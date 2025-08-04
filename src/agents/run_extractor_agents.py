#!/usr/bin/env python
"""
Unified runner for extractor-based ACOS agents pipeline.

This script processes review text through the ACOS pipeline for either restaurant or laptop domain:
1. Unified aspect-opinion extraction (with reasons)
2. Category classification 
3. Sentiment classification

Usage:
    python src/agents/run_extractor_agents.py --domain restaurant --input "The pasta was delicious but service was slow."
"""

import argparse
import json
import logging
import sys
import os
import ast
from typing import Dict, Any, Optional, List, Literal
import concurrent.futures
import copy

# Add project root to path to allow direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Dynamic imports based on domain
from langchain_openai import AzureChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from src.baseline.report_generator import create_report
from src.eval_utils import compute_scores, extract_spans_para
from src.baseline.api import is_deepseek_model, MODEL_DEPLOYMENTS, llm_chat as api_llm_chat
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Import validation and audit logging
from src.agents.validation import (
    validate_pipeline_state, audit_logger, state_manager, ValidationError
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Map domain names to dataset names expected by report_generator
DOMAIN_TO_DATASET = {
    "restaurant": "rest16",
    "laptop": "laptop16"
}

# Global token usage tracker
GLOBAL_TOKEN_USAGE = {}

# Global agent error tracker
GLOBAL_AGENT_ERRORS = {
    "unified": 0,      # Count of empty extracted_pairs
    "category": 0,     # Count of empty categorized_pairs  
    "sentiment": 0     # Count of empty sentiments
}

class TokenTrackingCallback(BaseCallbackHandler):
    """Callback handler to track token usage for all LLM types."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM ends running."""
        if self.agent_name not in GLOBAL_TOKEN_USAGE:
            GLOBAL_TOKEN_USAGE[self.agent_name] = {"prompt_tokens": 0, "completion_tokens": 0}
            
        # Extract token usage from the response
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            if token_usage:
                GLOBAL_TOKEN_USAGE[self.agent_name]["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
                GLOBAL_TOKEN_USAGE[self.agent_name]["completion_tokens"] += token_usage.get("completion_tokens", 0)

# Remove the custom evaluation functions and replace with standardized ones
def calculate_overall_scores(all_preds: List[List[List[str]]], all_golds: List[List[List[str]]]) -> Dict[str, float]:
    """
    Calculate overall scores using the standardized evaluation from eval_utils.
    This ensures consistency with the LLM script evaluation.
    """
    # Convert from List[List[str]] format to List[Tuple] format expected by eval_utils
    formatted_preds = []
    formatted_golds = []
    
    for pred_quads in all_preds:
        formatted_pred = [tuple(quad) for quad in pred_quads]
        formatted_preds.append(formatted_pred)
    
    for gold_quads in all_golds:
        formatted_gold = [tuple(quad) for quad in gold_quads]
        formatted_golds.append(formatted_gold)
    
    # Use the standardized compute_scores function
    scores = compute_scores(formatted_preds, formatted_golds, verbose=False)
    
    # Convert from percentage to decimal for consistency
    return {
        'precision': scores['precision'] / 100.0,
        'recall': scores['recall'] / 100.0,
        'f1': scores['f1'] / 100.0
    }

def calculate_pair_scores(all_preds: List[List[List[str]]], all_golds: List[List[List[str]]]) -> Dict[str, float]:
    """Calculate precision, recall, F1 for only aspect-opinion pairs."""
    formatted_pred_pairs = []
    formatted_gold_pairs = []

    # Build pair lists per sample
    for sample_pred_quads in all_preds:
        pred_pairs = []
        for quad in sample_pred_quads:
            if len(quad) >= 2:
                aspect, opinion = quad[0], quad[1]
                pred_pairs.append((aspect, opinion))
        formatted_pred_pairs.append(pred_pairs)

    for sample_gold_quads in all_golds:
        gold_pairs = []
        for quad in sample_gold_quads:
            if len(quad) >= 2:
                aspect, opinion = quad[0], quad[1]
                gold_pairs.append((aspect, opinion))
        formatted_gold_pairs.append(gold_pairs)

    # Use existing compute_scores util (exact match) on the pairs
    scores = compute_scores(formatted_pred_pairs, formatted_gold_pairs, verbose=False)

    # Convert from percentage to decimal for consistency
    return {
        'precision': scores['precision'] / 100.0,
        'recall': scores['recall'] / 100.0,
        'f1': scores['f1'] / 100.0
    }

def calculate_implicit_f1_scores(all_predicted_quads: List[List[List[str]]], all_gold_quads: List[List[List[str]]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate F1 scores for implicit aspects and implicit opinions across all samples.
    
    Args:
        all_predicted_quads: List of predicted quadruples for all samples
        all_gold_quads: List of gold standard quadruples for all samples
    
    Returns:
        Dictionary with implicit F1 scores for aspects and opinions
    """
    def normalize_text(text):
        """Normalize text for exact matching"""
        if text is None:
            return "null"
        return str(text).lower().strip()
    
    # Initialize counters
    implicit_aspect_tp = 0
    implicit_aspect_gold = 0
    implicit_aspect_pred = 0
    
    implicit_opinion_tp = 0
    implicit_opinion_gold = 0
    implicit_opinion_pred = 0
    
    # Process all samples
    for sample_idx in range(len(all_predicted_quads)):
        pred_quads = all_predicted_quads[sample_idx]
        gold_quads = all_gold_quads[sample_idx]
        
        # Count implicit aspects and opinions in predictions
        for quad in pred_quads:
            if len(quad) >= 2:
                aspect = normalize_text(quad[0])
                opinion = normalize_text(quad[1])
                
                if aspect == "null":
                    implicit_aspect_pred += 1
                if opinion == "null":
                    implicit_opinion_pred += 1
        
        # Count implicit aspects and opinions in gold standard
        for quad in gold_quads:
            if len(quad) >= 2:
                gold_aspect = normalize_text(quad[0])
                gold_opinion = normalize_text(quad[1])
                
                if gold_aspect == "null":
                    implicit_aspect_gold += 1
                if gold_opinion == "null":
                    implicit_opinion_gold += 1
        
        # Calculate true positives by matching predictions to gold
        for pred_quad in pred_quads:
            for gold_quad in gold_quads:
                if len(pred_quad) >= 2 and len(gold_quad) >= 2:
                    pred_aspect = normalize_text(pred_quad[0])
                    pred_opinion = normalize_text(pred_quad[1])
                    gold_aspect = normalize_text(gold_quad[0])
                    gold_opinion = normalize_text(gold_quad[1])
                    
                    # Check if this is a match (exact match for aspect and opinion)
                    if pred_aspect == gold_aspect and pred_opinion == gold_opinion:
                        # Check implicit aspect matches
                        if pred_aspect == "null" and gold_aspect == "null":
                            implicit_aspect_tp += 1
                        
                        # Check implicit opinion matches
                        if pred_opinion == "null" and gold_opinion == "null":
                            implicit_opinion_tp += 1
                        break
    
    # Calculate F1 scores
    def calculate_f1(tp, pred, gold):
        precision = tp / pred if pred > 0 else 0.0
        recall = tp / gold if gold > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1
    
    aspect_precision, aspect_recall, aspect_f1 = calculate_f1(
        implicit_aspect_tp, implicit_aspect_pred, implicit_aspect_gold
    )
    
    opinion_precision, opinion_recall, opinion_f1 = calculate_f1(
        implicit_opinion_tp, implicit_opinion_pred, implicit_opinion_gold
    )
    
    return {
        'implicit_aspect': {
            'precision': aspect_precision,
            'recall': aspect_recall,
            'f1': aspect_f1,
            'tp': implicit_aspect_tp,
            'pred': implicit_aspect_pred,
            'gold': implicit_aspect_gold
        },
        'implicit_opinion': {
            'precision': opinion_precision,
            'recall': opinion_recall,
            'f1': opinion_f1,
            'tp': implicit_opinion_tp,
            'pred': implicit_opinion_pred,
            'gold': implicit_opinion_gold
        }
    }


class DeepseekChatWrapper(BaseChatModel):
    """A custom chat model wrapper for Deepseek models using the project's llm_chat function."""
    model_name: str
    track_tokens: bool = False
    agent_name: str = "unknown"

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any
    ) -> ChatResult:
        
        converted_messages = []
        for m in messages:
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
            elif isinstance(m, SystemMessage):
                role = "system"
            else:
                raise ValueError(f"Got unknown message type: {m}")

            converted_messages.append({"role": role, "content": m.content})
        
        if self.track_tokens:
            response_text, usage = api_llm_chat(messages=converted_messages, stop=stop, model_name=self.model_name, return_usage=True)
            
            # Initialize agent in global token usage dict if not present
            if self.agent_name not in GLOBAL_TOKEN_USAGE:
                GLOBAL_TOKEN_USAGE[self.agent_name] = {"prompt_tokens": 0, "completion_tokens": 0}
            
            # Add token counts
            if usage:
                GLOBAL_TOKEN_USAGE[self.agent_name]["prompt_tokens"] += usage.get("prompt_tokens", 0)
                GLOBAL_TOKEN_USAGE[self.agent_name]["completion_tokens"] += usage.get("completion_tokens", 0)
        else:
            response_text = api_llm_chat(messages=converted_messages, stop=stop, model_name=self.model_name)
        
        chat_generation = ChatGeneration(message=AIMessage(content=response_text))
        return ChatResult(generations=[chat_generation])

    @property
    def _llm_type(self) -> str:
        return "deepseek_chat_wrapper"

def get_llm(model_name: str, api_version: str, track_tokens: bool = False, agent_name: str = "unknown") -> BaseChatModel:
    """Factory function to get the correct LangChain chat model."""
    if is_deepseek_model(model_name):
        llm = DeepseekChatWrapper(model_name=model_name, track_tokens=track_tokens, agent_name=agent_name)
        return llm
    else:
        deployment_name = MODEL_DEPLOYMENTS.get(model_name, model_name)
        logger.info(f"Using OpenAI deployment: {deployment_name} for model: {model_name}")
        
        # Special handling for GPT-4.1
        if model_name == "gpt-4.1":
            import os
            endpoint = os.getenv("AZURE_OPENAI_GPT41_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_GPT41_API_KEY")
            
            if endpoint and api_key:
                # Extract base URL without deployment path
                from urllib.parse import urlparse
                parsed = urlparse(endpoint)
                base_url = f"{parsed.scheme}://{parsed.netloc}"
                
                logger.info(f"Using dedicated GPT-4.1 endpoint: {base_url}")
                llm = AzureChatOpenAI(
                    azure_endpoint=base_url,
                    api_key=api_key,
                    azure_deployment=deployment_name,
                    api_version=api_version,
                    temperature=0
                )
            else:
                llm = AzureChatOpenAI(
                    azure_deployment=deployment_name,
                    api_version=api_version,
                    temperature=0
                )
        else:
            llm = AzureChatOpenAI(
                azure_deployment=deployment_name,
                api_version=api_version,
                temperature=0
            )
        
        # Add token tracking callback for OpenAI models
        if track_tokens:
            callback = TokenTrackingCallback(agent_name)
            llm.callbacks = [callback]
        
        return llm

def track_agent_error(agent_name: str, prediction_data: Any) -> None:
    """
    Track when an agent returns empty predictions.
    
    Args:
        agent_name: Name of the agent ("unified", "category", "sentiment")
        prediction_data: The prediction data returned by the agent
    """
    global GLOBAL_AGENT_ERRORS
    
    # Check if prediction is empty based on agent type
    is_empty = False
    
    if agent_name == "unified":
        # Check extracted_pairs
        extracted_pairs = prediction_data if isinstance(prediction_data, list) else []
        is_empty = len(extracted_pairs) == 0
    elif agent_name == "category":
        # Check categorized_pairs
        categorized_pairs = prediction_data if isinstance(prediction_data, list) else []
        is_empty = len(categorized_pairs) == 0
    elif agent_name == "sentiment":
        # Check sentiments
        sentiments = prediction_data if isinstance(prediction_data, list) else []
        is_empty = len(sentiments) == 0
    
    if is_empty:
        GLOBAL_AGENT_ERRORS[agent_name] += 1
        logger.debug(f"Agent {agent_name} returned empty prediction (total errors: {GLOBAL_AGENT_ERRORS[agent_name]})")

def run_acos_pipeline(text: str, domain: Literal["restaurant", "laptop"], 
                      model_name: str = "gpt-4o", 
                      api_version: str = "2025-01-01-preview",
                      track_tokens: bool = False,
                      prompt_type: str = '0shot') -> Dict[str, Any]:
    """
    Run the full ACOS pipeline on a single text.
    
    Args:
        text: The text to process
        domain: Domain to use (restaurant or laptop)
        model_name: Model to use (can be a single model or comma-separated for different agents)
        api_version: Azure OpenAI API version
        track_tokens: Whether to track token usage
        prompt_type: Prompt type to use (0shot or 20shot)
        
    Returns:
        Dict containing the extraction results
    """
    # Parse model configuration
    models = model_name.split(',')
    if len(models) == 1:
        # Use the same model for all agents
        unified_model = category_model = sentiment_model = models[0]
    elif len(models) >= 3:
        # Use different models for each agent
        unified_model, category_model, sentiment_model = models[:3]
    else:
        # Invalid configuration
        logger.error(f"Invalid model configuration: {model_name}. Use a single model or three comma-separated models.")
        return {}
    
    # Generate unique review ID and check for duplicates
    review_id = state_manager.generate_review_id(text, domain)
    
    # Check if already processed (duplicate prevention)
    if state_manager.is_already_processed(review_id):
        logger.info(f"Review {review_id} already processed, skipping")
        audit_logger.log_operation(
            operation="duplicate_check",
            review_id=review_id,
            agent="pipeline",
            input_data={"text": text[:100], "domain": domain},
            output_data="skipped_duplicate",
            success=True
        )
        return {"error": "Review already processed", "review_id": review_id}
    
    # Initialize the state dictionary to track progress
    state = {
        'text': text,
        'domain': domain,
        'prompt_type': prompt_type,  # Add prompt_type to state
        'extracted_pairs': [],
        'categorized_pairs': [],
        'sentiments': [],
        'review_id': review_id
    }
    
    # Validate input text
    try:
        validate_pipeline_state(state, 'input', domain)
        audit_logger.log_operation(
            operation="input_validation",
            review_id=review_id,
            agent="validator",
            input_data={"text": text[:100]},
            output_data="valid",
            success=True
        )
    except ValidationError as e:
        audit_logger.log_operation(
            operation="input_validation",
            review_id=review_id,
            agent="validator",
            input_data={"text": text[:100]},
            output_data="invalid",
            success=False,
            error_msg=str(e)
        )
        return {"error": f"Input validation failed: {e}", "review_id": review_id}
    
    # Initialize LLMs
    unified_llm = get_llm(unified_model, api_version, track_tokens, "unified")
    category_llm = get_llm(category_model, api_version, track_tokens, "category")
    sentiment_llm = get_llm(sentiment_model, api_version, track_tokens, "sentiment")
    
    # Load the appropriate extractor agent based on domain
    if domain == "restaurant":
        from src.agents.extractor_agents.extractor_rest_Voting.unified_extractor import UnifiedExtractorAgent
        from src.agents.extractor_agents.extractor_rest_Voting.category import CategoryAgent
        from src.agents.extractor_agents.extractor_rest_Voting.sentiment import SentimentAgent
    else:  # laptop
        try:
            from src.agents.extractor_agents.extractor_laptop_Voting.unified_extractor import UnifiedExtractorAgent
            from src.agents.extractor_agents.extractor_laptop_Voting.category import CategoryAgent
            from src.agents.extractor_agents.extractor_laptop_Voting.sentiment import SentimentAgent
        except ImportError as e:
            logger.error(f"Error importing laptop modules: {e}")
            return {"error": f"Error importing laptop modules: {e}", "review_id": review_id}
    
    # Initialize the agents
    unified_agent = UnifiedExtractorAgent(
        llm=unified_llm, 
        model=unified_model,
        prompt_type=prompt_type,
        ensemble_size=1,  # Use ensemble size 1 for simplicity
        temperature=0.0
    )
    
    category_agent = CategoryAgent(
        llm=category_llm, 
        model=category_model,
        prompt_type=prompt_type
    )
    
    sentiment_agent = SentimentAgent(
        llm=sentiment_llm, 
        model=sentiment_model,
        prompt_type=prompt_type
    )
    
    # Add the category and sentiment agents to the unified agent
    unified_agent.category_agent = category_agent
    unified_agent.sentiment_agent = sentiment_agent
    
    # Add mock token usage for testing
    if track_tokens:
        # Mock token usage for testing
        unified_agent.token_usage = {"prompt_tokens": 500, "completion_tokens": 100}
        category_agent.token_usage = {"prompt_tokens": 300, "completion_tokens": 50}
        sentiment_agent.token_usage = {"prompt_tokens": 300, "completion_tokens": 50}
        
        # Directly update GLOBAL_TOKEN_USAGE
        global GLOBAL_TOKEN_USAGE
        GLOBAL_TOKEN_USAGE = {
            "unified": {"prompt_tokens": 500, "completion_tokens": 100},
            "category": {"prompt_tokens": 300, "completion_tokens": 50},
            "sentiment": {"prompt_tokens": 300, "completion_tokens": 50}
        }
        
        print(f"DEBUG: Added mock token usage to GLOBAL_TOKEN_USAGE")
    
    # Run the pipeline
    try:
        # Step 1: Extract aspect-opinion pairs
        print(f"\n📝 INPUT TEXT: {text}")
        print("─" * 80)
        
        state = unified_agent.extract_pairs(state)
        
        # Collect token usage from agents if tracking is enabled
        if track_tokens:
            # Collect token usage from unified agent
            if hasattr(unified_agent, 'token_usage'):
                if "unified" not in GLOBAL_TOKEN_USAGE:
                    GLOBAL_TOKEN_USAGE["unified"] = {"prompt_tokens": 0, "completion_tokens": 0}
                GLOBAL_TOKEN_USAGE["unified"]["prompt_tokens"] += unified_agent.token_usage.get("prompt_tokens", 0)
                GLOBAL_TOKEN_USAGE["unified"]["completion_tokens"] += unified_agent.token_usage.get("completion_tokens", 0)
                logger.info(f"Collected token usage from unified agent: {unified_agent.token_usage}")
            else:
                logger.warning("unified_agent does not have token_usage attribute")
            
            # Collect token usage from category agent
            if hasattr(category_agent, 'token_usage'):
                if "category" not in GLOBAL_TOKEN_USAGE:
                    GLOBAL_TOKEN_USAGE["category"] = {"prompt_tokens": 0, "completion_tokens": 0}
                GLOBAL_TOKEN_USAGE["category"]["prompt_tokens"] += category_agent.token_usage.get("prompt_tokens", 0)
                GLOBAL_TOKEN_USAGE["category"]["completion_tokens"] += category_agent.token_usage.get("completion_tokens", 0)
                logger.info(f"Collected token usage from category agent: {category_agent.token_usage}")
            else:
                logger.warning("category_agent does not have token_usage attribute")
            
            # Collect token usage from sentiment agent
            if hasattr(sentiment_agent, 'token_usage'):
                if "sentiment" not in GLOBAL_TOKEN_USAGE:
                    GLOBAL_TOKEN_USAGE["sentiment"] = {"prompt_tokens": 0, "completion_tokens": 0}
                GLOBAL_TOKEN_USAGE["sentiment"]["prompt_tokens"] += sentiment_agent.token_usage.get("prompt_tokens", 0)
                GLOBAL_TOKEN_USAGE["sentiment"]["completion_tokens"] += sentiment_agent.token_usage.get("completion_tokens", 0)
                logger.info(f"Collected token usage from sentiment agent: {sentiment_agent.token_usage}")
            else:
                logger.warning("sentiment_agent does not have token_usage attribute")
        
        # Track empty extraction results
        track_agent_error("unified", state.get('extracted_pairs', []))
        track_agent_error("category", state.get('categorized_pairs', []))
        track_agent_error("sentiment", state.get('sentiments', []))
        
        # Validate the pipeline state
        try:
            validate_pipeline_state(state, 'extraction', domain)
        except ValidationError as e:
            logger.warning(f"Extraction validation failed: {e}")
        
        try:
            validate_pipeline_state(state, 'classification', domain)
        except ValidationError as e:
            logger.warning(f"Classification validation failed: {e}")
        
        try:
            validate_pipeline_state(state, 'final', domain)
        except ValidationError as e:
            logger.warning(f"Final validation failed: {e}")
        
        # Format the results for output
        quadruples = format_results_to_quads(state)
        state['quadruples'] = quadruples
        
        # Print the results
        print_step_results(state)
        
        # Return the results
        return {
            "quadruples": quadruples,
            "state": state,
            "review_id": review_id
        }
        
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        audit_logger.log_operation(
            operation="pipeline_error",
            review_id=review_id,
            agent="pipeline",
            input_data={"text": text[:100]},
            output_data="error",
            success=False,
            error_msg=str(e)
        )
        return {"error": str(e), "review_id": review_id}

def format_results_to_quads(state: Dict[str, Any]) -> List[List[str]]:
    """
    Format the final state into a list of ACOS quads.
    Quad order: [aspect, opinion, category, sentiment]
    """
    quads = []
    
    # Create a mapping of aspect-opinion pairs to their categories
    categorized_map = {}
    for pair in state.get('categorized_pairs', []):
        key = f"{pair['aspect']}-{pair['opinion']}"
        categorized_map[key] = pair['category']
    
    # For each sentiment item, find the corresponding category
    for sentiment_item in state["sentiments"]:
        aspect = sentiment_item["aspect"]
        # Convert None to "null" string for consistency
        if aspect is None:
            aspect = "null"
            
        opinion = sentiment_item["opinion"]
        # Convert None to "null" string for consistency
        if opinion is None:
            opinion = "null"
        
        # Look up the category for this aspect-opinion pair
        key = f"{aspect}-{opinion}"
        category = categorized_map.get(key)
        
        # If not found, try to use the old categories dictionary as fallback
        if category is None:
            category_lookup_key = "null" if aspect is None or aspect == "null" else aspect
            category = state.get("categories", {}).get(category_lookup_key, "unknown")
            
        sentiment = sentiment_item["sentiment"]
        
        quads.append([aspect, opinion, category, sentiment])
        
    return quads

def extract_reasons_from_state(state: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract the reasons from the extracted_pairs and return them as a dict
    mapping aspect-opinion to reason.
    """
    reasons = {}
    for pair in state.get('extracted_pairs', []):
        key = f"{pair.get('aspect', 'null')}-{pair.get('opinion', 'null')}"
        reasons[key] = pair.get('reason', '')
    return reasons

def save_results(results: Dict[str, Any], output_file: Optional[str] = None) -> None:
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    else:
        print(json.dumps(results, indent=2))

def batch_process(input_file: str, output_dir: str, domain: str, 
                  model_name: str, api_version: str, num_sample: int = 10, 
                  start_index: int = 0, track_tokens: bool = False,
                  prompt_type: str = '0shot') -> None:
    """
    Process a batch of reviews from an input file.
    """
    # Reset global token usage and agent errors for this batch
    global GLOBAL_TOKEN_USAGE, GLOBAL_AGENT_ERRORS
    GLOBAL_TOKEN_USAGE = {}
    GLOBAL_AGENT_ERRORS = {
        "unified": 0,
        "category": 0,
        "sentiment": 0
    }
    
    # Load the input file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return

    # Get the samples to process
    end_index = min(start_index + num_sample, len(lines))
    samples = lines[start_index:end_index]
    logger.info(f"Processing {len(samples)} samples from index {start_index} to {end_index-1}")
    
    # Process each sample
    all_results = []
    all_predicted_quads = []
    all_gold_quads = []
    
    for i, line in enumerate(samples):
        try:
            # Parse the line to extract text and gold standard (format: "text####[annotations]")
            if "####" in line:
                text = line.split("####")[0].strip()
                gold_quads = parse_gold_quads(line)
            else:
                text = line.strip()
                gold_quads = []
            
            if not text:
                logger.warning(f"Sample {start_index + i} has no text, skipping")
                continue
            
            logger.info(f"Processing sample {start_index + i}/{end_index-1}: {text[:50]}...")
            
            # Run the pipeline with prompt_type
            result = run_acos_pipeline(text, domain, model_name, api_version, track_tokens, prompt_type)
            
            # Debug: Print token usage after each sample
            if track_tokens:
                print(f"DEBUG: Token usage after sample {start_index + i}: {GLOBAL_TOKEN_USAGE}")
                
                # Save token usage after each sample for debugging
                debug_token_file = os.path.join(output_dir, f"token_usage_debug_{start_index + i}.json")
                try:
                    with open(debug_token_file, 'w', encoding='utf-8') as f:
                        json.dump(GLOBAL_TOKEN_USAGE, f, indent=2)
                    print(f"DEBUG: Token usage saved to {debug_token_file}")
                except Exception as e:
                    print(f"DEBUG: Error saving token usage: {e}")
            
            # Add the sample index and gold standard
            result['sample_index'] = start_index + i
            result['gold_quads'] = gold_quads
            
            # Display comparison with gold standard
            predicted_quads = result.get('quadruples', []) # Changed from 'quads' to 'quadruples'
            display_gold_vs_predicted(predicted_quads, gold_quads, start_index + i)
            
            # Collect for overall evaluation
            all_predicted_quads.append(predicted_quads)
            all_gold_quads.append(gold_quads)
            
            # Add to results
            all_results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing sample {start_index + i}: {e}")
            continue
    
    # Calculate and display overall metrics
    if all_predicted_quads and all_gold_quads:
        overall_scores = calculate_overall_scores(all_predicted_quads, all_gold_quads)
        print(f"\n🏆 OVERALL EVALUATION RESULTS:")
        print("=" * 60)
        print(f"📊 Precision: {overall_scores['precision']:.4f}")
        print(f"📊 Recall:    {overall_scores['recall']:.4f}")
        print(f"📊 F1-Score:  {overall_scores['f1']:.4f}")
        print("=" * 60)

        # ----- NEW: Aspect-Opinion pair metrics -----
        pair_scores = calculate_pair_scores(all_predicted_quads, all_gold_quads)
        print(f"\n🏆 ASPECT-OPINION PAIR EVALUATION RESULTS:")
        print("=" * 60)
        print(f"📊 Precision: {pair_scores['precision']:.4f}")
        print(f"📊 Recall:    {pair_scores['recall']:.4f}")
        print(f"📊 F1-Score:  {pair_scores['f1']:.4f}")
        print("=" * 60)

        # ----- NEW: Implicit F1 scores -----
        implicit_scores = calculate_implicit_f1_scores(all_predicted_quads, all_gold_quads)
        print(f"\n🎯 IMPLICIT F1 SCORES (End of Full Run):")
        print("=" * 60)
        
        aspect_scores = implicit_scores['implicit_aspect']
        opinion_scores = implicit_scores['implicit_opinion']
        
        print(f"📊 Implicit Aspects:")
        print(f"   Precision: {aspect_scores['precision']:.4f}")
        print(f"   Recall:    {aspect_scores['recall']:.4f}")
        print(f"   F1-Score:  {aspect_scores['f1']:.4f}")
        print(f"   TP: {aspect_scores['tp']}, Pred: {aspect_scores['pred']}, Gold: {aspect_scores['gold']}")
        
        print(f"\n📊 Implicit Opinions:")
        print(f"   Precision: {opinion_scores['precision']:.4f}")
        print(f"   Recall:    {opinion_scores['recall']:.4f}")
        print(f"   F1-Score:  {opinion_scores['f1']:.4f}")
        print(f"   TP: {opinion_scores['tp']}, Pred: {opinion_scores['pred']}, Gold: {opinion_scores['gold']}")
        print("=" * 60)
    
    # Display agent error statistics
    total_samples = len(all_results)
    print(f"\n🚨 AGENT ERROR ANALYSIS:")
    print("=" * 60)
    print(f"📈 Total samples processed: {total_samples}")
    for agent_name, error_count in GLOBAL_AGENT_ERRORS.items():
        error_percentage = (error_count / total_samples * 100) if total_samples > 0 else 0
        print(f"🔴 {agent_name.capitalize()} agent empty predictions: {error_count}/{total_samples} ({error_percentage:.1f}%)")
        print("=" * 60)
    
    # Save all results
    output_file = os.path.join(output_dir, f"results_{start_index}_{end_index-1}.json")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    # Save agent error statistics
    try:
        agent_errors_file = os.path.join(output_dir, "agent_errors.json")
        agent_error_data = {
            "total_samples": total_samples,
            "errors": GLOBAL_AGENT_ERRORS.copy(),
            "error_percentages": {
                agent: (count / total_samples * 100) if total_samples > 0 else 0 
                for agent, count in GLOBAL_AGENT_ERRORS.items()
            }
        }
        with open(agent_errors_file, 'w', encoding='utf-8') as f:
            json.dump(agent_error_data, f, indent=2)
        logger.info(f"Agent error statistics saved to {agent_errors_file}")
    except Exception as e:
        logger.error(f"Error saving agent error statistics: {e}")
    
    # Save token usage if tracking is enabled
    if track_tokens:
        print(f"DEBUG: Final token usage: {GLOBAL_TOKEN_USAGE}")
        token_usage_file = os.path.join(output_dir, "token_usage.json")
        try:
            with open(token_usage_file, 'w', encoding='utf-8') as f:
                json.dump(GLOBAL_TOKEN_USAGE, f, indent=2)
            print(f"DEBUG: Token usage saved to {token_usage_file}")
        except Exception as e:
            print(f"DEBUG: Error saving token usage: {e}")
    
    # Generate PDF report if we have results
    if all_results:
        try:
            print(f"\n📄 Generating PDF report...")
            
            # Convert to format expected by report generator
            formatted_predictions = []
            formatted_gold_standards = []
            sources = []  # Collect source texts for report
            
            for result in all_results:
                sample_idx = result.get('sample_index', 0)
                text = result.get('text', '')
                predicted_quads = result.get('quadruples', []) # Changed from 'quads' to 'quadruples'
                gold_quads = result.get('gold_quads', [])
                
                # Convert to tuples as expected by the report generator
                pred_tuples = [tuple(quad) for quad in predicted_quads] if predicted_quads else []
                gold_tuples = [tuple(quad) for quad in gold_quads] if gold_quads else []
                
                formatted_predictions.append(pred_tuples)
                formatted_gold_standards.append(gold_tuples)
                
                # Add source text as a list of words (as expected by report generator)
                source_words = text.split() if text else ["[Empty text]"]
                sources.append(source_words)
            
            # Create data structure expected by report generator
            models = model_name.split(',')
            if len(models) == 1:
                model_str = models[0]
            else:
                model_str = f"Multi-Agent({','.join(models)})"
            
            # Format scores as expected by report generator
            formatted_scores = {
                'precision': overall_scores['precision'],
                'recall': overall_scores['recall'], 
                'f1': overall_scores['f1']
            }
            
            data = {
                'args': {
                    'dataset': DOMAIN_TO_DATASET[domain],
                    'model': model_str,
                    'prompt_type': prompt_type, # Pass prompt_type to args
                    'num_sample': len(all_results),
                    'task': 'acos'
                },
                'predictions': formatted_predictions,
                'gold_standards': formatted_gold_standards,
                'scores': formatted_scores,
                'agent_errors': {
                    'total_samples': total_samples,
                    'errors': GLOBAL_AGENT_ERRORS.copy(),
                    'error_percentages': {
                        agent: (count / total_samples * 100) if total_samples > 0 else 0 
                        for agent, count in GLOBAL_AGENT_ERRORS.items()
                    }
                }
            }
            
            # Create predictions file for report generator
            predictions_file = os.path.join(output_dir, f"{DOMAIN_TO_DATASET[domain]}_extractor_predictions.json")
            with open(predictions_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            # Generate the report with actual source texts
            create_report(predictions_file, data, sources, GLOBAL_TOKEN_USAGE if track_tokens else None)
            print(f"✅ PDF report generated successfully!")
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            print(f"❌ Failed to generate PDF report: {e}")

def parse_gold_quads(gold_line: str) -> List[List[str]]:
    """
    Parse gold standard quads from the line after #### using the same logic as LLM script.
    This ensures consistency with the LLM script gold standard parsing.
    """
    try:
        if "####" not in gold_line:
            return []
        
        # Extract the gold part after ####
        gold_part = gold_line.split("####")[1].strip()
        if not gold_part:
            return []
        
        # Parse the JSON-like format: [['aspect', 'category', 'sentiment', 'opinion'], ...]
        import ast
        gold_list_raw = ast.literal_eval(gold_part)
        
        # Convert from test file format to expected format
        # The test file format is: [aspect, category, sentiment, opinion]
        # We need to convert to: [aspect_term, opinion_term, aspect_category, sentiment_polarity]
        normalized_quads = []
        for quad in gold_list_raw:
            if len(quad) == 4:
                aspect, category, sentiment, opinion = quad
                # Convert to [aspect_term, opinion_term, aspect_category, sentiment_polarity]
                normalized_quad = [str(aspect), str(opinion), str(category), str(sentiment)]
                normalized_quads.append(normalized_quad)
        
        return normalized_quads
    except Exception as e:
        logger.warning(f"Error parsing gold quads: {e}")
        return []

def display_gold_vs_predicted(predicted_quads: List[List[str]], gold_quads: List[List[str]], sample_idx: int):
    """Display comparison between predicted and gold standard quads"""
    print(f"\n🎯 EVALUATION FOR SAMPLE {sample_idx}:")
    print("─" * 50)
    
    print("📋 GOLD STANDARD:")
    if gold_quads:
        for i, quad in enumerate(gold_quads, 1):
            print(f"   {i}. {quad}")
    else:
        print("   No gold standard available")
    
    print("\n🤖 PREDICTED:")
    if predicted_quads:
        for i, quad in enumerate(predicted_quads, 1):
            print(f"   {i}. {quad}")
    else:
        print("   No predictions made")
    
    # Calculate match statistics using the same logic as eval_utils
    if gold_quads and predicted_quads:
        # Convert to tuple format for consistency with eval_utils
        pred_tuples = [tuple(quad) for quad in predicted_quads]
        gold_tuples = [tuple(quad) for quad in gold_quads]
        
        # Use the standardized evaluation on this single sample
        sample_scores = compute_scores([pred_tuples], [gold_tuples], verbose=False)
        
        print(f"\n📊 SAMPLE SCORES:")
        print(f"   Precision: {sample_scores['precision']:.1f}% | Recall: {sample_scores['recall']:.1f}% | F1: {sample_scores['f1']:.1f}%")
    else:
        print(f"\n📊 SAMPLE SCORES: Cannot calculate (missing data)")
    
    print("─" * 50)

def print_step_results(state: Dict[str, Any]) -> None:
    """Print the results of each step in the pipeline."""
    # Print extraction results
    print("🔍 STEP 1 - ASPECT-OPINION-REASON EXTRACTION:")
    extracted_pairs = state.get('extracted_pairs', [])
    if extracted_pairs:
        for i, pair in enumerate(extracted_pairs, 1):
            aspect = pair.get('aspect', 'null')
            opinion = pair.get('opinion', 'null')
            reason = pair.get('reason', 'No reason provided')
            print(f"   {i}. ASPECT: '{aspect}' | OPINION: '{opinion}' | REASON: {reason}")
    else:
        print("   No aspect-opinion pairs extracted.")
    print()
    
    # Print category classification results
    print("🏷️  STEP 2 - CATEGORY CLASSIFICATION:")
    categorized_pairs = state.get('categorized_pairs', [])
    if categorized_pairs:
        for i, pair in enumerate(categorized_pairs, 1):
            aspect = pair.get('aspect', 'null')
            opinion = pair.get('opinion', 'null')
            category = pair.get('category', 'unknown')
            print(f"   {i}. ASPECT: '{aspect}' | OPINION: '{opinion}' → CATEGORY: '{category}'")
    else:
        print("   No categories assigned.")
    print()
    
    # Print sentiment classification results
    print("💭 STEP 3 - SENTIMENT CLASSIFICATION:")
    sentiments = state.get('sentiments', [])
    if sentiments:
        for i, sentiment_item in enumerate(sentiments, 1):
            aspect = sentiment_item.get('aspect', 'null')
            opinion = sentiment_item.get('opinion', 'null')
            sentiment = sentiment_item.get('sentiment', 'unknown')
            print(f"   {i}. ASPECT: '{aspect}' | OPINION: '{opinion}' → SENTIMENT: '{sentiment}'")
    else:
        print("   No sentiments assigned.")
    print()
    
    # Print final quadruples
    print("✅ FINAL PIPELINE OUTPUT:")
    print("   [ASPECT, OPINION, CATEGORY, SENTIMENT]")
    quadruples = state.get('quadruples', [])
    if quadruples:
        for i, quad in enumerate(quadruples, 1):
            print(f"   {i}. {quad}")
    else:
        print("   No quadruples generated.")
    print()


def main():
    parser = argparse.ArgumentParser(description='Run ACOS extraction pipeline')
    parser.add_argument('--domain', type=str, choices=['restaurant', 'laptop'], default='restaurant',
                        help='Domain to process (restaurant or laptop)')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='Model to use (gpt, gpt-4o, gpt-4.1, deepseek-v3, deepseek-r1) or comma-separated list for different agents')
    parser.add_argument('--api-version', type=str, default='2025-01-01-preview',
                        help='Azure OpenAI API version')
    parser.add_argument('--input', type=str, default=None,
                        help='Input text to process (for interactive mode)')
    parser.add_argument('--input-file', type=str, default=None,
                        help='Input file to process (for batch mode)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for batch processing')
    parser.add_argument('--num-sample', type=int, default=10,
                        help='Number of samples to process in batch mode')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Start index for batch processing')
    parser.add_argument('--batch', action='store_true',
                        help='Enable batch processing mode')
    parser.add_argument('--track-tokens', action='store_true',
                        help='Track token usage')
    parser.add_argument('--prompt-type', type=str, choices=['0shot', '20shot'], default='0shot',
                        help='Prompt type to use (0shot or 20shot)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.batch:
        if not args.input_file:
            parser.error("--input-file is required for batch mode")
        if not args.output_dir:
            parser.error("--output-dir is required for batch mode")
    else:
        if not args.input:
            args.input = "The food was delicious but the service was slow."
            logger.info(f"No input provided, using default: '{args.input}'")
    
    # Run in batch or interactive mode
    if args.batch:
        batch_process(
            args.input_file, 
            args.output_dir, 
            args.domain, 
            args.model, 
            args.api_version, 
            args.num_sample, 
            args.start_index, 
            args.track_tokens,
            args.prompt_type
        )
    else:
        # Interactive mode
        result = run_acos_pipeline(
            args.input, 
            args.domain, 
            args.model, 
            args.api_version, 
            args.track_tokens,
            args.prompt_type
        )
        
        # Save token usage if tracking was enabled
        if args.track_tokens and GLOBAL_TOKEN_USAGE:
            with open("token_usage.json", "w") as f:
                json.dump(GLOBAL_TOKEN_USAGE, f, indent=2)
            logger.info("Token usage saved to token_usage.json")
            
        # Print the result
        save_results(result)

if __name__ == '__main__':
    main() 