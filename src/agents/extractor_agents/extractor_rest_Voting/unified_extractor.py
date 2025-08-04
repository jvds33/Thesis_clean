"""
Unified aspect-opinion extraction agent for restaurant reviews with Agent Forest voting.

Input state keys required:
• text: str

Output state keys added/overwritten:
• extracted_pairs: List[Dict[str, str]]  # List of {"aspect": "...", "opinion": "...", "reason": "..."}
"""

import logging
import os
import re
import difflib
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import copy
from .category import CategoryAgent
from .sentiment import SentimentAgent

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# Add project root to path to allow imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from src.baseline.api import llm_chat
from src.agents.validation import audit_logger, ValidationError

logger = logging.getLogger(__name__)

class ExtractionPair(BaseModel):
    aspect: Optional[str] = None
    opinion: Optional[str] = None
    reason: Optional[str] = None

class ExtractionResponse(BaseModel):
    extracted_pairs: List[ExtractionPair]

def display_voter_predictions(all_predictions: List[List[Dict[str, Any]]]):
    """Display the predictions from each voter."""
    print("\n📊 VOTER PREDICTIONS:")
    print("-" * 80)
    for i, pairs in enumerate(all_predictions):
        print(f"🗳️  Voter {i+1}:")
        if pairs:
            print(f"   Count: {len(pairs)} pairs")
            for j, pair in enumerate(pairs):
                print(f"   {j+1}. ({pair.get('aspect')}, {pair.get('opinion')})")
        else:
            print("   ❌ No valid predictions")
    print()

def count_pair_votes(all_predictions: List[List[Dict[str, Any]]]):
    """Count votes for each unique aspect-opinion pair."""
    pair_counts = Counter()
    original_pair_map = {}
    valid_predictions = []

    for voter_preds in all_predictions:
        # Filter out empty predictions
        if not voter_preds:
            continue
        
        valid_predictions.append(voter_preds)
        
        # Create a set of tuples for efficient counting
        seen_pairs_per_voter = set()
        for pair in voter_preds:
            aspect = pair.get('aspect', 'null')
            opinion = pair.get('opinion', 'null')
            
            # Normalize for counting
            key = (str(aspect).lower().strip(), str(opinion).lower().strip())
            
            # Store original casing and reasoning
            if key not in original_pair_map:
                original_pair_map[key] = {
                    'aspect': aspect,
                    'opinion': opinion,
                    'reasons': []
                }
            
            if pair.get('reason'):
                original_pair_map[key]['reasons'].append(pair.get('reason'))
            
            seen_pairs_per_voter.add(key)
            
        # Add to global counts
        for key in seen_pairs_per_voter:
            pair_counts[key] += 1
            
    return pair_counts, original_pair_map, valid_predictions

class UnifiedExtractorAgent:
    def __init__(self, llm=None, model="gpt-4o", prompt_type="fewshot", ensemble_size=1, temperature=0.0, parallel=True, category_agent: CategoryAgent = None, sentiment_agent: SentimentAgent = None):
        """
        Initialize the Unified Extractor agent.
        
        Args:
            llm: Language model instance.
            model: Name of the model to use.
            prompt_type: Type of prompt to use ('zeroshot' or 'fewshot').
            ensemble_size: Number of voters for ensemble extraction.
            temperature: Temperature for LLM sampling.
            parallel: Whether to run voters in parallel.
            category_agent: An instance of CategoryAgent.
            sentiment_agent: An instance of SentimentAgent.
        """
        self.llm = llm
        self.model_name = model.lower()
        self.prompt_type = prompt_type
        self.ensemble_size = ensemble_size
        self.temperature = temperature
        self.parallel = parallel
        self.category_agent = category_agent
        self.sentiment_agent = sentiment_agent
        
        # Set up output parser
        self.parser = PydanticOutputParser(pydantic_object=ExtractionResponse)
        
        # Load prompts
        self.prompt = self._load_prompts()
        self.retry_prompt = self._load_retry_prompts()
        
    def _load_prompts(self):
        """Load the appropriate prompts based on configuration."""
        try:
            # Get the directory of the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))

            if self.prompt_type in ["zeroshot", "0shot"]:
                prompt_file_name = "unified_extractor_prompt_0shot.txt"
            else:
                prompt_file_name = "unified_extractor_prompt.txt"

            prompt_file_path = os.path.join(current_dir, "prompts", prompt_file_name)
            
            with open(prompt_file_path, 'r', encoding='utf-8') as file:
                system_template = file.read()

            human_template = """Text: {text}

Extract all aspect-opinion pairs."""

            # Create the chat prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_template),
                ("human", human_template),
            ])

            return prompt
            
        except FileNotFoundError:
            logger.error(f"Prompt file not found at {prompt_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            raise

    def _load_retry_prompts(self):
        """Load the retry prompts for when extraction fails."""
        try:
            # Get the directory of the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))

            if self.prompt_type in ["zeroshot", "0shot"]:
                prompt_file_name = "unified_extractor_prompt_0shot.txt"
            else:
                prompt_file_name = "unified_extractor_prompt.txt"

            prompt_file_path = os.path.join(current_dir, "prompts", prompt_file_name)
            
            with open(prompt_file_path, 'r', encoding='utf-8') as file:
                base_system_template = file.read()

            # Add the retry-specific instruction to the base prompt
            system_template = base_system_template + "\n\nIMPORTANT: The previous extraction contained aspect/opinion terms that were not found in the original text. Please re-extract ensuring that aspect and opinion terms appear exactly as they are in the text."

            human_template = """Text: {text}

Extract all aspect-opinion pairs."""

            # Create the chat prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_template),
                ("human", human_template),
            ])

            return prompt
        
        except FileNotFoundError:
            logger.error(f"Prompt file not found at {prompt_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading retry prompts: {e}")
            raise

    def extract_pairs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract aspect-opinion pairs from the text in the state.
        This method orchestrates the full extraction and classification pipeline.
        """
        review_id = state.get('review_id', 'unknown')
        text = state.get('text')
        if not text:
            raise ValueError("Input state must contain a 'text' key")

        try:
            # Main path: extraction with voting
            all_predictions = self._extract_with_voting(state['text'])
            display_voter_predictions(all_predictions)
            
            # Count votes
            pair_votes, original_pair_map, valid_predictions = count_pair_votes(all_predictions)
            
            if not pair_votes:
                print("❌ No valid predictions from any voter!")
                raise ValueError("No valid predictions from any voter")
            
            if len(valid_predictions) == 1:
                winner = valid_predictions[0]
                print(f"🏆 VOTING RESULT: Only one valid prediction")
                print(f"   Winner: {len(winner)} pairs")
                state['extracted_pairs'] = winner
            else:
                # Apply fallback fixes only if ensemble_size > 1, otherwise return as is
                if self.ensemble_size > 1:
                    final_pairs = self._apply_fallback_fixes(valid_predictions, state['text'])
                else:
                    final_pairs = valid_predictions[0]
                
                state['extracted_pairs'] = final_pairs
            
            # Log successful extraction
            audit_logger.log_operation(
                operation="extraction_voting",
                review_id=review_id,
                agent="unified_extractor",
                input_data={"text": text[:100]},
                output_data={"pairs_count": len(state.get('extracted_pairs', []))},
                success=True
            )
            
        except Exception as e:
            logger.warning(f"Agent Forest voting failed, falling back to single extraction. Error: {e}")
            
            # Fallback to a single, non-voting extraction call
            try:
                # Use the same extraction logic as a single voter
                single_extraction_result = self._make_single_extraction(text, voter_id=0)
                state['extracted_pairs'] = single_extraction_result
                
                # Log fallback extraction
                audit_logger.log_operation(
                    operation="extraction_fallback",
                    review_id=review_id,
                    agent="unified_extractor",
                    input_data={"text": text[:100]},
                    output_data={"pairs_count": len(single_extraction_result)},
                    success=True,
                    metadata={"fallback_reason": str(e)}
                )
                
            except Exception as e2:
                logger.warning(f"Fallback extraction also failed, defaulting to empty list. Error: {e2}")
                state['extracted_pairs'] = []
                
                # Log extraction failure
                audit_logger.log_operation(
                    operation="extraction_error",
                    review_id=review_id,
                    agent="unified_extractor",
                    input_data={"text": text[:100]},
                    output_data="complete_failure",
                    success=False,
                    error_msg=str(e2)
                )

        # Add the extracted pairs to state for downstream agents
        state['all_pairs'] = []
        for pair in state.get('extracted_pairs', []):
            aspect = pair.get('aspect')
            opinion = pair.get('opinion')
            
            if aspect is None:
                aspect = "null"
            if opinion is None:
                opinion = "null"
                
            state['all_pairs'].append({
                'aspect': aspect,
                'opinion': opinion
            })

        # --- Start of Parallel Classification (Orchestrated by Extractor) ---
        if self.category_agent and self.sentiment_agent:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                state_for_classifiers = copy.deepcopy(state)
                
                future_category = executor.submit(self.category_agent.classify_category, state_for_classifiers)
                future_sentiment = executor.submit(self.sentiment_agent.classify_sentiment, state_for_classifiers)

                category_result_state = future_category.result()
                sentiment_result_state = future_sentiment.result()

            # Merge results back into the state
            state['categorized_pairs'] = category_result_state.get('categorized_pairs', [])
            state['sentiments'] = sentiment_result_state.get('sentiments', [])
            
            # Log classification results
            audit_logger.log_operation(
                operation="parallel_classification",
                review_id=review_id,
                agent="unified_extractor",
                input_data={"pairs_count": len(state.get('extracted_pairs', []))},
                output_data={
                    "categorized_count": len(state.get('categorized_pairs', [])),
                    "sentiment_count": len(state.get('sentiments', []))
                },
                success=True
            )
        # --- End of Parallel Classification ---

        return state

    def _extract_with_voting(self, text: str) -> List[List[Dict[str, Any]]]:
        """Run the extraction with voting."""
        if self.parallel:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._make_single_extraction, text, i) for i in range(self.ensemble_size)]
                return [future.result() for future in as_completed(futures)]
        else:
            return [self._make_single_extraction(text, i) for i in range(self.ensemble_size)]

    def _make_single_extraction(self, text: str, voter_id: int) -> List[Dict[str, Any]]:
        """Makes a single extraction call using the llms.api."""
        
        current_prompt = self.prompt
        
        for attempt in range(2): # Allow one retry
            try:
                # The llms.api functions handle client logic internally
                system_message = current_prompt.messages[0].prompt.template.format(format_instructions=self.parser.get_format_instructions())
                human_message = current_prompt.messages[1].prompt.template.format(text=text)

                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": human_message}
                ]
                
                # Use the working llm_chat function with token tracking
                raw_response, usage = llm_chat(messages, model_name=self.model_name, temperature=self.temperature, return_usage=True)
                
                # Store token usage for cost calculation
                if usage:
                    if not hasattr(self, 'token_usage'):
                        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
                    self.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    self.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                
                parsed_response = self.parser.parse(raw_response)

                # Validate that the extracted terms are in the text
                if not self._validate_pairs(parsed_response.extracted_pairs, text):
                    logger.warning(f"Voter {voter_id}, Attempt {attempt+1}: Validation failed, retrying...")
                    current_prompt = self.retry_prompt # Switch to retry prompt
                    if attempt == 0: continue # Continue to next attempt in the loop
                
                return [p.model_dump() for p in parsed_response.extracted_pairs]

            except Exception as e:
                logger.error(f"Voter {voter_id}: Failed to extract or parse on attempt {attempt+1}. Error: {e}")
                if attempt == 0:
                    current_prompt = self.retry_prompt
                    continue
        
        logger.warning(f"Voter {voter_id}: All extraction attempts failed, returning empty list.")
        return [] # Return empty list if all attempts fail
        
    def _validate_pairs(self, pairs: List[ExtractionPair], text: str) -> bool:
        """Validate that all extracted terms are present in the text."""
        for pair in pairs:
            if pair.aspect and not self._is_term_findable_in_text(pair.aspect, text):
                return False
            if pair.opinion and not self._is_term_findable_in_text(pair.opinion, text):
                return False
        return True

    def _is_term_findable_in_text(self, term, text):
        """
        Check if a term can be found in the text using case-insensitive regex.
        Returns True if term is null/NULL or if found in text.
        """
        if term is None or term.lower() in ["null", "NULL"]:
            return True
        
        # Regex to find the whole word, case-insensitive
        return bool(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE))

    def _find_most_similar_text(self, target_term, text):
        """Find the most similar substring in text to the target term."""
        if not target_term or not text:
            return None
        
        words = text.split()
        if not words:
            return None
        
        closest_match = difflib.get_close_matches(target_term, words, n=1, cutoff=0.8)
        
        return closest_match[0] if closest_match else None

    def _apply_fallback_fixes(self, pairs: List[ExtractionPair], original_text: str) -> List[ExtractionPair]:
        """Attempt to fix unfindable terms by finding the most similar text."""
        fixed_pairs = []
        for pair in pairs:
            fixed_aspect = pair.aspect
            fixed_opinion = pair.opinion
            
            if pair.aspect and not self._is_term_findable_in_text(pair.aspect, original_text):
                most_similar = self._find_most_similar_text(pair.aspect, original_text)
                if most_similar:
                    logger.info(f"Fallback Fix: Replaced aspect '{pair.aspect}' with most similar term '{most_similar}'")
                    fixed_aspect = most_similar
                    
            if pair.opinion and not self._is_term_findable_in_text(pair.opinion, original_text):
                most_similar = self._find_most_similar_text(pair.opinion, original_text)
                if most_similar:
                    logger.info(f"Fallback Fix: Replaced opinion '{pair.opinion}' with most similar term '{most_similar}'")
                    fixed_opinion = most_similar

            fixed_pairs.append(ExtractionPair(aspect=fixed_aspect, opinion=fixed_opinion, reason=pair.reason))
        
        return fixed_pairs

    def _needs_retry(self, pairs: List[ExtractionPair], original_text: str) -> bool:
        """Check if any extracted term is not findable in the original text."""
        for pair in pairs:
            # Check both aspect and opinion terms
            if pair.aspect and not self._is_term_findable_in_text(pair.aspect, original_text):
                logger.warning(f"Aspect '{pair.aspect}' not found in text. Triggering retry.")
                return True
            if pair.opinion and not self._is_term_findable_in_text(pair.opinion, original_text):
                logger.warning(f"Opinion '{pair.opinion}' not found in text. Triggering retry.")
                return True
        return False

    def _get_majority_pairs(self, all_pairs: List[List[ExtractionPair]], threshold: int) -> List[ExtractionPair]:
        """Get pairs that appear in at least threshold number of predictions."""
        # Convert pairs to tuples for counting
        pair_tuples = []
        for pairs in all_pairs:
            for pair in pairs:
                pair_tuples.append((
                    str(pair.aspect).lower().strip() if pair.aspect else None,
                    str(pair.opinion).lower().strip() if pair.opinion else None,
                    pair.reason if pair.reason else None
                ))
        
        # Count occurrences
        pair_counts = Counter(pair_tuples)
        
        # Get pairs that meet threshold
        final_pairs = []
        for (aspect, opinion, reason), count in pair_counts.items():
            if count >= threshold:
                final_pairs.append(ExtractionPair(
                    aspect=aspect,
                    opinion=opinion,
                    reason=reason
                ))
        
        return final_pairs

    def _format_pairs(self, pairs: List[ExtractionPair]) -> List[Dict[str, Any]]:
        """Format pairs for output."""
        return [p.model_dump() for p in pairs] 