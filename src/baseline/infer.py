import sys
import argparse
import time
import random
import os
import numpy as np
import json # For saving output
import ast # For safely evaluating string representation of list
import re # For regex parsing of JSON
import logging # For logging
from typing import List, Dict, Tuple, Optional, Any # For type hints

# Add imports for Pydantic model support
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(".")
from data_utils import get_transformed_io
from eval_utils import extract_spans_para, compute_scores # Added compute_scores
from baseline.api import llm_chat

# Define Pydantic models for structured extraction
class ACOSQuad(BaseModel):
    aspect_term: Optional[str] = Field(description="The aspect term extracted from the text. Use 'null' if implied.")
    opinion_term: Optional[str] = Field(description="The opinion term expressing sentiment. Use 'null' if no subjective words.")
    aspect_category: str = Field(description="The category the aspect belongs to.")
    sentiment_polarity: str = Field(description="The sentiment polarity: 'positive', 'negative', or 'neutral'.")

class ExtractionResponse(BaseModel):
    quads: List[ACOSQuad] = Field(description="List of aspect-opinion-category-sentiment quads")

# Create output parser
class PydanticOutputParser:
    def __init__(self, pydantic_object):
        self.pydantic_object = pydantic_object

    def get_format_instructions(self):
        schema = self.pydantic_object.schema()
        schema_str = json.dumps(schema, indent=2)
        return f"""
You must format your output as a JSON instance that conforms to the JSON schema below.

{schema_str}

The output should be properly formatted JSON and nothing else.
"""

    def parse(self, text):
        # Try to find JSON in the response
        try:
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.MULTILINE)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # If no code block, try to find JSON directly
                json_match = re.search(r"\{[\s\S]*\}", text)
                if json_match:
                    json_str = json_match.group(0).strip()
                else:
                    json_str = text.strip()
            
            # Parse the JSON
            data = json.loads(json_str)
            return self.pydantic_object.parse_obj(data)
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            logger.error(f"Raw response: {text}")
            raise ValueError(f"Failed to parse output: {e}")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../data/", type=str)
    parser.add_argument(
        "--task",
        default='acos',
        choices=["asqp", "acos", "aste", "tasd", "unified", "unified3"],
        type=str,
        help="The name of the task, selected from: [asqp, tasd, aste]")
    parser.add_argument(
        "--dataset",
        default='rest16',
        type=str,
        help="The name of the dataset, selected from: [rest15, rest16]")
    parser.add_argument(
        "--eval_data_split",
        default='test',
        choices=["test", "dev"],
        type=str,
    )
    parser.add_argument("--output_dir",
                        default='outputs/temp',
                        type=str,
                        help="Output directory")
    parser.add_argument(
        "--do_inference",
        default=True,
        help="Whether to run inference with trained checkpoints")
    parser.add_argument('--seed',
                        type=int,
                        default=25,
                        help="random seed for initialization")
    parser.add_argument("--top_k", default=1, type=int)
    parser.add_argument("--num_path", default=1, type=int)
    parser.add_argument("--save_top_k", default=0, type=int)
    parser.add_argument("--ctrl_token",
                        default="none",
                        choices=["post", "pre", "none"],
                        type=str)
    parser.add_argument("--sort_label",
                        action='store_true',
                        help="sort tuple by order of appearance")
    parser.add_argument("--lowercase", action='store_true')
    parser.add_argument("--num_sample", default=10000, type=int, help="Number of samples to infer.")
    parser.add_argument("--prompt_type", default="0shot", type=str, help="Prompt type, e.g., 0shot or 5shot.")
    parser.add_argument("--single_view_type",
                        default="heuristic",
                        choices=["heuristic"], # Only heuristic is supported now
                        type=str,
                        help="Strategy for view selection (heuristic is the only one after T5 removal)")
    parser.add_argument("--multi_task", action='store_true', help="Enable multi-task processing (defaults to False).")
    parser.add_argument("--model", default="gpt", 
                        choices=["gpt", "gpt-4o", "gpt-4.1", "gpt-4.1-nano", "deepseek-v3", "deepseek-r1"], 
                        help="The model to use for inference.")
    parser.add_argument("--track_tokens", action='store_true', help="Track token usage for cost calculation.")
    parser.add_argument("--use_pydantic", action='store_true', help="Use Pydantic model for structured output parsing.")

    args = parser.parse_args()
    return args


opinion2sentword = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}

def load_prompt(task, data, prompt_type, use_pydantic=False):
    if data == "rest16":
        data_folder_suffix = "rest"
    elif data == "laptop16":
        data_folder_suffix = "laptop"
    else:
        # Fallback or error handling if data is neither rest16 nor laptop16
        # For now, let's assume it will be one of these, or raise an error.
        raise ValueError(f"Unsupported dataset: {data}. Cannot determine prompt folder.")

    prompt_path = f"baseline/prompts_{data_folder_suffix}/{task}_{data}_{prompt_type}.txt"
    
    with open(prompt_path, 'r', encoding='utf-8') as fp:
        prompt_template = fp.read().strip() + "\n\n"
    
    # If using Pydantic, inject format instructions
    if use_pydantic:
        parser = PydanticOutputParser(pydantic_object=ExtractionResponse)
        format_instructions = parser.get_format_instructions()
        
        # Replace format_instructions placeholder if it exists
        if "{format_instructions}" in prompt_template:
            prompt = prompt_template.format(format_instructions=format_instructions)
        else:
            # If no placeholder, append format instructions
            prompt = prompt_template + "\n" + format_instructions + "\n\n"
        
        return prompt, parser
    else:
        return prompt_template, None


def inference(args, start_idx=0, end_idx=None): # Allow end_idx to be None for full dataset
    data_path = f'{args.data_path}/{args.task}/{args.dataset}/{args.data_type}.txt'
    sources, targets = get_transformed_io(data_path,
                                        args.dataset,
                                        args.data_type, top_k=args.top_k, args=args)

    print(f"Total examples = {len(sources)}")
    num_samples_to_process = min(args.num_sample, len(sources))
    print(f"Processing {num_samples_to_process} samples")
    
    # Use sequential samples instead of random sampling
    # Take the first num_samples_to_process samples sequentially
    samples = list(zip(sources, targets))[:num_samples_to_process]
    
    # If start_idx and end_idx are provided, further limit the range
    if end_idx is not None:
        end_idx = min(end_idx, num_samples_to_process)
    else:
        end_idx = num_samples_to_process
        
    if start_idx > 0:
        print(f"Starting from sample {start_idx}")
    if end_idx < num_samples_to_process:
        print(f"Ending at sample {end_idx}")

    prompt, parser = load_prompt(args.task, args.dataset, args.prompt_type, args.use_pydantic)
    
    all_gold_for_eval = []
    all_pred_for_eval = []
    
    # Initialize token tracking if enabled
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    for i in range(start_idx, end_idx):
        source, target = samples[i]
        print(f"Processing sample {i}/{end_idx-1}...")
        try:
            current_source_text = " ".join(source)
            
            # Process Gold Quads
            gold_list_raw = extract_spans_para(target, 'gold')
            gold_list_transformed = []
            if args.task in ['asqp', 'acos']:
                for (ac, at, sp, ot) in gold_list_raw:
                    sp_mapped = opinion2sentword.get(sp.lower() if sp else "", sp) # Handle None for sp
                    gold_list_transformed.append((at, ot, ac, sp_mapped))
            elif args.task == "aste":
                for (ac, at, sp, ot) in gold_list_raw:
                    sp_mapped = opinion2sentword.get(sp.lower() if sp else "", sp)
                    gold_list_transformed.append((at, ot, opinion2sentword.get(sp.lower(), sp)))
            elif args.task == "tasd":
                 for (ac, at, sp, ot) in gold_list_raw:
                    sp_mapped = opinion2sentword.get(sp.lower() if sp else "", sp)
                    gold_list_transformed.append((at, ac, opinion2sentword.get(sp.lower(), sp)))
            all_gold_for_eval.append(gold_list_transformed)

            # Prepare context for LLM and get prediction
            context = f"Text: {current_source_text}\n"
            context += "Sentiment Elements: "
            
            # Process Predicted Quads
            pred_list_raw = [] # Initialize as empty list
            max_retries = 1 if args.model.startswith("deepseek") else 3  # Use fewer retries for DeepSeek models
            retry_delay = 5
            success = False
            
            print(f"Using {max_retries} max retries for model {args.model}")
            
            for attempt in range(max_retries):
                if success:
                    print(f"Already got successful response on attempt {attempt}, skipping remaining retries.")
                    break
                    
                try:
                    # Prepare context for LLM and get prediction
                    context = f"Text: {current_source_text}\n"
                    context += "Sentiment Elements: "
                    
                    print(f"Making API call attempt {attempt+1}/{max_retries}...")
                    llm_prediction_str, usage = llm_chat([{"role": "user", "content": prompt + context}], model_name=args.model, return_usage=args.track_tokens)
                    print(f"API call attempt {attempt+1} completed successfully.")
                    
                    # Track token usage if enabled
                    if args.track_tokens and usage:
                        token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                        token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                    
                    # The LLM now outputs a string representation of a list of tuples
                    # e.g., "[('term', 'opinion', 'category', 'polarity'), ...]"
                    if args.use_pydantic:
                        try:
                            # Parse JSON output using Pydantic parser
                            parsed_response = parser.parse(llm_prediction_str)
                            
                            # Convert to list of tuples for compatibility with existing code
                            pred_list_raw = []
                            for quad in parsed_response.quads:
                                # Convert any "null" strings to None for consistent handling
                                aspect = None if quad.aspect_term in ["null", "NULL", None] else quad.aspect_term
                                opinion = None if quad.opinion_term in ["null", "NULL", None] else quad.opinion_term
                                pred_list_raw.append((aspect, opinion, quad.aspect_category, quad.sentiment_polarity))
                            
                            success = True
                            break  # Break out of the retry loop when successful
                        except ValueError as e:
                            print(f"Warning: Could not parse LLM output as JSON: {e}")
                            print(f"Raw output: {llm_prediction_str}")
                            if attempt == max_retries - 1:  # Last attempt
                                pred_list_raw = []  # Empty list on failure
                            continue
                    else:
                        # Traditional parsing with ast.literal_eval
                        try:
                            # First, try to clean up common JSON response formats
                            # Remove markdown code blocks if present
                            clean_output = re.sub(r'```json\s*(.*?)\s*```', r'\1', llm_prediction_str, flags=re.DOTALL)
                            clean_output = re.sub(r'```\s*(.*?)\s*```', r'\1', clean_output, flags=re.DOTALL)
                            
                            # Check if the output looks like JSON
                            if clean_output.strip().startswith('[{') or clean_output.strip().startswith('['):
                                try:
                                    # Try parsing as JSON
                                    json_data = json.loads(clean_output)
                                    # Convert JSON to tuples list format
                                    pred_list_raw = []
                                    for item in json_data:
                                        aspect = item.get('aspect_term')
                                        opinion = item.get('opinion_term')
                                        category = item.get('aspect_category')
                                        polarity = item.get('sentiment_polarity')
                                        # Convert "null" strings to None
                                        aspect = None if aspect in ["null", "NULL", None] else aspect
                                        opinion = None if opinion in ["null", "NULL", None] else opinion
                                        pred_list_raw.append((aspect, opinion, category, polarity))
                                    success = True
                                    break  # Break out of the retry loop when successful
                                except json.JSONDecodeError:
                                    print(f"Warning: Failed to parse as JSON: '{clean_output}'")
                            
                            # Fall back to ast.literal_eval for traditional tuple list format
                            if not success:
                                pred_list_raw = ast.literal_eval(clean_output)
                                # Ensure it's a list of tuples, and each tuple has 4 elements
                                if not isinstance(pred_list_raw, list) or not all(isinstance(item, tuple) and len(item) == 4 for item in pred_list_raw):
                                    print(f"Warning: LLM output was not a list of 4-element tuples after ast.literal_eval: {clean_output}")
                                    pred_list_raw = [] # Reset if format is incorrect
                                else:
                                    success = True
                                    break  # Break out of the retry loop when successful
                        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError) as e:
                            print(f"Warning: Could not parse LLM output string: '{llm_prediction_str}'. Error: {e}")
                            if attempt < max_retries - 1:
                                print(f"Retrying... (Attempt {attempt+1}/{max_retries})")
                            else:
                                print("Failed to get valid prediction after all attempts.")
                                pred_list_raw = []  # Empty list on failure
                        except Exception as e:
                            print(f">" * 30, "exception processing output:", e)
                            pred_list_raw = []
                except Exception as e:
                    print(f"API error on attempt {attempt+1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"Max retries reached. Skipping this sample.")
            
            if not success:
                print(f"Failed to get valid prediction after {max_retries} attempts.")
                pred_list_raw = []  # Ensure it's empty if all attempts failed
            
            # The rest of the transformation logic for pred_list_transformed remains the same
            pred_list_transformed = []
            if args.task in ['asqp', 'acos']:
                # LLM output is: (aspect_term, opinion_term, aspect_category, sentiment_polarity_word)
                for (at_llm, ot_llm, ac_llm, sp_llm_word) in pred_list_raw:
                    sp_mapped = opinion2sentword.get(sp_llm_word.lower() if sp_llm_word else "", sp_llm_word)
                    pred_list_transformed.append((at_llm, ot_llm, ac_llm, sp_mapped))
            elif args.task == "aste":
                # LLM output for ASTE might be different, assuming (aspect_term, opinion_term, sentiment_polarity_word)
                # This part needs verification if ASTE is used. For ACOS, the above is key.
                for (at_llm, ot_llm, sp_llm_word) in pred_list_raw: # Assuming 3 elements for ASTE from LLM if it matches gold's need
                    sp_mapped = opinion2sentword.get(sp_llm_word.lower() if sp_llm_word else "", sp_llm_word)
                    # Gold for ASTE is (at, ot, mapped_sp)
                    pred_list_transformed.append((at_llm, ot_llm, sp_mapped))
            elif args.task == "tasd":
                # LLM output for TASD might be different. For ACOS, the above is key.
                # Assuming (aspect_term, aspect_category, sentiment_polarity_word) from LLM
                for (at_llm, ac_llm, sp_llm_word) in pred_list_raw: # Assuming 3 elements for TASD
                    sp_mapped = opinion2sentword.get(sp_llm_word.lower() if sp_llm_word else "", sp_llm_word)
                    # Gold for TASD is (at, ac, mapped_sp)
                    pred_list_transformed.append((at_llm, ac_llm, sp_mapped))
            all_pred_for_eval.append(pred_list_transformed)

            # Print individual results (optional)
            print(f"Text: {current_source_text}")
            print(f"LLM Raw Output: {llm_prediction_str}")
            print(f"Predicted Quads: {pred_list_transformed}")
            print(f"Gold Quads: {gold_list_transformed}\n")
            
            # Show progress as percentage and count from 1 rather than 0 for more intuitive display
            progress = (i - start_idx + 1) / (end_idx - start_idx) * 100
            print(f"Progress: {progress:.1f}% ({i - start_idx + 1}/{end_idx - start_idx})")
            
            # Remove the mandatory sleep between samples
            # time.sleep(3) # Increased sleep time to 3 seconds to respect API rate limits
        except BaseException as e:
            print(f">" * 30, "exception processing sample {i}/{end_idx-1}:", e)
            # Optionally append empty predictions or skip sample in evaluation
            all_gold_for_eval.append(gold_list_transformed) # Keep gold if error in pred
            all_pred_for_eval.append([]) # No prediction for this sample
            continue
    
    # Calculate and print scores
    if not all_gold_for_eval and not all_pred_for_eval:
        print("No samples were processed or no predictions/gold labels available for evaluation.")
        return
        
    print("\nCalculating overall scores...")
    scores = compute_scores(all_pred_for_eval, all_gold_for_eval, verbose=False)
    print("Evaluation Scores:")
    for key, value in scores.items():
        print(f"  {key}: {value:.4f}")

    # Save results to a file
    output_results_path = os.path.join(args.output_dir, f"{args.dataset}_{args.prompt_type}_predictions.json")
    results_data = {
        'args': vars(args),
        'scores': scores,
        'predictions': all_pred_for_eval,
        'gold_standards': all_gold_for_eval,
        'num_samples': len(all_pred_for_eval)
    }
    try:
        with open(output_results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=4, ensure_ascii=False)
        print(f"\nDetailed predictions and scores saved to: {output_results_path}")
    except Exception as e:
        print(f"Error saving results to file: {e}")
    
    # Save token usage if tracking was enabled
    if args.track_tokens:
        token_usage_path = os.path.join(args.output_dir, "token_usage.json")
        try:
            with open(token_usage_path, 'w', encoding='utf-8') as f:
                json.dump(token_usage, f, indent=4)
            print(f"Token usage data saved to: {token_usage_path}")
        except Exception as e:
            print(f"Error saving token usage data: {e}")


if __name__ == "__main__":
    args = init_args()
    set_seed(args.seed)

    # default parameters - only set if not already provided via command line
    args.data_type = "test"
    
    # DO NOT override num_sample if it was passed on the command line
    # args.num_sample = 1000  # This line was causing the issue

    ## tasks:
    # args.task = "acos"
    # args.dataset = "rest16"

    # Commenting out overrides to respect command-line arguments
    # args.task = "asqp" 
    # args.dataset = "rest15"

    # args.task = "aste"
    # args.dataset = "laptop14"

    # args.task = "tasd"
    # args.dataset = "rest16"

    print(f"Will process up to {args.num_sample} samples")
    print(f"Using {'Pydantic model' if args.use_pydantic else 'traditional parsing'} for output")
    inference(args) # Process up to args.num_sample from the dataset
