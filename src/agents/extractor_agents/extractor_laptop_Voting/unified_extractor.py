"""
Unified aspect-opinion extraction agent for laptop reviews with Agent Forest voting.

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

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# --- V4 ---
# Use the proven, working API from the llms directory
import sys
# Add src to path to allow importing from llms
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from llms.api import llm_chat, reset_api_logging_for_sample


logger = logging.getLogger(__name__)

class AspectOpinionPair(BaseModel):
    aspect: Optional[str] = Field(description="The aspect term extracted or inferred from the text. Use null if there is no clear aspect.")
    opinion: Optional[str] = Field(description="The opinion phrase extracted or inferred from the text. Use null if there is no clear opinion.")
    reason: str = Field(description="A brief one-sentence explanation of why this aspect-opinion pair is present in the text.")

class ExtractionResponse(BaseModel):
    pairs: List[AspectOpinionPair] = Field(description="List of aspect-opinion pairs with reasoning")

class UnifiedExtractorAgent:
    def __init__(self, llm=None, ensemble_size=1, temperature=0.0, parallel=True, model="gpt-4o", prompt_type="20shot"):
        """Initialize the Unified Extractor agent with Agent Forest voting capabilities."""
        self.model_name = model.lower()
        self.prompt_type = prompt_type
        
        # Agent Forest voting parameters
        self.ensemble_size = ensemble_size
        self.temperature = temperature
        self.parallel = parallel
        
        self.parser = PydanticOutputParser(pydantic_object=ExtractionResponse)
        
        # Load the prompt from file
        self.system_prompt = self._load_prompt_from_file()
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", 
             "Text: {text}\n\n"
             "JSON list of pairs:")
        ])
        
        self.retry_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt + "\n\nIMPORTANT: The previous extraction contained aspect/opinion terms that were not found in the original text. Please re-extract ensuring that aspect and opinion terms appear exactly as they are written in the text (including any typos or variations)."),
            ("human", 
             "Text: {text}\n\n"
             "Previous extraction had unfindable terms. Please extract aspect-opinion pairs using EXACT terms from the text:\n\n"
             "JSON list of pairs:")
        ])
        
        self.prompt = self.prompt.partial(
            format_instructions=self.parser.get_format_instructions()
        )
        
        self.retry_prompt = self.retry_prompt.partial(
            format_instructions=self.parser.get_format_instructions()
        )

    def _load_prompt_from_file(self):
        """Load the system prompt from the txt file."""
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))

        if self.prompt_type in ["zeroshot", "0shot"]:
            prompt_file_name = "unified_extractor_prompt_0shot.txt"
        else:
            prompt_file_name = "unified_extractor_prompt.txt"

        prompt_file_path = os.path.join(current_dir, "prompts", prompt_file_name)
        
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as file:
                prompt_content = file.read()
            return prompt_content
        except FileNotFoundError:
            logger.error(f"Prompt file not found at {prompt_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt file: {e}")
            raise

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

    def _apply_fallback_fixes(self, pairs: List[AspectOpinionPair], original_text: str) -> List[AspectOpinionPair]:
        """Attempt to fix unfindable terms by finding the most similar text."""
        fixed_pairs = []
        for pair in pairs:
            fixed_aspect = pair.aspect
            fixed_opinion = pair.opinion
            
            if not self._is_term_findable_in_text(pair.aspect, original_text):
                most_similar = self._find_most_similar_text(pair.aspect, original_text)
                if most_similar:
                    logger.info(f"Fallback Fix: Replaced aspect '{pair.aspect}' with most similar term '{most_similar}'")
                    fixed_aspect = most_similar
                    
            if not self._is_term_findable_in_text(pair.opinion, original_text):
                most_similar = self._find_most_similar_text(pair.opinion, original_text)
                if most_similar:
                    logger.info(f"Fallback Fix: Replaced opinion '{pair.opinion}' with most similar term '{most_similar}'")
                    fixed_opinion = most_similar

            fixed_pairs.append(AspectOpinionPair(aspect=fixed_aspect, opinion=fixed_opinion, reason=pair.reason))
        
        return fixed_pairs

    def _needs_retry(self, pairs: List[AspectOpinionPair], original_text: str) -> bool:
        """Check if any extracted term is not findable in the original text."""
        for pair in pairs:
            # Check both aspect and opinion terms using attribute access
            if not self._is_term_findable_in_text(pair.aspect, original_text):
                logger.warning(f"Aspect '{pair.aspect}' not found in text. Triggering retry.")
                return True
            if not self._is_term_findable_in_text(pair.opinion, original_text):
                logger.warning(f"Opinion '{pair.opinion}' not found in text. Triggering retry.")
                return True
        return False

    def _make_single_extraction(self, text: str, voter_id: int) -> List[Dict[str, str]]:
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

                # Validate terms and retry if necessary - but only if ensemble_size > 1
                if self.ensemble_size > 1 and self._needs_retry(parsed_response.pairs, text):
                    logger.warning(f"Voter {voter_id}, Attempt {attempt+1}: Bad extraction, retrying...")
                    current_prompt = self.retry_prompt # Switch to retry prompt
                    if attempt == 0: continue # Continue to next attempt in the loop

                # Apply fallback fixes only if ensemble_size > 1, otherwise return as is
                if self.ensemble_size > 1:
                    final_pairs = self._apply_fallback_fixes(parsed_response.pairs, text)
                else:
                    final_pairs = parsed_response.pairs

                return [p.dict() for p in final_pairs]

            except Exception as e:
                logger.error(f"Voter {voter_id}: Failed to extract or parse on attempt {attempt+1}. Error: {e}")
        
        return [] # Return empty list if all attempts fail

    def _normalize_pair_for_voting(self, pair: Dict[str, str]) -> tuple:
        """Normalize aspect-opinion pair for voting comparison (exclude reason)."""
        return (str(pair.get('aspect', '')).lower().strip(), str(pair.get('opinion', '')).lower().strip())

    def _extract_with_voting(self, text: str) -> List[Dict[str, str]]:
        """
        Performs aspect-opinion extraction using the Agent Forest majority voting method.
        """
        print("=" * 80)
        print(f"🌲 AGENT FOREST VOTING - Aspect-Opinion Extraction (Ensemble Size: {self.ensemble_size})")
        print(f"📝 Input Text: '{text[:100]}...'")
        print(f"🎯 Temperature: {self.temperature} | Parallel: {self.parallel}")
        print("=" * 80)
        
        # Reset logging for the new sample to avoid clutter
        reset_api_logging_for_sample()
        
        def get_predictions_from_voters(num_voters: int, voter_offset: int = 0):
            """Helper to run extraction for a given number of voters, in parallel or sequentially."""
            all_predictions = []
            
            if self.parallel:
                with ThreadPoolExecutor(max_workers=num_voters) as executor:
                    future_to_voter = {executor.submit(self._make_single_extraction, text, i + 1 + voter_offset): i for i in range(num_voters)}
                    for future in as_completed(future_to_voter):
                        voter_id = future_to_voter[future]
                        try:
                            prediction = future.result()
                            all_predictions.append(prediction)
                        except Exception as exc:
                            print(f"Voter {voter_id + 1 + voter_offset} generated an exception: {exc}")
                            all_predictions.append([]) # Append empty list on failure
            else:
                for i in range(num_voters):
                    prediction = self._make_single_extraction(text, i + 1 + voter_offset)
                    all_predictions.append(prediction)
            return all_predictions
        
        def display_voter_predictions(predictions, voter_offset=0):
            """Helper to print out predictions from each voter."""
            print("\n📊 VOTER PREDICTIONS:")
            print("-" * 80)
            for i, prediction in enumerate(predictions):
                print(f"🗳️  Voter {i + 1 + voter_offset}:")
                if prediction:
                    print(f"   Count: {len(prediction)} pairs")
                    for j, pair in enumerate(prediction):
                        print(f"   {j+1}. ({pair.get('aspect')}, {pair.get('opinion')})")
                else:
                    print("   ❌ No valid predictions")
        
        def count_pair_votes(all_predictions):
            """Counts votes for each unique aspect-opinion pair."""
            pair_votes = Counter()
            original_pair_map = {} # To store the first occurrence of a pair with its reason
            
            for prediction in all_predictions:
                # Use a set to count each pair only once per voter
                voter_unique_pairs = set()
                for pair in prediction:
                    norm_pair = self._normalize_pair_for_voting(pair)
                    voter_unique_pairs.add(norm_pair)
                    
                    # Store the original pair dict (with reason) if it's the first time we see this pair
                    if norm_pair not in original_pair_map:
                        original_pair_map[norm_pair] = pair
                
                # Add the voter's unique pairs to the total vote count
                pair_votes.update(voter_unique_pairs)
            
            return pair_votes, original_pair_map
        
        def try_voting_with_threshold(pair_votes, original_pair_map, threshold, level_name):
            """Helper function to try voting with a specific threshold."""
            print(f"\n🗳️  {level_name} - THRESHOLD: {threshold} votes")
            print(f"{'-'*80}")
            
            # Sort by vote count (descending) for better display
            sorted_votes = sorted(pair_votes.items(), key=lambda x: x[1], reverse=True)
            
            winning_pairs = []
            for norm_pair, vote_count in sorted_votes:
                original_pair = original_pair_map[norm_pair]
                meets_threshold = vote_count >= threshold
                status = "✅ ACCEPTED" if meets_threshold else "❌ REJECTED"
                
                print(f"📋 ({original_pair['aspect']}, {original_pair['opinion']})")
                print(f"   Votes: {vote_count} | Threshold: {threshold} | {status}")
                
                if meets_threshold:
                    winning_pairs.append(original_pair)
            
            return winning_pairs
        
        # --- Main voting logic ---
        all_voter_predictions = get_predictions_from_voters(self.ensemble_size)
        
        # Filter out empty predictions for majority calculation
        valid_predictions = [p for p in all_voter_predictions if p]
        num_valid_voters = len(valid_predictions)
        
        display_voter_predictions(valid_predictions)
        
        if num_valid_voters == 0:
            print("\n❌ No valid predictions from any voter!")
            return []
        
        pair_votes, original_pair_map = count_pair_votes(valid_predictions)
        
        # --- LEVEL 0: NORMAL MAJORITY VOTING ---
        majority_threshold = (num_valid_voters // 2) + 1
        print(f"\n📊 Majority threshold: {majority_threshold} votes (out of {num_valid_voters} valid voters)")
        
        winning_pairs = try_voting_with_threshold(pair_votes, original_pair_map, majority_threshold, "NORMAL MAJORITY VOTING")
        if winning_pairs:
            print(f"\n🏆 SUCCESS: Normal majority voting found {len(winning_pairs)} pairs!")
            return winning_pairs
        
        # --- FALLBACK SYSTEM ---
        print("\n⚠️  Normal majority voting failed - entering fallback system...")
        
        # --- LEVEL 1: LOWER THRESHOLD ---
        if majority_threshold > 1:
            lower_threshold = majority_threshold - 1
            winning_pairs = try_voting_with_threshold(pair_votes, original_pair_map, lower_threshold, "LEVEL 1 FALLBACK: Lowered Threshold")
            if winning_pairs: return winning_pairs
        
        # --- LEVEL 2: HIGHEST VOTE COUNT (PLURALITY) ---
        if pair_votes:
            highest_vote_count = pair_votes.most_common(1)[0][1]
            if highest_vote_count > 1: # Only use plurality if there's some agreement
                winning_pairs = try_voting_with_threshold(pair_votes, original_pair_map, highest_vote_count, "LEVEL 2 FALLBACK: Highest Vote Count")
                if winning_pairs: return winning_pairs
        
        # --- LEVEL 3: ADD MORE VOTERS & REVOTE ---
        print("\n🛡️  LEVEL 3 FALLBACK: Adding more voters and revoting...")
        additional_voters = 4 # Add an even number to avoid ties
        additional_predictions = get_predictions_from_voters(additional_voters, voter_offset=self.ensemble_size)
        
        display_voter_predictions(additional_predictions, voter_offset=self.ensemble_size)
        
        combined_predictions = valid_predictions + additional_predictions
        num_total_voters = len(combined_predictions)
        new_majority_threshold = (num_total_voters // 2) + 1
        
        new_pair_votes, new_original_map = count_pair_votes(combined_predictions)
        
        winning_pairs = try_voting_with_threshold(new_pair_votes, new_original_map, new_majority_threshold, "REVOTE with more voters")
        if winning_pairs: return winning_pairs
        
        # --- FINAL FALLBACK: ANY PAIR WITH >1 VOTE ---
        print("\n🛡️  FINAL FALLBACK: Accepting any pair with more than one vote.")
        final_threshold = 2
        winning_pairs = try_voting_with_threshold(new_pair_votes, new_original_map, final_threshold, "FINAL FALLBACK")
        if winning_pairs: return winning_pairs

        print("\n❌ Voting failed at all levels. No consensus found.")
        return []

    def run(self, state):
        """Run the agent."""
        text = state.get('text')
        if not text:
            raise ValueError("Input state must contain a 'text' key")

        try:
            # Main path: extraction with voting
            winning_pairs = self._extract_with_voting(text)
            state['extracted_pairs'] = winning_pairs
        except Exception as e:
            logger.warning(f"Agent Forest voting failed, falling back to single extraction. Error: {e}")
            
            # Fallback to a single, non-voting extraction call
            try:
                # Use the same extraction logic as a single voter
                single_extraction_result = self._make_single_extraction(text, voter_id=0)
                state['extracted_pairs'] = single_extraction_result
            except Exception as e2:
                logger.warning(f"Fallback extraction also failed, defaulting to empty list. Error: {e2}")
                state['extracted_pairs'] = []
        
        return state 