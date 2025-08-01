"""
Unified aspect-opinion extraction agent for laptop reviews with Agent Forest voting.

Input state keys required:
• text: str

Output state keys added/overwritten:
• extracted_pairs: List[Dict[str, str]]  # List of {"aspect": "...", "opinion": "...", "reason": "..."}
"""

import os
import re
import difflib
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
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
import logging

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
        
        # Agent Forest voting parameters
        self.ensemble_size = ensemble_size
        self.temperature = temperature
        self.parallel = parallel
        self.prompt_type = prompt_type
        
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
        # This handles cases where the term might be a substring of another word
        # We need to escape the term in case it contains special regex characters
        return bool(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE))

    def _find_most_similar_text(self, target_term, text):
        """
        Find the most similar substring in the text to the target term.
        """
        if not target_term or not text:
            return None
        
        words = text.split()
        if not words:
            return None
        
        # Find the best match using difflib
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
        
        if self.ensemble_size < 1:
            raise ValueError("Ensemble size must be at least 1")
        
        def get_predictions_from_voters(num_voters: int, voter_offset: int = 0):
            """Helper function to get predictions from a specified number of voters."""
            predictions = []
            
            if self.parallel and num_voters > 1:
                print(f"🔄 Making {num_voters} parallel extraction calls...")
                
                # Build the executor without a context manager so we can shut it down
                # immediately after use (avoids blocking on long-running threads).
                executor = ThreadPoolExecutor(max_workers=min(8, num_voters))

                # Submit jobs and remember which voter index each future belongs to
                futures_map = {
                    executor.submit(self._make_single_extraction, text, i + 1 + voter_offset): i
                    for i in range(num_voters)
                }

                # Pre-allocate list to preserve output order (index == voter)
                predictions = [[] for _ in range(num_voters)]

                try:
                    # Collect results as they complete so a single slow call cannot block the rest
                    for future in as_completed(futures_map, timeout=65):
                        idx = futures_map[future]
                        try:
                            predictions[idx] = future.result()
                        except Exception as e:
                            print(f"❌ Voter {idx + 1 + voter_offset}: Call failed - {e}")
                            predictions[idx] = []
                except Exception as e:
                    # Handles TimeoutError from as_completed or other unexpected exceptions
                    print(f"⚠️  Parallel extraction interrupted: {e}")
                finally:
                    # Cancel whatever is still running and shutdown executor without waiting
                    for fut in futures_map:
                        if not fut.done():
                            fut.cancel()
                    executor.shutdown(wait=False)

            else:
                print(f"🔄 Making {num_voters} sequential extraction calls...")
                for i in range(num_voters):
                    try:
                        pred = self._make_single_extraction(text, i+1+voter_offset)
                        predictions.append(pred)
                    except Exception as e:
                        print(f"❌ Voter {i+1+voter_offset}: Call failed - {e}")
                        predictions.append([])
            
            return predictions
        
        def display_voter_predictions(predictions, voter_offset=0):
            """Helper function to display voter predictions."""
            print(f"\n📊 VOTER PREDICTIONS:")
            print(f"{'-'*80}")
            for i, pred in enumerate(predictions):
                print(f"🗳️  Voter {i+1+voter_offset}:")
                if pred:
                    print(f"   Count: {len(pred)} pairs")
                    for j, pair in enumerate(pred[:3]):  # Show first 3 pairs
                        print(f"   {j+1}. ({pair['aspect']}, {pair['opinion']})")
                    if len(pred) > 3:
                        print(f"   ... and {len(pred)-3} more")
                else:
                    print(f"   ❌ No valid predictions")
                print()
        
        def count_pair_votes(all_predictions):
            """Helper function to count votes for each aspect-opinion pair."""
            # Filter out empty predictions
            valid_predictions = [pred for pred in all_predictions if pred]
            
            if not valid_predictions:
                return Counter(), {}, valid_predictions
            
            # Count votes for each normalized pair (excluding reason)
            pair_votes = Counter()
            original_pair_map = {}  # Map normalized -> original form with reason
            
            for pred in valid_predictions:
                seen_in_this_prediction = set()  # Track pairs seen in this prediction
                for pair in pred:
                    norm_pair = self._normalize_pair_for_voting(pair)
                    
                    # Only count once per prediction (avoid double counting)
                    if norm_pair not in seen_in_this_prediction:
                        pair_votes[norm_pair] += 1
                        seen_in_this_prediction.add(norm_pair)
                        
                        # Keep track of original form with reason (prefer the first seen)
                        if norm_pair not in original_pair_map:
                            original_pair_map[norm_pair] = pair
            
            return pair_votes, original_pair_map, valid_predictions
        
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

        # ==== INITIAL VOTING ROUND ====
        all_predictions = get_predictions_from_voters(self.ensemble_size)
        display_voter_predictions(all_predictions)
        
        # Count votes
        pair_votes, original_pair_map, valid_predictions = count_pair_votes(all_predictions)
        
        if not pair_votes:
            print("❌ No valid predictions from any voter!")
            return []
        
        if len(valid_predictions) == 1:
            winner = valid_predictions[0]
            print(f"🏆 VOTING RESULT: Only one valid prediction")
            print(f"   Winner: {len(winner)} pairs")
            return winner
        
        valid_voters = len(valid_predictions)
        normal_threshold = (valid_voters + 1) // 2  # ⌈N/2⌉
        print(f"📊 Majority threshold: {normal_threshold} votes (out of {valid_voters} valid voters)")
        
        # ==== TRY NORMAL MAJORITY VOTING ====
        winning_pairs = try_voting_with_threshold(pair_votes, original_pair_map, normal_threshold, "NORMAL MAJORITY VOTING")
        
        if winning_pairs:
            print(f"\n🏆 SUCCESS: Normal majority voting found {len(winning_pairs)} pairs!")
            return winning_pairs
        
        print(f"\n⚠️  Normal majority voting failed - entering fallback system...")
        
        # ==== LEVEL 1 FALLBACK: Lower threshold by 1 ====
        level1_threshold = max(1, normal_threshold - 1)
        print(f"\n🛡️  LEVEL 1 FALLBACK: Lowering threshold from {normal_threshold} to {level1_threshold}")
        
        winning_pairs = try_voting_with_threshold(pair_votes, original_pair_map, level1_threshold, "LEVEL 1 FALLBACK")
        
        if winning_pairs:
            print(f"\n🏆 SUCCESS: Level 1 fallback found {len(winning_pairs)} pairs!")
            return winning_pairs
        
        # ==== LEVEL 2 FALLBACK: Accept highest vote count ====
        print(f"\n🛡️  LEVEL 2 FALLBACK: Accepting pairs with highest vote count")
        
        if pair_votes:
            max_votes = max(pair_votes.values())
            print(f"   Highest vote count found: {max_votes}")
            
            winning_pairs = try_voting_with_threshold(pair_votes, original_pair_map, max_votes, "LEVEL 2 FALLBACK")
            
            if winning_pairs:
                print(f"\n🏆 SUCCESS: Level 2 fallback found {len(winning_pairs)} pairs!")
                return winning_pairs
        
        # ==== LEVEL 3 FALLBACK: Add more agents and revote ====
        print(f"\n🛡️  LEVEL 3 FALLBACK: Adding more agents and revoting")
        
        additional_voters = max(2, min(5, self.ensemble_size // 2))
        print(f"   Adding {additional_voters} additional voters to break deadlock...")
        
        # Get predictions from additional voters
        additional_predictions = get_predictions_from_voters(additional_voters, voter_offset=self.ensemble_size)
        display_voter_predictions(additional_predictions, voter_offset=self.ensemble_size)
        
        # Combine all predictions
        combined_predictions = all_predictions + additional_predictions
        total_combined_voters = len(combined_predictions)
        
        # Recount votes with all voters
        combined_pair_votes, combined_original_map, combined_valid_predictions = count_pair_votes(combined_predictions)
        
        if not combined_pair_votes:
            print("❌ Level 3 fallback: Still no valid predictions!")
            return []
        
        # Calculate new threshold for combined voting
        combined_valid_voters = len(combined_valid_predictions)
        combined_threshold = (combined_valid_voters + 1) // 2
        print(f"\n📊 Combined voting with {combined_valid_voters} valid voters (threshold: {combined_threshold})")
        
        winning_pairs = try_voting_with_threshold(combined_pair_votes, combined_original_map, combined_threshold, "LEVEL 3 COMBINED VOTING")
        
        if winning_pairs:
            print(f"\n🏆 SUCCESS: Level 3 fallback found {len(winning_pairs)} pairs!")
            return winning_pairs
        
        # Final fallback - accept highest vote count from combined voting
        if combined_pair_votes:
            max_combined_votes = max(combined_pair_votes.values())
            print(f"\n🛡️  FINAL FALLBACK: Accepting highest vote count ({max_combined_votes}) from combined voting")
            
            winning_pairs = try_voting_with_threshold(combined_pair_votes, combined_original_map, max_combined_votes, "FINAL FALLBACK")
            
            if winning_pairs:
                print(f"\n🏆 SUCCESS: Final fallback found {len(winning_pairs)} pairs!")
                return winning_pairs
        
        # Absolute last resort
        print("\n❌ ALL FALLBACK LEVELS EXHAUSTED")
        print("🔍 SYSTEM DIAGNOSIS: Extremely low confidence - no consensus possible")
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