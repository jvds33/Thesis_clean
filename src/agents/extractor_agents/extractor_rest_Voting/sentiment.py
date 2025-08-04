"""
Sentiment classifier agent for the multi-agent ACOS framework, laptop domain.

Input state keys required:
• all_pairs: List[Dict[str, str]]  # Each dict contains 'aspect' and 'opinion' keys
• extracted_pairs: List[Dict[str, str]]  # From unified extractor, includes reasons
• text: str                      # Full review text
• lang: str (optional; defaults "en")

Output state keys added/overwritten:
• sentiments: List[Dict[str, str]]  # Each dict contains 'aspect', 'opinion', and 'sentiment' keys
                                   # where sentiment ∈ {"positive","negative","neutral"}
"""
import sys
import os
import logging

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Any
import logging
from dotenv import load_dotenv

# --- V4 ---
# Use the proven, working API from the baseline directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from baseline.api import llm_chat

# Load environment variables from the root .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

logger = logging.getLogger(__name__)

class SentimentPair(BaseModel):
    aspect: Optional[str] = None
    opinion: Optional[str] = None
    sentiment: Literal["positive", "negative", "neutral"] = "neutral"

class SentimentClassificationResponse(BaseModel):
    sentiments: List[SentimentPair]

class SentimentAgent:
    def __init__(self, llm=None, model="gpt-4o", prompt_type="0shot"):
        """Initialize the Sentiment Classification agent."""
        self.llm = llm
        self.model_name = model.lower()
        self.prompt_type = prompt_type
        
        # Set up output parser
        self.parser = PydanticOutputParser(pydantic_object=SentimentClassificationResponse)
        
        # Load the prompt from file
        self.system_prompt = self._load_prompt_from_file()
        
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", 
             "Full review text: {text}\n\n"
             "Opinions to classify with reasons: {opinions_with_reasons}\n\n"
             "Language: {lang}\n\n"
             "Classify the sentiment of each aspect-opinion pair:")
        ])
        
        self.prompt = self.prompt.partial(
            format_instructions=self.parser.get_format_instructions()
        )
        
    def _load_prompt_from_file(self):
        """Load the system prompt from the txt file."""
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))

        if self.prompt_type in ["zeroshot", "0shot"]:
            prompt_file_name = "sentiment_prompt_0shot.txt"
        else:
            prompt_file_name = "sentiment_prompt.txt"

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

    def classify_sentiment(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Classify sentiment for aspect-opinion pairs."""
        return self.run(state)
        
    def run(self, state):
        """
        Run the sentiment classification agent on the provided state.
        
        Args:
            state: A dictionary containing the current state with 'all_pairs', 'extracted_pairs', and 'text' keys
            
        Returns:
            Updated state with 'sentiments' key added
        """
        if 'all_pairs' not in state or 'text' not in state:
            raise ValueError("Input state must contain 'all_pairs' and 'text' keys")
        
        # Set default language if not provided
        lang = state.get('lang', 'en')
        
        # Create a mapping of aspect-opinion pairs to their reasons from extracted_pairs
        pair_reasons = {}
        if 'extracted_pairs' in state:
            for pair in state['extracted_pairs']:
                aspect = pair.get('aspect')
                opinion = pair.get('opinion')
                reason = pair.get('reason', '')
                key = (self._normalize_key(aspect), self._normalize_key(opinion))
                pair_reasons[key] = reason

        # Format the opinions for the prompt, including the reasons
        opinions_with_reasons = ""
        for pair in state['all_pairs']:
            aspect = pair.get('aspect')
            opinion = pair.get('opinion')
            key = (self._normalize_key(aspect), self._normalize_key(opinion))
            reason = pair_reasons.get(key, 'No specific reason provided.')
            opinions_with_reasons += f"- Aspect: {aspect}, Opinion: {opinion} (Reason: {reason})\n"

        if not opinions_with_reasons:
            logger.info("No aspect-opinion pairs to classify.")
            state['sentiments'] = []
            return state

        for attempt in range(2):  # Allow one retry
            try:
                system_message = self.prompt.messages[0].prompt.template.format(format_instructions=self.parser.get_format_instructions())
                human_message = self.prompt.messages[1].prompt.template.format(text=state['text'], opinions_with_reasons=opinions_with_reasons, lang=lang)

                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": human_message}
                ]
                
                # Use the working llm_chat function with token tracking
                raw_response, usage = llm_chat(messages, model_name=self.model_name, temperature=0, return_usage=True)
                
                # Store token usage for cost calculation
                if usage:
                    if not hasattr(self, 'token_usage'):
                        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
                    self.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    self.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)

                result = self.parser.parse(raw_response)
                
                state['sentiments'] = [item.model_dump() for item in result.sentiments]
                return state  # Success, exit the loop and return
                
            except Exception as e:
                logger.error(f"Error in sentiment classification (Attempt {attempt + 1}): {e}")
                if attempt == 0:
                    logger.info("Retrying sentiment classification...")
                    continue
        
        # If both attempts fail
        logger.error("Both attempts for sentiment classification failed.")
        state['sentiments'] = [] # Ensure the key exists even on failure
        return state
        
    def _normalize_key(self, value):
        """Helper to normalize keys for dictionary lookups."""
        return str(value).lower().strip() if value is not None else "null" 