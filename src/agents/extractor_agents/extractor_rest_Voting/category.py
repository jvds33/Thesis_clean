"""
Aspect-category classifier for the multi-agent ACOS framework, laptop domain.

Input state keys required:
• text: str
• all_pairs: List[Dict[str, str]]
• extracted_pairs: List[Dict[str, str]]  # From unified extractor, includes reasons

Output state keys added/overwritten:
• categorized_pairs: List[Dict[str, str]]  # List of {"aspect": "...", "opinion": "...", "category": "..."}
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, field_validator
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
import logging
import sys
import os

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from src.baseline.api import llm_chat

logger = logging.getLogger(__name__)

class CategoryPair(BaseModel):
    aspect: Optional[str] = None
    opinion: Optional[str] = None
    category: Optional[str] = None

    @field_validator('aspect', 'opinion', mode='before')
    def convert_none_to_null(cls, v: Optional[str]) -> str:
        """Convert None values to 'null' string."""
        return "null" if v is None else v

class CategoryClassificationResponse(BaseModel):
    categorized_pairs: List[CategoryPair]

class CategoryAgent:
    def __init__(self, llm=None, model="gpt-4o", prompt_type="0shot"):
        """Initialize the Category Classification agent."""
        self.llm = llm
        self.model_name = model.lower()
        self.prompt_type = prompt_type
        
        # Set up output parser
        self.parser = PydanticOutputParser(pydantic_object=CategoryClassificationResponse)
        
        # Load the prompt from file
        self.system_prompt = self._load_prompt_from_file()
        
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", 
             "Text:\n{text}\n\n"
             "Aspect-opinion pairs to classify with their reasons:\n{aspects_with_context}\n\n"
             "JSON output:")
        ])
        
        self.prompt = self.prompt.partial(
            format_instructions=self.parser.get_format_instructions()
        )
        
    def _load_prompt_from_file(self):
        """Load the system prompt from the txt file."""
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))

        if self.prompt_type in ["zeroshot", "0shot"]:
            prompt_file_name = "category_prompt_0shot.txt"
        else:
            prompt_file_name = "category_prompt.txt"

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

    def classify_category(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Classify the category for each aspect-opinion pair."""
        return self.run(state)
        
    def run(self, state):
        """Run the category classification agent."""
        if 'all_pairs' not in state or 'text' not in state:
            raise ValueError("Input state must contain 'all_pairs' and 'text' keys")

        # Create a mapping of aspect-opinion pairs to their reasons
        pair_reasons = {}
        if 'extracted_pairs' in state:
            for pair in state['extracted_pairs']:
                aspect = pair.get('aspect')
                opinion = pair.get('opinion')
                reason = pair.get('reason', '')
                key = (str(aspect).lower().strip() if aspect else 'null', str(opinion).lower().strip() if opinion else 'null')
                pair_reasons[key] = reason
        
        # Format the aspects with context for the prompt
        aspects_with_context = ""
        for pair in state['all_pairs']:
            aspect = pair.get('aspect')
            opinion = pair.get('opinion')
            key = (str(aspect).lower().strip() if aspect else 'null', str(opinion).lower().strip() if opinion else 'null')
            reason = pair_reasons.get(key, 'No specific reason provided.')
            aspects_with_context += f"- Aspect: {aspect}, Opinion: {opinion} (Reason: {reason})\n"

        if not aspects_with_context:
            logger.info("No aspect-opinion pairs to categorize.")
            state['categorized_pairs'] = []
            return state
        
        for attempt in range(2):  # Allow one retry
            try:
                system_message = self.prompt.messages[0].prompt.template.format(format_instructions=self.parser.get_format_instructions())
                human_message = self.prompt.messages[1].prompt.template.format(text=state['text'], aspects_with_context=aspects_with_context)

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

                state['categorized_pairs'] = [item.model_dump() for item in result.categorized_pairs]
                return state  # Success, exit the loop and return

            except Exception as e:
                logger.error(f"Error in category classification (Attempt {attempt + 1}): {e}")
                if attempt == 0:
                    logger.info("Retrying category classification...")
                    continue  # Go to next attempt
        
        # If both attempts fail
        logger.error("Both attempts for category classification failed.")
        state['categorized_pairs'] = []
        return state 