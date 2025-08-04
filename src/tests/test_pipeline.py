import unittest
from unittest.mock import MagicMock, patch
import json
import sys
import os
import logging
from typing import Any, Dict, List, Optional

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.run_extractor_agents import run_acos_pipeline
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import Field

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ACOS_TEST")

class DummyLLM(BaseChatModel):
    """A dummy LLM that always returns a fixed response for testing."""
    
    model_name: str = Field(default="dummy")
    temperature: float = Field(default=0.0)
    agent_name: str = Field(default="unknown")
    call_count: int = Field(default=0)
    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        """Return realistic responses for each agent in the pipeline."""
        self.call_count += 1
        prompt = messages[0].content.lower()
        
        # Determine which agent is calling based on the prompt content
        agent_type = "Unknown"
        if "extract aspect-opinion pairs" in prompt:
            agent_type = "Extractor Agent"
        elif "classify the sentiment" in prompt:
            agent_type = "Sentiment Classifier"
        elif "classify the category" in prompt:
            agent_type = "Category Classifier"
        
        logger.info(f"\n[DUMMY LLM] Call #{self.call_count} - {agent_type}")
        logger.info(f"[DUMMY LLM] System prompt: {messages[0].content[:100]}...")
        logger.info(f"[DUMMY LLM] User prompt: {messages[-1].content[:100]}...")
        
        if "extract aspect-opinion pairs" in prompt:
            # Detailed response for the extractor agent
            content = json.dumps({
                "extracted_pairs": [
                    {
                        "aspect": "food",
                        "opinion": "delicious",
                        "reason": "The sentence explicitly states that 'the food was delicious', creating a direct connection between the aspect 'food' and the opinion 'delicious'. This is a clear evaluative statement about the quality of the food."
                    }
                ]
            })
            logger.info("[DUMMY LLM] Extractor Agent Response:")
            logger.info("  - Extracted 1 aspect-opinion pair:")
            logger.info("    - Aspect: 'food'")
            logger.info("    - Opinion: 'delicious'")
            logger.info("    - Reason: The sentence explicitly states that 'the food was delicious'...")
            
        elif "classify the sentiment" in prompt:
            # Detailed response for the sentiment classifier
            content = json.dumps({
                "sentiments": [
                    {
                        "aspect": "food",
                        "opinion": "delicious",
                        "sentiment": "positive"
                    }
                ]
            })
            logger.info("[DUMMY LLM] Sentiment Classifier Response:")
            logger.info("  - Classified sentiment for 1 pair:")
            logger.info("    - Aspect: 'food'")
            logger.info("    - Opinion: 'delicious'")
            logger.info("    - Sentiment: 'positive'")
            
        elif "classify the category" in prompt:
            # Detailed response for the category classifier
            content = json.dumps({
                "categorized_pairs": [
                    {
                        "aspect": "food",
                        "opinion": "delicious",
                        "category": "food quality"
                    }
                ]
            })
            logger.info("[DUMMY LLM] Category Classifier Response:")
            logger.info("  - Classified category for 1 pair:")
            logger.info("    - Aspect: 'food'")
            logger.info("    - Opinion: 'delicious'")
            logger.info("    - Category: 'food quality'")
            
        else:
            content = json.dumps({"error": f"Unknown prompt type: {prompt[:50]}..."})
            logger.warning(f"[DUMMY LLM] Unknown prompt type: {prompt[:50]}...")
        
        logger.info(f"[DUMMY LLM] Raw JSON response: {content}")
        logger.info("-" * 60)
            
        # Create a ChatGeneration with the content
        generation = ChatGeneration(message=AIMessage(content=content))
        
        # Return a ChatResult with the generation
        return ChatResult(generations=[generation])
    
    @property
    def _llm_type(self) -> str:
        return "dummy"

class TestACOSPipeline(unittest.TestCase):
    """Test cases for the ACOS pipeline using a DummyLLM."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.dummy_llm = DummyLLM()
        
        # Create a patcher for the get_llm function
        self.get_llm_patcher = patch('agents.run_extractor_agents.get_llm')
        self.mock_get_llm = self.get_llm_patcher.start()
        self.mock_get_llm.return_value = self.dummy_llm
        
    def tearDown(self):
        """Clean up after each test method."""
        self.get_llm_patcher.stop()
        
    def test_pipeline_produces_quadruples(self):
        """Test that the pipeline produces at least one ACOS quadruple."""
        # Sample input text - use a unique text to avoid duplicate detection
        import time
        text = f"The food was delicious. {int(time.time())}"  # Add timestamp to make unique
        
        # Log test header
        logger.info("\n" + "="*80)
        logger.info("ACOS PIPELINE TEST")
        logger.info("="*80)
        logger.info(f"Input text: '{text}'")
        logger.info("-"*80)
        
        # Run pipeline with DummyLLM
        with patch('builtins.print'):  # Temporarily suppress prints during execution
            result = run_acos_pipeline(
                text=text,
                domain="restaurant",
                model_name="dummy",
                api_version="2025-01-01-preview",
                track_tokens=False,
                prompt_type="0shot"
            )
        
        # Handle case where duplicate prevention might return an error
        if 'error' in result:
            if 'already processed' in result.get('error', ''):
                logger.info("Test text was already processed (duplicate prevention working)")
                return  # Skip test if duplicate was detected
            else:
                self.fail(f"Pipeline returned error: {result['error']}")
        
        # Log detailed pipeline steps and outputs
        logger.info("PIPELINE EXECUTION DETAILS:")
        logger.info("-"*80)
        
        # Step 1: Extraction
        logger.info("STEP 1: ASPECT-OPINION EXTRACTION")
        extracted_pairs = result['state'].get('extracted_pairs', [])
        
        for i, pair in enumerate(extracted_pairs, 1):
            logger.info(f"  Pair {i}:")
            logger.info(f"    Aspect: '{pair.get('aspect')}'")
            logger.info(f"    Opinion: '{pair.get('opinion')}'")
            logger.info(f"    Reason: {pair.get('reason')}")
        
        # Step 2: Category Classification
        logger.info("\nSTEP 2: CATEGORY CLASSIFICATION")
        categorized_pairs = result['state'].get('categorized_pairs', [])
        
        for i, pair in enumerate(categorized_pairs, 1):
            logger.info(f"  Pair {i}:")
            logger.info(f"    Aspect: '{pair.get('aspect')}'")
            logger.info(f"    Opinion: '{pair.get('opinion')}'")
            logger.info(f"    Category: '{pair.get('category')}'")
        
        # Step 3: Sentiment Classification
        logger.info("\nSTEP 3: SENTIMENT CLASSIFICATION")
        sentiments = result['state'].get('sentiments', [])
        
        for i, sentiment_item in enumerate(sentiments, 1):
            logger.info(f"  Pair {i}:")
            logger.info(f"    Aspect: '{sentiment_item.get('aspect')}'")
            logger.info(f"    Opinion: '{sentiment_item.get('opinion')}'")
            logger.info(f"    Sentiment: '{sentiment_item.get('sentiment')}'")
        
        # Final Quadruples
        logger.info("\nFINAL ACOS QUADRUPLES:")
        quads = result.get('quadruples', [])
        
        for i, quad in enumerate(quads, 1):
            logger.info(f"  Quadruple {i}: {quad}")
            logger.info(f"    Aspect: '{quad[0]}'")
            logger.info(f"    Opinion: '{quad[1]}'")
            logger.info(f"    Category: '{quad[2]}'")
            logger.info(f"    Sentiment: '{quad[3]}'")
        logger.info("-"*80)
        
        # Assertions
        self.assertGreater(len(quads), 0, "Pipeline should produce at least one quadruple")
        
        # Verify the structure of the first quadruple
        first_quad = quads[0]
        self.assertEqual(len(first_quad), 4, "Each quadruple should have 4 elements")
        
        aspect, opinion, category, sentiment = first_quad
        
        # Verify expected values from DummyLLM
        self.assertEqual(aspect, "food")
        self.assertEqual(opinion, "delicious")
        self.assertEqual(category, "food quality")  # This is what DummyLLM returns
        self.assertEqual(sentiment, "positive")
        
        # Log test summary
        logger.info("\nTEST SUMMARY:")
        logger.info(f"  Input: '{text}'")
        logger.info(f"  Expected quadruple: ['food', 'delicious', 'food quality', 'positive']")
        logger.info(f"  Actual quadruple: {first_quad}")
        logger.info("  Test result: PASSED")
        logger.info("="*80)

if __name__ == '__main__':
    unittest.main() 