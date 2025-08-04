"""
Validation module for ACOS pipeline safety conditions.

Implements all pre-conditions and post-conditions from Chapter 3 of the thesis
to ensure robustness and fault tolerance.
"""

import logging
import json
import hashlib
import os
from typing import Dict, Any, List, Optional, Set
from datetime import datetime

# Import the correct category definitions
from src.categories import VALID_RESTAURANT_CATEGORIES, VALID_LAPTOP_CATEGORIES

logger = logging.getLogger(__name__)

# Valid sentiments for validation
VALID_SENTIMENTS = {"positive", "negative", "neutral"}

class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass

class AuditLogger:
    """Structured audit logging for pipeline operations."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.audit_file = os.path.join(log_dir, f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl")
    
    def log_operation(self, operation: str, review_id: str, agent: str, 
                     input_data: Any, output_data: Any, success: bool, 
                     error_msg: Optional[str] = None, metadata: Optional[Dict] = None):
        """Log a structured audit entry."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "review_id": review_id,
            "agent": agent,
            "input_data": str(input_data)[:500],  # Truncate for readability
            "output_data": str(output_data)[:500],
            "success": success,
            "error_msg": error_msg,
            "metadata": metadata or {}
        }
        
        try:
            with open(self.audit_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(audit_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

class ProcessingStateManager:
    """Manages processing state for duplicate prevention and fault-tolerant re-runs."""
    
    def __init__(self, state_dir: str = "processing_state"):
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.processed_file = os.path.join(state_dir, "processed_reviews.json")
        self.state_file = os.path.join(state_dir, "pipeline_state.json")
        
        # Load existing processed reviews
        self.processed_reviews = self._load_processed_reviews()
    
    def _load_processed_reviews(self) -> Set[str]:
        """Load set of already processed review IDs."""
        try:
            if os.path.exists(self.processed_file):
                with open(self.processed_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return set(data.get('processed_ids', []))
            return set()
        except Exception as e:
            logger.error(f"Failed to load processed reviews: {e}")
            return set()
    
    def _save_processed_reviews(self):
        """Save processed review IDs to disk."""
        try:
            with open(self.processed_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'processed_ids': list(self.processed_reviews),
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save processed reviews: {e}")
    
    def generate_review_id(self, text: str, domain: str) -> str:
        """Generate a unique review ID based on content and domain."""
        content = f"{domain}:{text}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
    
    def is_already_processed(self, review_id: str) -> bool:
        """Check if review has already been processed."""
        return review_id in self.processed_reviews
    
    def mark_as_processed(self, review_id: str):
        """Mark review as successfully processed."""
        self.processed_reviews.add(review_id)
        self._save_processed_reviews()
    
    def save_pipeline_state(self, review_id: str, state: Dict[str, Any], stage: str):
        """Save intermediate pipeline state for fault tolerance."""
        try:
            state_data = {
                'review_id': review_id,
                'stage': stage,
                'state': state,
                'timestamp': datetime.now().isoformat()
            }
            
            state_file = os.path.join(self.state_dir, f"state_{review_id}.json")
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save pipeline state: {e}")
    
    def load_pipeline_state(self, review_id: str) -> Optional[Dict[str, Any]]:
        """Load saved pipeline state for recovery."""
        try:
            state_file = os.path.join(self.state_dir, f"state_{review_id}.json")
            if os.path.exists(state_file):
                with open(state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Failed to load pipeline state: {e}")
            return None
    
    def cleanup_state(self, review_id: str):
        """Clean up intermediate state file after successful completion."""
        try:
            state_file = os.path.join(self.state_dir, f"state_{review_id}.json")
            if os.path.exists(state_file):
                os.remove(state_file)
        except Exception as e:
            logger.error(f"Failed to cleanup state file: {e}")

def validate_review_text(text: str) -> bool:
    """
    Validate input review text meets quality requirements.
    Implements: reviewTextValid = true
    """
    if not text or not isinstance(text, str):
        raise ValidationError("Review text must be a non-empty string")
    
    text = text.strip()
    
    if len(text) < 3:
        raise ValidationError("Review text too short (minimum 3 characters)")
    
    if len(text) > 10000:
        raise ValidationError("Review text too long (maximum 10000 characters)")
    
    # Check for basic text quality
    if not any(c.isalnum() for c in text):
        raise ValidationError("Review text must contain alphanumeric characters")
    
    return True

def validate_extracted_pairs(pairs: List[Dict[str, Any]], text: str) -> bool:
    """
    Validate extracted aspect-opinion pairs.
    Implements: extractedPairsValid = true
    """
    if not isinstance(pairs, list):
        raise ValidationError("Extracted pairs must be a list")
    
    for i, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ValidationError(f"Pair {i} must be a dictionary")
        
        # Check required fields
        if 'aspect' not in pair or 'opinion' not in pair:
            raise ValidationError(f"Pair {i} missing required fields (aspect, opinion)")
        
        # Validate aspect and opinion are strings or null
        aspect = pair.get('aspect')
        opinion = pair.get('opinion')
        
        if aspect is not None and not isinstance(aspect, str):
            raise ValidationError(f"Pair {i} aspect must be string or null")
        
        if opinion is not None and not isinstance(opinion, str):
            raise ValidationError(f"Pair {i} opinion must be string or null")
        
        # Check that at least one of aspect or opinion is not null/empty
        if (not aspect or aspect == "null") and (not opinion or opinion == "null"):
            raise ValidationError(f"Pair {i} must have at least one non-null aspect or opinion")
    
    return True

def validate_reasoning_provided(pairs: List[Dict[str, Any]]) -> bool:
    """
    Validate that reasoning is provided for extracted pairs.
    Implements: reasoningProvided = true
    """
    for i, pair in enumerate(pairs):
        reason = pair.get('reason')
        if not reason or reason.strip() == "":
            raise ValidationError(f"Pair {i} missing reasoning explanation")
        
        if len(reason.strip()) < 5:
            raise ValidationError(f"Pair {i} reasoning too short (minimum 5 characters)")
    
    return True

def validate_category_labels(categorized_pairs: List[Dict[str, Any]], domain: str) -> bool:
    """
    Validate category labels are from valid taxonomy.
    Implements: categoryLabelsValid = true, taxonomyCompliant = true
    """
    valid_categories = VALID_RESTAURANT_CATEGORIES if domain == "restaurant" else VALID_LAPTOP_CATEGORIES
    
    for i, pair in enumerate(categorized_pairs):
        category = pair.get('category')
        if not category:
            raise ValidationError(f"Categorized pair {i} missing category")
        
        if category not in valid_categories:
            raise ValidationError(f"Categorized pair {i} has invalid category '{category}'. Valid categories: {valid_categories}")
    
    return True

def validate_sentiment_labels(sentiments: List[Dict[str, Any]]) -> bool:
    """
    Validate sentiment labels are from valid set.
    Implements: sentimentLabelsValid = true
    """
    for i, sentiment_item in enumerate(sentiments):
        sentiment = sentiment_item.get('sentiment')
        if not sentiment:
            raise ValidationError(f"Sentiment item {i} missing sentiment")
        
        if sentiment not in VALID_SENTIMENTS:
            raise ValidationError(f"Sentiment item {i} has invalid sentiment '{sentiment}'. Valid sentiments: {VALID_SENTIMENTS}")
    
    return True

def validate_final_quadruples(quadruples: List[List[str]]) -> bool:
    """
    Validate final quadruples are complete and well-formed.
    Implements: finalQuadruplesComplete = true
    """
    if not isinstance(quadruples, list):
        raise ValidationError("Quadruples must be a list")
    
    for i, quad in enumerate(quadruples):
        if not isinstance(quad, list):
            raise ValidationError(f"Quadruple {i} must be a list")
        
        if len(quad) != 4:
            raise ValidationError(f"Quadruple {i} must have exactly 4 elements, got {len(quad)}")
        
        aspect, opinion, category, sentiment = quad
        
        # All elements should be strings
        if not all(isinstance(elem, str) for elem in quad):
            raise ValidationError(f"Quadruple {i} all elements must be strings")
        
        # At least aspect or opinion should be non-null
        if (not aspect or aspect == "null") and (not opinion or opinion == "null"):
            raise ValidationError(f"Quadruple {i} must have at least one non-null aspect or opinion")
    
    return True

def validate_pipeline_state(state: Dict[str, Any], stage: str, domain: str) -> bool:
    """
    Comprehensive validation of pipeline state at different stages.
    """
    try:
        # Always validate text input
        if 'text' in state:
            validate_review_text(state['text'])
        
        # Stage-specific validations
        if stage in ['extraction', 'classification', 'final']:
            if 'extracted_pairs' in state and state['extracted_pairs']:
                validate_extracted_pairs(state['extracted_pairs'], state.get('text', ''))
                validate_reasoning_provided(state['extracted_pairs'])
        
        if stage in ['classification', 'final']:
            if 'categorized_pairs' in state and state['categorized_pairs']:
                validate_category_labels(state['categorized_pairs'], domain)
            
            if 'sentiments' in state and state['sentiments']:
                validate_sentiment_labels(state['sentiments'])
        
        if stage == 'final':
            if 'quadruples' in state and state['quadruples']:
                validate_final_quadruples(state['quadruples'])
        
        return True
        
    except ValidationError as e:
        logger.error(f"Validation failed at stage '{stage}': {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected validation error at stage '{stage}': {e}")
        raise ValidationError(f"Validation failed: {e}")

# Global instances for easy access
audit_logger = AuditLogger()
state_manager = ProcessingStateManager() 