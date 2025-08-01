#!/usr/bin/env python
"""
Runner for the LangGraph ACOS pipeline.

This script processes review text through the LangGraph ACOS pipeline:
1. extract_aspects → Extract aspect terms from the text
2. extract_opinions → Extract opinion terms and their associated aspects
3. categorize_aspects → Classify aspects into predefined categories
4. decide_sentiment → Determine sentiment polarity for aspect-opinion pairs
5. link_and_check → Link aspects, opinions, categories, and sentiments into final quadruples

Usage:
    python run_agent_pipeline.py --domain restaurant --input "The pasta was delicious but service was slow." --output results.json
    python run_agent_pipeline.py --domain laptop --input "The battery life is excellent but the keyboard feels cheap." --output results.json
"""

import argparse
import json
import logging
import sys
import os
from typing import Dict, Any, Optional, List, Literal

# Add project root to path to allow direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import the pipeline implementation
from agents.graph import run_pipeline, format_results
from langchain_openai import AzureChatOpenAI
from llms.report_generator import main as generate_report, create_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Map domain names to dataset names expected by report_generator
DOMAIN_TO_DATASET = {
    "restaurant": "rest16",
    "laptop": "laptop16"
}

def save_results(results: Dict[str, Any], output_file: Optional[str] = None) -> None:
    """
    Save results to a file or print to stdout.
    
    Args:
        results: The results to save
        output_file: File to save to, or None to print to stdout
    """
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    else:
        json_str = json.dumps(results, indent=2)
        print(json_str)

def main():
    """Run the ACOS pipeline on a text input."""
    parser = argparse.ArgumentParser(description='Run the ACOS pipeline on a text input.')
    parser.add_argument('--domain', type=str, choices=['restaurant', 'laptop'], default='restaurant',
                        help='Domain of the review (restaurant or laptop)')
    parser.add_argument('--input', type=str,
                        help='Review text to analyze')
    parser.add_argument('--file', type=str,
                        help='File containing review text (one per line)')
    parser.add_argument('--output', type=str,
                        help='File to save results to (defaults to stdout)')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='OpenAI model to use')
    parser.add_argument('--api-version', type=str, default='2025-01-01-preview',
                        help='Azure OpenAI API version')
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input and not args.file:
        parser.error("Either --input or --file must be provided")
    if args.input and args.file:
        parser.error("Only one of --input or --file can be provided")
    
    # Process single review
    if args.input:
        logger.info(f"Processing {args.domain} domain review: {args.input[:50]}{'...' if len(args.input) > 50 else ''}")
        
        # Run the pipeline
        final_state = run_pipeline(
            text=args.input,
            domain=args.domain,
            lang="en"
        )
        
        # Format and save results
        results = format_results(final_state)
        save_results(results, args.output)
        
    # Process file of reviews
    else:
        logger.info(f"Processing reviews from file: {args.file}")
        
        # Determine output file base name
        output_base = args.output if args.output else 'results'
        if output_base.endswith('.json'):
            output_base = output_base[:-5]
        
        # Process each line in the file
        all_results = []
        with open(args.file, 'r') as f:
            for i, line in enumerate(f, 1):
                review = line.strip()
                if not review:
                    continue
                    
                logger.info(f"Processing review {i}: {review[:50]}{'...' if len(review) > 50 else ''}")
                
                # Run the pipeline
                final_state = run_pipeline(
                    text=review,
                    domain=args.domain,
                    lang="en"
                )
                
                # Format results
                results = format_results(final_state)
                all_results.append(results)
                
                # Save individual result
                if args.output:
                    individual_output = f"{output_base}_{i}.json"
                    save_results(results, individual_output)
        
        # Save all results
        if args.output:
            save_results({"results": all_results}, f"{output_base}_all.json")
            
            # Generate PDF report
            dataset = DOMAIN_TO_DATASET.get(args.domain, "rest16")
            predictions_file = f"{output_base}_all.json"
            report_file = f"{output_base}_report.pdf"
            logger.info(f"Generating report: {report_file}")
            create_report(predictions_file, report_file, dataset)

if __name__ == "__main__":
    main() 