#!/usr/bin/env python
"""
Test script for the LangGraph ACOS pipeline.

This script processes a simple example review text through the pipeline and prints the results.
"""

import sys
import os
import json

# Add project root to path to allow direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from agents.graph import run_pipeline, format_results

def main():
    """Run a test example through the pipeline."""
    # Restaurant example
    restaurant_text = "The pasta was delicious but service was slow."
    print(f"Testing restaurant review: {restaurant_text}")
    restaurant_state = run_pipeline(restaurant_text, domain="restaurant")
    restaurant_results = format_results(restaurant_state)
    print(json.dumps(restaurant_results, indent=2))
    
    print("\n" + "-" * 50 + "\n")
    
    # Laptop example
    laptop_text = "The battery life is excellent but the keyboard feels cheap."
    print(f"Testing laptop review: {laptop_text}")
    laptop_state = run_pipeline(laptop_text, domain="laptop")
    laptop_results = format_results(laptop_state)
    print(json.dumps(laptop_results, indent=2))

if __name__ == "__main__":
    main() 