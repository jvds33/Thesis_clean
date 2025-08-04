import json
import sys
import os

def calculate_cost(token_usage_file, model_name):
    # Pricing table
    prices = {
        "gpt-4o":      {"prompt": 0.0025,   "completion": 0.015},
        "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
        "deepseek-v3": {"prompt": 0.00114, "completion": 0.00456}
    }

    # Default to gpt-4o-mini prices if model is "gpt"
    model_key = model_name
    if model_key == "gpt":
        model_key = "gpt-4o-mini"

    try:
        # Load token usage data
        with open(token_usage_file, "r") as f:
            usage = json.load(f)
        
        # Check if the usage is grouped by agent (new format) or not (old format)
        if isinstance(usage, dict) and any(key in usage for key in ["unified", "category", "sentiment"]):
            # New format - grouped by agent
            total_prompt_tokens = 0
            total_completion_tokens = 0
            
            # Print per-agent breakdown
            print(f"\n===== TOKEN USAGE BY AGENT =====")
            for agent, tokens in usage.items():
                agent_prompt_tokens = tokens.get("prompt_tokens", 0)
                agent_completion_tokens = tokens.get("completion_tokens", 0)
                total_prompt_tokens += agent_prompt_tokens
                total_completion_tokens += agent_completion_tokens
                
                print(f"Agent: {agent}")
                print(f"  Prompt tokens: {agent_prompt_tokens:,}")
                print(f"  Completion tokens: {agent_completion_tokens:,}")
                
                if model_key in prices:
                    agent_prompt_cost = agent_prompt_tokens * prices[model_key]["prompt"] / 1000
                    agent_completion_cost = agent_completion_tokens * prices[model_key]["completion"] / 1000
                    agent_total_cost = agent_prompt_cost + agent_completion_cost
                    print(f"  Cost: ${agent_total_cost:.4f}")
                print()
            
            # Calculate total cost
            if model_key in prices:
                prompt_cost = total_prompt_tokens * prices[model_key]["prompt"] / 1000
                completion_cost = total_completion_tokens * prices[model_key]["completion"] / 1000
                total_cost = prompt_cost + completion_cost
                
                print(f"===== TOTAL COST SUMMARY =====")
                print(f"Model: {model_key}")
                print(f"Total prompt tokens: {total_prompt_tokens:,} (${prompt_cost:.4f})")
                print(f"Total completion tokens: {total_completion_tokens:,} (${completion_cost:.4f})")
                print(f"Total cost: ${total_cost:.4f}")
                print(f"=============================")
            else:
                print(f"Warning: No pricing information available for model {model_key}")
        else:
            # Old format - not grouped by agent
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            if model_key in prices:
                prompt_cost = prompt_tokens * prices[model_key]["prompt"] / 1000
                completion_cost = completion_tokens * prices[model_key]["completion"] / 1000
                total_cost = prompt_cost + completion_cost
                
                print(f"\n===== COST SUMMARY =====")
                print(f"Model: {model_key}")
                print(f"Prompt tokens: {prompt_tokens:,} (${prompt_cost:.4f})")
                print(f"Completion tokens: {completion_tokens:,} (${completion_cost:.4f})")
                print(f"Total cost: ${total_cost:.4f}")
                print(f"========================")
            else:
                print(f"Warning: No pricing information available for model {model_key}")
    except Exception as e:
        print(f"Error calculating cost: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python cost_calculator.py <token_usage_file> <model_name>")
        sys.exit(1)
    
    token_usage_file = sys.argv[1]
    model_name = sys.argv[2]
    calculate_cost(token_usage_file, model_name)
