#!/bin/bash
set -ex

# Default model is gpt, but can be overridden by passing MODEL=deepseek-v3 before running the script
MODEL=${MODEL:-gpt}
echo "Using model: $MODEL"

# Verify model is one of the supported models
if [[ ! "$MODEL" =~ ^(gpt|gpt-4o|deepseek-v3)$ ]]; then
    echo "Error: Unsupported model. Choose from: gpt, gpt-4o, deepseek-v3"
    exit 1
fi

# Default number of samples to process, can be overridden with NUM_SAMPLES=100 bash scripts/run_llm_laptop.sh
NUM_SAMPLES=${NUM_SAMPLES:-1000}
echo "Testing on $NUM_SAMPLES samples"

cd src

DATA=laptop16 # Fixed for laptop dataset
TASK=acos
K=1
INFER_PATH=$K
CTRL_TOKEN=none
FIXED_SEED=0 # Define a fixed seed for consistency

# Determine prompt type for directory name
if [[ "$PROMPT_TYPE" == "0shot" ]]; then
    PROMPT_DIR="0shot"
elif [[ "$PROMPT_TYPE" == "zeroshot" ]]; then
    PROMPT_DIR="0shot"
else
    PROMPT_DIR="20shot"
    # Override PROMPT_TYPE to match actual file name
    PROMPT_TYPE="20shot"
fi

# Updated output directory structure: output_single/laptop/{prompt_type}/{model}
OUT_DIR="../output_single/laptop/${PROMPT_DIR}/${MODEL}/top${K}_${CTRL_TOKEN}_data${DATA_RATIO}_seed${FIXED_SEED}_samples${NUM_SAMPLES}"

mkdir -p $OUT_DIR

# Create a temporary Python script for cost calculation
cat > ../cost_calculator.py << 'EOF'
import json
import sys
import os

def calculate_cost(token_usage_file, model_name):
    # Pricing table
    prices = {
        "gpt-4o":      {"prompt": 0.005,   "completion": 0.015},
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
        
        # Calculate cost
        if model_key in prices:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
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
EOF

python -u llms/infer.py \
    --data_path "../data/" \
    --dataset $DATA \
    --output_dir $OUT_DIR \
    --save_top_k 0 \
    --task $TASK \
    --top_k $K \
    --ctrl_token $CTRL_TOKEN \
    --num_path $INFER_PATH \
    --seed $FIXED_SEED \
    --lowercase \
    --sort_label \
    --single_view_type heuristic \
    --prompt_type $PROMPT_TYPE \
    --model $MODEL \
    --num_sample $NUM_SAMPLES \
    --track_tokens \
    --use_pydantic

# Check if inference completed successfully
if [ $? -eq 0 ]; then
    echo "Inference completed successfully. Generating report..."
    python -u llms/report_generator.py $OUT_DIR/${DATA}_${PROMPT_TYPE}_predictions.json
    
    # Calculate and display cost
    if [ -f "$OUT_DIR/token_usage.json" ]; then
        python ../cost_calculator.py "$OUT_DIR/token_usage.json" "$MODEL"
    else
        echo "Warning: Token usage data not found. Cost calculation skipped."
    fi
else
    echo "Error: Inference failed. Not generating report."
    exit 1
fi 