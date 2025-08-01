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

# Default number of samples to process, can be overridden with NUM_SAMPLES=100 bash scripts/run_llm_rest.sh
NUM_SAMPLES=${NUM_SAMPLES:-583}  # Changed default to 10 for testing
echo "Testing on $NUM_SAMPLES samples"

# Default prompt type, can be overridden with PROMPT_TYPE=0shot bash scripts/run_llm_rest.sh
PROMPT_TYPE=${PROMPT_TYPE:-improvedfewshot}
echo "Using prompt type: $PROMPT_TYPE"

cd src

DATA=rest16 # Fixed for restaurant dataset
TASK=acos
K=1
INFER_PATH=$K
CTRL_TOKEN=none
FIXED_SEED=0 # Define a fixed seed for consistency

# Determine prompt type for directory name
if [[ "$PROMPT_TYPE" == "0shot" ]]; then
    PROMPT_DIR="zeroshot"
elif [[ "$PROMPT_TYPE" == "zeroshot" ]]; then
    PROMPT_DIR="zeroshot"
    # Override PROMPT_TYPE to match actual file name
    PROMPT_TYPE="0shot"
else
    PROMPT_DIR="20shot"
    # Override PROMPT_TYPE to match actual file name
    PROMPT_TYPE="20shot"
fi

# Updated output directory structure: output_single/restaurant/{prompt_type}/{model}
OUT_DIR="../output_single/restaurant/${PROMPT_DIR}/${MODEL}/top${K}_${CTRL_TOKEN}_data${DATA_RATIO}_seed${FIXED_SEED}_samples${NUM_SAMPLES}"

mkdir -p $OUT_DIR


# Run inference
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
    # Generate report
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