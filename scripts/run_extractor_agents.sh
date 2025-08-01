#!/bin/bash
set -e

# Define default parameters
MODEL="gpt-4o"
NUM_SAMPLE=1000
DOMAIN="restaurant"
BATCH=false
DATA_DIR="data"
OUTPUT_DIR="output_multi"  # Changed to output_multi to match the LLM output structure
START_INDEX=0
PROMPT_TYPE="0shot"  # Added prompt type parameter

# Help function
function show_help {
    echo "Usage: ./run_extractor_agents.sh [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help               Show this help message"
    echo "  -m, --model MODEL        Specify model name(s) to use (default: $MODEL)"
    echo "                           Can specify a single model for all agents: -m deepseek-v3"
    echo "                           Or different models per agent (unified,category,sentiment):"
    echo "                           -m deepseek-v3,gpt-4o,gpt-4.1"
    echo "  -n, --num-sample N       Process first N samples (default: $NUM_SAMPLE)"
    echo "  -d, --domain DOMAIN      Domain to process: restaurant or laptop (default: $DOMAIN)"
    echo "  -b, --batch              Enable batch mode (default: $BATCH)"
    echo "  -p, --prompt-type TYPE   Prompt type: 0shot or 20shot (default: $PROMPT_TYPE)"
    echo "  --data-dir DIR           Data directory (default: $DATA_DIR)"
    echo "  --output-dir DIR         Output directory (default: $OUTPUT_DIR)"
    echo "  --start-index N          Start processing from index N (default: $START_INDEX)"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -n|--num-sample)
            NUM_SAMPLE="$2"
            shift 2
            ;;
        -d|--domain)
            DOMAIN="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH=true
            shift
            ;;
        -p|--prompt-type)
            PROMPT_TYPE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --start-index)
            START_INDEX="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate model name(s)
if [[ "$MODEL" == *","* ]]; then
    # Multi-model mode - validate each model
    IFS=',' read -ra MODELS <<< "$MODEL"
    for model in "${MODELS[@]}"; do
        if [[ ! "$model" =~ ^(gpt|gpt-4o|gpt-4.1|deepseek-v3|deepseek-r1)$ ]]; then
            echo "Error: Unsupported model '$model'. Choose from: gpt, gpt-4o, gpt-4.1, deepseek-v3, deepseek-r1"
            exit 1
        fi
    done
    echo "Using multiple models: $MODEL"
    # Use the first model for directory naming in multi-model mode
    MODEL_DIR="${MODELS[0]}"
else
    # Single model mode
    if [[ ! "$MODEL" =~ ^(gpt|gpt-4o|gpt-4.1|deepseek-v3|deepseek-r1)$ ]]; then
        echo "Error: Unsupported model. Choose from: gpt, gpt-4o, gpt-4.1, deepseek-v3, deepseek-r1"
        exit 1
    fi
    echo "Using single model: $MODEL for all agents"
    MODEL_DIR="$MODEL"
fi

# Validate domain
if [[ ! "$DOMAIN" =~ ^(restaurant|laptop)$ ]]; then
    echo "Error: Unsupported domain. Choose from: restaurant, laptop"
    exit 1
fi

# Validate prompt type and set prompt directory
if [[ "$PROMPT_TYPE" == "0shot" ]]; then
    PROMPT_DIR="0shot"
elif [[ "$PROMPT_TYPE" == "20shot" ]]; then
    PROMPT_DIR="20shot"
else
    echo "Error: Unsupported prompt type. Choose from: 0shot, 20shot"
    exit 1
fi

# Set input file path based on domain
if [[ "$DOMAIN" == "restaurant" ]]; then
    INPUT_FILE="$DATA_DIR/acos/rest16/test.txt"
else
    INPUT_FILE="$DATA_DIR/acos/laptop16/test.txt"
fi

# Set output directory for this run - mirror the LLM output structure
RUN_OUTPUT_DIR="$OUTPUT_DIR/${DOMAIN}/${PROMPT_DIR}/${MODEL_DIR}/top1_none_data_seed0_samples${NUM_SAMPLE}"

# Create output directory
mkdir -p "$RUN_OUTPUT_DIR"


# Run command
if [[ "$BATCH" == "true" ]]; then
    echo "Running in batch mode for $DOMAIN domain with $MODEL model(s), using $PROMPT_TYPE prompt type"
    echo "Starting from index $START_INDEX, processing $NUM_SAMPLE samples"
    echo "Output will be saved to: $RUN_OUTPUT_DIR"
    
    PYTHONPATH=. python src/agents/run_extractor_agents.py \
        --input-file "$INPUT_FILE" \
        --output-dir "$RUN_OUTPUT_DIR" \
        --domain "$DOMAIN" \
        --model "$MODEL" \
        --num-sample "$NUM_SAMPLE" \
        --start-index "$START_INDEX" \
        --prompt-type "$PROMPT_TYPE" \
        --batch \
        --track-tokens
        
    # Calculate and display cost after completion
    if [ -f "$RUN_OUTPUT_DIR/token_usage.json" ]; then
        python agent_cost_calculator.py "$RUN_OUTPUT_DIR/token_usage.json" "$MODEL"
    else
        echo "Warning: Token usage data not found. Cost calculation skipped."
    fi
else
    echo "Running in interactive mode for $DOMAIN domain with $MODEL model(s), using $PROMPT_TYPE prompt type"
    PYTHONPATH=. python src/agents/run_extractor_agents.py \
        --domain "$DOMAIN" \
        --model "$MODEL" \
        --prompt-type "$PROMPT_TYPE" \
        --input "The food was great but the service was slow." \
        --track-tokens
        
    # For interactive mode, display cost if token usage is available
    if [ -f "token_usage.json" ]; then
        python agent_cost_calculator.py "token_usage.json" "$MODEL"
    else
        echo "Warning: Token usage data not found. Cost calculation skipped."
    fi
fi 