# Multi-Agent ACOS Pipeline

This repository contains a Python-based pipeline for performing Aspect-Category-Opinion-Sentiment (ACOS) analysis on customer reviews. The pipeline uses a multi-agent architecture with voting capabilities to extract and analyze aspects, opinions, categories, and sentiments from text. It currently supports reviews for restaurants and laptops.

## Features

- **Unified Aspect-Opinion Extraction**: Identifies aspect terms and their associated opinions from review text
- **Category Classification**: Maps aspects to predefined domain-specific categories
- **Sentiment Analysis**: Determines the sentiment (positive, negative, neutral) for each aspect-opinion pair
- **Voting Mechanism**: Supports ensemble-based extraction for improved reliability
- **Detailed Reporting**: Generates comprehensive PDF reports with analysis results and statistics
- **Token Usage Tracking**: Optional tracking of API token usage for cost analysis

## Architecture

The pipeline uses three specialized agents:

1. **UnifiedExtractorAgent**: Extracts aspect-opinion pairs with explanatory reasoning
2. **CategoryAgent**: Classifies aspects into domain-specific categories
3. **SentimentAgent**: Determines sentiment polarity for aspect-opinion pairs

Each agent can operate in either:
- Zero-shot mode (`0shot`)
- Few-shot mode (`20shot`) with domain-specific examples

## Supported Domains

### Restaurant Domain
- Analyzes restaurant reviews
- Categories include: food quality, service, ambience, etc.
- Located in `extractor_rest_Voting/`

### Laptop Domain
- Analyzes laptop reviews
- Categories include: performance, build quality, battery life, etc.
- Located in `extractor_laptop_Voting/`

## Getting Started

### Running the Pipeline

Use the provided shell script to run the pipeline:

```bash
# Process restaurant reviews
bash scripts/run_extractor_agents.sh --domain restaurant --num-sample 3 --batch

# Process laptop reviews
bash scripts/run_extractor_agents.sh --domain laptop --num-sample 3 --batch
```

### Command Line Options

- `--domain`: Choose domain (`restaurant` or `laptop`)
- `--num-sample`: Number of samples to process
- `--model`: LLM model to use (`gpt-4o`, `gpt`, `deepseek-v3`, etc.)
- `--prompt-type`: Prompt type (`0shot` or `20shot`)
- `--batch`: Enable batch processing mode
- `--track-tokens`: Enable token usage tracking

### Interactive Mode

For single review analysis:

```bash
bash scripts/run_extractor_agents.sh --domain restaurant --input "The pasta was delicious but service was slow."
```

### Batch Mode

For processing multiple reviews:

```bash
bash scripts/run_extractor_agents.sh --domain restaurant \
    --batch \
    --input-file data/acos/rest16/test.txt \
    --output-dir output_multi/restaurant \
    --num-sample 100
```

## Output

The pipeline generates:
1. Structured JSON output with extracted ACOS quadruples
2. Detailed PDF report with:
   - Overall performance metrics
   - Confusion matrices
   - Error analysis
   - Token usage statistics (if enabled)
3. Token usage tracking (optional)

## Example Output

For the input "The pasta was delicious but service was slow":
```json
{
    "text": "The pasta was delicious but service was slow",
    "quads": [
        ["pasta", "delicious", "food quality", "positive"],
        ["service", "slow", "service general", "negative"]
    ]
}
```

## Performance Metrics

The pipeline evaluates:
- Precision, Recall, and F1 scores
- Aspect-Opinion pair extraction accuracy
- Category classification accuracy
- Sentiment classification accuracy
- Implicit aspect/opinion handling
