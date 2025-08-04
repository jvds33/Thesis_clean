# ACOS Multi-Agent System

A Multi-Agent System for Aspect-Category-Opinion-Sentiment (ACOS) extraction from product reviews, following the Gaia methodology.

## Overview

This repository implements a Multi-Agent System for Aspect-based Sentiment Analysis that extracts complete ACOS quadruples from product reviews. The system follows the Gaia methodology, treating the MAS as an organized society of roles with a step-wise set of artifacts.

### System Architecture

The system consists of three specialized agents:

1. **Aspect-Opinion Extractor Agent**: Identifies aspect-opinion pairs from review text using ensemble voting with multiple LLM calls.
   - Exposes `extract_pairs` service
   - Implements `InitiateExtraction`, `VotingConsensus`, and `ProvideExtractionReasoning` protocols

2. **Sentiment Classifier Agent**: Classifies sentiment polarity (positive/negative/neutral) for each aspect-opinion pair.
   - Exposes `classify_sentiment` service
   - Implements `ClassifySentiment` and `ContextualAnalysis` protocols

3. **Category Classifier Agent**: Assigns aspect categories from a predefined taxonomy.
   - Exposes `classify_category` service
   - Implements `ClassifyCategory` and `TaxonomyMapping` protocols

The pipeline follows an EXTRACT-CLASSIFY structure with a single extraction phase followed by parallel classifications and a final merge in the coordinator.

## Setup

### Environment Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the root directory with your Azure OpenAI API credentials:
   ```
   # Azure OpenAI API Configuration
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
   AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
   OPENAI_API_VERSION=2025-01-01-preview
   
   # OpenAI Model Deployments
   AZURE_OPENAI_DEPLOYMENT_NAME_CHAT=gpt-4o-mini
   AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME=gpt-4o
   AZURE_OPENAI_GPT4O_MINI_DEPLOYMENT_NAME=gpt-4o-mini
   ```

3. You can use the provided `env_template.txt` as a starting point.

## Usage

### Running the Pipeline

The system supports both restaurant and laptop domains:

```bash
# Run the extractor agents pipeline in batch mode
bash scripts/run_extractor_agents.sh --domain restaurant --num-sample 10 --batch

# Run in interactive mode with custom input
bash scripts/run_extractor_agents.sh --domain laptop --input "The battery life is excellent but the keyboard is uncomfortable."
```

### Running the Baseline LLM Inference

```bash
# For restaurant domain
bash scripts/run_llm_rest.sh

# For laptop domain
bash scripts/run_llm_laptop.sh
```

### Testing

Run the test suite to verify the pipeline's functionality:

```bash
python src/tests/run_test_with_output.py
```

## Implementation Details

The implementation follows the Gaia methodology's design phase:

1. **Agent Model**: Maps roles to executable agent types
2. **Services Model**: Exposes each agent's public API
3. **Acquaintance Model**: Depicts mandatory communication links

The system uses a fan-out architecture where the extractor agent communicates with both classifier agents, while the classifiers themselves are purely reactive.
