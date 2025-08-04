# ACOS Pipeline Tests

This directory contains tests for the ACOS (Aspect-Category-Opinion-Sentiment) pipeline.

## Test Pipeline

The `test_pipeline.py` file contains a test that verifies the ACOS pipeline produces at least one quadruple when given an input text. It uses a `DummyLLM` that returns predefined responses to simulate the behavior of the real LLMs.

### Running the Test

You can run the test using pytest:

```bash
# Run with minimal output
python -m pytest src/tests/test_pipeline.py -v

# Run with detailed output
python src/tests/run_test_with_output.py
```

### What the Test Shows

The test demonstrates the complete flow of the ACOS pipeline:

1. **Input Text**: A simple sentence "The food was delicious."

2. **Extraction Phase**:
   - The `UnifiedExtractorAgent` extracts aspect-opinion pairs
   - Output: `(food, delicious)` with reasoning

3. **Classification Phase** (runs in parallel):
   - The `CategoryAgent` classifies the aspect into a category
   - Output: `food quality` for the aspect `food`
   - The `SentimentAgent` determines the sentiment of the opinion
   - Output: `positive` for the opinion `delicious`

4. **Quadruple Formation**:
   - The `run_acos_pipeline` function (Review Processing Coordinator) combines the results
   - Final quadruple: `['food', 'delicious', 'food quality', 'positive']`

### Pipeline Architecture

The test demonstrates the pipeline's architecture as described in the thesis:

- **Extract-Classify Structure**: Single extraction followed by parallel classifications
- **Acquaintance Model**: The extractor agent fans out to both classifiers
- **Service Model**: Each agent provides a specific service (`extract_pairs`, `classify_category`, `classify_sentiment`)
- **Role Responsibilities**: Each agent has clear responsibilities and permissions

This test verifies that the pipeline correctly implements the design specified in the thesis, with the proper flow of data between agents. 