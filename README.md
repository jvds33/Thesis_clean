# Aspect-Based Sentiment Analysis with LangGraph

This repository contains a Python-based pipeline for performing Aspect-Based Sentiment Analysis (ABSA) on customer reviews. The pipeline is built using LangGraph and is designed to extract aspects, opinions, and their associated sentiments from text. It currently supports reviews for restaurants and laptops.

## Features

- **Aspect Extraction**: Identifies aspect terms mentioned in the review text.
- **Opinion Extraction**: Extracts opinion terms associated with the identified aspects.
- **Aspect Categorization**: Classifies aspects into predefined categories.
- **Sentiment Analysis**: Determines the sentiment (positive, negative, neutral) for each aspect-opinion pair.
- **PDF Report Generation**: Generates a PDF report summarizing the analysis.

## How it Works

The pipeline processes text through a series of steps:

1.  **Extract Aspects**: Identifies the specific features or topics being discussed (e.g., "service", "pasta").
2.  **Extract Opinions**: Finds the words that describe the sentiment towards those aspects (e.g., "slow", "delicious").
3.  **Categorize Aspects**: Groups aspects into broader categories (e.g., "SERVICE", "FOOD").
4.  **Determine Sentiment**: Assigns a sentiment to each aspect-opinion pair.
5.  **Link and Check**: Consolidates the information into a final, structured output.

## Getting Started

**From a string:**

```bash
python src/agents/run_agent_pipeline.py --domain restaurant --input "The pasta was delicious but the service was slow."
```

**From a file:**

```bash
python src/agents/run_agent_pipeline.py --domain laptop --file path/to/your/reviews.txt --output results.json
```

This will generate a `results.json` file with the extracted information and a `results_report.pdf` with a summary of the analysis.
