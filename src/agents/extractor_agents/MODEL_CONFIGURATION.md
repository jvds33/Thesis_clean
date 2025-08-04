# Model Configuration System

This document describes the model configuration system for the refined extraction pipelines in both the restaurant and laptop domains.

## Overview

The model configuration system provides a unified way to manage and use different language models across both pipelines. It maps user-friendly model names to Azure OpenAI deployment configurations.

## Available Models

| Model Name | Azure Deployment | Description |
|------------|------------------|-------------|
| `gpt-4o` | `gpt-4o` | High performance - Working deployment |

## Usage

### Command Line

```bash
# Restaurant pipeline
cd src/agents/extractor_agents/extractor_rest_REFINE
python run_refined_pipeline.py --model gpt-4o --num-samples 10

# Laptop pipeline  
cd src/agents/extractor_agents/extractor_laptop_REFINE
python run_refined_pipeline.py --model gpt-4o --num-samples 10
```

### Shell Scripts

```bash
# Restaurant pipeline
cd src/agents/extractor_agents/extractor_rest_REFINE
bash run_refined_pipeline.sh gpt-4o 10

# Laptop pipeline
cd src/agents/extractor_agents/extractor_laptop_REFINE  
bash run_refined_pipeline.sh gpt-4o 10
```

## Configuration Details

### Model Configuration Structure

Each model in the configuration has the following structure:

```python
"model_name": {
    "azure_deployment": "actual_deployment_name",
    "api_version": "2025-01-01-preview",
    "temperature": 0,
    "description": "Human-readable description"
}
```

### Adding New Models

To add a new model:

1. **Verify the Azure deployment exists** in your Azure OpenAI service
2. Add the configuration to `MODEL_CONFIGURATIONS` in both:
   - `src/agents/extractor_agents/extractor_rest_REFINE/model_config.py`
   - `src/agents/extractor_agents/extractor_laptop_REFINE/model_config.py`

```python
MODEL_CONFIGURATIONS = {
    "gpt-4o": {
        "azure_deployment": "gpt-4o",
        "api_version": "2025-01-01-preview",
        "temperature": 0,
        "description": "GPT-4o - High performance (Working deployment)"
    },
    # Add your new model here
    "new_model": {
        "azure_deployment": "your_azure_deployment_name",
        "api_version": "2025-01-01-preview",
        "temperature": 0,
        "description": "Your model description"
    }
}
```

## Testing

### Test Model Configuration

```bash
# Test restaurant pipeline models
cd src/agents/extractor_agents/extractor_rest_REFINE
python test_models.py

# Test laptop pipeline models
cd src/agents/extractor_agents/extractor_laptop_REFINE
python test_models.py
```

### Verify Model Availability

```bash
# Check available models
python -c "from model_config import list_models; list_models()"
```

## Error Handling

The system includes comprehensive error handling:

- **Model validation**: Checks if requested model exists
- **Azure deployment verification**: Ensures deployment is available
- **Graceful fallbacks**: Provides helpful error messages with available options

## Common Issues

### DeploymentNotFound Error

```
Error code: 404 - {'error': {'code': 'DeploymentNotFound', 'message': 'The API deployment for this resource does not exist'}}
```

**Solution**: The Azure deployment name in the configuration doesn't match what's deployed in your Azure OpenAI service. Check your Azure portal and update the configuration accordingly.

### Invalid Model Name

```
Model 'invalid_model' not found. Available models: ['gpt-4o']
```

**Solution**: Use one of the available models listed in the error message.

## File Structure

```
src/agents/extractor_agents/
├── extractor_rest_REFINE/
│   ├── model_config.py          # Model configuration
│   ├── run_refined_pipeline.py  # Main pipeline script
│   ├── run_refined_pipeline.sh  # Shell wrapper
│   └── test_models.py          # Test script
├── extractor_laptop_REFINE/
│   ├── model_config.py          # Model configuration
│   ├── run_refined_pipeline.py  # Main pipeline script
│   ├── run_refined_pipeline.sh  # Shell wrapper
│   └── test_models.py          # Test script
└── MODEL_CONFIGURATION.md      # This documentation
```

## Support

For issues or questions:
1. Check that your Azure OpenAI deployments are active
2. Verify model names in the configuration match your deployments
3. Test with `test_models.py` to verify configuration

## Migration Notes

The system was updated to only include working Azure deployments. Previously configured models that gave 404 errors have been removed. Only `gpt-4o` is currently available as it's the only verified working deployment in the Azure OpenAI service. 