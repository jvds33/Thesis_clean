import os
from dotenv import load_dotenv
import json
import requests
from openai import AzureOpenAI
import logging
from urllib.parse import urlparse
import re
import threading
import time

# For Deepseek model
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Set up logging - change default level to WARNING
# Note: This hides INFO level messages to reduce console output clutter
# To see all API logs again, change both levels below to logging.INFO
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Only show warnings and errors, not info

# Thread-local storage to track API logs per sample
_thread_local = threading.local()

# Added to load .env file from project root
# Assumes api.py is in src/llms/, so .env is two levels up
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

def extract_base_endpoint(endpoint):
    """Extract just the base URL from an endpoint URL."""
    # Parse the URL
    parsed = urlparse(endpoint)
    
    # Extract just the base URL (scheme + netloc)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    
    # Only log this once per sample
    if not hasattr(_thread_local, 'logged_endpoints') or endpoint not in _thread_local.logged_endpoints:
        if not hasattr(_thread_local, 'logged_endpoints'):
            _thread_local.logged_endpoints = set()
        _thread_local.logged_endpoints.add(endpoint)
        logger.info(f"Extracted base URL: {base_url} from endpoint: {endpoint}")
    
    return base_url

# Dictionary of supported models and their deployment names
MODEL_DEPLOYMENTS = {
    "gpt": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CHAT", "gpt-4o-mini"),
    "gpt-4o": os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME", "gpt-4o"),
    "gpt-4o-mini": os.getenv("AZURE_OPENAI_GPT4O_MINI_DEPLOYMENT_NAME", "gpt-4o-mini"),
    "gpt-4.1": os.getenv("AZURE_OPENAI_GPT41_DEPLOYMENT_NAME", "gpt-4.1-preview"),
    "gpt-4.1-nano": os.getenv("AZURE_OPENAI_GPT41NANO_DEPLOYMENT_NAME", "gpt-4.1-nano"),
    "deepseek-v3": os.getenv("AZURE_DEEPSEEK_V3_DEPLOYMENT_NAME", "DeepSeek-V3-0324"),
    "deepseek-r1": os.getenv("AZURE_DEEPSEEK_R1_DEPLOYMENT_NAME", "DeepSeek-R1-0325")
}

# Get and clean endpoints
openai_endpoint = extract_base_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT"))
deepseek_endpoint = os.getenv("AZURE_DEEPSEEK_ENDPOINT")

# Log these just once at module import time
logger.info(f"OpenAI endpoint: {openai_endpoint}")
logger.info(f"Deepseek endpoint: {deepseek_endpoint}")

# Create client for OpenAI models
default_client = None
try:
    default_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION", "2025-01-01-preview"),
        azure_endpoint=openai_endpoint
    )
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")

# Create client for Deepseek models
deepseek_client = None
if deepseek_endpoint and os.getenv("AZURE_DEEPSEEK_API_KEY"):
    try:
        logger.warning(f"Initializing Deepseek client with endpoint: {deepseek_endpoint}")
        deepseek_client = ChatCompletionsClient(
            endpoint=deepseek_endpoint,
            credential=AzureKeyCredential(os.getenv("AZURE_DEEPSEEK_API_KEY")),
            api_version="2024-05-01-preview",  # Use the correct API version for Deepseek
            # Add connection options for better performance
            connection_timeout=30,
            read_timeout=60
        )
        logger.warning("Deepseek client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Deepseek client: {str(e)}")

# Function to reset the API logging for a new sample
def reset_api_logging_for_sample():
    """Reset the API logging flags for a new sample."""
    if hasattr(_thread_local, 'logged_endpoints'):
        delattr(_thread_local, 'logged_endpoints')
    if hasattr(_thread_local, 'logged_model_configs'):
        delattr(_thread_local, 'logged_model_configs')
    if hasattr(_thread_local, 'logged_models'):
        delattr(_thread_local, 'logged_models')

def get_client_for_model(model_name):
    """Get the appropriate client for the given model name."""
    if is_deepseek_model(model_name):
        if deepseek_client is None:
            raise ValueError("Deepseek client not initialized. Check your environment variables.")
            
        # Only log this once per sample
        if not hasattr(_thread_local, 'logged_models') or model_name not in _thread_local.logged_models:
            if not hasattr(_thread_local, 'logged_models'):
                _thread_local.logged_models = set()
            _thread_local.logged_models.add(model_name)
            logger.info(f"Using Deepseek client for model: {model_name}")
            
        return deepseek_client
    else:
        # For OpenAI models (gpt, gpt-4o, gpt-4.1, gpt-4.1-nano)
        if model_name in ["gpt-4o", "gpt-4.1", "gpt-4.1-nano"]:
            # Check if specific model configs are defined
            model_key = f"AZURE_OPENAI_{model_name.upper().replace('-', '')}_API_KEY"
            model_endpoint = f"AZURE_OPENAI_{model_name.upper().replace('-', '')}_ENDPOINT"
            
            # Only log this once per sample
            if not hasattr(_thread_local, 'logged_model_configs') or model_name not in _thread_local.logged_model_configs:
                if not hasattr(_thread_local, 'logged_model_configs'):
                    _thread_local.logged_model_configs = set()
                _thread_local.logged_model_configs.add(model_name)
                logger.info(f"Checking for model-specific config: {model_key} and {model_endpoint}")
            
            model_endpoint_value = os.getenv(model_endpoint)
            if os.getenv(model_key) and model_endpoint_value:
                # Check if the endpoint is a placeholder
                if "your-resource" in model_endpoint_value:
                    logger.warning(f"Model-specific endpoint for {model_name} contains placeholder 'your-resource'. Falling back to default OpenAI client.")
                else:
                    try:
                        # Create a dedicated client for this specific model
                        endpoint = extract_base_endpoint(model_endpoint_value)
                        
                        # Only log this once per sample
                        if not hasattr(_thread_local, 'logged_models') or f"{model_name}_dedicated" not in _thread_local.logged_models:
                            if not hasattr(_thread_local, 'logged_models'):
                                _thread_local.logged_models = set()
                            _thread_local.logged_models.add(f"{model_name}_dedicated")
                            logger.info(f"Creating dedicated client for {model_name} with endpoint: {endpoint}")
                            
                        return AzureOpenAI(
                            api_key=os.getenv(model_key),
                            api_version=os.getenv("OPENAI_API_VERSION", "2025-01-01-preview"),
                            azure_endpoint=endpoint
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create dedicated client for {model_name}: {str(e)}")
                        logger.warning(f"Falling back to default OpenAI client")
        
        # Use default client for base GPT model or as fallback
        if default_client is None:
            raise ValueError("OpenAI client not initialized. Check your environment variables.")
            
        # Only log this once per sample
        if not hasattr(_thread_local, 'logged_models') or f"{model_name}_default" not in _thread_local.logged_models:
            if not hasattr(_thread_local, 'logged_models'):
                _thread_local.logged_models = set()
            _thread_local.logged_models.add(f"{model_name}_default")
            logger.info(f"Using default OpenAI client for model: {model_name}")
            
        return default_client

def is_deepseek_model(model_name):
    """Check if the model is a Deepseek model."""
    return model_name.startswith("deepseek")

def llm_completion(prompt, stop=None, model_name="gpt"):
    try:
        if is_deepseek_model(model_name):
            raise NotImplementedError("Completion API not supported for Deepseek models. Use llm_chat instead.")
        
        client = get_client_for_model(model_name)
        deployment = MODEL_DEPLOYMENTS.get(model_name, MODEL_DEPLOYMENTS["gpt"])
        
        # Only log this once per sample
        if not hasattr(_thread_local, 'logged_models') or f"{model_name}_completion" not in _thread_local.logged_models:
            if not hasattr(_thread_local, 'logged_models'):
                _thread_local.logged_models = set()
            _thread_local.logged_models.add(f"{model_name}_completion")
            logger.info(f"Using model: {model_name}, deployment: {deployment}")
        
        response = client.completions.create(
            model=deployment,
            prompt=prompt,
            temperature=0,
            max_tokens=10000,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop
        )
        return response.choices[0].text
    except Exception as e:
        logger.error(f"Error in llm_completion for model {model_name}: {str(e)}")
        raise

def llm_chat(messages, stop=None, model_name="gpt", temperature=0, return_usage=False):
    """
    Send a chat request to the specified model.

    Args:
        messages: A list of message dictionaries, e.g., [{"role": "user", "content": "..."}]
        stop: Optional stop sequences
        model_name: The name of the model to use
        temperature: Sampling temperature (0 for deterministic, higher for more randomness)
        return_usage: Whether to return token usage information
    
    Returns:
        If return_usage is False: The response content from the model.
        If return_usage is True: A tuple of (response content, token usage dict)
    """
    try:
        if is_deepseek_model(model_name):
            # Use the Azure AI Inference SDK for Deepseek models
            start_time = time.time()
            logger.warning(f"Starting Deepseek API call for model {model_name}")
            
            client = get_client_for_model(model_name)
            deployment = MODEL_DEPLOYMENTS.get(model_name)
            
            # Only log this once per sample
            if not hasattr(_thread_local, 'logged_models') or f"{model_name}_chat" not in _thread_local.logged_models:
                if not hasattr(_thread_local, 'logged_models'):
                    _thread_local.logged_models = set()
                _thread_local.logged_models.add(f"{model_name}_chat")
                logger.warning(f"Using Deepseek model: {model_name}, deployment: {deployment}")
            
            # Convert messages to the format expected by the Deepseek client
            from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
            messages_to_send = []
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                if role == "user":
                    messages_to_send.append(UserMessage(content=content))
                elif role == "system":
                    messages_to_send.append(SystemMessage(content=content))
                elif role == "assistant":
                    messages_to_send.append(AssistantMessage(content=content))
            
            response = client.complete(
                messages=messages_to_send,
                max_tokens=10000,
                temperature=temperature,
                top_p=1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                model=deployment,
                timeout=60  # Add a 60-second timeout to prevent hanging
            )
            
            end_time = time.time()
            logger.warning(f"Deepseek API call completed in {end_time - start_time:.2f} seconds")
            
            content = response.choices[0].message.content

            if model_name == "deepseek-r1":
                match = re.match(r"<think>(.*?)</think>(.*)", content, re.DOTALL)
                if match:
                    reasoning = match.group(1).strip()
                    answer = match.group(2).strip()
                    logger.info(f"DeepSeek-R1 Reasoning: {reasoning}")
                    logger.info(f"DeepSeek-R1 Answer: {answer}")
                    content = answer
            
            # Extract usage information if available and requested
            usage = None
            if return_usage and hasattr(response, 'usage'):
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            return (content, usage) if return_usage else content
        else:
            # Use the OpenAI SDK for GPT models
            client = get_client_for_model(model_name)
            deployment = MODEL_DEPLOYMENTS.get(model_name, MODEL_DEPLOYMENTS["gpt"])
            
            # Only log this once per sample
            if not hasattr(_thread_local, 'logged_models') or f"{model_name}_chat" not in _thread_local.logged_models:
                if not hasattr(_thread_local, 'logged_models'):
                    _thread_local.logged_models = set()
                _thread_local.logged_models.add(f"{model_name}_chat")
                logger.info(f"Using OpenAI model: {model_name}, deployment: {deployment}")
            
            response = client.chat.completions.create(
                model=deployment,
                messages=messages,
                temperature=temperature,
                max_tokens=10000,
                stop=stop
            )

            result = ''
            for choice in response.choices:
                result += choice.message.content
            
            # Extract usage information if available and requested
            usage = None
            if return_usage and hasattr(response, 'usage'):
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            return (result, usage) if return_usage else result
    except Exception as e:
        logger.error(f"Error in llm_chat for model {model_name}: {str(e)}")
        raise

if __name__ == "__main__":
    print("Attempting a test call to Azure models using llm_chat...")
    print("Please ensure your Azure environment variables are set correctly:")
    print("For OpenAI models:")
    print(f"  AZURE_OPENAI_ENDPOINT: {os.getenv('AZURE_OPENAI_ENDPOINT', 'Not set')}")
    print(f"  AZURE_OPENAI_API_KEY: {'Set' if os.getenv('AZURE_OPENAI_API_KEY') else 'Not set'}")
    print("For GPT-4.1-nano:")
    print(f"  AZURE_OPENAI_GPT41NANO_ENDPOINT: {os.getenv('AZURE_OPENAI_GPT41NANO_ENDPOINT', 'Not set')}")
    print(f"  AZURE_OPENAI_GPT41NANO_API_KEY: {'Set' if os.getenv('AZURE_OPENAI_GPT41NANO_API_KEY') else 'Not set'}")
    print(f"  AZURE_OPENAI_GPT41NANO_DEPLOYMENT_NAME: {os.getenv('AZURE_OPENAI_GPT41NANO_DEPLOYMENT_NAME', 'Not set')}")
    print("For Deepseek models:")
    print(f"  AZURE_DEEPSEEK_ENDPOINT: {os.getenv('AZURE_DEEPSEEK_ENDPOINT', 'Not set')}")
    print(f"  AZURE_DEEPSEEK_API_KEY: {'Set' if os.getenv('AZURE_DEEPSEEK_API_KEY') else 'Not set'}")
    
    try:
        test_response = llm_chat(
            [{"role": "user", "content": "Say hello in one word."}], 
            model_name="gpt"
        )
        print(f"Response from model: {test_response}")
    except Exception as e:
        print(f"Test failed: {str(e)}")
