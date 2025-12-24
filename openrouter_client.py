"""
OpenRouter API client using OpenAI library.

OpenRouter is an API gateway that provides access to multiple LLM models.
This module provides functions to interact with OpenRouter using the OpenAI library.
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI


def call_openrouter(
    messages: List[Dict[str, str]],
    model: str = "openai/gpt-3.5-turbo",
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    base_url: str = "https://openrouter.ai/api/v1",
    **kwargs
) -> Dict[str, Any]:
    """
    Call OpenRouter API using OpenAI library.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.
                 Example: [{"role": "user", "content": "Hello!"}]
        model: Model identifier. Examples:
               - "openai/gpt-3.5-turbo"
               - "openai/gpt-4"
               - "anthropic/claude-3-opus"
               - "google/gemini-pro"
               See https://openrouter.ai/models for full list
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens to generate (None for no limit)
        api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY env var
        base_url: API base URL (default: OpenRouter endpoint)
        **kwargs: Additional arguments to pass to chat.completions.create()
    
    Returns:
        Dictionary containing:
        - 'content': Generated text content
        - 'model': Model used
        - 'usage': Token usage information
        - 'full_response': Full API response object
    
    Raises:
        ValueError: If API key is not provided
        Exception: For API errors
    
    Example:
        >>> messages = [{"role": "user", "content": "What is 2+2?"}]
        >>> result = call_openrouter(messages, model="openai/gpt-3.5-turbo")
        >>> print(result['content'])
    """
    # Get API key from parameter or environment variable
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError(
                "API key not provided. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    # Initialize OpenAI client with OpenRouter endpoint
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    # Prepare request parameters
    request_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        **kwargs
    }
    
    if max_tokens is not None:
        request_params["max_tokens"] = max_tokens
    
    # Make API call
    try:
        response = client.chat.completions.create(**request_params)
        
        # Extract content from response
        content = response.choices[0].message.content
        
        # Extract usage information
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        } if response.usage else None
        
        return {
            "content": content,
            "model": response.model,
            "usage": usage,
            "full_response": response,
        }
    
    except Exception as e:
        raise Exception(f"OpenRouter API call failed: {str(e)}")


def call_openrouter_simple(
    prompt: str,
    model: str = "openai/gpt-3.5-turbo",
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> str:
    """
    Simplified function to call OpenRouter with a single prompt.
    
    Args:
        prompt: User prompt text
        model: Model identifier (default: "openai/gpt-3.5-turbo")
        temperature: Sampling temperature (default: 1.0)
        max_tokens: Maximum tokens to generate (None for no limit)
        system_prompt: Optional system prompt
        api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
        **kwargs: Additional arguments for call_openrouter()
    
    Returns:
        Generated text content as string
    
    Example:
        >>> response = call_openrouter_simple("What is 2+2?")
        >>> print(response)
    """
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    result = call_openrouter(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        **kwargs
    )
    
    return result["content"]


def list_openrouter_models(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List available models from OpenRouter.
    
    Args:
        api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
    
    Returns:
        List of model dictionaries with model information
    
    Note:
        This requires calling OpenRouter's models endpoint directly.
        The OpenAI library doesn't have a direct method for this,
        so we use requests or similar.
    """
    import requests
    
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError(
                "API key not provided. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    try:
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    except Exception as e:
        raise Exception(f"Failed to list OpenRouter models: {str(e)}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Check if API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Warning: OPENROUTER_API_KEY environment variable not set.")
        print("Set it with: export OPENROUTER_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Example 1: Simple call
    print("Example 1: Simple call")
    try:
        response = call_openrouter_simple(
            "What is 2+2?",
            model="openai/gpt-3.5-turbo",
            temperature=0.7
        )
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 2: Full call with messages
    print("Example 2: Full call with messages")
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing in one sentence."}
        ]
        result = call_openrouter(
            messages=messages,
            model="openai/gpt-3.5-turbo",
            temperature=0.7
        )
        print(f"Response: {result['content']}")
        print(f"Model: {result['model']}")
        print(f"Usage: {result['usage']}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 3: List available models
    print("Example 3: List available models (first 5)")
    try:
        models = list_openrouter_models()
        for model in models[:5]:
            print(f"- {model.get('id', 'N/A')}: {model.get('name', 'N/A')}")
    except Exception as e:
        print(f"Error: {e}")

