"""
OpenRouter API client using OpenAI library.

OpenRouter is an API gateway that provides access to multiple LLM models.
This module provides functions to interact with OpenRouter using the OpenAI library.
"""

import os
import time
from typing import Optional, Dict, List, Any
from openai import OpenAI

def call_openrouter(
    messages: List[Dict[str, str]],
    model: str = "openai/gpt-3.5-turbo",
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    base_url: str = "https://openrouter.ai/api/v1",
    timeout: float = 120.0,  # 添加超时参数（秒），增加到 120 秒以支持慢速模型
    max_retries: int = 3,    # 添加重试次数
    retry_delay: float = 1.0,  # 重试延迟（秒）
    **kwargs
) -> Dict[str, Any]:
    """
    Call OpenRouter API using OpenAI library with retry and timeout support.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.
        model: Model identifier
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens to generate (None for no limit)
        api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY env var
        base_url: API base URL (default: OpenRouter endpoint)
        timeout: Request timeout in seconds (default: 60.0)
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Delay between retries in seconds (default: 1.0)
        **kwargs: Additional arguments to pass to chat.completions.create()
    
    Returns:
        Dictionary containing:
        - 'content': Generated text content
        - 'model': Model used
        - 'usage': Token usage information
        - 'full_response': Full API response object
    """
    # Get API key from parameter or environment variable
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError(
                "API key not provided. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    # Initialize OpenAI client with OpenRouter endpoint and timeout
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,  # 设置超时
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
    
    # 重试逻辑
    last_exception = None
    for attempt in range(max_retries):
        try:
            # Make API call
            response = client.chat.completions.create(**request_params)
            
            # Extract content from response
            content = response.choices[0].message.content
            
            # 检查 content 是否为空（None 或空字符串）
            if content is None or (isinstance(content, str) and not content.strip()):
                print(f"Warning: API returned empty content (None or empty string) on attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # 指数退避
                    print(f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    content = ""  # 最后一次尝试，返回空字符串
                    print(f"Warning: All retries exhausted, returning empty string")
            
            # Extract usage information
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if response.usage else None
            
            return {
                "content": content or "",  # 确保返回字符串
                "model": response.model,
                "usage": usage,
                "full_response": response,
            }
        
        except Exception as e:
            last_exception = e
            print(f"API call failed on attempt {attempt + 1}/{max_retries}: {str(e)}")
            
            # 如果不是最后一次尝试，等待后重试
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # 指数退避
                print(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                # 最后一次尝试失败，抛出异常
                raise Exception(f"OpenRouter API call failed after {max_retries} attempts: {str(last_exception)}")
    
    # 理论上不会到达这里，但为了安全
    raise Exception(f"OpenRouter API call failed: {str(last_exception)}")


def call_openrouter_simple(
    prompt: str,
    model: str = "openai/gpt-3.5-turbo",
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 120.0,      # 添加超时参数，增加到 120 秒以支持慢速模型
    max_retries: int = 3,        # 添加重试参数
    retry_delay: float = 1.0,   # 添加重试延迟参数
    **kwargs
) -> str:
    """
    Simplified function to call OpenRouter with a single prompt.
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
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
        **kwargs
    )
    
    return result["content"] or ""  # 确保返回字符串而不是 None


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

