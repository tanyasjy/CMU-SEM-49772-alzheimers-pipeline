import openai
from typing import List, Dict, Generator, Optional
import json
import os

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Detect OpenAI SDK version
def _get_openai_version():
    """Get the major version of the OpenAI SDK"""
    try:
        version = openai.__version__
        major_version = int(version.split('.')[0])
        return major_version, version
    except Exception:
        # If we can't determine version, assume old SDK
        return 0, "unknown"

OPENAI_MAJOR_VERSION, OPENAI_VERSION = _get_openai_version()
print(f"OpenAI SDK version detected: {OPENAI_VERSION} (major: {OPENAI_MAJOR_VERSION})")

# Initialize based on SDK version
if OPENAI_MAJOR_VERSION >= 1:
    # New SDK (1.x+)
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    USE_NEW_SDK = True
    print("Using OpenAI SDK 1.x+ API")
else:
    # Old SDK (0.28.x)
    openai.api_key = OPENAI_API_KEY
    client = None
    USE_NEW_SDK = False
    print("Using OpenAI SDK 0.28.x API")


def safe_extract_content(chunk) -> Optional[str]:
    """Safely extract content from OpenAI streaming response chunk"""
    try:
        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content is not None:
                return delta.content
        return None
    except Exception as e:
        print(f"Error extracting content: {e}")
        return None


def query_openai_api(instruction: str, inp: str, model: str = "gpt-4o"):
    """
    Query OpenAI API (supports both 0.28.x and 1.x+ SDKs)
    """
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": inp}
    ]
    
    if USE_NEW_SDK:
        # New SDK (1.x+)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    else:
        # Old SDK (0.28.x)
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content


async def get_openai_streaming_response(messages: List[Dict[str, str]], model: str = "gpt-4o"):
    """
    Get streaming response from OpenAI API (async version)
    Supports both OpenAI SDK 0.28.x and 1.x+
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: OpenAI model to use
    
    Yields:
        str: Content chunks from the streaming response
    """
    try:
        print(f"Starting OpenAI stream for model: {model} (SDK: {OPENAI_VERSION})")
        
        if USE_NEW_SDK:
            # New SDK (1.x+) streaming
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=True
            )
            
            chunk_count = 0
            for chunk in response:
                content = safe_extract_content(chunk)
                if content is not None:
                    chunk_count += 1
                    yield content
            
            print(f"Stream completed. Total chunks: {chunk_count}")
        else:
            # Old SDK (0.28.x) streaming
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=True
            )
            
            chunk_count = 0
            for chunk in response:
                content = safe_extract_content(chunk)
                if content is not None:
                    chunk_count += 1
                    yield content
            
            print(f"Stream completed. Total chunks: {chunk_count}")
                
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        yield f"Error: {str(e)}"


def format_messages(user_input: str, chat_history: List[Dict] = None) -> List[Dict[str, str]]:
    """
    Format messages for OpenAI API
    
    Args:
        user_input: Current user message
        chat_history: Previous conversation history
    
    Returns:
        List of formatted messages
    """
    messages = []
    
    # Add system message with markdown formatting instructions
    messages.append({
        "role": "system",
        "content": """You are a helpful AI assistant. Provide clear, accurate, and helpful responses.

Format your responses using markdown for better readability:
- Use **bold** for emphasis on important terms
- Use *italics* for subtle emphasis
- Use bullet points and numbered lists for organization
- Use `code` for technical terms, function names, or code snippets
- Use code blocks (```) for multi-line code
- Use [links](url) for references
- Use headings (##) to structure longer responses
- Use tables when presenting structured data

Always structure your responses for maximum clarity and readability."""
    })
    
    # Add chat history if provided
    if chat_history:
        for msg in chat_history:
            if msg.get('role') in ['user', 'assistant']:
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
    
    # Add current user message
    messages.append({
        "role": "user",
        "content": user_input
    })
    
    return messages
