import os
from typing import List, Dict, Any
from insidellm.mistral_integration import MistralAIWrapper
from insidellm.client import InsideLLMClient
import insidellm


def initialize_client():
    """Initialize InsideLLM client"""
    return insidellm.initialize(
            api_key=os.getenv("INSIDELLM_API_KEY", "iilmn-sample-key"),
            local_testing= True,
        )

def main():
    # Initialize InsideLLM client
    client = initialize_client()
    
    # Initialize Mistral AI wrapper with your integration
    mistral = MistralAIWrapper(
        client=client,
        mistral_api_key="your_mistral_api_key_here",  
        user_id="test_user",
        model="mistral-small-latest"  
    )
    
    # Example 1: Simple chat completion using dictionary format
    print("\nExample 1: Simple chat completion")
    messages = [
        {
            "role": "user", 
            "content": "What is the capital of France?"
        }
    ]
    
    try:
        response = mistral.chat(
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error in Example 1: {str(e)}")
    
    # Example 2: Multi-turn conversation
    print("\nExample 2: Multi-turn conversation")
    conversation = [
        {
            "role": "system", 
            "content": "You are a helpful AI assistant."
        },
        {
            "role": "user", 
            "content": "Hi, I need help with Python programming."
        },
        {
            "role": "assistant", 
            "content": "Hello! I'd be happy to help you with Python programming. What specific topic would you like to learn about?"
        },
        {
            "role": "user", 
            "content": "Can you explain list comprehensions?"
        }
    ]
    
    try:
        response = mistral.chat(
            messages=conversation,
            temperature=0.7,
            max_tokens=200
        )
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error in Example 2: {str(e)}")
    
    # Example 3: Different model
    print("\nExample 3: Using different model")
    messages = [
        {
            "role": "user", 
            "content": "Explain quantum computing in simple terms."
        }
    ]
    
    try:
        response = mistral.chat(
            messages=messages,
            model="mistral-medium-latest",  # Different model
            temperature=0.5,
            max_tokens=150
        )
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error in Example 3: {str(e)}")
    
    # Example 4: With custom metadata
    print("\nExample 4: With custom metadata")
    mistral_with_metadata = MistralAIWrapper(
        client=client,
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        user_id="test_user",
        model="mistral-small-latest",
        metadata={
            "session_type": "demo",
            "experiment_id": "example_4",
            "user_context": "learning"
        }
    )
    
    messages = [
        {
            "role": "user", 
            "content": "Tell me a short joke about programming."
        }
    ]
    
    try:
        response = mistral_with_metadata.chat(
            messages=messages,
            temperature=0.8,
            max_tokens=100
        )
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error in Example 4: {str(e)}")
    
    # Example 5: Error handling demonstration
    print("\nExample 5: Error handling")
    try:
        # This should fail with invalid API key
        bad_mistral = MistralAIWrapper(
            client=client,
            mistral_api_key="invalid_key_12345",
            user_id="test_user"
        )
        response = bad_mistral.chat(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7
        )
    except Exception as e:
        print(f"Expected error caught: {str(e)}")
    
    # Example 6: Custom parameters
    print("\nExample 6: Custom parameters")
    messages = [
        {
            "role": "user", 
            "content": "Write a very brief summary of machine learning."
        }
    ]
    
    try:
        response = mistral.chat(
            messages=messages,
            temperature=0.3,  # Lower temperature for more focused response
            max_tokens=80,    # Shorter response
            top_p=0.9
        )
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error in Example 6: {str(e)}")

    # Example 7: Test direct Mistral client (for debugging)
    print("\nExample 7: Direct Mistral client test")
    try:
        from mistralai import Mistral
        
        direct_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        response = direct_client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {
                    "content": "Hello from direct client!",
                    "role": "user",
                }
            ]
        )
        print(f"Direct client response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Direct client error: {str(e)}")


   

def check_mistral_client():
    """Check what's available in the Mistral client"""
    try:
        from mistralai import Mistral
        print("✓ Mistral client imported successfully")
        
        # Try to see what's available
        import mistralai
        print(f"✓ Mistral AI version: {getattr(mistralai, '__version__', 'unknown')}")
        
        return True
    except ImportError as e:
        print(f"✗ Mistral client import failed: {e}")
        return False

if __name__ == "__main__":
    print("Mistral AI InsideLLM Integration Example")
    print("=" * 50)
    
    if not check_mistral_client():
        print("Please install mistralai: pip install mistralai")
        exit(1)

    
    try:
        main()
        print("\n" + "=" * 50)
        print("Examples completed successfully!")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        print("Please check your API keys and network connection.")
        import traceback
        traceback.print_exc()