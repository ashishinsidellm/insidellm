import logging
import os
import sys
import time
from typing import Dict, Any, List
from dotenv import load_dotenv
from insidellm.utils import generate_uuid

# Load environment variables from .env file
load_dotenv()

# Get Mistral API key from environment
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in environment variables. Please set it in your .env file.")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Setup test environment and check dependencies."""
    print("=" * 60)
    print("InsideLLM Mistral Integration Test Suite (Real Client)")
    print("=" * 60)

    # Check if Mistral AI is available
    try:
        from mistralai import Mistral
        print("✅ Mistral AI client imported successfully")
    except ImportError:
        print("❌ Mistral AI not found. Install with: pip install mistralai")
        return False

    # Check if InsideLLM is properly configured
    try:
        from insidellm.client import InsideLLMClient
        from insidellm.models import Event, EventType
        from insidellm.mistral_integration import InsideLLMMistralIntegration

        print("✅ InsideLLM modules imported successfully")
    except ImportError as e:
        print(f"❌ InsideLLM modules not found: {e}")
        return False

    return True


class TestMistralChat:
    """Test Mistral chat functionality with real InsideLLMClient."""

    def __init__(self, client):
        self.client = client

    def test_basic_chat_flow(self):
        """Test basic chat completion flow."""
        print("\n🧪 Testing Basic Chat Flow (Real Client)...")

        try:
            from mistralai import Mistral
            from insidellm.mistral_integration import InsideLLMMistralIntegration

            # Initialize Mistral client
            mistral_client = Mistral(api_key=MISTRAL_API_KEY)

            # Initialize InsideLLM Mistral integration
            mistral_integration = InsideLLMMistralIntegration(
                insidellm_client=self.client,
                mistral_client=mistral_client,
                user_id=generate_uuid(),
                run_id=generate_uuid(),
                metadata={"test": "basic_chat_flow"}
            )

            # Test chat completion
            with mistral_integration.tracked_chat_completion(
                model="mistral-tiny",
                messages=[{"role": "user", "content": "What is the capital of France?"}]
            ) as response:
                if response and response.choices:
                    print(f"✅ Chat response received: {response.choices[0].message.content}")
                    return True
                else:
                    print("❌ No response received from Mistral")
                    return False

        except Exception as e:
            print(f"❌ Basic chat flow test failed: {e}")
            return False


class TestMistralEmbeddings:
    """Test Mistral embeddings functionality with real client."""

    def __init__(self, client):
        self.client = client

    def test_embeddings_flow(self):
        """Test embeddings generation flow."""
        print("\n🧪 Testing Embeddings Flow (Real Client)...")

        try:
            from mistralai import Mistral
            from insidellm.mistral_integration import InsideLLMMistralIntegration

            # Initialize Mistral client
            mistral_client = Mistral(api_key=MISTRAL_API_KEY)

            # Initialize InsideLLM Mistral integration
            mistral_integration = InsideLLMMistralIntegration(
                insidellm_client=self.client,
                mistral_client=mistral_client,
                user_id=generate_uuid(),
                run_id=generate_uuid(),
                metadata={"test": "embeddings_flow"}
            )

            # Test embeddings generation
            with mistral_integration.tracked_embeddings(
                model="mistral-embed",
                inputs=["What is the capital of France?"]
            ) as response:
                if response and response.data:
                    print(f"✅ Embeddings generated successfully (dimensions: {len(response.data[0].embedding)})")
                    return True
                else:
                    print("❌ No embeddings received from Mistral")
                    return False

        except Exception as e:
            print(f"❌ Embeddings flow test failed: {e}")
            return False


class TestMistralStreaming:
    """Test Mistral streaming functionality with real client."""

    def __init__(self, client):
        self.client = client

    def test_streaming_flow(self):
        """Test streaming chat completion flow."""
        print("\n🧪 Testing Streaming Flow (Real Client)...")

        try:
            from mistralai import Mistral
            from insidellm.mistral_integration import InsideLLMMistralIntegration

            # Initialize Mistral client
            mistral_client = Mistral(api_key=MISTRAL_API_KEY)

            # Initialize InsideLLM Mistral integration
            mistral_integration = InsideLLMMistralIntegration(
                insidellm_client=self.client,
                mistral_client=mistral_client,
                user_id=generate_uuid(),
                run_id=generate_uuid(),
                metadata={"test": "streaming_flow"}
            )

            # Test streaming chat completion
            with mistral_integration.tracked_streaming_chat(
                model="mistral-tiny",
                messages=[{"role": "user", "content": "Write a short poem about AI."}]
            ) as stream:
                if stream:
                    # print("✅ Streaming response chunks:")
                    # for chunk in stream:
                        # if chunk.data.choices and chunk.data.choices[0].delta.content:
                            # print(chunk.data.choices[0].delta.content, end="", flush=True)
                    print("\n✅ Streaming test completed successfully")
                    return True
                else:
                    print("❌ No streaming response received from Mistral")
                    return False

        except Exception as e:
            print(f"❌ Streaming flow test failed: {e}")
            return False


def main():
    """Main test runner with real InsideLLMClient."""
    if not setup_environment():
        sys.exit(1)

    # Initialize REAL InsideLLM client (replace with your API key)
    from insidellm.client import InsideLLMClient
    import insidellm

    client = insidellm.initialize(
        api_key=os.getenv("INSIDELLM_API_KEY", "iilmn-sample-key"),
        local_testing=True,
    )

    # Run tests
    chat_tests = TestMistralChat(client)
    embeddings_tests = TestMistralEmbeddings(client)
    streaming_tests = TestMistralStreaming(client)

    # Track results
    test_results = [
        chat_tests.test_basic_chat_flow(),
        embeddings_tests.test_embeddings_flow(),
        streaming_tests.test_streaming_flow()
    ]

    # Print summary
    passed = sum(test_results)
    total = len(test_results)

    print("\n" + "=" * 60)
    print(f"TESTS PASSED: {passed}/{total}")
    print("=" * 60)

    if passed == total:
        print("🎉 All tests passed! Check InsideLLM for logged events.")
    else:
        print("⚠️  Some tests failed. See logs above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 