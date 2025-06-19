import logging
import os
import sys
import time
from typing import Dict, Any, List
from insidellm.utils import generate_uuid

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Setup test environment and check dependencies."""
    print("=" * 60)
    print("InsideLLM LangChain Integration Test Suite (Real Client)")
    print("=" * 60)

    # Check if LangChain is available
    try:
        import langchain

        print(f"✅ LangChain version: {langchain.__version__}")
    except ImportError:
        print("❌ LangChain not found. Install with: pip install langchain")
        return False

    # Check if InsideLLM is properly configured
    try:
        from insidellm.client import InsideLLMClient
        from insidellm.models import Event, EventType

        print("✅ InsideLLM modules imported successfully")
    except ImportError as e:
        print(f"❌ InsideLLM modules not found: {e}")
        return False

    return True


class TestLLMCallbacks:
    """Test LLM callback functionality with real InsideLLMClient."""

    def __init__(self, client):
        self.client = client

    def test_basic_llm_flow(self):
        """Test basic LLM request/response flow."""
        print("\n🧪 Testing Basic LLM Flow (Real Client)...")

        try:
            from insidellm.langchain_integration import InsideLLMCallback
            from langchain.schema import LLMResult, Generation

            # Initialize real InsideLLM callback
            callback = InsideLLMCallback(
                client=self.client,
                user_id=generate_uuid(),
                run_id=generate_uuid(),
                metadata={
                    "test": "basic_llm_flow",
                    "framework": "langchain",
                    "test_type": "llm"
                },
            )

            # Simulate LLM start with system message
            run_id = generate_uuid()
            callback.on_llm_start(
                serialized={
                    "name": "gpt-3.5-turbo",
                    "_type": "openai",
                    "metadata": {
                        "model_type": "chat",
                        "provider": "openai"
                    }
                },
                prompts=["What is the capital of France?"],
                run_id=run_id,
                metadata={
                    "system_message": "You are a helpful assistant.",
                    "temperature": 0.7
                }
            )

            # Simulate LLM end with detailed response if you want to use real LangChain response, uncomment the code below and use open_ai_api_key
            mock_result = LLMResult(
                generations=[[Generation(text="The capital of France is Paris.")]],
                llm_output={
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 10,
                        "total_tokens": 20
                    },
                    "finish_reason": "stop"
                },
            )


            # # Get real response from LangChain
            # from langchain.chat_models import ChatOpenAI
            # from langchain.schema import HumanMessage

            # chat = ChatOpenAI(
            #     temperature=0.7,
            #     openai_api_key=os.getenv("OPENAI_API_KEY")
            # )
            # messages = [HumanMessage(content="What is the capital of France?")]
            # result = chat.generate([messages]) 

            callback.on_llm_end(
                response=mock_result,
                run_id=run_id,
                metadata={
                    "response_time_ms": 1000,
                    "success": True
                }
            )

            print("✅ Basic LLM flow test passed (check InsideLLM dashboard for events)")
            return True

        except Exception as e:
            print(f"❌ Basic LLM flow test failed: {e}")
            return False


class TestToolCallbacks:
    """Test tool callback functionality with real client."""

    def __init__(self, client):
        self.client = client

    def test_tool_usage_flow(self):
        """Test tool call/response flow."""
        print("\n🧪 Testing Tool Usage Flow (Real Client)...")

        try:
            from insidellm.langchain_integration import InsideLLMCallback

            callback = InsideLLMCallback(
                client=self.client,
                user_id=generate_uuid(),
                run_id=generate_uuid(),
                metadata={
                    "test": "tool_flow",
                    "framework": "langchain",
                    "test_type": "tool"
                },
            )

            # Simulate tool start with detailed metadata
            run_id = generate_uuid()
            callback.on_tool_start(
                serialized={
                    "name": "calculator",
                    "_type": "math_tool",
                    "metadata": {
                        "tool_type": "calculator",
                        "input_type": "mathematical_expression"
                    }
                },
                input_str="2 + 2",
                run_id=run_id,
                metadata={
                    "context": "testing",
                    "start_time": time.time()
                }
            )

            time.sleep(0.1)  # Simulate processing time

            # Simulate tool end with detailed response
            callback.on_tool_end(
                output="4",
                run_id=run_id,
                metadata={
                    "execution_time_ms": 100,
                    "success": True,
                    "end_time": time.time()
                }
            )

            print("✅ Tool usage flow test passed (check InsideLLM dashboard)")
            return True

        except Exception as e:
            print(f"❌ Tool usage flow test failed: {e}")
            return False


class TestChainCallbacks:
    """Test chain callback functionality with real client."""

    def __init__(self, client):
        self.client = client

    def test_chain_execution_flow(self):
        """Test chain start/end flow."""
        print("\n🧪 Testing Chain Execution Flow (Real Client)...")

        try:
            import json
            from insidellm.langchain_integration import InsideLLMCallback

            callback = InsideLLMCallback(
                client=self.client,
                user_id=generate_uuid(),
                run_id=generate_uuid(),
                metadata={
                    "test": "chain_flow",
                    "framework": "langchain",
                    "test_type": "chain",
                    "chain_type": "qa"
                },
            )

            # Simulate chain start with detailed metadata
            run_id = generate_uuid()
            callback.on_chain_start(
                serialized={
                    "name": "qa_chain",
                    "_type": "sequential",
                    "metadata": json.dumps({"chain_type": "qa", "components": ["retriever", "llm"]})
                },
                inputs={
                    "question": "What is the capital of France?",
                    "context": "This is a test context."
                },
                run_id=run_id,
                metadata={
                    "start_time": time.time(),
                    "chain_config": json.dumps({"temperature": 0.7, "max_tokens": 100})
                },
                tags="test,qa"  # Added as comma-separated string instead of list
            )

            time.sleep(0.1)  # Simulate processing time

            # Simulate chain end with detailed response
            callback.on_chain_end(
                outputs={
                    "answer": "The capital of France is Paris.",
                    "confidence": 0.95,
                    "sources": ["test_context"]
                },
                run_id=run_id,
                metadata={
                    "execution_time_ms": 1000,
                    "success": True,
                    "end_time": time.time()
                }
            )

            print("✅ Chain execution flow test passed (check InsideLLM dashboard)")
            return True

        except Exception as e:
            print(f"❌ Chain execution flow test failed: {e}")
            return False

import insidellm


def main():
    """Main test runner with real InsideLLMClient."""
    if not setup_environment():
        sys.exit(1)

    # Initialize REAL InsideLLM client (replace with your API key)
    from insidellm.client import InsideLLMClient

    client = insidellm.initialize(
        api_key=os.getenv("INSIDELLM_API_KEY", "iilmn-sample-key"),
        local_testing=True,
    )

    # Run tests
    llm_tests = TestLLMCallbacks(client)
    tool_tests = TestToolCallbacks(client)
    chain_tests = TestChainCallbacks(client)

    # Track results
    test_results = [
        llm_tests.test_basic_llm_flow(),
        tool_tests.test_tool_usage_flow(),
        chain_tests.test_chain_execution_flow()
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
