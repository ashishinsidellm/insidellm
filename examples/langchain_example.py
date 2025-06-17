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
                metadata={"test": "basic_llm_flow"},
            )

            # Simulate LLM start
            run_id = generate_uuid()
            callback.on_llm_start(
                serialized={"name": "gpt-3.5-turbo", "_type": "openai"},
                prompts=["What is the capital of France?"],
                run_id=run_id,
            )

            # Simulate LLM end
            mock_result = LLMResult(
                generations=[[Generation(text="The capital of France is Paris.")]],
                llm_output={"usage": {"total_tokens": 20}},
            )
            callback.on_llm_end(response=mock_result, run_id=run_id)

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
                metadata={"test": "tool_flow"},
            )

            # Simulate tool start
            run_id = generate_uuid()
            callback.on_tool_start(
                serialized={"name": "calculator", "_type": "math_tool"},
                input_str="2 + 2",
                run_id=run_id,
            )

            time.sleep(0.1)  # Simulate processing time

            # Simulate tool end
            callback.on_tool_end(output="4", run_id=run_id)

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
            from insidellm.langchain_integration import InsideLLMCallback

            callback = InsideLLMCallback(
                client=self.client,
                user_id=generate_uuid(),
                run_id=generate_uuid(),
                metadata={"test": "chain_flow", "chain_type": "qa"},
            )

            # Simulate chain start
            run_id = generate_uuid()
            # Convert tags list to string if needed
            tags_list = ["test", "qa"]
            tags_str = ",".join(tags_list) if isinstance(tags_list, list) else tags_list
            
            callback.on_chain_start(
                serialized={"name": "qa_chain", "_type": "sequential"},
                inputs={"question": "What is the capital of France?"},
                run_id=run_id,
                tags=tags_str
            )

            time.sleep(0.1)  # Simulate processing time

            # Simulate chain end
            callback.on_chain_end(
                outputs={"answer": "The capital of France is Paris."},
                run_id=run_id
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
