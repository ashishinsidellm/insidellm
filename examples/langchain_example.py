"""
InsideLLM LangChain Integration Test Suite
==========================================

This file provides comprehensive tests and examples for the InsideLLM LangChain integration.
It tests all callback functionalities including LLM calls, tool usage, agent actions, and error handling.
"""

import asyncio
import logging
import os
import sys
import time
from typing import Dict, Any, List
from unittest.mock import Mock, patch
from insidellm.utils import generate_uuid

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup test environment and check dependencies."""
    print("=" * 60)
    print("InsideLLM LangChain Integration Test Suite")
    print("=" * 60)
    
    # Check if LangChain is available
    try:
        import langchain
        print(f"✅ LangChain version: {langchain.__version__}")
    except ImportError:
        print("❌ LangChain not found. Install with: pip install langchain")
        return False
    
    # Check if InsideLLM modules are available
    try:
        from insidellm.langchain_integration import InsideLLMCallback
        from insidellm.client import InsideLLMClient
        from insidellm.models import Event, EventType
        print("✅ InsideLLM modules imported successfully")
    except ImportError as e:
        print(f"❌ InsideLLM modules not found: {e}")
        return False
    
    return True

from insidellm.models import Event
class MockInsideLLMClient:
    """Mock client for testing purposes."""
    
    def __init__(self):
        self.logged_events: List[Event] = []
        self.call_count = 0
    
    def log_event(self, event: Event):
        """Mock event logging."""
        self.logged_events.append(event)
        self.call_count += 1
        print(f"📝 Event logged: {event.event_type.value} (ID: {event.event_id[:8]}...)")
        
        # Print event details for debugging
        if hasattr(event, 'payload') and event.payload:
            if 'model_name' in event.payload:
                print(f"   Model: {event.payload['model_name']}")
            if 'tool_name' in event.payload:
                print(f"   Tool: {event.payload['tool_name']}")
            if 'error_message' in event.payload:
                print(f"   Error: {event.payload['error_message']}")
    
    def get_events_by_type(self, event_type: str) -> List[Event]:
        """Get events by type for testing."""
        return [e for e in self.logged_events if e.event_type.value == event_type]
    
    def clear_events(self):
        """Clear logged events."""
        self.logged_events.clear()
        self.call_count = 0


class TestLLMCallbacks:
    """Test LLM callback functionality."""
    
    def __init__(self, client: MockInsideLLMClient):
        self.client = client
    
    def test_basic_llm_flow(self):
        """Test basic LLM request/response flow."""
        print("\n🧪 Testing Basic LLM Flow...")
        
        try:
            from insidellm.langchain_integration import InsideLLMCallback
            from langchain.schema import LLMResult, Generation
            from uuid import uuid4
            
            # Create callback
            callback = InsideLLMCallback(
                client=self.client,
                user_id=generate_uuid(),
                run_id=generate_uuid()
            )
            # Simulate LLM start
            run_id = generate_uuid()
            callback.on_llm_start(
                serialized={"name": "gpt-3.5-turbo", "_type": "openai"},
                prompts=["What is the capital of France?"],
                run_id=run_id
            )
            
            # Simulate LLM end
            mock_result = LLMResult(
                generations=[[Generation(text="The capital of France is Paris.")]],
                llm_output={"usage": {"total_tokens": 20}}
            )
            callback.on_llm_end(response=mock_result, run_id=run_id)
            
            # Verify events
            request_events = self.client.get_events_by_type("llm.request")
            response_events = self.client.get_events_by_type("llm.response")

            assert len(request_events) == 1, f"Expected 1 request event, got {len(request_events)}"
            assert len(response_events) == 1, f"Expected 1 response event, got {len(response_events)}"
            
            print("✅ Basic LLM flow test passed")
            
        except Exception as e:
            print(f"❌ Basic LLM flow test failed: {e}")
            return False
        
        return True
    
    def test_llm_error_handling(self):
        """Test LLM error handling."""
        print("\n🧪 Testing LLM Error Handling...")
        
        try:
            from insidellm.langchain_integration import InsideLLMCallback
            
            callback = InsideLLMCallback(
                client=self.client,
                user_id=generate_uuid(),
                run_id=generate_uuid()
            )
            
            # Simulate LLM error
            run_id = generate_uuid()
            callback.on_llm_start(
                serialized={"name": "gpt-3.5-turbo", "_type": "openai"},
                prompts=["Test prompt"],
                run_id=run_id
            )
            
            # Simulate error
            test_error = ValueError("Test API error")
            callback.on_llm_error(error=test_error, run_id=run_id)
            
            # Verify error event
            error_events = self.client.get_events_by_type("error")
            assert len(error_events) >= 1, "Expected at least 1 error event"
            
            error_event = error_events[-1]  # Get the latest error event
            assert "Test API error" in error_event.payload["error_message"]
            
            print("✅ LLM error handling test passed")
            
        except Exception as e:
            print(f"❌ LLM error handling test failed: {e}")
            return False
        
        return True


class TestToolCallbacks:
    """Test tool callback functionality."""
    
    def __init__(self, client: MockInsideLLMClient):
        self.client = client
    
    def test_tool_usage_flow(self):
        """Test tool call/response flow."""
        print("\n🧪 Testing Tool Usage Flow...")
        
        try:
            from insidellm.langchain_integration import InsideLLMCallback
            
            callback = InsideLLMCallback(
                client=self.client,
                user_id=generate_uuid(),
                run_id=generate_uuid()
            )
            
            # Simulate tool start
            run_id = generate_uuid()
            callback.on_tool_start(
                serialized={"name": "calculator", "_type": "math_tool"},
                input_str="2 + 2",
                run_id=run_id
            )
            
            # Small delay to simulate execution time
            time.sleep(0.1)
            
            # Simulate tool end
            callback.on_tool_end(output="4", run_id=run_id)
            
            # Verify events
            tool_events = self.client.get_events_by_type("tool.call")
            response_events = self.client.get_events_by_type("tool.response")
            
            assert len(tool_events) >= 1, "Expected at least 1 tool call event"
            assert len(response_events) >= 1, "Expected at least 1 tool response event"
            
            print("✅ Tool usage flow test passed")
            
        except Exception as e:
            print(f"❌ Tool usage flow test failed: {e}")
            return False
        
        return True
    
    def test_tool_error_handling(self):
        """Test tool error handling."""
        print("\n🧪 Testing Tool Error Handling...")
        
        try:
            from insidellm.langchain_integration import InsideLLMCallback
       
            
            callback = InsideLLMCallback(
                client=self.client,
                user_id=generate_uuid(),
                run_id=generate_uuid()
            )
            
            # Simulate tool start
            run_id = generate_uuid()
            callback.on_tool_start(
                serialized={"name": "web_search", "_type": "search_tool"},
                input_str="search query",
                run_id=run_id
            )
            
            # Simulate tool error
            test_error = ConnectionError("Network timeout")
            callback.on_tool_error(error=test_error, run_id=run_id)
            
            # Verify error event
            error_events = self.client.get_events_by_type("error")
            tool_error_events = [e for e in error_events if e.payload.get("error_type") == "tool_error"]
            
            assert len(tool_error_events) >= 1, "Expected at least 1 tool error event"
            
            print("✅ Tool error handling test passed")
            
        except Exception as e:
            print(f"❌ Tool error handling test failed: {e}")
            return False
        
        return True


class TestAgentCallbacks:
    """Test agent callback functionality."""
    
    def __init__(self, client: MockInsideLLMClient):
        self.client = client
    
    def test_agent_actions(self):
        """Test agent action tracking."""
        print("\n🧪 Testing Agent Actions...")
        
        try:
            from insidellm.langchain_integration import InsideLLMCallback
            from langchain.schema import AgentAction, AgentFinish
            
            callback = InsideLLMCallback(
                client=self.client,
                user_id=generate_uuid(),
                run_id=generate_uuid()
            )
            
            # Create mock agent action
            mock_action = AgentAction(
                tool="search",
                tool_input="python programming",
                log="I need to search for information about python programming"
            )
            # Simulate agent action
            run_id = generate_uuid()
            callback.on_agent_action(action=mock_action, run_id=run_id)

            # Create mock agent finish
            mock_finish = AgentFinish(
                return_values={"output": "Python is a programming language"},
                log="Task completed successfully"
            )
            
            # Simulate agent finish
            callback.on_agent_finish(finish=mock_finish, run_id=run_id)
            
            # Verify events
            reasoning_events = self.client.get_events_by_type("agent.reasoning")
            response_events = self.client.get_events_by_type("agent.response")
            
            assert len(reasoning_events) >= 1, "Expected at least 1 agent reasoning event"
            assert len(response_events) >= 1, "Expected at least 1 agent response event"
            
            print("✅ Agent actions test passed")
            
        except Exception as e:
            print(f"❌ Agent actions test failed: {e}")
            return False
        
        return True


class TestConfigurationOptions:
    """Test callback configuration options."""
    
    def __init__(self, client: MockInsideLLMClient):
        self.client = client
    
    def test_selective_tracking(self):
        """Test selective event tracking configuration."""
        print("\n🧪 Testing Selective Tracking Configuration...")
        
        try:
            from insidellm.langchain_integration import InsideLLMCallback
            
            # Create callback with only LLM tracking enabled
            callback = InsideLLMCallback(
                client=self.client,
                user_id=generate_uuid(),
                run_id=generate_uuid(),
                track_llm_calls=True,
                track_tool_calls=False,
                track_agent_actions=False,
                track_errors=False
            )
            
            initial_count = self.client.call_count
            
            # Try LLM call (should be tracked)
            run_id = generate_uuid()
            callback.on_llm_start(
                serialized={"name": "test-model", "_type": "test"},
                prompts=["test"],
                run_id=run_id
            )
            
            # Try tool call (should NOT be tracked)
            tool_run_id = generate_uuid()
            callback.on_tool_start(
                serialized={"name": "test-tool", "_type": "test"},
                input_str="test",
                run_id=tool_run_id
            )
            
            # Only LLM event should be logged
            expected_new_events = 1
            actual_new_events = self.client.call_count - initial_count
            
            assert actual_new_events == expected_new_events, \
                f"Expected {expected_new_events} new events, got {actual_new_events}"
            
            print("✅ Selective tracking test passed")
            
        except Exception as e:
            print(f"❌ Selective tracking test failed: {e}")
            return False
        
        return True



def main():
    """Main test runner."""
    if not setup_environment():
        sys.exit(1)
    
    # Create mock client for testing
    mock_client = MockInsideLLMClient()
    
    # Initialize test classes
    llm_tests = TestLLMCallbacks(mock_client)
    tool_tests = TestToolCallbacks(mock_client)
    agent_tests = TestAgentCallbacks(mock_client)
    config_tests = TestConfigurationOptions(mock_client)
    
    # Track test results
    test_results = []
    
    print("\n" + "=" * 60)
    print("RUNNING CALLBACK TESTS")
    print("=" * 60)
    
    # Run LLM tests
    test_results.append(llm_tests.test_basic_llm_flow())
    test_results.append(llm_tests.test_llm_error_handling())
    
    # Run tool tests
    test_results.append(tool_tests.test_tool_usage_flow())
    test_results.append(tool_tests.test_tool_error_handling())
    
    # Run agent tests
    test_results.append(agent_tests.test_agent_actions())
    
    # Run configuration tests
    test_results.append(config_tests.test_selective_tracking())
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Total Events Logged: {mock_client.call_count}")
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your LangChain integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        
    # Print event breakdown
    print(f"\nEvent Breakdown:")
    for event_type in ["llm_request", "llm_response", "tool_call", "tool_response", 
                      "agent_reasoning", "agent_response", "error"]:
        count = len(mock_client.get_events_by_type(event_type))
        if count > 0:
            print(f"  {event_type}: {count}")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)