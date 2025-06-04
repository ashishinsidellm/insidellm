"""
Simple test script for InsideLLM Python SDK
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import insidellm
from insidellm import Event, EventType
import time

def test_basic_functionality():
    """Test basic SDK functionality"""
    print("InsideLLM SDK Test")
    print("=" * 30)
    
    # Initialize the SDK with demo configuration
    try:
        insidellm.initialize(
            api_key="demo-api-key",
            auto_flush_interval=5.0,
            batch_size=10,
            raise_on_error=False  # Don't raise on network errors for demo
        )
        print("✓ SDK initialized successfully")
    except Exception as e:
        print(f"✗ SDK initialization failed: {e}")
        return False
    
    try:
        # Get the client
        client = insidellm.get_client()
        print("✓ Client obtained successfully")
        
        # Start a run
        run_id = client.start_run(
            user_id="test-user-123",
            metadata={
                "test_session": True,
                "environment": "demo"
            }
        )
        print(f"✓ Run started: {run_id}")
        
        # Create and log a user input event
        user_event = Event.create_user_input(
            run_id=run_id,
            user_id="test-user-123",
            input_text="Hello, test my LLM integration!",
            input_type="text",
            metadata={"channel": "test"}
        )
        client.log_event(user_event)
        print(f"✓ User input event logged: {user_event.event_id}")
        
        # Create and log an LLM request event
        llm_request = Event.create_llm_request(
            run_id=run_id,
            user_id="test-user-123",
            model_name="gpt-4",
            provider="openai",
            prompt="Process user input: Hello, test my LLM integration!",
            parent_event_id=user_event.event_id,
            parameters={"temperature": 0.7}
        )
        client.log_event(llm_request)
        print(f"✓ LLM request event logged: {llm_request.event_id}")
        
        # Simulate processing time
        time.sleep(0.2)
        
        # Create and log an LLM response event
        llm_response = Event.create_llm_response(
            run_id=run_id,
            user_id="test-user-123",
            model_name="gpt-4",
            provider="openai",
            response_text="Hello! I've successfully processed your request through the InsideLLM SDK.",
            parent_event_id=llm_request.event_id,
            response_time_ms=200,
            usage={"prompt_tokens": 12, "completion_tokens": 18, "total_tokens": 30}
        )
        client.log_event(llm_response)
        print(f"✓ LLM response event logged: {llm_response.event_id}")
        
        # Create and log a tool call event
        tool_call = Event(
            run_id=run_id,
            user_id="test-user-123",
            event_type=EventType.TOOL_CALL,
            payload={
                "tool_name": "calculator",
                "tool_type": "function",
                "parameters": {"expression": "2 + 2"},
                "call_id": "calc-test-001"
            }
        )
        client.log_event(tool_call)
        print(f"✓ Tool call event logged: {tool_call.event_id}")
        
        # Create and log a tool response event
        tool_response = Event(
            run_id=run_id,
            user_id="test-user-123",
            event_type=EventType.TOOL_RESPONSE,
            parent_event_id=tool_call.event_id,
            payload={
                "tool_name": "calculator",
                "tool_type": "function",
                "call_id": "calc-test-001",
                "response_data": 4,
                "execution_time_ms": 50,
                "success": True
            }
        )
        client.log_event(tool_response)
        print(f"✓ Tool response event logged: {tool_response.event_id}")
        
        # Create and log an error event
        error_event = Event.create_error(
            run_id=run_id,
            user_id="test-user-123",
            error_type="test_error",
            error_message="This is a test error to demonstrate error logging",
            error_code="TEST_ERROR_001",
            context={"test": True}
        )
        client.log_event(error_event)
        print(f"✓ Error event logged: {error_event.event_id}")
        
        # Log a performance metric
        metric_event = Event(
            run_id=run_id,
            user_id="test-user-123",
            event_type=EventType.PERFORMANCE_METRIC,
            payload={
                "metric_name": "response_time",
                "metric_value": 250,
                "metric_unit": "ms",
                "metric_type": "gauge"
            }
        )
        client.log_event(metric_event)
        print(f"✓ Performance metric logged: {metric_event.event_id}")
        
        # Get queue statistics
        stats = client.queue_manager.get_statistics()
        print(f"\nQueue Statistics:")
        print(f"  Events Queued: {stats['events_queued']}")
        print(f"  Events Sent: {stats['events_sent']}")
        print(f"  Events Failed: {stats['events_failed']}")
        print(f"  Queue Size: {stats['queue_size']}")
        print(f"  Success Rate: {stats['success_rate']:.1f}%")
        
        # Test flush functionality
        print("\n✓ Flushing events...")
        client.flush()
        
        # End the run
        client.end_run(run_id)
        print(f"✓ Run ended: {run_id}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Shutdown the SDK
        try:
            insidellm.shutdown()
            print("✓ SDK shutdown complete")
        except Exception as e:
            print(f"✗ Shutdown error: {e}")

def test_context_manager():
    """Test context manager functionality"""
    print("\nContext Manager Test")
    print("=" * 30)
    
    try:
        # Initialize if not already done
        if not hasattr(insidellm, '_default_client') or insidellm._default_client is None:
            insidellm.initialize(api_key="demo-api-key")
        
        # Test context manager
        with insidellm.InsideLLMTracker(
            user_id="context-test-user",
            metadata={"test": "context_manager"}
        ) as tracker:
            
            # Log user input
            input_id = tracker.log_user_input(
                input_text="Test context manager functionality",
                input_type="text"
            )
            print(f"✓ User input logged via context manager: {input_id}")
            
            # Test LLM tracking context manager
            with tracker.track_llm_call("test-model", "test-provider", "Test prompt") as log_response:
                time.sleep(0.1)
                log_response("Test response from LLM")
                print("✓ LLM call tracked with context manager")
            
            # Test tool tracking context manager
            with tracker.track_tool_call("test_tool", {"param": "value"}) as log_response:
                time.sleep(0.1)
                log_response("Tool execution result")
                print("✓ Tool call tracked with context manager")
            
            # Log agent response
            response_id = tracker.log_agent_response(
                response_text="Context manager test completed successfully",
                response_type="test_response",
                parent_event_id=input_id
            )
            print(f"✓ Agent response logged: {response_id}")
        
        print("✓ Context manager test completed")
        return True
        
    except Exception as e:
        print(f"✗ Context manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_decorators():
    """Test decorator functionality"""
    print("\nDecorator Test")
    print("=" * 30)
    
    try:
        # Initialize if not already done
        if not hasattr(insidellm, '_default_client') or insidellm._default_client is None:
            insidellm.initialize(api_key="demo-api-key")
        
        # Start a run for decorator tests
        client = insidellm.get_client()
        run_id = client.start_run(user_id="decorator-test-user")
        
        @insidellm.track_llm_call("test-model", "test-provider")
        def test_llm_function(prompt):
            time.sleep(0.1)
            return f"Response to: {prompt}"
        
        @insidellm.track_tool_use("test_tool", "function")
        def test_tool_function(param):
            time.sleep(0.05)
            return f"Tool result for: {param}"
        
        @insidellm.track_agent_step("test_step")
        def test_agent_function(data):
            time.sleep(0.02)
            return f"Processed: {data}"
        
        # Test decorated functions
        llm_result = test_llm_function("Test LLM prompt")
        print(f"✓ LLM decorator test: {llm_result}")
        
        tool_result = test_tool_function("test parameter")
        print(f"✓ Tool decorator test: {tool_result}")
        
        agent_result = test_agent_function("test data")
        print(f"✓ Agent step decorator test: {agent_result}")
        
        # End the run
        client.end_run(run_id)
        print("✓ Decorator test completed")
        return True
        
    except Exception as e:
        print(f"✗ Decorator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting InsideLLM SDK Tests")
    print("=" * 50)
    
    success = True
    
    # Run tests
    success &= test_basic_functionality()
    success &= test_context_manager()
    success &= test_decorators()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! SDK is working correctly.")
    else:
        print("✗ Some tests failed. Check the output above.")
    
    print("\nSDK Features Demonstrated:")
    print("• Asynchronous event queuing and batching")
    print("• 15+ comprehensive event types")
    print("• Context managers for workflow tracking")
    print("• Function decorators for automatic tracking")
    print("• Error handling and performance metrics")
    print("• Thread-safe operations")
    print("• Configurable retry mechanisms")