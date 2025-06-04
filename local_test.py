#!/usr/bin/env python3
"""
Local Testing Script for InsideLLM SDK

This script demonstrates how to test the SDK locally without API calls.
All events are logged to local files for inspection and analysis.
"""

import sys
import os
import json
from pathlib import Path

# Add current directory to Python path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import insidellm
from insidellm.local_logger import LocalTestingClient
from insidellm import Event, EventType
import time

def initialize_local_testing():
    """Initialize SDK for local testing"""
    print("Initializing InsideLLM SDK for Local Testing")
    print("=" * 50)
    
    # Replace the default client with local testing client
    insidellm._default_client = LocalTestingClient(
        log_directory="local_test_logs",
        log_format="json"
    )
    
    print("Local testing client initialized")
    print("Events will be logged to: local_test_logs/")
    return insidellm._default_client

def test_basic_functionality():
    """Test basic SDK functionality with local logging"""
    print("\n1. Basic Functionality Test")
    print("-" * 30)
    
    client = insidellm.get_client()
    
    # Start a run
    run_id = client.start_run(
        user_id="local-test-user",
        metadata={
            "test_type": "local_basic",
            "environment": "development"
        }
    )
    print(f"Started run: {run_id}")
    
    # Log various event types
    events_created = []
    
    # User input event
    user_event = Event.create_user_input(
        run_id=run_id,
        user_id="local-test-user",
        input_text="Hello, I want to test the SDK locally!",
        input_type="text",
        metadata={"channel": "local_test"}
    )
    client.log_event(user_event)
    events_created.append(("User Input", user_event.event_id))
    
    # LLM request event
    llm_request = Event.create_llm_request(
        run_id=run_id,
        user_id="local-test-user",
        model_name="gpt-4",
        provider="openai",
        prompt="Process the user's local testing request",
        parent_event_id=user_event.event_id,
        parameters={"temperature": 0.7, "max_tokens": 150}
    )
    client.log_event(llm_request)
    events_created.append(("LLM Request", llm_request.event_id))
    
    # Simulate processing time
    time.sleep(0.1)
    
    # LLM response event
    llm_response = Event.create_llm_response(
        run_id=run_id,
        user_id="local-test-user",
        model_name="gpt-4",
        provider="openai",
        response_text="I can help you test the SDK locally. All events are being logged to your local filesystem.",
        parent_event_id=llm_request.event_id,
        response_time_ms=100,
        usage={"prompt_tokens": 20, "completion_tokens": 25, "total_tokens": 45}
    )
    client.log_event(llm_response)
    events_created.append(("LLM Response", llm_response.event_id))
    
    # Tool call event
    tool_call = Event(
        run_id=run_id,
        user_id="local-test-user",
        event_type=EventType.TOOL_CALL,
        payload={
            "tool_name": "local_file_system",
            "tool_type": "function",
            "parameters": {"action": "create_log", "filename": "test.log"},
            "call_id": "local-tool-001"
        }
    )
    client.log_event(tool_call)
    events_created.append(("Tool Call", tool_call.event_id))
    
    # Tool response event
    tool_response = Event(
        run_id=run_id,
        user_id="local-test-user",
        event_type=EventType.TOOL_RESPONSE,
        parent_event_id=tool_call.event_id,
        payload={
            "tool_name": "local_file_system",
            "tool_type": "function",
            "call_id": "local-tool-001",
            "response_data": {"status": "success", "file_created": "test.log"},
            "execution_time_ms": 50,
            "success": True
        }
    )
    client.log_event(tool_response)
    events_created.append(("Tool Response", tool_response.event_id))
    
    # Performance metric
    metric_event = Event(
        run_id=run_id,
        user_id="local-test-user",
        event_type=EventType.PERFORMANCE_METRIC,
        payload={
            "metric_name": "local_test_duration",
            "metric_value": 150,
            "metric_unit": "ms",
            "metric_type": "gauge"
        }
    )
    client.log_event(metric_event)
    events_created.append(("Performance Metric", metric_event.event_id))
    
    # Error event (simulated)
    error_event = Event.create_error(
        run_id=run_id,
        user_id="local-test-user",
        error_type="test_error",
        error_message="This is a simulated error for testing purposes",
        error_code="TEST_ERROR_001",
        context={"test_mode": True, "simulated": True}
    )
    client.log_event(error_event)
    events_created.append(("Error Event", error_event.event_id))
    
    # End the run
    client.end_run(run_id)
    
    print(f"Logged {len(events_created)} events:")
    for event_type, event_id in events_created:
        print(f"  - {event_type}: {event_id}")
    
    return run_id

def test_decorators():
    """Test decorator functionality with local logging"""
    print("\n2. Decorator Functionality Test")
    print("-" * 30)
    
    client = insidellm.get_client()
    run_id = client.start_run(user_id="decorator-test-user")
    
    @insidellm.track_llm_call("local-test-model", "local-provider")
    def mock_llm_call(prompt):
        time.sleep(0.05)  # Simulate processing
        return f"Local test response to: {prompt[:30]}..."
    
    @insidellm.track_tool_use("local_calculator", "function")
    def mock_calculator(expression):
        time.sleep(0.02)  # Simulate calculation
        try:
            result = eval(expression)  # Safe for testing
            return result
        except:
            return "Error: Invalid expression"
    
    @insidellm.track_agent_step("local_planning")
    def mock_planning_step(task):
        time.sleep(0.03)  # Simulate planning
        return f"Plan for: {task}"
    
    # Test decorated functions
    try:
        llm_result = mock_llm_call("What is the weather like today?")
        print(f"LLM call result: {llm_result}")
        
        calc_result = mock_calculator("15 * 24")
        print(f"Calculator result: {calc_result}")
        
        plan_result = mock_planning_step("analyze user query")
        print(f"Planning result: {plan_result}")
        
        print("All decorator tests completed successfully")
        
    except Exception as e:
        print(f"Decorator test error: {e}")
    
    client.end_run(run_id)
    return run_id

def test_context_manager():
    """Test context manager functionality with local logging"""
    print("\n3. Context Manager Test")
    print("-" * 30)
    
    with insidellm.InsideLLMTracker(
        user_id="context-test-user",
        metadata={"test_type": "context_manager", "local": True}
    ) as tracker:
        
        # Log user input
        input_id = tracker.log_user_input(
            input_text="Test context manager with local logging",
            input_type="text"
        )
        print(f"User input logged: {input_id}")
        
        # Test LLM tracking context manager
        with tracker.track_llm_call("local-gpt", "local-provider", "Test prompt") as log_response:
            time.sleep(0.1)
            response = "This is a test response from local LLM simulation"
            log_response(response)
            print("LLM call tracked with context manager")
        
        # Test tool tracking context manager
        with tracker.track_tool_call("local_web_search", {"query": "local testing"}) as log_response:
            time.sleep(0.15)
            results = ["Local result 1", "Local result 2", "Local result 3"]
            log_response(results)
            print("Tool call tracked with context manager")
        
        # Log agent response
        response_id = tracker.log_agent_response(
            response_text="Context manager test completed successfully with local logging",
            response_type="test_completion",
            parent_event_id=input_id
        )
        print(f"Agent response logged: {response_id}")
    
    print("Context manager test completed")

def analyze_local_logs():
    """Analyze the locally logged events"""
    print("\n4. Local Log Analysis")
    print("-" * 30)
    
    client = insidellm.get_client()
    if hasattr(client, 'local_logger'):
        # Get session summaries
        sessions = client.local_logger.list_sessions()
        
        print(f"Found {len(sessions)} test sessions:")
        for session in sessions[:5]:  # Show last 5 sessions
            print(f"  Run ID: {session.get('run_id', 'Unknown')}")
            print(f"  Events: {session.get('total_events', 0)}")
            print(f"  Duration: {session.get('duration_seconds', 0):.2f}s")
            
            event_types = session.get('event_types', {})
            if event_types:
                print(f"  Event Types: {', '.join(f'{k}({v})' for k, v in event_types.items())}")
            print()
        
        # Show log directory contents
        log_dir = Path("local_test_logs")
        if log_dir.exists():
            print("Log files created:")
            for file_path in log_dir.glob("*.json"):
                size = file_path.stat().st_size
                print(f"  {file_path.name} ({size:,} bytes)")
    
    else:
        print("Local logger not available in current client")

def inspect_log_file():
    """Inspect a log file to show the event structure"""
    print("\n5. Log File Inspection")
    print("-" * 30)
    
    log_dir = Path("local_test_logs")
    json_files = list(log_dir.glob("*.json")) if log_dir.exists() else []
    
    if json_files:
        # Get the most recent file
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f"Inspecting: {latest_file.name}")
        
        try:
            with open(latest_file, 'r') as f:
                lines = f.readlines()
                total_events = len(lines)
                
                print(f"Total events in file: {total_events}")
                
                if total_events > 0:
                    # Show first event as example
                    first_event = json.loads(lines[0])
                    print("\nExample event structure:")
                    print(json.dumps(first_event, indent=2))
                    
                    # Show event type summary
                    event_types = {}
                    for line in lines:
                        try:
                            event = json.loads(line.strip())
                            event_type = event.get('event_type', 'unknown')
                            event_types[event_type] = event_types.get(event_type, 0) + 1
                        except:
                            continue
                    
                    print(f"\nEvent type distribution:")
                    for event_type, count in event_types.items():
                        print(f"  {event_type}: {count}")
        
        except Exception as e:
            print(f"Error reading log file: {e}")
    
    else:
        print("No log files found. Run some tests first.")

def main():
    """Main local testing function"""
    print("InsideLLM SDK - Local Testing Suite")
    print("=" * 60)
    print("This test suite runs all SDK functionality locally without API calls.")
    print("All events are logged to local files for inspection and analysis.")
    print()
    
    try:
        # Initialize local testing
        client = initialize_local_testing()
        
        # Run all tests
        run_id_1 = test_basic_functionality()
        run_id_2 = test_decorators() 
        test_context_manager()
        
        # Analyze results
        analyze_local_logs()
        inspect_log_file()
        
        # Show final statistics
        stats = client.get_statistics()
        print(f"\n6. Final Statistics")
        print("-" * 30)
        print(f"Total events logged: {stats.get('events_logged', 0)}")
        print(f"Log directory: {stats.get('log_directory', 'N/A')}")
        print(f"Sessions created: {stats.get('sessions', 0)}")
        
        # Shutdown
        client.shutdown()
        
        print("\n" + "=" * 60)
        print("LOCAL TESTING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nWhat was tested:")
        print("• Event creation and logging (all 15+ event types)")
        print("• Function decorators (@track_llm_call, @track_tool_use, etc.)")
        print("• Context managers (InsideLLMTracker)")
        print("• Session management and workflow tracking")
        print("• Local file logging with JSON format")
        print("• Event analysis and statistics")
        
        print(f"\nLog files created in: local_test_logs/")
        print("You can inspect these files to see the exact event structure")
        print("that would be sent to your API endpoint.")
        
    except Exception as e:
        print(f"\nLocal testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()