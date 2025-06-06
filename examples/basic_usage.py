"""
InsideLLM Basic Usage Example

This example demonstrates the basic functionality of the InsideLLM SDK,
including initialization, event logging, and basic tracking features.
"""

import os
import time
import insidellm
from insidellm import Event, EventType


def basic_event_logging():
    """Demonstrate basic event logging functionality."""
    print("1. Basic Event Logging")
    print("-" * 30)
    
    # Get the client
    client = insidellm.get_client()
    
    # Start a run
    run_id = client.start_run(
        user_id="basic-demo-user",
        metadata={
            "session_type": "demo",
            "environment": "development"
        }
    )
    print(f"Started run: {run_id}")
    
    # Log user input
    user_input_event = Event.create_user_input(
        run_id=run_id,
        user_id="basic-demo-user",
        input_text="Hello, how can you help me today?",
        input_type="text",
        metadata={"channel": "web"}
    )
    client.log_event(user_input_event)
    print(f"Logged user input event: {user_input_event.event_id}")
    
    # Log LLM request
    llm_request_event = Event.create_llm_request(
        run_id=run_id,
        user_id="basic-demo-user",
        model_name="gpt-4",
        provider="openai",
        prompt="User asks: Hello, how can you help me today?",
        parent_event_id=user_input_event.event_id,
        parameters={"temperature": 0.7, "max_tokens": 150}
    )
    client.log_event(llm_request_event)
    print(f"Logged LLM request event: {llm_request_event.event_id}")
    
    # Simulate processing time
    time.sleep(0.5)
    
    # Log LLM response
    llm_response_event = Event.create_llm_response(
        run_id=run_id,
        user_id="basic-demo-user",
        model_name="gpt-4",
        provider="openai",
        response_text="Hello! I'm an AI assistant. I can help you with various tasks like answering questions, writing, analysis, and more.",
        parent_event_id=llm_request_event.event_id,
        response_time_ms=500,
        usage={"prompt_tokens": 15, "completion_tokens": 32, "total_tokens": 47}
    )
    client.log_event(llm_response_event)
    print(f"Logged LLM response event: {llm_response_event.event_id}")
    
    # End the run
    client.end_run(run_id)
    print(f"Ended run: {run_id}")


def tool_usage_example():
    """Demonstrate tool usage tracking."""
    print("\n2. Tool Usage Tracking")
    print("-" * 30)
    
    client = insidellm.get_client()
    run_id = client.start_run(user_id="tool-demo-user")
    
    # Log tool call
    tool_call_event = Event(
        run_id=run_id,
        user_id="tool-demo-user",
        event_type=EventType.TOOL_CALL,
        payload={
            "tool_name": "calculator",
            "tool_type": "function",
            "parameters": {"expression": "15 * 24"},
            "call_id": "calc-001"
        }
    )
    client.log_event(tool_call_event)
    print(f"Logged tool call: {tool_call_event.event_id}")
    
    # Simulate tool execution
    time.sleep(0.2)
    
    # Log tool response
    tool_response_event = Event(
        run_id=run_id,
        user_id="tool-demo-user",
        event_type=EventType.TOOL_RESPONSE,
        parent_event_id=tool_call_event.event_id,
        payload={
            "tool_name": "calculator",
            "tool_type": "function",
            "call_id": "calc-001",
            "response_data": 360,
            "execution_time_ms": 200,
            "success": True
        }
    )
    client.log_event(tool_response_event)
    print(f"Logged tool response: {tool_response_event.event_id}")
    
    client.end_run(run_id)


def error_handling_example():
    """Demonstrate error event logging."""
    print("\n3. Error Handling")
    print("-" * 30)
    
    client = insidellm.get_client()
    run_id = client.start_run(user_id="error-demo-user")
    
    try:
        # Simulate an operation that might fail
        raise ValueError("Invalid input parameter")
    except Exception as e:
        # Log the error
        error_event = Event.create_error(
            run_id=run_id,
            user_id="error-demo-user",
            error_type="validation_error",
            error_message=str(e),
            error_code="INVALID_INPUT",
            context={"operation": "data_validation", "input": "invalid_value"}
        )
        client.log_event(error_event)
        print(f"Logged error event: {error_event.event_id}")
    
    client.end_run(run_id)


def performance_metrics_example():
    """Demonstrate performance metrics logging."""
    print("\n4. Performance Metrics")
    print("-" * 30)
    
    client = insidellm.get_client()
    run_id = client.start_run(user_id="metrics-demo-user")
    
    # Log various performance metrics
    metrics = [
        ("response_time", 250, "ms"),
        ("memory_usage", 128, "MB"),
        ("token_count", 150, "tokens"),
        ("accuracy_score", 0.95, "percentage")
    ]
    
    for metric_name, value, unit in metrics:
        metric_event = Event(
            run_id=run_id,
            user_id="metrics-demo-user",
            event_type=EventType.PERFORMANCE_METRIC,
            payload={
                "metric_name": metric_name,
                "metric_value": value,
                "metric_unit": unit,
                "metric_type": "gauge"
            }
        )
        client.log_event(metric_event)
        print(f"Logged metric {metric_name}: {value} {unit}")
    
    client.end_run(run_id)


def context_manager_example():
    """Demonstrate context manager usage."""
    print("\n5. Context Manager Usage")
    print("-" * 30)
    
    # Use InsideLLMTracker as context manager
    with insidellm.InsideLLMTracker(
        user_id="context-demo-user",
        metadata={"demo_type": "context_manager"}
    ) as tracker:
        
        # Log user input
        input_id = tracker.log_user_input(
            input_text="What's the weather like today?",
            input_type="text"
        )
        print(f"User input logged: {input_id}")
        
        # Use LLM tracking context manager
        with tracker.track_llm_call("gpt-3.5-turbo", "openai", "Get weather information") as log_response:
            # Simulate LLM call
            time.sleep(0.3)
            response_text = "I'd be happy to help with weather information, but I need your location first."
            log_response(response_text)
            print("LLM call tracked with context manager")
        
        # Use tool tracking context manager
        with tracker.track_tool_call("weather_api", {"location": "New York"}) as log_response:
            # Simulate API call
            time.sleep(0.4)
            weather_data = {
                "temperature": 72,
                "condition": "sunny",
                "humidity": 45
            }
            log_response(weather_data)
            print("Tool call tracked with context manager")
        
        # Log final response
        response_id = tracker.log_agent_response(
            response_text="The weather in New York is currently 72°F and sunny with 45% humidity.",
            response_type="weather_info",
            parent_event_id=input_id
        )
        print(f"Agent response logged: {response_id}")


def decorator_example():
    """Demonstrate decorator usage."""
    print("\n6. Decorator Usage")
    print("-" * 30)
    
    @insidellm.track_llm_call("custom-model", "custom-provider")
    def call_custom_llm(prompt, temperature=0.7):
        """Mock LLM call function."""
        time.sleep(0.2)
        return f"Response to: {prompt[:30]}..."
    
    @insidellm.track_tool_use("text_analyzer", "nlp")
    def analyze_text(text):
        """Mock text analysis function."""
        time.sleep(0.1)
        return {
            "sentiment": "positive",
            "confidence": 0.85,
            "word_count": len(text.split())
        }
    
    @insidellm.track_agent_step("text_processing")
    def process_text(text):
        """Mock text processing function."""
        return text.upper()
    
    # Start a run for decorator examples
    client = insidellm.get_client()
    run_id = client.start_run(user_id="decorator-demo-user")
    
    try:
        # These function calls will be automatically tracked
        llm_result = call_custom_llm("Analyze this text for sentiment")
        print(f"LLM result: {llm_result}")
        
        analysis_result = analyze_text("This is a great example of SDK usage!")
        print(f"Analysis result: {analysis_result}")
        
        processed_text = process_text("hello world")
        print(f"Processed text: {processed_text}")
        
    except Exception as e:
        print(f"Error in decorator example: {e}")
    finally:
        client.end_run(run_id)


def statistics_and_monitoring():
    """Show statistics and monitoring capabilities."""
    print("\n7. Statistics and Monitoring")
    print("-" * 30)
    
    client = insidellm.get_client()
    
    # Get queue statistics
    stats = client.queue_manager.get_statistics()
    print("Queue Statistics:")
    print(f"  Events Queued: {stats['events_queued']}")
    print(f"  Events Sent: {stats['events_sent']}")
    print(f"  Events Failed: {stats['events_failed']}")
    print(f"  Success Rate: {stats['success_rate']:.1f}%")
    print(f"  Current Queue Size: {stats['queue_size']}")
    
    # Check client health
    health_status = client.is_healthy()
    print(f"\nClient Health: {'Healthy' if health_status else 'Unhealthy'}")
    
    # Show current run context
    current_run = client.get_current_run_id()
    current_user = client.get_current_user_id()
    print(f"Current Run ID: {current_run or 'None'}")
    print(f"Current User ID: {current_user or 'None'}")


def main():
    """Main example function."""
    print("InsideLLM Basic Usage Example")
    print("=" * 50)
    
    # Initialize the SDK
    try:
        insidellm.initialize(
            api_key=os.getenv("INSIDELLM_API_KEY", "iilmn-sample-key"),
            local_testing= True,
            auto_flush_interval=10.0,  # Flush every 10 seconds
            batch_size=20,
            max_retries=2,
            raise_on_error=False  # Don't raise on network errors for demo
        )
        print("InsideLLM SDK initialized successfully")
    except Exception as e:
        print(f"Failed to initialize SDK: {e}")
        return
    
    try:
        # Run all examples
        basic_event_logging()
        tool_usage_example()
        error_handling_example()
        performance_metrics_example()
        context_manager_example()
        decorator_example()
        statistics_and_monitoring()
        
        # Manual flush to ensure all events are sent
        print("\n8. Flushing Events")
        print("-" * 30)
        insidellm.flush()
        print("All events flushed successfully")
        
    except Exception as e:
        print(f"Error during example execution: {e}")
    
    finally:
        # Shutdown the SDK
        print("\n9. Shutting Down")
        print("-" * 30)
        insidellm.shutdown()
        print("SDK shutdown complete")


if __name__ == "__main__":
    main()
