"""
InsideLLM Custom Agent Integration Example

This example demonstrates how to integrate InsideLLM with custom agents
using decorators and context managers.
"""

import os
import time
import random
import insidellm
from typing import Dict, Any, List


class CustomAgent:
    """
    Example custom agent that uses InsideLLM for tracking.
    """
    
    def __init__(self, name: str, model: str = "custom-llm-v1"):
        self.name = name
        self.model = model
        self.tools = {
            "calculator": self._calculator_tool,
            "weather": self._weather_tool,
            "memory": self._memory_tool
        }
        self.memory = {}
    
    @insidellm.track_llm_call(
        model_name="custom-llm-v1",
        provider="custom",
        extract_prompt=lambda self, prompt, **kwargs: prompt,
        extract_response=lambda response: response.get("text", "")
    )
    def _call_llm(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Simulate LLM call with tracking."""
        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.3))
        
        # Mock response
        return {
            "text": f"Mock response to: {prompt[:50]}...",
            "confidence": random.uniform(0.8, 1.0),
            "tokens_used": random.randint(50, 200)
        }
    
    @insidellm.track_tool_use(
        tool_name="calculator",
        tool_type="function",
        extract_parameters=lambda self, expression: {"expression": expression},
        extract_response=lambda result: result
    )
    def _calculator_tool(self, expression: str) -> float:
        """Calculator tool with tracking."""
        try:
            # Safe evaluation
            allowed_chars = set('0123456789+-*/.() ')
            if all(c in allowed_chars for c in expression):
                return eval(expression)
            else:
                raise ValueError("Invalid expression")
        except Exception as e:
            raise ValueError(f"Calculation error: {e}")
    
    @insidellm.track_tool_use(
        tool_name="weather",
        tool_type="api",
        extract_parameters=lambda self, location: {"location": location}
    )
    def _weather_tool(self, location: str) -> Dict[str, Any]:
        """Mock weather API tool with tracking."""
        # Simulate API call delay
        time.sleep(random.uniform(0.2, 0.5))
        
        return {
            "location": location,
            "temperature": random.randint(15, 30),
            "condition": random.choice(["sunny", "cloudy", "rainy"]),
            "humidity": random.randint(30, 80)
        }
    
    @insidellm.track_tool_use(
        tool_name="memory",
        tool_type="storage"
    )
    def _memory_tool(self, action: str, key: str = None, value: str = None) -> Any:
        """Memory tool for storing/retrieving information."""
        if action == "store" and key and value:
            self.memory[key] = value
            return f"Stored {key}: {value}"
        elif action == "retrieve" and key:
            return self.memory.get(key, "Not found")
        elif action == "list":
            return list(self.memory.keys())
        else:
            raise ValueError("Invalid memory operation")
    
    @insidellm.track_agent_step("planning")
    def _plan_task(self, task: str) -> List[str]:
        """Plan how to approach a task."""
        # Simulate planning time
        time.sleep(0.1)
        
        # Simple planning logic
        if "calculate" in task.lower() or any(op in task for op in ['+', '-', '*', '/']):
            return ["use_calculator", "format_response"]
        elif "weather" in task.lower():
            return ["get_weather", "format_response"]
        elif "remember" in task.lower() or "store" in task.lower():
            return ["use_memory", "confirm_storage"]
        else:
            return ["call_llm", "format_response"]
    
    @insidellm.track_agent_step("execution")
    def _execute_step(self, step: str, context: Dict[str, Any]) -> Any:
        """Execute a planned step."""
        if step == "use_calculator":
            # Extract expression from context
            expression = context.get("expression", "1+1")
            return self._calculator_tool(expression)
        
        elif step == "get_weather":
            location = context.get("location", "New York")
            return self._weather_tool(location)
        
        elif step == "use_memory":
            return self._memory_tool(
                context.get("action", "list"),
                context.get("key"),
                context.get("value")
            )
        
        elif step == "call_llm":
            prompt = context.get("prompt", "Hello")
            return self._call_llm(prompt)
        
        elif step == "format_response":
            return f"Formatted: {context.get('data', 'No data')}"
        
        else:
            return f"Unknown step: {step}"
    
    def process_task(self, task: str, user_id: str) -> str:
        """
        Process a task using InsideLLM context manager for tracking.
        """
        with insidellm.InsideLLMTracker(
            user_id=user_id,
            metadata={
                "agent_name": self.name,
                "agent_model": self.model,
                "task_type": "general"
            }
        ) as tracker:
            try:
                # Log user input
                input_event_id = tracker.log_user_input(
                    input_text=task,
                    input_type="text",
                    channel="custom_agent"
                )
                
                # Plan the task
                plan = self._plan_task(task)
                
                # Log planning
                planning_event = insidellm.Event(
                    run_id=tracker.run_id,
                    user_id=user_id,
                    event_type=insidellm.EventType.AGENT_PLANNING,
                    parent_event_id=input_event_id,
                    payload={
                        "plan_type": "sequential",
                        "planned_actions": [{"step": step, "order": i} for i, step in enumerate(plan)],
                        "planning_time_ms": 100
                    }
                )
                tracker.client.log_event(planning_event)
                
                # Execute steps
                results = []
                context = {"task": task}
                
                for step in plan:
                    try:
                        result = self._execute_step(step, context)
                        results.append(result)
                        context["data"] = result
                    except Exception as e:
                        error_id = tracker.log_error(
                            error_type="step_execution_error",
                            error_message=str(e),
                            error_code=type(e).__name__
                        )
                        results.append(f"Error in {step}: {e}")
                
                # Generate final response
                final_response = f"Task completed. Results: {results[-1] if results else 'No results'}"
                
                # Log agent response
                tracker.log_agent_response(
                    response_text=final_response,
                    response_type="task_completion",
                    parent_event_id=planning_event.event_id,
                    response_confidence=0.9
                )
                
                return final_response
                
            except Exception as e:
                # Log any unexpected errors
                tracker.log_error(
                    error_type="task_processing_error",
                    error_message=str(e),
                    error_code=type(e).__name__
                )
                return f"Task failed: {e}"


def demonstrate_context_managers():
    """Demonstrate using context managers for detailed tracking."""
    print("\nContext Manager Example")
    print("-" * 30)
    
    # Get default client
    client = insidellm.get_client()
    
    with insidellm.InsideLLMTracker(
        user_id="context-demo-user",
        metadata={"demo": "context_managers"}
    ) as tracker:
        
        # Track LLM call with context manager
        with tracker.track_llm_call("gpt-4", "openai", "What is AI?") as log_response:
            time.sleep(0.1)  # Simulate processing
            response = "AI is artificial intelligence..."
            log_response(response)
        
        # Track tool call with context manager
        with tracker.track_tool_call("web_search", {"query": "AI news"}) as log_response:
            time.sleep(0.2)  # Simulate API call
            results = ["Article 1", "Article 2", "Article 3"]
            log_response(results)
        
        # Log performance metrics
        tracker.log_performance_metric(
            metric_name="processing_time",
            metric_value=300,
            metric_unit="ms"
        )
        
        print("Context manager tracking completed")


def main():
    """Main example function."""
    # Initialize InsideLLM
    insidellm.initialize(
        api_key=os.getenv("INSIDELLM_API_KEY", "demo-key"),
        local_testing= True,
        auto_flush_interval=3.0,
        batch_size=5
    )
    
    print("InsideLLM Custom Agent Integration Example")
    print("=" * 50)
    
    # Create custom agent
    agent = CustomAgent("MyCustomAgent")
    
    # Example tasks
    tasks = [
        "Calculate 15 * 24",
        "What's the weather in San Francisco?",
        "Remember that my favorite color is blue",
        "What did I tell you about my favorite color?",
        "Tell me a joke about programming"
    ]
    
    # Process tasks
    for i, task in enumerate(tasks, 1):
        print(f"\n{i}. Processing Task: {task}")
        print("-" * 30)
        
        result = agent.process_task(task, f"user-{i}")
        print(f"Result: {result}")
    
    # Demonstrate context managers
    demonstrate_context_managers()
    
    # Show statistics
    print("\nQueue Statistics")
    print("-" * 30)
    
    client = insidellm.get_client()
    stats = client.queue_manager.get_statistics()
    
    print(f"Events Queued: {stats['events_queued']}")
    print(f"Events Sent: {stats['events_sent']}")
    print(f"Events Failed: {stats['events_failed']}")
    print(f"Success Rate: {stats['success_rate']:.1f}%")
    
    # Flush and shutdown
    print("\nFlushing and shutting down...")
    insidellm.flush()
    insidellm.shutdown()
    print("Example completed!")


if __name__ == "__main__":
    main()
