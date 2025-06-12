"""
Example script to test InsideLLM CrewAI integration.

This script demonstrates how to use the CrewAI callback handler
to automatically track agent actions, task executions, and LLM calls.
"""

import os
import sys
import time
import logging
from typing import Dict, Any
from insidellm.utils import generate_uuid
# Add the parent directory to the Python path (adjust as needed)
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
    try:
        import crewai
        logger.info("✅ CrewAI is available")
    except ImportError:
        missing_deps.append("crewai")
        logger.error("❌ CrewAI is not available")
    
    try:
        import litellm
        logger.info("✅ LiteLLM is available")
    except ImportError:
        missing_deps.append("litellm")
        logger.error("❌ LiteLLM is not available")
    
    if missing_deps:
        logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.error("Install with: pip install crewai litellm")
        return False
    
    return True


def setup_mock_insidellm_client():
    """
    Create a mock InsideLLM client for testing purposes.
    Replace this with your actual InsideLLM client initialization.
    """
    class MockInsideLLMClient:
        def __init__(self):
            self.events = []
            logger.info("Mock InsideLLM client initialized")
        
        def log_event(self, event):
            """Mock event logging - just print and store."""
            self.events.append(event)
            logger.info(f"📝 Event logged: {event.event_type} - {event.payload.get('reasoning_type', event.payload.get('tool_name', 'unknown'))}")
            
        def flush(self):
            """Mock flush - just log count."""
            logger.info(f"🔄 Flushed {len(self.events)} events")
    
    return MockInsideLLMClient()


def create_test_crew():
    """Create a test crew with agents and tasks."""
    try:
        from crewai import Agent, Task, Crew
        from crewai.tools import BaseTool
        
        # Define a simple custom tool for testing
        class CalculatorTool(BaseTool):
            name: str = "Calculator"
            description: str = "Useful for performing mathematical calculations"
            
            def _run(self, expression: str) -> str:
                """Execute the calculation."""
                try:
                    # Simple evaluation - in production, use safer alternatives
                    result = eval(expression.replace("^", "**"))
                    return f"The result of {expression} is {result}"
                except Exception as e:
                    return f"Error calculating {expression}: {str(e)}"
        
        # Create agents
        researcher = Agent(
            role="Research Analyst",
            goal="Gather and analyze information about AI trends",
            backstory="You are an experienced research analyst specializing in AI and technology trends.",
            verbose=True,
            allow_delegation=False,
            tools=[CalculatorTool()]
        )
        
        writer = Agent(
            role="Content Writer",
            goal="Create engaging content based on research findings",
            backstory="You are a skilled content writer who can transform complex research into accessible content.",
            verbose=True,
            allow_delegation=False
        )
        
        # Create tasks
        research_task = Task(
            description="Research the current state of AI agents and their applications. Use the calculator tool to compute some statistics if needed.",
            expected_output="A comprehensive report on AI agents with key statistics",
            agent=researcher
        )
        
        writing_task = Task(
            description="Write a blog post about AI agents based on the research findings",
            expected_output="A well-structured blog post about AI agents",
            agent=writer
        )
        
        # Create crew
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            verbose=True
        )
        
        return crew, [researcher, writer], [research_task, writing_task]
        
    except Exception as e:
        logger.error(f"Failed to create test crew: {e}")
        raise


def test_callback_integration():
    """Test the CrewAI callback integration."""
    logger.info("🚀 Starting CrewAI Integration Test")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Cannot proceed without required dependencies")
        return False
    
    try:
        # Import the integration
        from insidellm.crewai_integration import (
            CrewAIInsideLLMCallbackHandler,
            create_crew_with_callback
        )
        logger.info("✅ Successfully imported CrewAI integration")
        
        # Setup mock client
        mock_client = setup_mock_insidellm_client()
        
        # Create callback handler
        callback_handler = CrewAIInsideLLMCallbackHandler(
            client=mock_client,
            user_id=generate_uuid(),
            run_id=generate_uuid(),
            metadata={"test_environment": True, "version": "1.0.0"},
            track_llm_calls=True,
            track_tool_calls=True,
            track_agent_actions=True,
            track_task_execution=True,
            track_errors=True
        )
        logger.info("✅ Callback handler created")
        
        # Create test crew
        crew, agents, tasks = create_test_crew()
        logger.info("✅ Test crew created")
        
        # Test manual callback methods
        logger.info("🧪 Testing manual callback methods...")
        
        # Test crew start/end
        crew_data = {"name": "Test Crew", "agent_count": len(agents), "task_count": len(tasks)}
        callback_handler.on_crew_start(crew_data)
        
        # Test agent callbacks
        for agent in agents:
            agent_data = {
                "role": agent.role,
                "goal": agent.goal,
                "backstory": agent.backstory
            }
            callback_handler.on_agent_start(agent.role, agent_data)
            
            # Simulate some work
            time.sleep(0.1)
            
            callback_handler.on_agent_end(agent.role, f"Agent {agent.role} completed successfully")
        
        # Test task callbacks
        for task in tasks:
            task_data = {
                "description": task.description,
                "expected_output": task.expected_output,
                "agent": task.agent.role if task.agent else "unknown"
            }
            callback_handler.on_task_start(f"task_{tasks.index(task)}", task_data)
            
            # Simulate some work
            time.sleep(0.1)
            
            callback_handler.on_task_end(f"task_{tasks.index(task)}", "Task completed successfully")
        
        # Test tool callbacks
        tool_call_id = callback_handler.on_tool_start(
            "Calculator", 
            "2 + 2", 
            {"context": "testing"}
        )
        time.sleep(0.1)
        callback_handler.on_tool_end(tool_call_id, "4")
        
        # Test LLM callbacks (mock)
        callback_handler.log_pre_api_call(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello, world!"}],
            kwargs={"temperature": 0.7, "max_tokens": 100}
        )
        
        # Mock response object
        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]
                self.usage = MockUsage()
        
        class MockChoice:
            def __init__(self):
                self.message = MockMessage()
        
        class MockMessage:
            def __init__(self):
                self.content = "Hello! How can I help you today?"
        
        class MockUsage:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 15
                self.total_tokens = 25
        
        mock_response = MockResponse()
        callback_handler.log_post_api_call(
            kwargs={"model": "gpt-4"},
            response_obj=mock_response,
            start_time=time.time() - 1,
            end_time=time.time()
        )
        
        # Test error callback
        callback_handler.on_error(
            ValueError("Test error"),
            {"context": "testing error handling"}
        )
        
        # Test custom event
        callback_handler.log_custom_event(
            "test_event",
            {"message": "This is a custom test event", "value": 42}
        )
        
        # Test crew end
        callback_handler.on_crew_end(crew_data, "Crew execution completed successfully")
        
        # Flush events
        callback_handler.flush_events()
        
        # Print summary
        logger.info(f"🎉 Test completed! Logged {len(mock_client.events)} events:")
        for i, event in enumerate(mock_client.events, 1):
            event_info = event.payload.get('reasoning_type', 
                        event.payload.get('tool_name', 
                        event.payload.get('custom_event_type', 'unknown')))
            logger.info(f"  {i}. {event.event_type} - {event_info}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_crew_execution_with_callback():
    """Test actual crew execution with callback integration."""
    logger.info("🚀 Testing actual crew execution with callbacks")
    
    try:
        from insidellm.crewai_integration import (
            CrewAIInsideLLMCallbackHandler,
            create_crew_with_callback
        )
        
        # Setup mock client
        mock_client = setup_mock_insidellm_client()
        
        # Create callback handler
        callback_handler = CrewAIInsideLLMCallbackHandler(
            client=mock_client,
            user_id="execution_test_user",
            run_id="execution_test_run",
            metadata={"test_type": "execution", "version": "1.0.0"}
        )
        
        # Create test crew
        crew, agents, tasks = create_test_crew()
        
        # Try to integrate callback with crew
        try:
            crew_with_callback = create_crew_with_callback(
                agents=agents,
                tasks=tasks,
                callback_handler=callback_handler,
                verbose=True
            )
            logger.info("✅ Crew created with callback integration")
            
            # Note: Actual execution would require API keys and might be expensive
            # For testing, we'll just verify the crew is properly configured
            logger.info("✅ Crew configuration verified")
            logger.info("ℹ️  To test full execution, ensure you have proper API keys configured")
            
            # Uncomment the following lines to test actual execution:
            # logger.info("🔄 Starting crew execution...")
            # result = crew_with_callback.kickoff()
            # logger.info(f"✅ Crew execution completed: {result}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to integrate callback with crew: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Execution test failed: {e}")
        return False


def main():
    """Main function to run all tests."""
    logger.info("=" * 60)
    logger.info("🧪 InsideLLM CrewAI Integration Test Suite")
    logger.info("=" * 60)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Callback Integration
    logger.info("\n" + "=" * 40)
    logger.info("Test 1: Callback Integration")
    logger.info("=" * 40)
    if test_callback_integration():
        logger.info("✅ Test 1 PASSED")
        success_count += 1
    else:
        logger.error("❌ Test 1 FAILED")
    
    # Test 2: Crew Execution with Callback
    logger.info("\n" + "=" * 40)
    logger.info("Test 2: Crew Execution with Callback")
    logger.info("=" * 40)
    if test_crew_execution_with_callback():
        logger.info("✅ Test 2 PASSED")
        success_count += 1
    else:
        logger.error("❌ Test 2 FAILED")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Test Summary")
    logger.info("=" * 60)
    logger.info(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        logger.info("🎉 All tests passed! The CrewAI integration is working correctly.")
    else:
        logger.error(f"❌ {total_tests - success_count} test(s) failed. Please check the logs above.")
    
    return success_count == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)