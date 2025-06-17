"""
InsideLLM CrewAI Integration Test Suite (Using Real Client)
==========================================================
This version uses the real InsideLLMClient instead of mocks.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, List, Tuple
from insidellm.utils import generate_uuid
import insidellm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
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


def setup_real_insidellm_client() -> insidellm.InsideLLMClient:
    """Initialize the real InsideLLM client for testing."""
    try:
        client = insidellm.initialize(
            api_key=os.getenv("INSIDELLM_API_KEY", "your_api_key_here"),
            local_testing=True  # Set to False for production
        )
        logger.info("✅ Real InsideLLM client initialized")
        return client
    except Exception as e:
        logger.error(f"❌ Failed to initialize InsideLLM client: {e}")
        raise


def create_test_crew() -> Tuple[Any, List[Any], List[Any]]:
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
        
        # Create agents with detailed metadata
        researcher = Agent(
            role="Research Analyst",
            goal="Gather and analyze information about AI trends",
            backstory="You are an experienced research analyst specializing in AI and technology trends.",
            verbose=True,
            allow_delegation=False,
            tools=[CalculatorTool()],
            metadata={
                "expertise": ["AI", "Technology Trends", "Data Analysis"],
                "experience_level": "Senior",
                "specialization": "AI Research"
            }
        )
        
        writer = Agent(
            role="Content Writer",
            goal="Create engaging content based on research findings",
            backstory="You are a skilled content writer who can transform complex research into accessible content.",
            verbose=True,
            allow_delegation=False,
            metadata={
                "expertise": ["Content Writing", "Technical Writing", "Blog Posts"],
                "experience_level": "Senior",
                "specialization": "Technical Content"
            }
        )
        
        # Create tasks with detailed metadata
        research_task = Task(
            description="Research the current state of AI agents and their applications. Use the calculator tool to compute some statistics if needed.",
            expected_output="A comprehensive report on AI agents with key statistics",
            agent=researcher,
            metadata={
                "task_type": "research",
                "complexity": "high",
                "estimated_duration": "30 minutes",
                "required_tools": ["Calculator"]
            }
        )
        
        writing_task = Task(
            description="Write a blog post about AI agents based on the research findings",
            expected_output="A well-structured blog post about AI agents",
            agent=writer,
            metadata={
                "task_type": "content_creation",
                "complexity": "medium",
                "estimated_duration": "20 minutes",
                "content_format": "blog_post"
            }
        )
        
        # Create crew with metadata
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            verbose=True,
            metadata={
                "project_name": "AI Agents Blog Post",
                "project_type": "content_creation",
                "team_size": 2,
                "estimated_completion_time": "1 hour"
            }
        )
        
        return crew, [researcher, writer], [research_task, writing_task]
        
    except Exception as e:
        logger.error(f"Failed to create test crew: {e}")
        raise


def test_callback_integration(client: insidellm.InsideLLMClient) -> bool:
    """Test the CrewAI callback integration with real client."""
    logger.info("🚀 Starting CrewAI Integration Test (Real Client)")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Cannot proceed without required dependencies")
        return False
    
    try:
        from insidellm.crewai_integration import CrewAIInsideLLMCallbackHandler
        
        # Create callback handler with real client
        callback_handler = CrewAIInsideLLMCallbackHandler(
            client=client,
            user_id=generate_uuid(),
            run_id=generate_uuid(),
            metadata={
                "test_environment": True,
                "version": "1.0.0",
                "test_type": "integration",
                "framework": "crewai"
            },
            track_llm_calls=True,
            track_tool_calls=True,
            track_agent_actions=True,
            track_task_execution=True,
            track_errors=True
        )
        logger.info("✅ Callback handler created with real InsideLLM client")
        
        # Create test crew
        crew, agents, tasks = create_test_crew()
        logger.info("✅ Test crew created")
        
        # Test manual callback methods
        logger.info("🧪 Testing callback methods with real client...")
        
        # Test crew start
        callback_handler.on_crew_start({
            "name": crew.name,
            "agent_count": len(crew.agents),
            "task_count": len(crew.tasks),
            "crew_type": "research_and_writing",
            "crew_metadata": {
                "project_name": "AI Research Project",
                "project_type": "Research and Content Creation",
                "team_size": len(crew.agents),
                "estimated_completion_time": "2 hours"
            }
        })
        
        # Test agent callbacks
        for agent in agents:
            agent_data = {
                "role": agent.role,
                "goal": agent.goal,
                "backstory": agent.backstory,
                "metadata": getattr(agent, "metadata", {}),
                "tools": [tool.name for tool in agent.tools] if hasattr(agent, 'tools') else []
            }
            callback_handler.on_agent_start(agent.role, agent_data)
            time.sleep(0.1)
            callback_handler.on_agent_end(agent.role, {
                "result": f"Agent {agent.role} completed successfully",
                "execution_time": time.time() - agent_data.get('start_time', time.time()),
                "metadata": getattr(agent, "metadata", {})
            })
        
        # Test task callbacks
        for task in tasks:
            task_data = {
                "description": task.description,
                "expected_output": task.expected_output,
                "agent": task.agent.role if task.agent else "unknown",
                "metadata": getattr(task, "metadata", {}),
                "start_time": time.time()
            }
            callback_handler.on_task_start(f"task_{tasks.index(task)}", task_data)
            time.sleep(0.1)
            callback_handler.on_task_end(f"task_{tasks.index(task)}", {
                "result": "Task completed successfully",
                "execution_time": time.time() - task_data.get('start_time', time.time()),
                "metadata": getattr(task, "metadata", {})
            })
        
        # Test tool callbacks
        tool_call_id = callback_handler.on_tool_start(
            "Calculator", 
            "2 + 2", 
            {
                "context": "testing",
                "tool_type": "calculator",
                "input_type": "mathematical_expression"
            }
        )
        time.sleep(0.1)
        callback_handler.on_tool_end(tool_call_id, {
            "result": "4",
            "execution_time": 0.1,
            "success": True
        })
        
        # Test LLM callbacks
        callback_handler.log_pre_api_call(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, world!"}
            ],
            kwargs={
                "temperature": 0.7,
                "max_tokens": 100,
                "metadata": {
                    "request_type": "chat_completion",
                    "purpose": "testing"
                }
            }
        )
        
        # Mock response object
        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]
                self.usage = MockUsage()
        
        class MockChoice:
            def __init__(self):
                self.message = MockMessage()
                self.finish_reason = "stop"
        
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
            {
                "context": "testing error handling",
                "error_type": "test_error",
                "error_code": "TEST_ERROR",
                "timestamp": time.time()
            }
        )
        
        # Test custom event
        callback_handler.log_custom_event(
            "test_event",
            {
                "message": "This is a custom test event",
                "value": 42,
                "metadata": {
                    "event_type": "test",
                    "timestamp": time.time()
                }
            }
        )
        
        # Test crew end
        callback_handler.on_crew_end(
            {
                "name": crew.name,
                "agent_count": len(crew.agents),
                "task_count": len(crew.tasks),
                "crew_type": "research_and_writing",
                "crew_metadata": {
                    "project_name": "AI Research Project",
                    "project_type": "Research and Content Creation",
                    "team_size": len(crew.agents),
                    "estimated_completion_time": "2 hours"
                }
            },
            "Crew execution completed successfully"
        )
        
        # Flush events
        callback_handler.flush_events()
        
        logger.info("✅ All events sent to real InsideLLM client")
        logger.info("ℹ️  Check your InsideLLM dashboard for the logged events")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_crew_execution_with_callback(client: insidellm.InsideLLMClient) -> bool:
    """Test actual crew execution with real InsideLLM client."""
    logger.info("🚀 Testing actual crew execution with real callbacks")
    
    try:
        from insidellm.crewai_integration import (
            CrewAIInsideLLMCallbackHandler,
            create_crew_with_callback
        )
        
        # Create callback handler with real client
        callback_handler = CrewAIInsideLLMCallbackHandler(
            client=client,
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
            
            # Note: Actual execution would require API keys
            logger.info("ℹ️  To test full execution, uncomment the kickoff() line")
            logger.info("ℹ️  Ensure you have proper API keys configured")
            
            # Uncomment to test actual execution:
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


def main() -> bool:
    """Main function to run all tests with real client."""
    logger.info("=" * 60)
    logger.info("🧪 InsideLLM CrewAI Integration Test Suite (Real Client)")
    logger.info("=" * 60)
    
    # Initialize real InsideLLM client
    try:
        client = setup_real_insidellm_client()
    except Exception:
        return False
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Callback Integration
    logger.info("\n" + "=" * 40)
    logger.info("Test 1: Callback Integration")
    logger.info("=" * 40)
    if test_callback_integration(client):
        logger.info("✅ Test 1 PASSED")
        success_count += 1
    else:
        logger.error("❌ Test 1 FAILED")
    
    # Test 2: Crew Execution with Callback
    logger.info("\n" + "=" * 40)
    logger.info("Test 2: Crew Execution with Callback")
    logger.info("=" * 40)
    if test_crew_execution_with_callback(client):
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
        logger.info("🎉 All tests passed! Check InsideLLM dashboard for events.")
    else:
        logger.error(f"❌ {total_tests - success_count} test(s) failed.")
    
    return success_count == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)