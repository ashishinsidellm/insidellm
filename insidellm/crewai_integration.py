"""
InsideLLM CrewAI Integration - Callback handlers for automatic event tracking
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

# Check for CrewAI availability
try:
    import crewai
    from crewai.agent import Agent
    from crewai.task import Task
    from crewai.crew import Crew
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    # Create dummy classes when CrewAI is not available
    Agent = object
    Task = object
    Crew = object

# Check for LiteLLM availability (used by CrewAI)
try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

# Fix for relative imports - use absolute imports instead
try:
    # Try relative imports first (when running as module)
    from .models import Event, EventType
    from .utils import generate_uuid, get_iso_timestamp
    from .client import InsideLLMClient
except ImportError:
    # Fall back to absolute imports (when running directly)
    try:
        from insidellm.models import Event, EventType
        from insidellm.utils import generate_uuid, get_iso_timestamp
        from insidellm.client import InsideLLMClient
    except ImportError:
        # If still failing, add current directory to path
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from models import Event, EventType
        from utils import generate_uuid, get_iso_timestamp
        from client import InsideLLMClient

logger = logging.getLogger(__name__)


class CrewAIInsideLLMCallbackHandler:
    """
    CrewAI callback handler for automatic InsideLLM event tracking.
    
    Integrates with CrewAI's execution flow to automatically log
    agent actions, task executions, tool usage, and LLM calls.
    """
    
    # Class attributes for availability checks
    CREWAI_AVAILABLE = CREWAI_AVAILABLE
    LITELLM_AVAILABLE = LITELLM_AVAILABLE
    
    def __init__(
        self,
        client: InsideLLMClient,
        user_id: str,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        track_llm_calls: bool = True,
        track_tool_calls: bool = True,
        track_agent_actions: bool = True,
        track_task_execution: bool = True,
        track_errors: bool = True
    ):
        """
        Initialize CrewAI callback handler.
        
        Args:
            client: InsideLLM client instance
            user_id: User identifier
            run_id: Run identifier (auto-generated if not provided)
            metadata: Additional metadata for events
            track_llm_calls: Whether to track LLM requests/responses
            track_tool_calls: Whether to track tool calls
            track_agent_actions: Whether to track agent actions
            track_task_execution: Whether to track task execution
            track_errors: Whether to track errors
        """
        self.client = client
        self.user_id = user_id
        self.run_id = run_id or generate_uuid()
        self.metadata = metadata or {}
        
        # Tracking configuration
        self.track_llm_calls = track_llm_calls
        self.track_tool_calls = track_tool_calls
        self.track_agent_actions = track_agent_actions
        self.track_task_execution = track_task_execution
        self.track_errors = track_errors
        
        # Internal state tracking
        self._call_stack: Dict[str, Dict[str, Any]] = {}
        self._agent_context: Dict[str, Any] = {}
        self._task_context: Dict[str, Any] = {}
        
        logger.info(f"InsideLLM CrewAI callback initialized for run: {self.run_id}")
    
    # CrewAI-specific callback methods
    # These methods are called by CrewAI during execution
    def on_crew_start(self, crew_data: Dict[str, Any]) -> None:
        """Called when crew execution starts."""
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.AGENT_REASONING,
            metadata=self.metadata,
            payload={
                'reasoning_type': 'crew_start',
                'crew_info': crew_data,
                'start_time': get_iso_timestamp()
            }
        )
        
        self.client.log_event(event)
        logger.debug("Crew start event logged")
    
    # CrewAI crew end callback
    # This method is called when the crew execution ends
    def on_crew_end(self, crew_data: Dict[str, Any], result: Any) -> None:
        """Called when crew execution ends."""
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.AGENT_RESPONSE,
            metadata=self.metadata,
            payload={
                'response_type': 'crew_end',
                'crew_info': crew_data,
                'result': str(result),
                'end_time': get_iso_timestamp()
            }
        )
        
        self.client.log_event(event)
        logger.debug("Crew end event logged")
    
    # Agent and task execution callbacks
    # These methods are called when agents and tasks start/end execution
    def on_agent_start(self, agent_name: str, agent_data: Dict[str, Any]) -> None:
        """Called when an agent starts execution."""
        if not self.track_agent_actions:
            return
        
        self._agent_context[agent_name] = {
            'start_time': time.time(),
            'data': agent_data
        }
        
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.AGENT_REASONING,
            metadata=self.metadata,
            payload={
                'reasoning_type': 'agent_start',
                'agent_name': agent_name,
                'agent_role': agent_data.get('role', 'unknown'),
                'agent_goal': agent_data.get('goal', ''),
                'agent_backstory': agent_data.get('backstory', ''),
                'start_time': get_iso_timestamp()
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"Agent start event logged: {agent_name}")
    
    # Agent end callback
    # This method is called when an agent ends execution
    def on_agent_end(self, agent_name: str, result: Any) -> None:
        """Called when an agent ends execution."""
        if not self.track_agent_actions:
            return
        
        # Get context and calculate execution time
        context = self._agent_context.pop(agent_name, {})
        start_time = context.get('start_time', time.time())
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.AGENT_RESPONSE,
            metadata=self.metadata,
            payload={
                'response_type': 'agent_end',
                'agent_name': agent_name,
                'result': str(result),
                'execution_time_ms': execution_time_ms,
                'end_time': get_iso_timestamp()
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"Agent end event logged: {agent_name}")
    
    # Task execution callbacks
    # These methods are called when tasks start/end execution
    def on_task_start(self, task_name: str, task_data: Dict[str, Any]) -> None:
        """Called when a task starts execution."""
        if not self.track_task_execution:
            return
        
        self._task_context[task_name] = {
            'start_time': time.time(),
            'data': task_data
        }
        
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.TOOL_CALL,  # Tasks are similar to tool calls in context
            metadata=self.metadata,
            payload={
                'tool_name': f"task_{task_name}",
                'tool_type': 'crewai_task',
                'parameters': {
                    'description': task_data.get('description', ''),
                    'expected_output': task_data.get('expected_output', ''),
                    'agent': task_data.get('agent', 'unknown')
                },
                'start_time': get_iso_timestamp()
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"Task start event logged: {task_name}")
    
    # Task end callback
    # This method is called when a task ends execution
    def on_task_end(self, task_name: str, result: Any) -> None:
        """Called when a task ends execution."""
        if not self.track_task_execution:
            return
        
        # Get context and calculate execution time
        context = self._task_context.pop(task_name, {})
        start_time = context.get('start_time', time.time())
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.TOOL_RESPONSE,
            metadata=self.metadata,
            payload={
                'tool_name': f"task_{task_name}",
                'tool_type': 'crewai_task',
                'response_data': str(result),
                'execution_time_ms': execution_time_ms,
                'success': True,
                'end_time': get_iso_timestamp()
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"Task end event logged: {task_name}")
    
    # Tool call tracking methods
    # These methods are called when tools start/end execution
    def on_tool_start(self, tool_name: str, tool_input: str, context: Dict[str, Any] = None) -> None:
        """Called when a tool starts execution."""
        if not self.track_tool_calls:
            return
        
        call_id = generate_uuid()
        self._call_stack[call_id] = {
            'type': 'tool',
            'start_time': time.time(),
            'tool_name': tool_name,
            'input': tool_input
        }
        
        event = Event(
            event_id=call_id,
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.TOOL_CALL,
            metadata=self.metadata,
            payload={
                'tool_name': tool_name,
                'tool_type': 'crewai_tool',
                'parameters': {'input': tool_input},
                'context': context or {},
                'call_id': call_id
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"Tool call event logged: {tool_name}")
        
        return call_id
    
    # Tool end callback
    # This method is called when a tool ends execution
    def on_tool_end(self, call_id: str, result: Any) -> None:
        """Called when a tool ends execution."""
        if not self.track_tool_calls:
            return
        
        # Get call information
        call_info = self._call_stack.pop(call_id, {})
        start_time = call_info.get('start_time', time.time())
        execution_time_ms = int((time.time() - start_time) * 1000)
        tool_name = call_info.get('tool_name', 'unknown')
        
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.TOOL_RESPONSE,
            parent_event_id=call_id,
            metadata=self.metadata,
            payload={
                'tool_name': tool_name,
                'tool_type': 'crewai_tool',
                'call_id': call_id,
                'response_data': str(result),
                'execution_time_ms': execution_time_ms,
                'success': True
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"Tool response event logged: {tool_name}")
    
    # LiteLLM callback methods (for LLM call tracking)
    
    def log_pre_api_call(self, model: str, messages: List[Dict], kwargs: Dict[str, Any]) -> None:
        """Called before LLM API call (LiteLLM integration)."""
        if not self.track_llm_calls or not self.LITELLM_AVAILABLE:
            return
        
        call_id = generate_uuid()
        self._call_stack[call_id] = {
            'type': 'llm',
            'start_time': time.time(),
            'model': model,
            'messages': messages
        }
        
        # Extract prompt from messages
        prompt = ""
        if messages:
            if isinstance(messages[-1], dict) and 'content' in messages[-1]:
                prompt = messages[-1]['content']
            else:
                prompt = str(messages[-1])
        
        event = Event(
            event_id=call_id,
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.LLM_REQUEST,
            metadata=self.metadata,
            payload={
                'model_name': model,
                'provider': 'crewai_llm',
                'prompt': prompt,
                'parameters': kwargs,
                'messages': messages
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"LLM request event logged: {model}")
        
        return call_id
    
    # LLM response callback
    # This method is called after LLM API call
    # It logs the response and usage information
    def log_post_api_call(self, kwargs: Dict[str, Any], response_obj: Any, 
                         start_time: float, end_time: float) -> None:
        """Called after LLM API call (LiteLLM integration)."""
        if not self.track_llm_calls or not self.LITELLM_AVAILABLE:
            return
        
        response_time_ms = int((end_time - start_time) * 1000)
        
        # Extract response text
        response_text = ""
        if hasattr(response_obj, 'choices') and response_obj.choices:
            choice = response_obj.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                response_text = choice.message.content
            elif hasattr(choice, 'text'):
                response_text = choice.text
        
        # Extract usage information
        usage = {}
        if hasattr(response_obj, 'usage'):
            usage = {
                'prompt_tokens': getattr(response_obj.usage, 'prompt_tokens', 0),
                'completion_tokens': getattr(response_obj.usage, 'completion_tokens', 0),
                'total_tokens': getattr(response_obj.usage, 'total_tokens', 0)
            }
        
        model_name = kwargs.get('model', 'unknown')
        
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.LLM_RESPONSE,
            metadata=self.metadata,
            payload={
                'model_name': model_name,
                'provider': 'crewai_llm',
                'response_text': response_text,
                'response_time_ms': response_time_ms,
                'usage': usage,
                'response_metadata': str(response_obj)
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"LLM response event logged: {model_name}")
    
    def log_failure_event(self, kwargs: Dict[str, Any], response_obj: Any,
                         start_time: float, end_time: float) -> None:
        """Called when LLM API call fails (LiteLLM integration)."""
        if not self.track_errors:
            return
        
        model_name = kwargs.get('model', 'unknown')
        error_message = str(response_obj) if response_obj else "Unknown error"
        
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.ERROR,
            metadata=self.metadata,
            payload={
                'error_type': 'llm_error',
                'error_message': error_message,
                'error_code': 'api_failure',
                'context': {
                    'model_name': model_name,
                    'provider': 'crewai_llm',
                    'duration_ms': int((end_time - start_time) * 1000)
                }
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"LLM failure event logged: {model_name}")
    
    def on_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Called when any error occurs during CrewAI execution."""
        if not self.track_errors:
            return
        
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.ERROR,
            metadata=self.metadata,
            payload={
                'error_type': 'crewai_error',
                'error_message': str(error),
                'error_code': type(error).__name__,
                'context': context or {},
                'timestamp': get_iso_timestamp()
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"Error event logged: {type(error).__name__}")
    
    # Utility methods
    def log_custom_event(self, event_type: str, payload: Dict[str, Any], 
                        parent_event_id: Optional[str] = None) -> None:
        """Log a custom event."""
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.CUSTOM_EVENT,
            parent_event_id=parent_event_id,
            metadata=self.metadata,
            payload={
                'custom_event_type': event_type,
                **payload
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"Custom event logged: {event_type}")
    
    def flush_events(self) -> None:
        """Flush any pending events."""
        try:
            if hasattr(self.client, 'flush'):
                self.client.flush()
        except Exception as e:
            logger.error(f"Failed to flush events: {e}")


# Helper function to create a callback-enabled CrewAI setup
def create_crew_with_callback(
    agents: List[Agent],
    tasks: List[Task],
    callback_handler: CrewAIInsideLLMCallbackHandler,
    **crew_kwargs
) -> Crew:
    """
    Create a CrewAI Crew with InsideLLM callback integration.
    
    Args:
        agents: List of CrewAI agents
        tasks: List of CrewAI tasks
        callback_handler: InsideLLM callback handler
        **crew_kwargs: Additional arguments for Crew
    
    Returns:
        Configured Crew instance
    """
    if not CREWAI_AVAILABLE:
        raise ImportError("CrewAI is not available. Install it with: pip install crewai")
    
    # Create crew
    crew = Crew(
        agents=agents,
        tasks=tasks,
        **crew_kwargs
    )
    
    # Try to integrate callback
    if hasattr(crew, 'callbacks'):
        if not crew.callbacks:
            crew.callbacks = []
        crew.callbacks.append(callback_handler)
    elif hasattr(crew, 'manager_callbacks'):
        if not crew.manager_callbacks:
            crew.manager_callbacks = []
        crew.manager_callbacks.append(callback_handler)
    else:
        logger.warning("CrewAI version doesn't support direct callback integration")
    
    return crew


# Main execution block for testing
if __name__ == "__main__":
    print("InsideLLM CrewAI Integration loaded successfully!")
    print(f"CrewAI Available: {CREWAI_AVAILABLE}")
    print(f"LiteLLM Available: {LITELLM_AVAILABLE}")
    print("To use this module, import it as: from insidellm.crewai_integration import CrewAIInsideLLMCallbackHandler")