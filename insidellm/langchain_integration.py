"""
InsideLLM LangChain Integration - Callback handlers for automatic event tracking
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

try:
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.schema import AgentAction, AgentFinish, LLMResult
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object

from .models import Event, EventType
from .utils import generate_uuid, get_iso_timestamp
from .client import InsideLLMClient

logger = logging.getLogger(__name__)


class InsideLLMCallback(BaseCallbackHandler):
    """
    LangChain callback handler for automatic InsideLLM event tracking.
    
    Integrates with LangChain's callback system to automatically log
    LLM calls, tool usage, agent actions, and errors.
    """
    
    def __init__(
        self,
        client: InsideLLMClient,
        user_id: str,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        track_llm_calls: bool = True,
        track_tool_calls: bool = True,
        track_agent_actions: bool = True,
        track_errors: bool = True
    ):
        """
        Initialize LangChain callback handler.
        
        Args:
            client: InsideLLM client instance
            user_id: User identifier
            run_id: Run identifier (auto-generated if not provided)
            metadata: Additional metadata for events
            track_llm_calls: Whether to track LLM requests/responses
            track_tool_calls: Whether to track tool calls
            track_agent_actions: Whether to track agent actions
            track_errors: Whether to track errors
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not installed. Install it with: pip install langchain")
        
        super().__init__()
        
        self.client = client
        self.user_id = user_id
        self.run_id = run_id or generate_uuid()
        self.metadata = metadata or {}
        
        # Tracking configuration
        self.track_llm_calls = track_llm_calls
        self.track_tool_calls = track_tool_calls
        self.track_agent_actions = track_agent_actions
        self.track_errors = track_errors
        
        # Internal state tracking
        self._call_stack: Dict[str, Dict[str, Any]] = {}
        self._parent_run_map: Dict[str, Optional[str]] = {}
        
        logger.info(f"InsideLLM LangChain callback initialized for run: {self.run_id}")
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts running."""
        if not self.track_llm_calls:
            return
        
        run_id_str = str(run_id)
        parent_run_id_str = str(parent_run_id) if parent_run_id else None
        
        # Store parent relationship
        self._parent_run_map[run_id_str] = parent_run_id_str
        
        # Track start time
        self._call_stack[run_id_str] = {
            'type': 'llm',
            'start_time': time.time(),
            'serialized': serialized,
            'prompts': prompts
        }
        
        # Create LLM request event
        model_name = serialized.get('name', 'unknown')
        provider = serialized.get('_type', 'unknown')
        
        # Combine all prompts
        combined_prompt = '\n'.join(prompts) if len(prompts) > 1 else (prompts[0] if prompts else '')
        
        event_metadata = self.metadata.copy()
        if metadata:
            event_metadata.update(metadata)
        if tags:
            event_metadata['tags'] = tags
        
        event = Event(
            event_id=run_id_str,
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.LLM_REQUEST,
            parent_event_id=self._get_parent_event_id(parent_run_id_str),
            metadata=event_metadata,
            payload={
                'model_name': model_name,
                'provider': provider,
                'prompt': combined_prompt,
                'parameters': kwargs
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"LLM request event logged: {run_id_str}")
    
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM ends running."""
        if not self.track_llm_calls:
            return
        
        run_id_str = str(run_id)
        
        # Get call information
        call_info = self._call_stack.pop(run_id_str, {})
        start_time = call_info.get('start_time', time.time())
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Extract response information
        response_text = ''
        if response.generations:
            response_text = '\n'.join([
                gen.text for generation in response.generations 
                for gen in generation
            ])
        
        model_name = call_info.get('serialized', {}).get('name', 'unknown')
        provider = call_info.get('serialized', {}).get('_type', 'unknown')
        
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.LLM_RESPONSE,
            parent_event_id=run_id_str,  # Parent is the request event
            metadata=self.metadata,
            payload={
                'model_name': model_name,
                'provider': provider,
                'response_text': response_text,
                'response_time_ms': response_time_ms,
                'usage': response.llm_output.get('usage', {}) if response.llm_output else {}
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"LLM response event logged for request: {run_id_str}")
    
    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM errors."""
        if not self.track_errors:
            return
        
        run_id_str = str(run_id)
        
        # Clean up call stack
        self._call_stack.pop(run_id_str, None)
        
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.ERROR,
            parent_event_id=run_id_str,
            metadata=self.metadata,
            payload={
                'error_type': 'llm_error',
                'error_message': str(error),
                'error_code': type(error).__name__,
                'context': {'llm_run_id': run_id_str}
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"LLM error event logged: {run_id_str}")
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool starts running."""
        if not self.track_tool_calls:
            return
        
        run_id_str = str(run_id)
        parent_run_id_str = str(parent_run_id) if parent_run_id else None
        
        # Store parent relationship and call info
        self._parent_run_map[run_id_str] = parent_run_id_str
        self._call_stack[run_id_str] = {
            'type': 'tool',
            'start_time': time.time(),
            'serialized': serialized,
            'input': input_str
        }
        
        tool_name = serialized.get('name', 'unknown_tool')
        
        event_metadata = self.metadata.copy()
        if metadata:
            event_metadata.update(metadata)
        if tags:
            event_metadata['tags'] = tags
        
        event = Event(
            event_id=run_id_str,
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.TOOL_CALL,
            parent_event_id=self._get_parent_event_id(parent_run_id_str),
            metadata=event_metadata,
            payload={
                'tool_name': tool_name,
                'tool_type': serialized.get('_type', 'unknown'),
                'parameters': {'input': input_str},
                'call_id': run_id_str
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"Tool call event logged: {tool_name} - {run_id_str}")
    
    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool ends running."""
        if not self.track_tool_calls:
            return
        
        run_id_str = str(run_id)
        
        # Get call information
        call_info = self._call_stack.pop(run_id_str, {})
        start_time = call_info.get('start_time', time.time())
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        tool_name = call_info.get('serialized', {}).get('name', 'unknown_tool')
        tool_type = call_info.get('serialized', {}).get('_type', 'unknown')
        
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.TOOL_RESPONSE,
            parent_event_id=run_id_str,  # Parent is the tool call event
            metadata=self.metadata,
            payload={
                'tool_name': tool_name,
                'tool_type': tool_type,
                'call_id': run_id_str,
                'response_data': output,
                'execution_time_ms': execution_time_ms,
                'success': True
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"Tool response event logged: {tool_name} - {run_id_str}")
    
    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool errors."""
        if not self.track_errors:
            return
        
        run_id_str = str(run_id)
        
        # Get call information
        call_info = self._call_stack.pop(run_id_str, {})
        tool_name = call_info.get('serialized', {}).get('name', 'unknown_tool')
        
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.ERROR,
            parent_event_id=run_id_str,
            metadata=self.metadata,
            payload={
                'error_type': 'tool_error',
                'error_message': str(error),
                'error_code': type(error).__name__,
                'context': {
                    'tool_name': tool_name,
                    'tool_run_id': run_id_str
                }
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"Tool error event logged: {tool_name} - {run_id_str}")
    
    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent takes an action."""
        if not self.track_agent_actions:
            return
        
        run_id_str = str(run_id)
        
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.AGENT_REASONING,
            parent_event_id=self._get_parent_event_id(str(parent_run_id) if parent_run_id else None),
            metadata=self.metadata,
            payload={
                'reasoning_type': 'agent_action',
                'reasoning_steps': [action.log] if action.log else [],
                'reasoning_time_ms': None
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"Agent action event logged: {run_id_str}")
    
    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent finishes."""
        if not self.track_agent_actions:
            return
        
        run_id_str = str(run_id)
        
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.AGENT_RESPONSE,
            parent_event_id=self._get_parent_event_id(str(parent_run_id) if parent_run_id else None),
            metadata=self.metadata,
            payload={
                'response_text': str(finish.return_values),
                'response_type': 'agent_finish',
                'response_metadata': finish.return_values
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"Agent finish event logged: {run_id_str}")
    
    def _get_parent_event_id(self, parent_run_id: Optional[str]) -> Optional[str]:
        """Get the appropriate parent event ID for event chaining."""
        if not parent_run_id:
            return None
        
        # If parent_run_id is in our call stack, use it directly
        if parent_run_id in self._call_stack:
            return parent_run_id
        
        # Otherwise, try to find it in our parent mapping
        return self._parent_run_map.get(parent_run_id)
