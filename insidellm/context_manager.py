"""
InsideLLM Context Manager - Context manager for tracking agent workflows
"""

import logging
import time
from typing import Any, Dict, Optional
from contextlib import contextmanager

from .models import Event, EventType
from .utils import generate_uuid, get_iso_timestamp
from .client import InsideLLMClient
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class InsideLLMTracker:
    """
    Context manager for tracking agent workflows and LLM operations.
    
    Provides a clean way to track complex agent workflows with automatic
    event logging and context management.
    """
    
    def __init__(
        self,
        client: Optional[InsideLLMClient] = None,
        run_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_start_session: bool = True
    ):
        """
        Initialize the tracker.
        
        Args:
            client: InsideLLM client instance (uses default if not provided)
            run_id: Run identifier (auto-generated if not provided)
            user_id: User identifier
            metadata: Additional metadata for events
            auto_start_session: Whether to automatically start/end session
        """
        # Get client
        if client is None:
            from . import get_client
            client = get_client()
        
        self.client = client
        self.run_id = run_id or generate_uuid()
        self.user_id = user_id
        self.metadata = metadata or {}
        self.auto_start_session = auto_start_session
        
        # Context tracking
        self._context_stack = []
        self._active_events = {}
        self._session_started = False
        
        logger.info(f"InsideLLM tracker initialized for run: {self.run_id}")
    
    def __enter__(self):
        """Enter context manager."""
        if self.auto_start_session:
            self.start_session()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        # Handle any exception
        if exc_type is not None:
            self.log_error(
                error_type='context_error',
                error_message=str(exc_val),
                error_code=exc_type.__name__
            )
        
        if self.auto_start_session:
            self.end_session()
    
    def start_session(self) -> str:
        """Start a session."""
        if not self._session_started:
            self.client.start_run(
                run_id=self.run_id,
                user_id=self.user_id,
                metadata=self.metadata
            )
            self._session_started = True
            logger.info(f"Session started: {self.run_id}")
        
        return self.run_id
    
    def end_session(self) -> None:
        """End the session."""
        if self._session_started:
            self.client.end_run(self.run_id)
            self._session_started = False
            logger.info(f"Session ended: {self.run_id}")
    
    def log_user_input(
        self,
        input_text: str,
        input_type: str = "text",
        **kwargs
    ) -> str:
        """
        Log user input event.
        
        Args:
            input_text: The user input text
            input_type: Type of input (text, voice, image, etc.)
            **kwargs: Additional payload parameters
            
        Returns:
            Event ID
        """
        event = Event.create_user_input(
            run_id=self.run_id,
            user_id=self.user_id,
            input_text=input_text,
            input_type=input_type,
            metadata=self.metadata,
            **kwargs
        )
        
        self.client.log_event(event)
        logger.debug(f"User input logged: {event.event_id}")
        return event.event_id
    
    def log_agent_response(
        self,
        response_text: str,
        response_type: str = "text",
        parent_event_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Log agent response event.
        
        Args:
            response_text: The agent response text
            response_type: Type of response
            parent_event_id: ID of the parent event
            **kwargs: Additional payload parameters
            
        Returns:
            Event ID
        """
        event = Event(
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.AGENT_RESPONSE,
            parent_event_id=parent_event_id,
            metadata=self.metadata,
            payload={
                'response_text': response_text,
                'response_type': response_type,
                **kwargs
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"Agent response logged: {event.event_id}")
        return event.event_id
    
    def log_llm_request(
        self,
        model_name: str,
        provider: str,
        prompt: str,
        parent_event_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Log LLM request event.
        
        Args:
            model_name: Name of the model
            provider: LLM provider
            prompt: The prompt sent to the LLM
            parent_event_id: ID of the parent event
            **kwargs: Additional payload parameters
            
        Returns:
            Event ID
        """
        event = Event.create_llm_request(
            run_id=self.run_id,
            user_id=self.user_id,
            model_name=model_name,
            provider=provider,
            prompt=prompt,
            parent_event_id=parent_event_id,
            metadata=self.metadata,
            **kwargs
        )
        
        self.client.log_event(event)
        logger.debug(f"LLM request logged: {event.event_id}")
        return event.event_id
    
    def log_llm_response(
        self,
        model_name: str,
        provider: str,
        response_text: str,
        parent_event_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Log LLM response event.
        
        Args:
            model_name: Name of the model
            provider: LLM provider
            response_text: The response from the LLM
            parent_event_id: ID of the parent event (usually the request)
            **kwargs: Additional payload parameters
            
        Returns:
            Event ID
        """
        event = Event.create_llm_response(
            run_id=self.run_id,
            user_id=self.user_id,
            model_name=model_name,
            provider=provider,
            response_text=response_text,
            parent_event_id=parent_event_id,
            metadata=self.metadata,
            **kwargs
        )
        
        self.client.log_event(event)
        logger.debug(f"LLM response logged: {event.event_id}")
        return event.event_id
    
    def log_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        tool_type: str = "function",
        parent_event_id: Optional[str] = None
    ) -> str:
        """
        Log tool call event.
        
        Args:
            tool_name: Name of the tool
            parameters: Parameters passed to the tool
            tool_type: Type of tool
            parent_event_id: ID of the parent event
            
        Returns:
            Event ID
        """
        event = Event(
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.TOOL_CALL,
            parent_event_id=parent_event_id,
            metadata=self.metadata,
            payload={
                'tool_name': tool_name,
                'tool_type': tool_type,
                'parameters': parameters
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"Tool call logged: {tool_name} - {event.event_id}")
        return event.event_id
    
    def log_tool_response(
        self,
        tool_name: str,
        response_data: Any,
        tool_type: str = "function",
        parent_event_id: Optional[str] = None,
        execution_time_ms: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> str:
        """
        Log tool response event.
        
        Args:
            tool_name: Name of the tool
            response_data: Response data from the tool
            tool_type: Type of tool
            parent_event_id: ID of the parent event (usually the tool call)
            execution_time_ms: Execution time in milliseconds
            success: Whether the tool call was successful
            error_message: Error message if tool call failed
            
        Returns:
            Event ID
        """
        event = Event(
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.TOOL_RESPONSE,
            parent_event_id=parent_event_id,
            metadata=self.metadata,
            payload={
                'tool_name': tool_name,
                'tool_type': tool_type,
                'response_data': response_data,
                'execution_time_ms': execution_time_ms,
                'success': success,
                'error_message': error_message
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"Tool response logged: {tool_name} - {event.event_id}")
        return event.event_id
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        error_code: Optional[str] = None,
        parent_event_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Log error event.
        
        Args:
            error_type: Type of error
            error_message: Error message
            error_code: Error code
            parent_event_id: ID of the parent event
            **kwargs: Additional payload parameters
            
        Returns:
            Event ID
        """
        event = Event.create_error(
            run_id=self.run_id,
            user_id=self.user_id,
            error_type=error_type,
            error_message=error_message,
            error_code=error_code,
            parent_event_id=parent_event_id,
            metadata=self.metadata,
            **kwargs
        )
        
        self.client.log_event(event)
        logger.debug(f"Error logged: {error_type} - {event.event_id}")
        return event.event_id
    
    def log_performance_metric(
        self,
        metric_name: str,
        metric_value: float,
        metric_unit: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Log performance metric event.
        
        Args:
            metric_name: Name of the metric
            metric_value: Value of the metric
            metric_unit: Unit of measurement
            **kwargs: Additional payload parameters
            
        Returns:
            Event ID
        """
        event = Event(
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.PERFORMANCE_METRIC,
            metadata=self.metadata,
            payload={
                'metric_name': metric_name,
                'metric_value': metric_value,
                'metric_unit': metric_unit,
                **kwargs
            }
        )
        
        self.client.log_event(event)
        logger.debug(f"Performance metric logged: {metric_name} = {metric_value}")
        return event.event_id
    
    @contextmanager
    def track_llm_call(self, model_name: str, provider: str, prompt: str):
        """
        Context manager for tracking LLM calls.
        
        Args:
            model_name: Name of the model
            provider: LLM provider
            prompt: The prompt sent to the LLM
            
        Yields:
            A callable to log the response
            
        Example:
            with tracker.track_llm_call('gpt-4', 'openai', 'Hello') as log_response:
                response = call_llm(prompt)
                log_response(response)
        """
        # Log request
        request_id = self.log_llm_request(model_name, provider, prompt)
        start_time = time.time()
        
        def log_response(response_text: str, **kwargs):
            """Log the LLM response."""
            response_time_ms = int((time.time() - start_time) * 1000)
            return self.log_llm_response(
                model_name=model_name,
                provider=provider,
                response_text=response_text,
                parent_event_id=request_id,
                response_time_ms=response_time_ms,
                **kwargs
            )
        
        try:
            yield log_response
        except Exception as e:
            # Log error
            self.log_error(
                error_type='llm_call_error',
                error_message=str(e),
                error_code=type(e).__name__,
                parent_event_id=request_id
            )
            raise
    
    @contextmanager
    def track_tool_call(self, tool_name: str, parameters: Dict[str, Any], tool_type: str = "function"):
        """
        Context manager for tracking tool calls.
        
        Args:
            tool_name: Name of the tool
            parameters: Parameters for the tool
            tool_type: Type of tool
            
        Yields:
            A callable to log the response
            
        Example:
            with tracker.track_tool_call('web_search', {'query': 'AI news'}) as log_response:
                results = search_web(query)
                log_response(results)
        """
        # Log tool call
        call_id = self.log_tool_call(tool_name, parameters, tool_type)
        start_time = time.time()
        
        def log_response(response_data: Any, success: bool = True, error_message: Optional[str] = None):
            """Log the tool response."""
            execution_time_ms = int((time.time() - start_time) * 1000)
            return self.log_tool_response(
                tool_name=tool_name,
                response_data=response_data,
                tool_type=tool_type,
                parent_event_id=call_id,
                execution_time_ms=execution_time_ms,
                success=success,
                error_message=error_message
            )
        
        try:
            yield log_response
        except Exception as e:
            # Log error response
            execution_time_ms = int((time.time() - start_time) * 1000)
            self.log_tool_response(
                tool_name=tool_name,
                response_data=None,
                tool_type=tool_type,
                parent_event_id=call_id,
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=str(e)
            )
            raise
    
    def flush(self):
        """Flush all pending events."""
        self.client.flush()
