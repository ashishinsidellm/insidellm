"""
InsideLLM Context Manager - Context manager for tracking agent workflows
"""

import logging
import time
from typing import Any, Dict, Optional, List, Union
from contextlib import contextmanager

from .models import Event, EventType, MultimodalContent
from .utils import generate_uuid, get_iso_timestamp, current_parent_event_id_var
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
        content: Union[MultimodalContent, str],
        content_type: str = "text",
        **kwargs
    ) -> str:
        """
        Log user input event.
        
        Args:
            content: The user input content (text or MultimodalContent object)
            content_type: Type of content (e.g., "text", "image")
            **kwargs: Additional payload parameters
            
        Returns:
            Event ID
        """
        p_id = kwargs.pop('parent_event_id', current_parent_event_id_var.get())
        event = Event.create_user_input(
            run_id=self.run_id,
            user_id=self.user_id,
            content=content,
            content_type=content_type,
            parent_event_id=p_id,
            metadata=self.metadata,
            **kwargs
        )
        
        self.client.log_event(event)
        current_parent_event_id_var.set(event.event_id)
        logger.debug(f"User input logged: {event.event_id}")
        return event.event_id
    
    def log_agent_response(
        self,
        response: Union[MultimodalContent, str],
        response_type: str = "text",  # Corresponds to AgentResponsePayload.response_type
        parent_event_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Log agent response event.
        
        Args:
            response: The agent response (text or MultimodalContent object).
                      If MultimodalContent, its .text will be used.
            response_type: Type of response (e.g., "text", "action").
                           This aligns with AgentResponsePayload's response_type.
            parent_event_id: ID of the parent event.
            **kwargs: Additional payload parameters for AgentResponsePayload.
            
        Returns:
            Event ID
        """
        p_id = parent_event_id or current_parent_event_id_var.get()

        actual_response_text: Optional[str] = None
        if isinstance(response, MultimodalContent):
            actual_response_text = response.text
            # If response_type is "text" but MultimodalContent has no text, it might be an issue.
            # However, AgentResponsePayload expects a string for response_text.
            # We could also warn if response.uri is present but response_type is just "text".
            # For now, we prioritize extracting .text if it's MultimodalContent.
        elif isinstance(response, str):
            actual_response_text = response
        else:
            raise TypeError("response must be a string or MultimodalContent object")

        # Ensure 'response_text' and 'response_type' are not duplicated in kwargs
        kwargs.pop('response_text', None)
        kwargs.pop('response_type', None)

        event = Event(
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.AGENT_RESPONSE,
            parent_event_id=p_id,
            metadata=self.metadata,
            payload={
                'response_text': actual_response_text,
                'response_type': response_type,
                **kwargs
            }
        )
        
        self.client.log_event(event)
        current_parent_event_id_var.set(event.event_id)
        logger.debug(f"Agent response logged: {event.event_id}")
        return event.event_id
    
    def log_llm_request(
        self,
        model_name: str,
        provider: str,
        input: Union[List[MultimodalContent], MultimodalContent, str],
        messages: Optional[List[MultimodalContent]] = None,
        parent_event_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Log LLM request event.
        
        Args:
            model_name: Name of the model
            provider: LLM provider
            input: The input to the LLM (text, MultimodalContent, or list of MultimodalContent)
            messages: Optional list of messages for chat models.
            parent_event_id: ID of the parent event
            **kwargs: Additional payload parameters
            
        Returns:
            Event ID
        """
        p_id = parent_event_id or current_parent_event_id_var.get()
        # Ensure 'input' and 'messages' are not duplicated if passed in kwargs
        kwargs.pop('input', None)
        kwargs.pop('messages', None)

        event = Event.create_llm_request(
            run_id=self.run_id,
            user_id=self.user_id,
            model_name=model_name,
            provider=provider,
            input=input,
            messages=messages,
            parent_event_id=p_id,
            metadata=self.metadata,
            **kwargs
        )
        
        self.client.log_event(event)
        current_parent_event_id_var.set(event.event_id)
        logger.debug(f"LLM request logged: {event.event_id}")
        return event.event_id
    
    def log_llm_response(
        self,
        model_name: str,
        provider: str,
        output: Union[MultimodalContent, str],
        parent_event_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Log LLM response event.
        
        Args:
            model_name: Name of the model
            provider: LLM provider
            output: The response from the LLM (text or MultimodalContent)
            parent_event_id: ID of the parent event (usually the request)
            **kwargs: Additional payload parameters
            
        Returns:
            Event ID
        """
        p_id = parent_event_id or current_parent_event_id_var.get()
        # Ensure 'output' is not duplicated if passed in kwargs
        kwargs.pop('output', None)

        event = Event.create_llm_response(
            run_id=self.run_id,
            user_id=self.user_id,
            model_name=model_name,
            provider=provider,
            output=output,
            parent_event_id=p_id,
            metadata=self.metadata,
            **kwargs
        )
        
        self.client.log_event(event)
        current_parent_event_id_var.set(event.event_id)
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
        p_id = parent_event_id or current_parent_event_id_var.get()
        event = Event(
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.TOOL_CALL,
            parent_event_id=p_id,
            metadata=self.metadata,
            payload={
                'tool_name': tool_name,
                'tool_type': tool_type,
                'parameters': parameters
            }
        )
        
        self.client.log_event(event)
        current_parent_event_id_var.set(event.event_id)
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
        p_id = parent_event_id or current_parent_event_id_var.get()
        event = Event(
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.TOOL_RESPONSE,
            parent_event_id=p_id,
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
        current_parent_event_id_var.set(event.event_id)
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
        p_id = parent_event_id or current_parent_event_id_var.get()
        event = Event.create_error(
            run_id=self.run_id,
            user_id=self.user_id,
            error_type=error_type,
            error_message=error_message,
            error_code=error_code,
            parent_event_id=p_id,
            metadata=self.metadata,
            **kwargs
        )
        
        self.client.log_event(event)
        current_parent_event_id_var.set(event.event_id)
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
        p_id = kwargs.pop('parent_event_id', current_parent_event_id_var.get())
        event = Event(
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.PERFORMANCE_METRIC,
            parent_event_id=p_id,
            metadata=self.metadata,
            payload={
                'metric_name': metric_name,
                'metric_value': metric_value,
                'metric_unit': metric_unit,
                **kwargs
            }
        )
        
        self.client.log_event(event)
        current_parent_event_id_var.set(event.event_id)
        logger.debug(f"Performance metric logged: {metric_name} = {metric_value}")
        return event.event_id
    
    @contextmanager
    def track_llm_call(self, model_name: str, provider: str,
                       input: Union[List[MultimodalContent], MultimodalContent, str],
                       messages: Optional[List[MultimodalContent]] = None):
        """
        Context manager for tracking LLM calls.
        
        Args:
            model_name: Name of the model
            provider: LLM provider
            input: The input to the LLM (text, MultimodalContent, or list of MultimodalContent)
            messages: Optional list of messages for chat models.
            
        Yields:
            A callable to log the response
            
        Example:
            with tracker.track_llm_call('gpt-4', 'openai', 'Hello') as log_response:
                response_content = call_llm(input_text) # or call_llm_multimodal(input_object)
                log_response(response_content)

            # For chat models
            with tracker.track_llm_call('claude-3', 'anthropic', input=None, messages=[...]) as log_response:
                response_content = call_chat_model(messages)
                log_response(response_content)

        """
        # Log request
        request_id = self.log_llm_request(model_name, provider, input=input, messages=messages)
        start_time = time.time()
        
        def log_response(output: Union[MultimodalContent, str], **kwargs):
            """Log the LLM response."""
            response_time_ms = int((time.time() - start_time) * 1000)
            return self.log_llm_response(
                model_name=model_name,
                provider=provider,
                output=output,
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
