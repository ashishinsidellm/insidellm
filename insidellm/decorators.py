"""
InsideLLM Decorators - Function decorators for automatic event tracking
"""

import functools
import logging
import time
import traceback
from typing import Any, Callable, Dict, Optional, TypeVar, Union, List

from .models import Event, EventType, MultimodalContent
from .utils import generate_uuid, get_iso_timestamp, current_parent_event_id_var
from .client import InsideLLMClient
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


def get_default_client() -> InsideLLMClient:
    """Get the default client instance."""
    from . import get_client
    return get_client()


def track_llm_call(
    model_name: str,
    provider: str,
    client: Optional[InsideLLMClient] = None,
    extract_prompt: Optional[Callable] = None,
    extract_response: Optional[Callable] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Callable[[F], F]:
    """
    Decorator to automatically track LLM calls.
    
    Args:
        model_name: Name of the LLM model
        provider: LLM provider (e.g., 'openai', 'anthropic')
        client: InsideLLM client instance (uses default if not provided)
        extract_prompt: Function to extract prompt from function arguments
        extract_response: Function to extract response from function result
        metadata: Additional metadata for events
        
    Example:
        @track_llm_call('gpt-4', 'openai')
        def call_openai(prompt, **kwargs):
            return openai.chat.completions.create(...)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get client
            tracking_client = client or get_default_client()
            current_run_id = tracking_client.get_current_run_id()
            current_user_id = tracking_client.get_current_user_id()
            
            if not current_run_id or not current_user_id:
                logger.warning("No active run context, LLM call not tracked")
                return func(*args, **kwargs)
            
            # Generate event IDs
            request_event_id = generate_uuid()
            parent_id_from_context = current_parent_event_id_var.get()
            
            # Extract prompt/input
            raw_input_data: Any
            if extract_prompt:
                raw_input_data = extract_prompt(*args, **kwargs)
            else:
                # Default extraction - look for common prompt parameter names
                raw_input_data = kwargs.get('prompt') or kwargs.get('input') or \
                                 kwargs.get('messages') or (args[0] if args else None)

            processed_input: List[MultimodalContent]
            if isinstance(raw_input_data, str):
                processed_input = [MultimodalContent(text=raw_input_data)]
            elif isinstance(raw_input_data, MultimodalContent):
                processed_input = [raw_input_data]
            elif isinstance(raw_input_data, list) and all(isinstance(item, MultimodalContent) for item in raw_input_data):
                processed_input = raw_input_data
            elif raw_input_data is None: # Case where messages might be handled separately or input is truly None
                processed_input = [] # Or handle as error, but LLMRequestPayload expects List[MultimodalContent] or messages
            else:
                # Attempt to stringify if it's not a recognized multimodal type, then wrap
                logger.warning(f"LLM input data type {type(raw_input_data)} not directly supported for MultimodalContent list, converting to string.")
                processed_input = [MultimodalContent(text=str(raw_input_data))]

            # Log LLM request event
            # Note: The 'messages' field of LLMRequestPayload is not explicitly handled here yet by a separate
            # extract_messages. If raw_input_data was meant to be 'messages', it needs to be List[MultimodalContent].
            # For now, 'input' field is prioritized.
            request_payload_dict = {
                'model_name': model_name,
                'provider': provider,
                'input': processed_input,
                 # Remove 'prompt' and 'input' from parameters to avoid duplication if they were in kwargs
                'parameters': {k: v for k, v in kwargs.items() if k not in ['prompt', 'input', 'messages']}
            }
            # If processed_input is empty and there's no 'messages' alternative (not yet supported here),
            # this might be an issue depending on LLMRequestPayload validation.
            # However, create_llm_request in models.py handles input string to List[MultimodalContent]

            request_event = Event(
                event_id=request_event_id,
                run_id=current_run_id,
                user_id=current_user_id,
                event_type=EventType.LLM_REQUEST,
                parent_event_id=parent_id_from_context,
                metadata=metadata,
                payload=request_payload_dict
            )
            tracking_client.log_event(request_event)
            current_parent_event_id_var.set(request_event_id)
            
            # Execute function and track timing
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                response_time_ms = int((end_time - start_time) * 1000)
                
                # Extract response
                raw_output_data: Any
                if extract_response:
                    raw_output_data = extract_response(result)
                else:
                    # Default extraction
                    raw_output_data = result # Keep as is, convert to string only if not MultimodalContent

                processed_output: MultimodalContent
                if isinstance(raw_output_data, MultimodalContent):
                    processed_output = raw_output_data
                elif isinstance(raw_output_data, str):
                    processed_output = MultimodalContent(text=raw_output_data)
                else:
                    # Attempt to stringify if it's not a recognized multimodal type
                    logger.warning(f"LLM output data type {type(raw_output_data)} not directly MultimodalContent, converting to string and wrapping.")
                    processed_output = MultimodalContent(text=str(raw_output_data))

                # Log LLM response event
                response_event = Event(
                    event_id=generate_uuid(),
                    run_id=current_run_id,
                    user_id=current_user_id,
                    event_type=EventType.LLM_RESPONSE,
                    parent_event_id=request_event_id,
                    metadata=metadata,
                    payload={
                        'model_name': model_name,
                        'provider': provider,
                        'output': processed_output, # Changed from response_text
                        'response_time_ms': response_time_ms
                        # Other fields like 'usage' or 'finish_reason' can be added via extract_response
                        # if extract_response returns a dict that is then spread into the payload.
                        # For now, this is direct.
                    }
                )
                tracking_client.log_event(response_event)
                current_parent_event_id_var.set(response_event.event_id)
                
                return result
                
            except Exception as e:
                end_time = time.time()
                
                # Log error event
                error_event = Event(
                    event_id=generate_uuid(),
                    run_id=current_run_id,
                    user_id=current_user_id,
                    event_type=EventType.ERROR,
                    parent_event_id=request_event_id,
                    metadata=metadata,
                    payload={
                        'error_type': 'llm_call_error',
                        'error_message': str(e),
                        'error_code': type(e).__name__,
                        'stack_trace': traceback.format_exc(),
                        'context': {
                            'model_name': model_name,
                            'provider': provider,
                            'function_name': func.__name__
                        }
                    }
                )
                tracking_client.log_event(error_event)
                current_parent_event_id_var.set(error_event.event_id)
                
                raise
        
        return wrapper
    return decorator


def track_tool_use(
    tool_name: str,
    tool_type: str = 'function',
    client: Optional[InsideLLMClient] = None,
    extract_parameters: Optional[Callable] = None,
    extract_response: Optional[Callable] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Callable[[F], F]:
    """
    Decorator to automatically track tool usage.
    
    Args:
        tool_name: Name of the tool
        tool_type: Type of tool (e.g., 'function', 'api', 'database')
        client: InsideLLM client instance (uses default if not provided)
        extract_parameters: Function to extract parameters from function arguments
        extract_response: Function to extract response from function result
        metadata: Additional metadata for events
        
    Example:
        @track_tool_use('web_search', 'api')
        def search_web(query, limit=10):
            return search_api.search(query, limit)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get client
            tracking_client = client or get_default_client()
            current_run_id = tracking_client.get_current_run_id()
            current_user_id = tracking_client.get_current_user_id()
            
            if not current_run_id or not current_user_id:
                logger.warning("No active run context, tool use not tracked")
                return func(*args, **kwargs)
            
            # Generate event IDs
            call_event_id = generate_uuid()
            parent_id_from_context = current_parent_event_id_var.get()
            
            # Extract parameters
            if extract_parameters:
                parameters = extract_parameters(*args, **kwargs)
            else:
                # Default extraction - combine args and kwargs
                parameters = {
                    'args': args,
                    'kwargs': kwargs
                }
            
            # Log tool call event
            call_event = Event(
                event_id=call_event_id,
                run_id=current_run_id,
                user_id=current_user_id,
                event_type=EventType.TOOL_CALL,
                parent_event_id=parent_id_from_context,
                metadata=metadata,
                payload={
                    'tool_name': tool_name,
                    'tool_type': tool_type,
                    'parameters': parameters,
                    'call_id': call_event_id
                }
            )
            tracking_client.log_event(call_event)
            current_parent_event_id_var.set(call_event_id)
            
            # Execute function and track timing
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time_ms = int((end_time - start_time) * 1000)
                
                # Extract response
                if extract_response:
                    response_data = extract_response(result)
                else:
                    response_data = result
                
                # Log tool response event
                response_event = Event(
                    event_id=generate_uuid(),
                    run_id=current_run_id,
                    user_id=current_user_id,
                    event_type=EventType.TOOL_RESPONSE,
                    parent_event_id=call_event_id,
                    metadata=metadata,
                    payload={
                        'tool_name': tool_name,
                        'tool_type': tool_type,
                        'call_id': call_event_id,
                        'response_data': response_data,
                        'execution_time_ms': execution_time_ms,
                        'success': True
                    }
                )
                tracking_client.log_event(response_event)
                current_parent_event_id_var.set(response_event.event_id)
                
                return result
                
            except Exception as e:
                end_time = time.time()
                execution_time_ms = int((end_time - start_time) * 1000)
                
                # Log tool error response
                response_event = Event(
                    event_id=generate_uuid(),
                    run_id=current_run_id,
                    user_id=current_user_id,
                    event_type=EventType.TOOL_RESPONSE,
                    parent_event_id=call_event_id,
                    metadata=metadata,
                    payload={
                        'tool_name': tool_name,
                        'tool_type': tool_type,
                        'call_id': call_event_id,
                        'response_data': None,
                        'execution_time_ms': execution_time_ms,
                        'success': False,
                        'error_message': str(e)
                    }
                )
                tracking_client.log_event(response_event)
                current_parent_event_id_var.set(response_event.event_id)
                
                raise
        
        return wrapper
    return decorator


def track_agent_step(
    step_name: str,
    client: Optional[InsideLLMClient] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Callable[[F], F]:
    """
    Decorator to track agent reasoning/planning steps.
    
    Args:
        step_name: Name of the agent step
        client: InsideLLM client instance (uses default if not provided)
        metadata: Additional metadata for events
        
    Example:
        @track_agent_step('analyze_query')
        def analyze_user_query(query):
            # Agent reasoning logic
            return analysis_result
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get client
            tracking_client = client or get_default_client()
            current_run_id = tracking_client.get_current_run_id()
            current_user_id = tracking_client.get_current_user_id()
            
            if not current_run_id or not current_user_id:
                logger.warning("No active run context, agent step not tracked")
                return func(*args, **kwargs)
            parent_id_from_context = current_parent_event_id_var.get()
            # Execute function and track timing
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                reasoning_time_ms = int((end_time - start_time) * 1000)
                
                # Log agent reasoning event
                reasoning_event = Event(
                    event_id=generate_uuid(),
                    run_id=current_run_id,
                    user_id=current_user_id,
                    event_type=EventType.AGENT_REASONING,
                    parent_event_id=parent_id_from_context,
                    metadata=metadata,
                    payload={
                        'reasoning_type': step_name,
                        'reasoning_steps': [f"Executed {step_name}"],
                        'reasoning_time_ms': reasoning_time_ms
                    }
                )
                tracking_client.log_event(reasoning_event)
                current_parent_event_id_var.set(reasoning_event.event_id)
                
                return result
                
            except Exception as e:
                end_time = time.time()
                
                # Log error event
                error_event = Event(
                    event_id=generate_uuid(),
                    run_id=current_run_id,
                    user_id=current_user_id,
                    event_type=EventType.ERROR,
                    parent_event_id=parent_id_from_context,
                    metadata=metadata,
                    payload={
                        'error_type': 'agent_step_error',
                        'error_message': str(e),
                        'error_code': type(e).__name__,
                        'stack_trace': traceback.format_exc(),
                        'context': {
                            'step_name': step_name,
                            'function_name': func.__name__
                        }
                    }
                )
                tracking_client.log_event(error_event)
                current_parent_event_id_var.set(error_event.event_id)
                
                raise
        
        return wrapper
    return decorator


def track_function_execution(
    function_name: Optional[str] = None,
    client: Optional[InsideLLMClient] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Callable[[F], F]:
    """
    Generic decorator to track function execution.
    
    Args:
        function_name: Name for the function (uses actual name if not provided)
        client: InsideLLM client instance (uses default if not provided)
        metadata: Additional metadata for events
        
    Example:
        @track_function_execution()
        def complex_calculation(data):
            return process_data(data)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get client
            tracking_client = client or get_default_client()
            current_run_id = tracking_client.get_current_run_id()
            current_user_id = tracking_client.get_current_user_id()
            
            if not current_run_id or not current_user_id:
                logger.warning("No active run context, function execution not tracked")
                return func(*args, **kwargs)
            parent_id_from_context = current_parent_event_id_var.get()
            # Use provided name or function name
            name = function_name or func.__name__
            
            # Execute function and track timing
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time_ms = int((end_time - start_time) * 1000)
                
                # Log function execution event
                execution_event = Event(
                    event_id=generate_uuid(),
                    run_id=current_run_id,
                    user_id=current_user_id,
                    event_type=EventType.FUNCTION_EXECUTION,
                    parent_event_id=parent_id_from_context,
                    metadata=metadata,
                    payload={
                        'function_name': name,
                        'function_args': {
                            'args': args,
                            'kwargs': kwargs
                        },
                        'return_value': str(result),
                        'execution_time_ms': execution_time_ms,
                        'success': True
                    }
                )
                tracking_client.log_event(execution_event)
                current_parent_event_id_var.set(execution_event.event_id)
                
                return result
                
            except Exception as e:
                end_time = time.time()
                execution_time_ms = int((end_time - start_time) * 1000)
                
                # Log function execution error
                execution_event = Event(
                    event_id=generate_uuid(),
                    run_id=current_run_id,
                    user_id=current_user_id,
                    event_type=EventType.FUNCTION_EXECUTION,
                    parent_event_id=parent_id_from_context,
                    metadata=metadata,
                    payload={
                        'function_name': name,
                        'function_args': {
                            'args': args,
                            'kwargs': kwargs
                        },
                        'return_value': None,
                        'execution_time_ms': execution_time_ms,
                        'success': False,
                        'error_message': str(e)
                    }
                )
                tracking_client.log_event(execution_event)
                current_parent_event_id_var.set(execution_event.event_id)
                
                raise
        
        return wrapper
    return decorator
