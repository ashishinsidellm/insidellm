"""
InsideLLM Decorators - Function decorators for automatic event tracking
"""

import functools
import logging
import time
import traceback
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from .models import Event, EventType
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
            
            # Extract prompt
            if extract_prompt:
                prompt = extract_prompt(*args, **kwargs)
            else:
                # Default extraction - look for common prompt parameter names
                prompt = kwargs.get('prompt') or kwargs.get('messages') or (args[0] if args else '')
            
            # Log LLM request event
            request_event = Event(
                event_id=request_event_id,
                run_id=current_run_id,
                user_id=current_user_id,
                event_type=EventType.LLM_REQUEST,
                parent_event_id=parent_id_from_context,
                metadata=metadata,
                payload={
                    'model_name': model_name,
                    'provider': provider,
                    'prompt': str(prompt),
                    'parameters': {k: v for k, v in kwargs.items() if k != 'prompt'}
                }
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
                if extract_response:
                    response_text = extract_response(result)
                else:
                    # Default extraction
                    response_text = str(result)
                
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
                        'response_text': response_text,
                        'response_time_ms': response_time_ms
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

def track_chain_start(
    chain_name: str,
    chain_type: str,
    inputs: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> Callable[[F], F]:
    """
    Decorator to track chain start.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get client
            tracking_client = client or get_default_client()
            current_run_id = tracking_client.get_current_run_id()
            current_user_id = tracking_client.get_current_user_id()
            
            if not current_run_id or not current_user_id:
                logger.warning("No active run context, chain start not tracked")
                return func(*args, **kwargs)
            parent_id_from_context = current_parent_event_id_var.get()
            # Execute function and track timing
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time_ms = int((end_time - start_time) * 1000)
                
                # Log chain start event
                chain_start_event = Event(
                    event_id=generate_uuid(),
                    run_id=current_run_id,
                    user_id=current_user_id,
                    event_type=EventType.CHAIN_START,
                    parent_event_id=parent_id_from_context,
                    metadata=metadata,
                    payload={
                        'chain_name': chain_name,
                        'chain_type': chain_type,
                        'inputs': inputs
                    }
                )
                tracking_client.log_event(chain_start_event)
                current_parent_event_id_var.set(chain_start_event.event_id)
                return result
            except Exception as e:
                end_time = time.time()
                execution_time_ms = int((end_time - start_time) * 1000)
                raise
        return wrapper
    return decorator

def track_chain_end(
    chain_name: str,
    chain_type: str,
    outputs: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> Callable[[F], F]:
    """
    Decorator to track chain end.
    """
    def decorator(func: F) -> F:    
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get client
            tracking_client = client or get_default_client()
            current_run_id = tracking_client.get_current_run_id()
            current_user_id = tracking_client.get_current_user_id()
            
            if not current_run_id or not current_user_id:
                logger.warning("No active run context, chain end not tracked")
                return func(*args, **kwargs)
            parent_id_from_context = current_parent_event_id_var.get()
            # Execute function and track timing
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time_ms = int((end_time - start_time) * 1000)
                
                # Log chain end event
                chain_end_event = Event(
                    event_id=generate_uuid(),
                    run_id=current_run_id,
                    user_id=current_user_id,
                    event_type=EventType.CHAIN_END,
                    parent_event_id=parent_id_from_context,
                    metadata=metadata,
                    payload={
                        'chain_name': chain_name,
                        'chain_type': chain_type,
                        'outputs': outputs
                    }
                )
                tracking_client.log_event(chain_end_event)
                current_parent_event_id_var.set(chain_end_event.event_id)
                return result
            except Exception as e:
                end_time = time.time()  
