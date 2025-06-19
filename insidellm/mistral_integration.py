"""
InsideLLM Mistral AI Integration - Comprehensive integration with Mistral AI
"""

import logging
import time
import asyncio
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Generator
from contextlib import contextmanager, asynccontextmanager

try:
    from mistralai import Mistral
    from mistralai.models.chatcompletionresponse import ChatCompletionResponse
    from mistralai.models.embeddingresponse import EmbeddingResponse
    from mistralai.extra.exceptions import MistralClientException
    MISTRAL_AVAILABLE = True
except ImportError as e:
    MISTRAL_AVAILABLE = False
    Mistral = None
    logger = logging.getLogger(__name__)
    logger.warning(f"Mistral AI import error: {str(e)}")

from .models import Event, EventType
from .utils import generate_uuid, get_iso_timestamp
from .client import InsideLLMClient

logger = logging.getLogger(__name__)

class InsideLLMMistralIntegration:
    """
    Comprehensive Mistral AI integration for automatic InsideLLM event tracking.
    
    Integrates with Mistral AI's Python client to automatically log:
    - Chat completions (sync/async, streaming/non-streaming)
    - Embeddings generation
    - Function/tool calls
    - Errors and exceptions
    - Performance metrics
    """
    
    def __init__(
        self,
        insidellm_client: InsideLLMClient,
        mistral_client: Optional["Mistral"] = None,
        user_id: str = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        track_chat_completions: bool = True,
        track_embeddings: bool = True,
        track_function_calls: bool = True,
        track_streaming: bool = True,
        track_errors: bool = True,
        track_performance: bool = True,
        auto_detect_functions: bool = True,
    ):
        """
        Initialize Mistral AI integration.
        
        Args:
            insidellm_client: InsideLLM client instance
            mistral_client: Mistral client instance (optional, can be passed later)
            user_id: User identifier
            run_id: Run identifier (auto-generated if not provided)
            metadata: Additional metadata for events
            track_chat_completions: Whether to track chat completion requests/responses
            track_embeddings: Whether to track embedding requests/responses
            track_function_calls: Whether to track function/tool calls
            track_streaming: Whether to track streaming responses
            track_errors: Whether to track errors
            track_performance: Whether to track performance metrics
            auto_detect_functions: Whether to automatically detect function calls in messages
        """
        if not MISTRAL_AVAILABLE:
            raise ImportError("Mistral AI is not installed. Install it with: pip install mistralai")
        
        self.insidellm_client = insidellm_client
        self.mistral_client = mistral_client
        self.user_id = user_id or generate_uuid()
        self.run_id = run_id or generate_uuid()
        self.metadata = metadata or {}
        
        # Tracking configuration
        self.track_chat_completions = track_chat_completions
        self.track_embeddings = track_embeddings
        self.track_function_calls = track_function_calls
        self.track_streaming = track_streaming
        self.track_errors = track_errors
        self.track_performance = track_performance
        self.auto_detect_functions = auto_detect_functions
        
        # Internal state
        self._active_requests: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"InsideLLM Mistral integration initialized for run: {self.run_id}")

    def set_mistral_client(self, client: "Mistral"):
        """Set or update the Mistral client."""
        self.mistral_client = client

    @contextmanager
    def tracked_chat_completion(self, **kwargs):
        """Context manager for tracking synchronous chat completions."""
        if not self.track_chat_completions or not self.mistral_client:
            yield self.mistral_client.chat.complete(**kwargs) if self.mistral_client else None
            return

        request_id = generate_uuid()
        start_time = time.time()
        
        try:
            # Log request start
            self._log_chat_request_start(request_id, kwargs, start_time)
            
            # Make the actual request
            response = self.mistral_client.chat.complete(**kwargs)
            
            # Log successful response
            end_time = time.time()
            self._log_chat_response_success(request_id, response, start_time, end_time, kwargs)
            
            yield response
            
        except Exception as e:
            # Log error
            end_time = time.time()
            self._log_error(request_id, e, 'chat_completion', start_time, end_time, kwargs)
            raise

    @asynccontextmanager
    async def tracked_async_chat_completion(self, **kwargs):
        """Context manager for tracking asynchronous chat completions."""
        if not self.track_chat_completions or not self.mistral_client:
            if self.mistral_client:
                yield await self.mistral_client.chat.complete_async(**kwargs)
            else:
                yield None
            return

        request_id = generate_uuid()
        start_time = time.time()
        
        try:
            # Log request start
            self._log_chat_request_start(request_id, kwargs, start_time)
            
            # Make the actual request
            response = await self.mistral_client.chat.complete_async(**kwargs)
            
            # Log successful response
            end_time = time.time()
            self._log_chat_response_success(request_id, response, start_time, end_time, kwargs)
            
            yield response
            
        except Exception as e:
            # Log error
            end_time = time.time()
            self._log_error(request_id, e, 'async_chat_completion', start_time, end_time, kwargs)
            raise

    @contextmanager
    def tracked_streaming_chat(self, **kwargs):
        """Context manager for tracking streaming chat completions."""
        if not self.track_streaming or not self.mistral_client:
            yield self.mistral_client.chat.stream(**kwargs) if self.mistral_client else None
            return

        request_id = generate_uuid()
        start_time = time.time()
        
        try:
            # Log request start
            kwargs_copy = kwargs.copy()
            kwargs_copy['stream'] = True
            self._log_chat_request_start(request_id, kwargs_copy, start_time)
            
            # Make the streaming request
            stream = self.mistral_client.chat.stream(**kwargs)
            
            # Wrap the stream to log chunks
            wrapped_stream = self._wrap_streaming_response(request_id, stream, start_time, kwargs)
            
            yield wrapped_stream
            
        except Exception as e:
            # Log error
            end_time = time.time()
            self._log_error(request_id, e, 'streaming_chat', start_time, end_time, kwargs)
            raise

    @asynccontextmanager
    async def tracked_async_streaming_chat(self, **kwargs):
        """Context manager for tracking async streaming chat completions."""
        if not self.track_streaming or not self.mistral_client:
            if self.mistral_client:
                yield self.mistral_client.chat.stream_async(**kwargs)
            else:
                yield None
            return

        request_id = generate_uuid()
        start_time = time.time()
        
        try:
            # Log request start
            kwargs_copy = kwargs.copy()
            kwargs_copy['stream'] = True
            self._log_chat_request_start(request_id, kwargs_copy, start_time)
            
            # Make the streaming request
            stream = self.mistral_client.chat.stream_async(**kwargs)
            
            # Wrap the stream to log chunks
            wrapped_stream = self._wrap_async_streaming_response(request_id, stream, start_time, kwargs)
            
            yield wrapped_stream
            
        except Exception as e:
            # Log error
            end_time = time.time()
            self._log_error(request_id, e, 'async_streaming_chat', start_time, end_time, kwargs)
            raise

    @contextmanager
    def tracked_embeddings(self, **kwargs):
        """Context manager for tracking embeddings generation."""
        if not self.track_embeddings or not self.mistral_client:
            yield self.mistral_client.embeddings.create(**kwargs) if self.mistral_client else None
            return

        request_id = generate_uuid()
        start_time = time.time()
        
        try:
            # Log request start
            self._log_embeddings_request_start(request_id, kwargs, start_time)
            
            # Make the actual request
            response = self.mistral_client.embeddings.create(**kwargs)
            
            # Log successful response
            end_time = time.time()
            self._log_embeddings_response_success(request_id, response, start_time, end_time, kwargs)
            
            yield response
            
        except Exception as e:
            # Log error
            end_time = time.time()
            self._log_error(request_id, e, 'embeddings', start_time, end_time, kwargs)
            raise

    def _log_chat_request_start(self, request_id: str, kwargs: Dict[str, Any], start_time: float):
        """Log chat completion request start."""
        messages = kwargs.get('messages', [])
        model = kwargs.get('model', 'unknown')
        
        # Extract function calls if present
        function_info = None
        if self.auto_detect_functions:
            function_info = self._extract_function_info(kwargs)
        
        # Store request info for later use
        self._active_requests[request_id] = {
            'type': 'chat_completion',
            'start_time': start_time,
            'kwargs': kwargs,
            'function_info': function_info
        }
        
        event = Event(
            event_id=request_id,
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.LLM_REQUEST,
            metadata=self.metadata,
            payload={
                'model_name': model,
                'provider': 'mistral',
                'messages': [self._serialize_message(msg) for msg in messages],
                'parameters': {k: v for k, v in kwargs.items() if k != 'messages'},
                'system_message': next((msg['content'] for msg in messages if msg.get('role') == 'system'), None),
                'function_info': function_info
            }
        )
        
        self.insidellm_client.log_event(event)
        logger.debug(f"Chat request start logged: {request_id}")

    def _log_chat_response_success(self, request_id: str, response: ChatCompletionResponse, 
                                 start_time: float, end_time: float, original_kwargs: Dict[str, Any]):
        """Log successful chat completion response."""
        response_time_ms = int((end_time - start_time) * 1000)
        request_info = self._active_requests.pop(request_id, {})
        
        # Extract response content
        response_text = ""
        function_calls = []
        
        if response.choices:
            choice = response.choices[0]
            if choice.message.content:
                response_text = choice.message.content
            
            # Check for function calls
            if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                function_calls = [
                    {
                        'id': tool_call.id,
                        'function': tool_call.function.name,
                        'arguments': tool_call.function.arguments
                    }
                    for tool_call in choice.message.tool_calls
                ]

        # Log main response event
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.LLM_RESPONSE,
            parent_event_id=request_id,
            metadata=self.metadata,
            payload={
                'model_name': original_kwargs.get('model', 'unknown'),
                'provider': 'mistral',
                'response_text': response_text,
                'response_time_ms': response_time_ms,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                    'completion_tokens': response.usage.completion_tokens if response.usage else 0,
                    'total_tokens': response.usage.total_tokens if response.usage else 0
                },
                'finish_reason': response.choices[0].finish_reason if response.choices else None,
                'function_calls': function_calls,
                'response_metadata': {
                    'function_info': request_info.get('function_info')
                }
            }
        )
        
        self.insidellm_client.log_event(event)
        
        # Log individual function calls if present
        if self.track_function_calls and function_calls:
            for func_call in function_calls:
                self._log_function_call(request_id, func_call)
        
        # Log performance metrics
        if self.track_performance:
            self._log_performance_metrics(request_id, response_time_ms, response.usage if response.usage else None)
        
        logger.debug(f"Chat response success logged: {request_id}")

    def _log_embeddings_request_start(self, request_id: str, kwargs: Dict[str, Any], start_time: float):
        """Log embeddings request start."""
        input_data = kwargs.get('input', [])
        model = kwargs.get('model', 'unknown')
        
        # Store request info
        self._active_requests[request_id] = {
            'type': 'embeddings',
            'start_time': start_time,
            'kwargs': kwargs
        }
        
        event = Event(
            event_id=request_id,
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.API_REQUEST,
            metadata=self.metadata,
            payload={
                'api_name': 'mistral_embeddings',
                'model_name': model,
                'provider': 'mistral',
                'input_texts': input_data if isinstance(input_data, list) else [input_data],
                'parameters': {k: v for k, v in kwargs.items() if k != 'input'},
                'request_metadata': {
                    'input_count': len(input_data) if isinstance(input_data, list) else 1
                }
            }
        )
        
        self.insidellm_client.log_event(event)
        logger.debug(f"Embeddings request start logged: {request_id}")

    def _log_embeddings_response_success(self, request_id: str, response: EmbeddingResponse, 
                                       start_time: float, end_time: float, original_kwargs: Dict[str, Any]):
        """Log successful embeddings response."""
        response_time_ms = int((end_time - start_time) * 1000)
        request_info = self._active_requests.pop(request_id, {})
        
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.API_RESPONSE,
            parent_event_id=request_id,
            metadata=self.metadata,
            payload={
                'api_name': 'mistral_embeddings',
                'model_name': original_kwargs.get('model', 'unknown'),
                'provider': 'mistral',
                'response_time_ms': response_time_ms,
                'embeddings_count': len(response.data),
                'embedding_dimensions': len(response.data[0].embedding) if response.data else 0,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                    'total_tokens': response.usage.total_tokens if response.usage else 0
                },
                'response_metadata': {
                    'request_info': request_info
                }
            }
        )
        
        self.insidellm_client.log_event(event)
        
        # Log performance metrics
        if self.track_performance:
            self._log_performance_metrics(request_id, response_time_ms, response.usage if response.usage else None)
        
        logger.debug(f"Embeddings response success logged: {request_id}")

    def _wrap_streaming_response(self, request_id: str, stream: Generator, start_time: float, 
                               original_kwargs: Dict[str, Any]) -> Generator:
        """Wrap streaming response to log chunks."""
        chunk_index = 0
        accumulated_content = ""
        
        try:
            for chunk in stream:
                if chunk.data.choices and chunk.data.choices[0].delta.content:
                    content = chunk.data.choices[0].delta.content
                    accumulated_content += content
                    
                    # Log streaming chunk
                    if self.track_streaming:
                        self._log_streaming_chunk(request_id, content, chunk_index, False)
                    
                    chunk_index += 1
                
                yield chunk
            
            # Log final streaming event
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)
            
            if self.track_streaming:
                self._log_streaming_chunk(request_id, "", chunk_index, True)
            
            # Log complete response
            self._log_streaming_complete(request_id, accumulated_content, response_time_ms, original_kwargs)
            
        except Exception as e:
            end_time = time.time()
            self._log_error(request_id, e, 'streaming_response', start_time, end_time, original_kwargs)
            raise

    async def _wrap_async_streaming_response(self, request_id: str, stream: AsyncGenerator, 
                                           start_time: float, original_kwargs: Dict[str, Any]) -> AsyncGenerator:
        """Wrap async streaming response to log chunks."""
        chunk_index = 0
        accumulated_content = ""
        
        try:
            async for chunk in stream:
                if chunk.data.choices and chunk.data.choices[0].delta.content:
                    content = chunk.data.choices[0].delta.content
                    accumulated_content += content
                    
                    # Log streaming chunk
                    if self.track_streaming:
                        self._log_streaming_chunk(request_id, content, chunk_index, False)
                    
                    chunk_index += 1
                
                yield chunk
            
            # Log final streaming event
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)
            
            if self.track_streaming:
                self._log_streaming_chunk(request_id, "", chunk_index, True)
            
            # Log complete response
            self._log_streaming_complete(request_id, accumulated_content, response_time_ms, original_kwargs)
            
        except Exception as e:
            end_time = time.time()
            self._log_error(request_id, e, 'async_streaming_response', start_time, end_time, original_kwargs)
            raise

    def _log_streaming_chunk(self, request_id: str, content: str, chunk_index: int, is_final: bool):
        """Log individual streaming chunk."""
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.LLM_STREAMING_CHUNK,
            parent_event_id=request_id,
            metadata=self.metadata,
            payload={
                'provider': 'mistral',
                'chunk_text': content,
                'chunk_index': chunk_index,
                'is_final': is_final,
                'chunk_metadata': {
                    'timestamp': get_iso_timestamp()
                }
            }
        )
        
        self.insidellm_client.log_event(event)

    def _log_streaming_complete(self, request_id: str, full_content: str, response_time_ms: int, 
                              original_kwargs: Dict[str, Any]):
        """Log streaming completion."""
        request_info = self._active_requests.pop(request_id, {})
        
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.LLM_RESPONSE,
            parent_event_id=request_id,
            metadata=self.metadata,
            payload={
                'model_name': original_kwargs.get('model', 'unknown'),
                'provider': 'mistral',
                'response_text': full_content,
                'response_time_ms': response_time_ms,
                'streaming': True,
                'response_metadata': {
                    'request_info': request_info,
                    'chunk_count': request_info.get('chunk_count', 0)
                }
            }
        )
        
        self.insidellm_client.log_event(event)

    def _log_function_call(self, parent_event_id: str, func_call: Dict[str, Any]):
        """Log individual function call."""
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.TOOL_CALL,
            parent_event_id=parent_event_id,
            metadata=self.metadata,
            payload={
                'tool_name': func_call['function'],
                'tool_type': 'mistral_function',
                'call_id': func_call.get('id'),
                'parameters': func_call.get('arguments', {}),
                'tool_metadata': {
                    'timestamp': get_iso_timestamp()
                }
            }
        )
        
        self.insidellm_client.log_event(event)

    def _log_error(self, request_id: str, error: Exception, operation_type: str, 
                   start_time: float, end_time: float, context: Dict[str, Any]):
        """Log error event."""
        if not self.track_errors:
            return
        
        # Clean up active request
        self._active_requests.pop(request_id, None)
        
        error_type = type(error).__name__
        if isinstance(error, MistralClientException):
            error_code = f"API_{error.http_status}" if hasattr(error, 'http_status') else "API_ERROR"
        elif isinstance(error, MistralClientException):
            error_code = "CONNECTION_ERROR"
        else:
            error_code = error_type
        
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.ERROR,
            parent_event_id=request_id,
            metadata=self.metadata,
            payload={
                'error_type': f'mistral_{operation_type}_error',
                'error_message': str(error),
                'error_code': error_code,
                'operation_duration_ms': int((end_time - start_time) * 1000),
                'context': {
                    'operation_type': operation_type,
                    'model': context.get('model', 'unknown'),
                    'request_params': {k: v for k, v in context.items() if k not in ['messages', 'input']},
                    'timestamp': get_iso_timestamp()
                }
            }
        )
        
        self.insidellm_client.log_event(event)
        logger.debug(f"Error logged for request: {request_id}")

    def _log_performance_metrics(self, request_id: str, response_time_ms: int, usage: Optional[Any]):
        """Log performance metrics."""
        if not self.track_performance:
            return
        
        # Response time metric
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.PERFORMANCE_METRIC,
            parent_event_id=request_id,
            metadata=self.metadata,
            payload={
                'metric_name': 'mistral_response_time',
                'metric_value': response_time_ms,
                'metric_unit': 'ms',
                'metric_type': 'gauge',
                'provider': 'mistral',
                'metric_metadata': {
                    'timestamp': get_iso_timestamp()
                }
            }
        )
        
        self.insidellm_client.log_event(event)
        
        # Token usage metrics
        if usage:
            token_metrics = {
                'prompt_tokens': usage.prompt_tokens,
                'completion_tokens': getattr(usage, 'completion_tokens', 0),
                'total_tokens': usage.total_tokens
            }
            
            for metric_name, metric_value in token_metrics.items():
                if metric_value > 0:
                    event = Event(
                        event_id=generate_uuid(),
                        run_id=self.run_id,
                        user_id=self.user_id,
                        event_type=EventType.PERFORMANCE_METRIC,
                        parent_event_id=request_id,
                        metadata=self.metadata,
                        payload={
                            'metric_name': f'mistral_{metric_name}',
                            'metric_value': metric_value,
                            'metric_unit': 'tokens',
                            'metric_type': 'counter',
                            'provider': 'mistral',
                            'metric_metadata': {
                                'timestamp': get_iso_timestamp()
                            }
                        }
                    )
                    
                    self.insidellm_client.log_event(event)

    def _extract_function_info(self, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract function/tool information from request."""
        if 'tools' in kwargs and kwargs['tools']:
            return {
                'available_functions': [
                    tool.get('function', {}).get('name', 'unknown') 
                    for tool in kwargs['tools']
                ],
                'tool_choice': kwargs.get('tool_choice', 'auto')
            }
        return None

    def _serialize_message(self, message) -> Dict[str, Any]:
        """Serialize a chat message for logging."""
        if isinstance(message, dict):
            return message
        else:
            return {'content': str(message), 'role': 'unknown'}

    def log_custom_event(self, event_type: EventType, payload: Dict[str, Any], 
                        parent_event_id: Optional[str] = None):
        """Log a custom event."""
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=event_type,
            parent_event_id=parent_event_id,
            metadata=self.metadata,
            payload=payload
        )
        
        self.insidellm_client.log_event(event)
        logger.debug(f"Custom event logged: {event_type}")

    def create_session_start_event(self, session_context: Optional[Dict[str, Any]] = None):
        """Create a session start event."""
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.SESSION_START,
            metadata=self.metadata,
            payload={
                'session_id': self.run_id,
                'session_type': 'mistral_chat',
                'initial_context': session_context or {},
                'session_metadata': {
                    'start_time': get_iso_timestamp()
                }
            }
        )
        
        self.insidellm_client.log_event(event)
        logger.info(f"Session started: {self.run_id}")

    def create_session_end_event(self, session_summary: Optional[Dict[str, Any]] = None, 
                                exit_reason: str = "completed"):
        """Create a session end event."""
        event = Event(
            event_id=generate_uuid(),
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.SESSION_END,
            metadata=self.metadata,
            payload={
                'session_id': self.run_id,
                'session_summary': session_summary or {},
                'exit_reason': exit_reason,
                'session_metadata': {
                    'end_time': get_iso_timestamp()
                }
            }
        )
        
        self.insidellm_client.log_event(event)
        logger.info(f"Session ended: {self.run_id}")
