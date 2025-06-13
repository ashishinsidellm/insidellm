import logging
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

from .models import Event, EventType
from .utils import generate_uuid, get_iso_timestamp
from .client import InsideLLMClient

logger = logging.getLogger(__name__)

class MistralAIWrapper:
    """
    Wrapper for Mistral AI's API that integrates with InsideLLM event tracking.
    """
    
    def __init__(
        self,
        client: InsideLLMClient,
        mistral_api_key: str,
        user_id: str,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model: str = "mistral-small-latest",
    ):
        """
        Initialize Mistral AI wrapper.
        
        Args:
            client: InsideLLM client instance
            mistral_api_key: Mistral AI API key
            user_id: User identifier
            run_id: Run identifier (auto-generated if not provided)
            metadata: Additional metadata for events
            model: Default model to use
        """
        if not MISTRAL_AVAILABLE:
            raise ImportError("Mistral AI client is not installed. Install it with: pip install mistralai")
        
        self.client = client
        self.mistral_client = Mistral(api_key=mistral_api_key)
        self.user_id = user_id
        self.run_id = run_id or generate_uuid()
        self.metadata = metadata or {}
        self.model = model
        
        logger.info(f"Mistral AI wrapper initialized for run: {self.run_id}")
    
    def _normalize_messages(self, messages: List[Union[Dict[str, str], Any]]) -> List[Dict[str, str]]:
        """Normalize messages to dictionary format"""
        normalized = []
        for msg in messages:
            if isinstance(msg, dict):
                # Already in dict format
                if "role" in msg and "content" in msg:
                    normalized.append(msg)
                else:
                    logger.warning(f"Message missing required fields: {msg}")
                    normalized.append({
                        "role": msg.get("role", "user"),
                        "content": str(msg.get("content", ""))
                    })
            else:
                # Handle objects with attributes
                try:
                    normalized.append({
                        "role": getattr(msg, 'role', 'user'),
                        "content": getattr(msg, 'content', str(msg))
                    })
                except Exception:
                    # Last resort - convert to string
                    normalized.append({
                        "role": "user",
                        "content": str(msg)
                    })
        return normalized
    
    def chat(
        self,
        messages: List[Union[Dict[str, str], Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Send a chat completion request to Mistral AI.
        
        Args:
            messages: List of chat messages (dict format with 'role' and 'content')
            model: Model to use (defaults to instance default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to Mistral AI
            
        Returns:
            Chat completion response
        """
        model = model or self.model
        request_id = generate_uuid()
        
        # Normalize messages to dict format
        normalized_messages = self._normalize_messages(messages)
        
        # Log request event
        request_event = Event(
            event_id=request_id,
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.LLM_REQUEST,
            metadata=self.metadata,
            payload={
                'model_name': model,
                'provider': 'mistral',
                'messages': normalized_messages,
                'parameters': {
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'top_p': top_p,
                    'stream': stream,
                    **kwargs
                }
            }
        )
        self.client.log_event(request_event)
        
        try:
            # Make API call
            if stream:
                response = self.mistral_client.chat.stream(
                    model=model,
                    messages=normalized_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    **kwargs
                )
            else:
                response = self.mistral_client.chat.complete(
                    model=model,
                    messages=normalized_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    **kwargs
                )
            
            # Log response event
            response_payload = {}
            try:
                if hasattr(response, 'model_dump'):
                    response_payload = response.model_dump()
                elif hasattr(response, 'dict'):
                    response_payload = response.dict()
                elif hasattr(response, '__dict__'):
                    response_payload = response.__dict__
                else:
                    response_payload = {'response': str(response)}
            except Exception as e:
                logger.warning(f"Could not serialize response: {e}")
                response_payload = {'response': str(response)}
            
            response_event = Event(
                event_id=generate_uuid(),
                run_id=self.run_id,
                user_id=self.user_id,
                event_type=EventType.LLM_RESPONSE,
                parent_event_id=request_id,
                metadata=self.metadata,
                payload={
                    'model_name': model,
                    'provider': 'mistral',
                    'response': response_payload,
                }
            )
            self.client.log_event(response_event)
            
            return response
            
        except Exception as e:
            # Log error event
            error_event = Event(
                event_id=generate_uuid(),
                run_id=self.run_id,
                user_id=self.user_id,
                event_type=EventType.ERROR,
                parent_event_id=request_id,
                metadata=self.metadata,
                payload={
                    'error_type': 'mistral_error',
                    'error_message': str(e),
                    'error_code': type(e).__name__,
                    'model_name': model,
                    'provider': 'mistral'
                }
            )
            self.client.log_event(error_event)
            raise
    
    def chat_stream(
        self,
        messages: List[Union[Dict[str, str], Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        **kwargs: Any,
    ):
        """
        Convenience method for streaming chat completion.
        
        Args:
            messages: List of chat messages
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            **kwargs: Additional parameters
            
        Yields:
            Stream chunks
        """
        return self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True,
            **kwargs
        )