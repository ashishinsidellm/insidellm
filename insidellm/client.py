"""
InsideLLM Client - Main client class for interacting with the InsideLLM API
"""

import json
import logging
import os
import time
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime

import requests
from threading import Lock, Event as ThreadEvent

from .config import InsideLLMConfig
from .models import Event, EventType
from .queue_manager import QueueManager
from .exceptions import InsideLLMError, NetworkError, ConfigurationError
from .utils import generate_uuid, get_iso_timestamp

logger = logging.getLogger(__name__)


class InsideLLMClient:
    """
    Main client class for InsideLLM SDK.
    
    Provides event logging capabilities with asynchronous processing,
    batching, and retry mechanisms.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.insideLLM.com",
        config: Optional[InsideLLMConfig] = None,
        **kwargs
    ):
        """
        Initialize InsideLLM client.
        
        Args:
            api_key: API key for authentication (defaults to INSIDELLM_API_KEY env var)
            base_url: Base URL for the API
            config: Custom configuration object
            **kwargs: Additional configuration parameters
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("INSIDELLM_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "API key is required. Provide it as parameter or set INSIDELLM_API_KEY environment variable."
            )
        
        self.base_url = base_url.rstrip('/')
        self.config = config or InsideLLMConfig(**kwargs)
        
        # Initialize HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': f'InsideLLM-Python-SDK/1.0.0'
        })
        
        # Initialize queue manager for async processing
        self.queue_manager = QueueManager(
            client=self,
            config=self.config
        )
        
        # Thread safety
        self._lock = Lock()
        self._shutdown_event = ThreadEvent()
        
        # Current run tracking
        self._current_run_id: Optional[str] = None
        self._current_user_id: Optional[str] = None
        self._run_metadata: Dict[str, Any] = {}
        
        logger.info(f"InsideLLM client initialized with base_url: {self.base_url}")
    
    def start_run(
        self,
        run_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new run/session.
        
        Args:
            run_id: Optional run ID (auto-generated if not provided)
            user_id: User identifier
            metadata: Additional metadata for the run
            
        Returns:
            The run ID
        """
        with self._lock:
            self._current_run_id = run_id or generate_uuid()
            self._current_user_id = user_id
            self._run_metadata = metadata or {}
        
        # Log session start event
        session_start_event = Event(
            event_id=generate_uuid(),
            run_id=self._current_run_id,
            timestamp=get_iso_timestamp(),
            event_type=EventType.SESSION_START,
            user_id=self._current_user_id,
            metadata=self._run_metadata,
            payload={"session_id": self._current_run_id}
        )
        
        self.log_event(session_start_event)
        
        logger.info(f"Started run: {self._current_run_id}")
        return self._current_run_id
    
    def end_run(self, run_id: Optional[str] = None) -> None:
        """
        End a run/session.
        
        Args:
            run_id: Run ID to end (defaults to current run)
        """
        target_run_id = run_id or self._current_run_id
        if not target_run_id:
            logger.warning("No active run to end")
            return
        
        # Log session end event
        session_end_event = Event(
            event_id=generate_uuid(),
            run_id=target_run_id,
            timestamp=get_iso_timestamp(),
            event_type=EventType.SESSION_END,
            user_id=self._current_user_id,
            metadata=self._run_metadata,
            payload={"session_id": target_run_id}
        )
        
        self.log_event(session_end_event)
        
        if target_run_id == self._current_run_id:
            with self._lock:
                self._current_run_id = None
                self._current_user_id = None
                self._run_metadata = {}
        
        logger.info(f"Ended run: {target_run_id}")
    
    def log_event(self, event: Event) -> None:
        """
        Log an event to the queue for async processing.
        
        Args:
            event: Event object to log
        """
        if self._shutdown_event.is_set():
            logger.warning("Client is shutting down, event not logged")
            return
        
        # Set run context if not provided
        if not event.run_id and self._current_run_id:
            event.run_id = self._current_run_id
        
        if not event.user_id and self._current_user_id:
            event.user_id = self._current_user_id
        
        # Merge run metadata with event metadata
        if self._run_metadata:
            combined_metadata = self._run_metadata.copy()
            if event.metadata:
                combined_metadata.update(event.metadata)
            event.metadata = combined_metadata
        
        # Validate event
        try:
            event.validate()
        except Exception as e:
            logger.error(f"Event validation failed: {e}")
            if self.config.strict_validation:
                raise
            return
        
        # Add to queue
        self.queue_manager.add_event(event)
        logger.debug(f"Event queued: {event.event_type} - {event.event_id}")
    
    def send_events(self, events: List[Event]) -> bool:
        """
        Send a batch of events to the API.
        
        Args:
            events: List of events to send
            
        Returns:
            True if successful, False otherwise
        """
        if not events:
            return True
        
        url = f"{self.base_url}/v1/runs/ingest"
        
        # Convert events to API format
        payload = []
        for event in events:
            try:
                event_data = event.to_dict()
                payload.append(event_data)
            except Exception as e:
                logger.error(f"Failed to serialize event {event.event_id}: {e}")
                continue
        
        if not payload:
            logger.warning("No valid events to send")
            return False
        
        # Send with retry logic
        max_retries = self.config.max_retries
        backoff_factor = self.config.backoff_factor
        
        for attempt in range(max_retries + 1):
            try:
                # If single event, send as object; if multiple, send as array
                if len(payload) == 1:
                    response = self.session.post(url, json=payload[0], timeout=self.config.request_timeout)
                else:
                    response = self.session.post(url, json=payload, timeout=self.config.request_timeout)
                
                response.raise_for_status()
                
                logger.info(f"Successfully sent {len(events)} events")
                return True
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}")
                
                if attempt < max_retries:
                    # Exponential backoff
                    delay = backoff_factor * (2 ** attempt)
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to send events after {max_retries + 1} attempts: {e}")
                    if self.config.raise_on_error:
                        raise NetworkError(f"Failed to send events: {e}")
                    return False
        
        return False
    
    def flush(self, timeout: Optional[float] = None) -> None:
        """
        Flush all pending events immediately.
        
        Args:
            timeout: Maximum time to wait for flush completion
        """
        logger.info("Flushing pending events")
        self.queue_manager.flush(timeout=timeout)
    
    def get_current_run_id(self) -> Optional[str]:
        """Get the current run ID."""
        return self._current_run_id
    
    def get_current_user_id(self) -> Optional[str]:
        """Get the current user ID."""
        return self._current_user_id
    
    def get_queue_size(self) -> int:
        """Get the current queue size."""
        return self.queue_manager.get_queue_size()
    
    def is_healthy(self) -> bool:
        """Check if the client is healthy and can send events."""
        try:
            url = f"{self.base_url}/health"
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def shutdown(self, timeout: Optional[float] = None) -> None:
        """
        Shutdown the client and flush all pending events.
        
        Args:
            timeout: Maximum time to wait for shutdown completion
        """
        logger.info("Shutting down InsideLLM client")
        self._shutdown_event.set()
        
        # Flush remaining events
        self.flush(timeout=timeout)
        
        # Shutdown queue manager
        self.queue_manager.shutdown()
        
        # Close HTTP session
        self.session.close()
        
        logger.info("InsideLLM client shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown()
