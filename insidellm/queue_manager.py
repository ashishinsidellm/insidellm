"""
InsideLLM Queue Manager - Handles asynchronous event processing and batching
"""

import logging
import queue
import threading
import time
from typing import List, Optional, TYPE_CHECKING

from .models import Event
from .config import InsideLLMConfig
from .exceptions import QueueError

if TYPE_CHECKING:
    from .client import InsideLLMClient

logger = logging.getLogger(__name__)


class QueueManager:
    """
    Manages asynchronous event queuing, batching, and flushing.
    
    Handles thread-safe event collection and periodic batch processing
    with configurable flush intervals and batch sizes.
    """
    
    def __init__(self, client: 'InsideLLMClient', config: InsideLLMConfig):
        """
        Initialize queue manager.
        
        Args:
            client: InsideLLM client instance
            config: Configuration object
        """
        self.client = client
        self.config = config
        
        # Thread-safe queue for events
        self.event_queue = queue.Queue(maxsize=config.max_queue_size)
        
        # Threading controls
        self._worker_thread: Optional[threading.Thread] = None
        self._flush_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._flush_event = threading.Event()
        self._lock = threading.Lock()
        
        # Statistics
        self._events_queued = 0
        self._events_sent = 0
        self._events_failed = 0
        
        # Start background threads
        self._start_threads()
    
    def _start_threads(self):
        """Start background processing threads."""
        # Worker thread for processing queue
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="InsideLLM-QueueWorker",
            daemon=True
        )
        self._worker_thread.start()
        
        # Flush thread for periodic flushing
        if self.config.auto_flush_interval > 0:
            self._flush_thread = threading.Thread(
                target=self._flush_loop,
                name="InsideLLM-FlushScheduler",
                daemon=True
            )
            self._flush_thread.start()
        
        logger.info("Queue manager threads started")
    
    def add_event(self, event: Event) -> None:
        """
        Add an event to the processing queue.
        
        Args:
            event: Event to add to queue
        """
        if self._shutdown_event.is_set():
            logger.warning("Queue manager is shutting down, event not added")
            return
        
        try:
            # Try to add event to queue without blocking
            self.event_queue.put_nowait(event)
            
            with self._lock:
                self._events_queued += 1
            
            logger.debug(f"Event added to queue: {event.event_id}")
            
            # Check if we should flush immediately
            if self.event_queue.qsize() >= self.config.batch_size:
                self._flush_event.set()
                
        except queue.Full:
            logger.error("Event queue is full, dropping event")
            with self._lock:
                self._events_failed += 1
            
            if self.config.raise_on_error:
                raise QueueError("Event queue is full")
    
    def _worker_loop(self):
        """Main worker loop for processing events."""
        logger.info("Queue worker thread started")
        
        batch = []
        last_flush_time = time.time()
        
        while not self._shutdown_event.is_set():
            try:
                # Check for flush signal or timeout
                if (self._flush_event.is_set() or 
                    (batch and time.time() - last_flush_time >= self.config.auto_flush_interval)):
                    
                    # Collect any remaining events for this batch
                    while len(batch) < self.config.batch_size:
                        try:
                            event = self.event_queue.get_nowait()
                            batch.append(event)
                        except queue.Empty:
                            break
                    
                    # Send batch if we have events
                    if batch:
                        self._send_batch(batch)
                        batch = []
                        last_flush_time = time.time()
                    
                    self._flush_event.clear()
                
                # Get next event from queue
                try:
                    event = self.event_queue.get(timeout=1.0)
                    batch.append(event)
                    
                    # Send batch when it reaches configured size
                    if len(batch) >= self.config.batch_size:
                        self._send_batch(batch)
                        batch = []
                        last_flush_time = time.time()
                        
                except queue.Empty:
                    # Timeout - continue loop to check shutdown
                    continue
                    
            except Exception as e:
                logger.error(f"Error in queue worker loop: {e}")
                time.sleep(1)  # Brief pause before retrying
        
        # Process remaining events on shutdown
        if batch:
            logger.info(f"Processing {len(batch)} remaining events on shutdown")
            self._send_batch(batch)
        
        # Drain remaining queue
        remaining_events = []
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                remaining_events.append(event)
            except queue.Empty:
                break
        
        if remaining_events:
            logger.info(f"Processing {len(remaining_events)} final events")
            self._send_batch(remaining_events)
        
        logger.info("Queue worker thread stopped")
    
    def _flush_loop(self):
        """Periodic flush loop."""
        logger.info("Flush scheduler thread started")
        
        while not self._shutdown_event.is_set():
            # Wait for flush interval or shutdown
            if self._shutdown_event.wait(timeout=self.config.auto_flush_interval):
                break  # Shutdown requested
            
            # Trigger flush if we have events
            if not self.event_queue.empty():
                self._flush_event.set()
        
        logger.info("Flush scheduler thread stopped")
    
    def _send_batch(self, events: List[Event]) -> None:
        """
        Send a batch of events.
        
        Args:
            events: List of events to send
        """
        if not events:
            return
        
        logger.debug(f"Sending batch of {len(events)} events")
        
        try:
            success = self.client.send_events(events)
            
            with self._lock:
                if success:
                    self._events_sent += len(events)
                    logger.info(f"Successfully sent batch of {len(events)} events")
                else:
                    self._events_failed += len(events)
                    logger.error(f"Failed to send batch of {len(events)} events")
                    
        except Exception as e:
            logger.error(f"Exception sending batch: {e}")
            with self._lock:
                self._events_failed += len(events)
    
    def flush(self, timeout: Optional[float] = None) -> None:
        """
        Flush all pending events immediately.
        
        Args:
            timeout: Maximum time to wait for flush completion
        """
        logger.info("Manual flush requested")
        
        # Signal flush
        self._flush_event.set()
        
        # Wait for queue to empty
        start_time = time.time()
        while not self.event_queue.empty():
            if timeout and (time.time() - start_time) > timeout:
                logger.warning("Flush timeout reached")
                break
            time.sleep(0.1)
        
        logger.info("Manual flush completed")
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.event_queue.qsize()
    
    def get_statistics(self) -> dict:
        """Get queue statistics."""
        with self._lock:
            return {
                'events_queued': self._events_queued,
                'events_sent': self._events_sent,
                'events_failed': self._events_failed,
                'queue_size': self.get_queue_size(),
                'success_rate': (
                    self._events_sent / max(1, self._events_sent + self._events_failed)
                ) * 100
            }
    
    def shutdown(self, timeout: Optional[float] = None) -> None:
        """
        Shutdown queue manager and wait for threads to complete.
        
        Args:
            timeout: Maximum time to wait for thread completion
        """
        logger.info("Shutting down queue manager")
        
        # Signal shutdown
        self._shutdown_event.set()
        self._flush_event.set()
        
        # Wait for threads to complete
        threads = [self._worker_thread, self._flush_thread]
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logger.warning(f"Thread {thread.name} did not shutdown cleanly")
        
        logger.info("Queue manager shutdown complete")
