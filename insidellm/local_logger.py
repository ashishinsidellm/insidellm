"""
InsideLLM Local Logger - For local testing without API calls
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from .models import Event

logger = logging.getLogger(__name__)


class LocalEventLogger:
    """
    Local event logger that saves events to files instead of sending to API.
    Useful for testing and development without API connectivity.
    """
    
    def __init__(
        self,
        log_directory: str = "insidellm_logs",
        log_format: str = "json",  # json, csv, or pretty
        create_session_files: bool = True,
        max_file_size_mb: int = 10
    ):
        """
        Initialize local logger.
        
        Args:
            log_directory: Directory to save log files
            log_format: Format for log files (json, csv, pretty)
            create_session_files: Create separate files per session
            max_file_size_mb: Maximum file size before rotation
        """
        self.log_directory = Path(log_directory)
        self.log_format = log_format
        self.create_session_files = create_session_files
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        
        # Create log directory
        self.log_directory.mkdir(exist_ok=True)
        
        # Session tracking
        self.current_sessions: Dict[str, Any] = {}
        
        logger.info(f"Local event logger initialized: {self.log_directory}")
    
    def log_events(self, events: List[Event]) -> bool:
        """
        Log events to local files.
        
        Args:
            events: List of events to log
            
        Returns:
            True if successful, False otherwise
        """
        if not events:
            return True
        
        try:
            for event in events:
                self._log_single_event(event)
            
            logger.info(f"Logged {len(events)} events locally")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log events locally: {e}")
            return False
    
    def _log_single_event(self, event: Event) -> None:
        """Log a single event to appropriate file(s)"""
        
        # Determine output file(s)
        files_to_write = []
        
        # Main log file
        main_file = self._get_main_log_file()
        files_to_write.append(main_file)
        
        # Session-specific file
        if self.create_session_files and event.run_id:
            session_file = self._get_session_log_file(event.run_id)
            files_to_write.append(session_file)
        
        # Write to all target files
        for file_path in files_to_write:
            self._write_event_to_file(event, file_path)
    
    def _get_main_log_file(self) -> Path:
        """Get the main log file path"""
        timestamp = datetime.now().strftime("%Y-%m-%d")
        filename = f"events_{timestamp}.{self.log_format}"
        return self.log_directory / filename
    
    def _get_session_log_file(self, run_id: str) -> Path:
        """Get session-specific log file path"""
        safe_run_id = run_id.replace("-", "_")[:16]  # Sanitize for filename
        filename = f"session_{safe_run_id}.{self.log_format}"
        return self.log_directory / filename
    
    def _write_event_to_file(self, event: Event, file_path: Path) -> None:
        """Write event to specific file"""
        
        # Check file size and rotate if needed
        if file_path.exists() and file_path.stat().st_size > self.max_file_size_bytes:
            self._rotate_file(file_path)
        
        # Write based on format
        if self.log_format == "json":
            self._write_json_event(event, file_path)
        elif self.log_format == "csv":
            self._write_csv_event(event, file_path)
        elif self.log_format == "pretty":
            self._write_pretty_event(event, file_path)
        else:
            raise ValueError(f"Unsupported log format: {self.log_format}")
    
    def _write_json_event(self, event: Event, file_path: Path) -> None:
        """Write event in JSON format"""
        event_data = event.to_dict()
        event_data["_logged_at"] = datetime.now().isoformat()
        
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_data, ensure_ascii=False) + "\n")
    
    def _write_csv_event(self, event: Event, file_path: Path) -> None:
        """Write event in CSV format"""
        import csv
        
        # Prepare CSV row
        row = [
            event.event_id,
            event.run_id,
            event.timestamp,
            event.event_type,
            event.user_id,
            event.parent_event_id or "",
            json.dumps(event.metadata or {}),
            json.dumps(event.payload),
            datetime.now().isoformat()
        ]
        
        # Write header if file doesn't exist
        write_header = not file_path.exists()
        
        with open(file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            if write_header:
                writer.writerow([
                    "event_id", "run_id", "timestamp", "event_type", 
                    "user_id", "parent_event_id", "metadata", "payload", "logged_at"
                ])
            
            writer.writerow(row)
    
    def _write_pretty_event(self, event: Event, file_path: Path) -> None:
        """Write event in human-readable format"""
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Event: {event.event_type.upper()}\n")
            f.write(f"Time: {event.timestamp}\n")
            f.write(f"Event ID: {event.event_id}\n")
            f.write(f"Run ID: {event.run_id}\n")
            f.write(f"User ID: {event.user_id}\n")
            
            if event.parent_event_id:
                f.write(f"Parent: {event.parent_event_id}\n")
            
            if event.metadata:
                f.write(f"Metadata: {json.dumps(event.metadata, indent=2)}\n")
            
            f.write(f"Payload:\n{json.dumps(event.payload, indent=2)}\n")
            f.write(f"Logged at: {datetime.now().isoformat()}\n")
    
    def _rotate_file(self, file_path: Path) -> None:
        """Rotate log file when it gets too large"""
        timestamp = int(time.time())
        backup_path = file_path.with_suffix(f".{timestamp}{file_path.suffix}")
        file_path.rename(backup_path)
        logger.info(f"Rotated log file: {file_path} -> {backup_path}")
    
    def get_session_summary(self, run_id: str) -> Dict[str, Any]:
        """Get summary statistics for a session"""
        session_file = self._get_session_log_file(run_id)
        
        if not session_file.exists():
            return {"error": "Session file not found"}
        
        summary = {
            "run_id": run_id,
            "total_events": 0,
            "event_types": {},
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0,
            "file_path": str(session_file)
        }
        
        try:
            if self.log_format == "json":
                with open(session_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            event_data = json.loads(line)
                            summary["total_events"] += 1
                            
                            event_type = event_data.get("event_type", "unknown")
                            summary["event_types"][event_type] = summary["event_types"].get(event_type, 0) + 1
                            
                            timestamp = event_data.get("timestamp")
                            if timestamp:
                                if not summary["start_time"] or timestamp < summary["start_time"]:
                                    summary["start_time"] = timestamp
                                if not summary["end_time"] or timestamp > summary["end_time"]:
                                    summary["end_time"] = timestamp
            
            # Calculate duration
            if summary["start_time"] and summary["end_time"]:
                from datetime import datetime
                start = datetime.fromisoformat(summary["start_time"].replace('Z', '+00:00'))
                end = datetime.fromisoformat(summary["end_time"].replace('Z', '+00:00'))
                summary["duration_seconds"] = (end - start).total_seconds()
        
        except Exception as e:
            summary["error"] = str(e)
        
        return summary
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions"""
        sessions = []
        
        for file_path in self.log_directory.glob("session_*.json"):
            run_id = file_path.stem.replace("session_", "").replace("_", "-")
            summary = self.get_session_summary(run_id)
            sessions.append(summary)
        
        return sorted(sessions, key=lambda x: x.get("start_time", ""), reverse=True)
    
    def export_session(self, run_id: str, export_format: str = "json") -> Optional[str]:
        """Export session data in specified format"""
        session_file = self._get_session_log_file(run_id)
        
        if not session_file.exists():
            return None
        
        export_file = session_file.with_suffix(f".export.{export_format}")
        
        if export_format == "json" and self.log_format == "json":
            # Just copy the file
            import shutil
            shutil.copy2(session_file, export_file)
        else:
            # Convert format
            events = []
            with open(session_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))
            
            if export_format == "json":
                with open(export_file, "w", encoding="utf-8") as f:
                    json.dump(events, f, indent=2, ensure_ascii=False)
            elif export_format == "csv":
                import csv
                with open(export_file, "w", newline="", encoding="utf-8") as f:
                    if events:
                        writer = csv.DictWriter(f, fieldnames=events[0].keys())
                        writer.writeheader()
                        writer.writerows(events)
        
        return str(export_file)


class LocalTestingClient:
    """
    A testing client that uses local logging instead of API calls.
    Drop-in replacement for InsideLLMClient for local testing.
    """
    
    def __init__(self, **kwargs):
        """Initialize local testing client"""
        self.local_logger = LocalEventLogger(
            log_directory=kwargs.get("log_directory", "insidellm_logs"),
            log_format=kwargs.get("log_format", "json")
        )
        
        self._current_run_id: Optional[str] = None
        self._current_user_id: Optional[str] = None
        self._run_metadata: Dict[str, Any] = {}
        self._events_logged = 0
        
        logger.info("Local testing client initialized")
    
    def start_run(self, run_id: Optional[str] = None, user_id: Optional[str] = None, 
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a new run"""
        from .utils import generate_uuid
        
        self._current_run_id = run_id or generate_uuid()
        self._current_user_id = user_id
        self._run_metadata = metadata or {}
        
        logger.info(f"Started local testing run: {self._current_run_id}")
        return self._current_run_id
    
    def end_run(self, run_id: Optional[str] = None) -> None:
        """End a run"""
        target_run_id = run_id or self._current_run_id
        if target_run_id:
            summary = self.local_logger.get_session_summary(target_run_id)
            logger.info(f"Ended local testing run: {target_run_id}")
            logger.info(f"Session summary: {summary.get('total_events', 0)} events logged")
    
    def log_event(self, event: Event) -> None:
        """Log an event locally"""
        # Set run context if not provided
        if not event.run_id and self._current_run_id:
            event.run_id = self._current_run_id
        
        if not event.user_id and self._current_user_id:
            event.user_id = self._current_user_id
        
        # Merge run metadata
        if self._run_metadata:
            combined_metadata = self._run_metadata.copy()
            if event.metadata:
                combined_metadata.update(event.metadata)
            event.metadata = combined_metadata
        
        # Log locally
        self.local_logger.log_events([event])
        self._events_logged += 1
        
        logger.debug(f"Event logged locally: {event.event_type} - {event.event_id}")
    
    def send_events(self, events: List[Event]) -> bool:
        """Send events (locally log them)"""
        return self.local_logger.log_events(events)
    
    def flush(self, timeout: Optional[float] = None) -> None:
        """Flush (no-op for local testing)"""
        logger.info(f"Local flush completed - {self._events_logged} events logged")
    
    def get_current_run_id(self) -> Optional[str]:
        """Get current run ID"""
        return self._current_run_id
    
    def get_current_user_id(self) -> Optional[str]:
        """Get current user ID"""
        return self._current_user_id
    
    def get_queue_size(self) -> int:
        """Get queue size (always 0 for local)"""
        return 0
    
    def is_healthy(self) -> bool:
        """Check if healthy (always true for local)"""
        return True
    
    def shutdown(self, timeout: Optional[float] = None) -> None:
        """Shutdown the local client"""
        if self._current_run_id:
            summary = self.local_logger.get_session_summary(self._current_run_id)
            logger.info(f"Local testing complete: {summary}")
        
        logger.info("Local testing client shutdown")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get local testing statistics"""
        return {
            "events_logged": self._events_logged,
            "current_run": self._current_run_id,
            "log_directory": str(self.local_logger.log_directory),
            "sessions": len(self.local_logger.list_sessions())
        }