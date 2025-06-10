"""
InsideLLM Utilities - Utility functions for the SDK
"""

import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


def get_iso_timestamp() -> str:
    """Get current timestamp in ISO 8601 format (UTC)."""
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def validate_uuid(uuid_string: str) -> bool:
    """
    Validate UUID string format.
    
    Args:
        uuid_string: String to validate
        
    Returns:
        True if valid UUID, False otherwise
    """
    try:
        uuid.UUID(uuid_string)
        return True
    except (ValueError, TypeError):
        return False


def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
    """
    Sanitize metadata to ensure only primitive types.
    
    Args:
        metadata: Raw metadata dictionary
        
    Returns:
        Sanitized metadata with only primitive types
    """
    if not metadata:
        return {}
    
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif value is None:
            continue  # Skip None values
        else:
            # Convert complex types to string
            sanitized[key] = str(value)
    
    return sanitized


def merge_metadata(*metadata_dicts: Optional[Dict[str, Any]]) -> Dict[str, Union[str, int, float, bool]]:
    """
    Merge multiple metadata dictionaries with sanitization.
    
    Args:
        *metadata_dicts: Variable number of metadata dictionaries
        
    Returns:
        Merged and sanitized metadata
    """
    merged = {}
    for metadata in metadata_dicts:
        if metadata:
            merged.update(metadata)
    
    return sanitize_metadata(merged)


def truncate_string(text: str, max_length: int = 10000) -> str:
    """
    Truncate string to maximum length.
    
    Args:
        text: String to truncate
        max_length: Maximum allowed length
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - 3] + "..."


def extract_error_context(exception: Exception) -> Dict[str, Any]:
    """
    Extract context information from an exception.
    
    Args:
        exception: Exception to extract context from
        
    Returns:
        Dictionary with error context
    """
    import traceback
    
    return {
        'error_type': type(exception).__name__,
        'error_message': str(exception),
        'error_module': getattr(exception, '__module__', None),
        'stack_trace': traceback.format_exc()
    }


def safe_serialize(obj: Any) -> Any:
    """
    Safely serialize an object to JSON-compatible format.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-compatible representation
    """
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [safe_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}
    else:
        # Convert complex objects to string
        return str(obj)


def calculate_batch_delay(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """
    Calculate exponential backoff delay.
    
    Args:
        attempt: Attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        
    Returns:
        Delay in seconds
    """
    delay = base_delay * (2 ** attempt)
    return min(delay, max_delay)


def parse_iso_timestamp(timestamp_str: str) -> datetime:
    """
    Parse ISO 8601 timestamp string to datetime object.
    
    Args:
        timestamp_str: ISO 8601 timestamp string
        
    Returns:
        Datetime object
    """
    # Handle both Z suffix and +00:00 suffix
    if timestamp_str.endswith('Z'):
        timestamp_str = timestamp_str[:-1] + '+00:00'
    
    return datetime.fromisoformat(timestamp_str)


def format_duration(duration_ms: int) -> str:
    """
    Format duration in milliseconds to human-readable string.
    
    Args:
        duration_ms: Duration in milliseconds
        
    Returns:
        Formatted duration string
    """
    if duration_ms < 1000:
        return f"{duration_ms}ms"
    elif duration_ms < 60000:
        return f"{duration_ms / 1000:.1f}s"
    else:
        minutes = duration_ms // 60000
        seconds = (duration_ms % 60000) / 1000
        return f"{minutes}m {seconds:.1f}s"


def get_function_signature(func) -> str:
    """
    Get function signature as string.
    
    Args:
        func: Function object
        
    Returns:
        Function signature string
    """
    import inspect
    
    try:
        sig = inspect.signature(func)
        return f"{func.__name__}{sig}"
    except Exception:
        return func.__name__


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result

#Context variable for current parent event ID
current_parent_event_id_var: ContextVar[Optional[str]] = ContextVar(
    "current_parent_event_id",
    default=None
)