"""
InsideLLM Configuration - Configuration management for the SDK
"""

import os
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class InsideLLMConfig:
    """
    Configuration class for InsideLLM SDK.
    
    Contains all configurable parameters for the SDK including
    queue management, network settings, and retry behavior.
    """
    
    # Queue Management
    max_queue_size: int = field(default=10000)
    batch_size: int = field(default=50)
    auto_flush_interval: float = field(default=30.0)  # seconds
    
    # Network Settings
    request_timeout: float = field(default=30.0)  # seconds
    max_retries: int = field(default=3)
    backoff_factor: float = field(default=2.0)
    
    # Error Handling
    raise_on_error: bool = field(default=False)
    strict_validation: bool = field(default=True)
    
    # Logging
    log_level: str = field(default="INFO")
    enable_debug_logging: bool = field(default=False)
    
    @classmethod
    def from_env(cls) -> 'InsideLLMConfig':
        """
        Create configuration from environment variables.
        
        Environment variables:
            INSIDELLM_MAX_QUEUE_SIZE: Maximum queue size
            INSIDELLM_BATCH_SIZE: Batch size for event processing
            INSIDELLM_AUTO_FLUSH_INTERVAL: Auto flush interval in seconds
            INSIDELLM_REQUEST_TIMEOUT: Request timeout in seconds
            INSIDELLM_MAX_RETRIES: Maximum number of retries
            INSIDELLM_BACKOFF_FACTOR: Backoff factor for retries
            INSIDELLM_RAISE_ON_ERROR: Whether to raise on errors (true/false)
            INSIDELLM_STRICT_VALIDATION: Whether to use strict validation (true/false)
            INSIDELLM_LOG_LEVEL: Log level (DEBUG, INFO, WARNING, ERROR)
            INSIDELLM_ENABLE_DEBUG_LOGGING: Enable debug logging (true/false)
        """
        def get_bool(key: str, default: bool) -> bool:
            value = os.getenv(key, str(default)).lower()
            return value in ('true', '1', 'yes', 'on')
        
        def get_int(key: str, default: int) -> int:
            try:
                return int(os.getenv(key, str(default)))
            except ValueError:
                return default
        
        def get_float(key: str, default: float) -> float:
            try:
                return float(os.getenv(key, str(default)))
            except ValueError:
                return default
        
        return cls(
            max_queue_size=get_int('INSIDELLM_MAX_QUEUE_SIZE', 10000),
            batch_size=get_int('INSIDELLM_BATCH_SIZE', 50),
            auto_flush_interval=get_float('INSIDELLM_AUTO_FLUSH_INTERVAL', 30.0),
            request_timeout=get_float('INSIDELLM_REQUEST_TIMEOUT', 30.0),
            max_retries=get_int('INSIDELLM_MAX_RETRIES', 3),
            backoff_factor=get_float('INSIDELLM_BACKOFF_FACTOR', 2.0),
            raise_on_error=get_bool('INSIDELLM_RAISE_ON_ERROR', False),
            strict_validation=get_bool('INSIDELLM_STRICT_VALIDATION', True),
            log_level=os.getenv('INSIDELLM_LOG_LEVEL', 'INFO'),
            enable_debug_logging=get_bool('INSIDELLM_ENABLE_DEBUG_LOGGING', False)
        )
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.batch_size > self.max_queue_size:
            raise ValueError("batch_size cannot exceed max_queue_size")
        
        if self.auto_flush_interval < 0:
            raise ValueError("auto_flush_interval must be non-negative")
        
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        
        if self.backoff_factor <= 0:
            raise ValueError("backoff_factor must be positive")
        
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"log_level must be one of: {valid_log_levels}")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'max_queue_size': self.max_queue_size,
            'batch_size': self.batch_size,
            'auto_flush_interval': self.auto_flush_interval,
            'request_timeout': self.request_timeout,
            'max_retries': self.max_retries,
            'backoff_factor': self.backoff_factor,
            'raise_on_error': self.raise_on_error,
            'strict_validation': self.strict_validation,
            'log_level': self.log_level,
            'enable_debug_logging': self.enable_debug_logging
        }
    
    def __post_init__(self):
        """Post-initialization validation."""
        self.validate()
