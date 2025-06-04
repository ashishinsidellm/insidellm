"""
InsideLLM Exceptions - Custom exception classes for the SDK
"""


class InsideLLMError(Exception):
    """Base exception class for InsideLLM SDK."""
    pass


class ConfigurationError(InsideLLMError):
    """Raised when there are configuration issues."""
    pass


class ValidationError(InsideLLMError):
    """Raised when event validation fails."""
    pass


class NetworkError(InsideLLMError):
    """Raised when network operations fail."""
    pass


class QueueError(InsideLLMError):
    """Raised when queue operations fail."""
    pass


class AuthenticationError(InsideLLMError):
    """Raised when authentication fails."""
    pass


class RateLimitError(InsideLLMError):
    """Raised when rate limits are exceeded."""
    pass


class PayloadError(InsideLLMError):
    """Raised when payload structure is invalid."""
    pass


class TimeoutError(InsideLLMError):
    """Raised when operations timeout."""
    pass
