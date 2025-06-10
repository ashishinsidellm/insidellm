"""
InsideLLM Python SDK - LLM/Agent Analytics Platform Client

A comprehensive Python SDK for ingesting LLM and Agent events with async processing,
LangChain integration, and custom agent support.
"""

from .client import InsideLLMClient
from .models import (
    Event,
    EventType,
    UserInputPayload,
    UserFeedbackPayload,
    AgentReasoningPayload,
    AgentPlanningPayload,
    AgentResponsePayload,
    LLMRequestPayload,
    LLMResponsePayload,
    LLMStreamingChunkPayload,
    ToolCallPayload,
    ToolResponsePayload,
    FunctionExecutionPayload,
    APIRequestPayload,
    APIResponsePayload,
    ErrorPayload,
    ValidationErrorPayload,
    TimeoutErrorPayload,
    SessionStartPayload,
    SessionEndPayload,
    PerformanceMetricPayload
)
#from .langchain_integration import InsideLLMCallback
from .crewai_integration import CrewAIInsideLLMCallbackHandler
from .vertex_ai_agent_builder_integration import InsideLLMVertexAgentBuilderConversationalSearchClient
from .decorators import track_llm_call, track_tool_use, track_agent_step
from .context_manager import InsideLLMTracker
from .config import InsideLLMConfig
from .exceptions import (
    InsideLLMError,
    ConfigurationError,
    ValidationError,
    NetworkError,
    QueueError
)

__version__ = "1.0.0"
__author__ = "InsideLLM Team"
__description__ = "Python SDK for LLM/Agent Analytics Platform"

# Default client instance
_default_client = None

def initialize(api_key: str = None, local_testing: bool = False, **kwargs):
    """Initialize the default InsideLLM client."""
    global _default_client
    
    if local_testing:
        from .local_logger import LocalTestingClient
        _default_client = LocalTestingClient(**kwargs)
    else:
        _default_client = InsideLLMClient(api_key=api_key, **kwargs)
    
    return _default_client

def get_client():
    """Get the default client instance."""
    if _default_client is None:
        raise ConfigurationError("InsideLLM client not initialized. Call initialize() first.")
    return _default_client

# Convenience functions using default client
def log_event(event: Event):
    """Log an event using the default client."""
    return get_client().log_event(event)

def start_run(run_id: str = None, user_id: str = None, metadata: dict = None):
    """Start a new run using the default client."""
    return get_client().start_run(run_id=run_id, user_id=user_id, metadata=metadata)

def end_run(run_id: str):
    """End a run using the default client."""
    return get_client().end_run(run_id)

def flush():
    """Flush all pending events using the default client."""
    return get_client().flush()

def shutdown():
    """Shutdown the default client."""
    if _default_client:
        _default_client.shutdown()

__all__ = [
    'InsideLLMClient',
    'Event',
    'EventType',
    'InsideLLMCallback', # This was for Langchain, commented out above but kept in __all__
    'CrewAIInsideLLMCallbackHandler',
    'InsideLLMVertexAgentBuilderConversationalSearchClient',
    'InsideLLMTracker',
    'InsideLLMConfig',
    'track_llm_call',
    'track_tool_use', 
    'track_agent_step',
    'initialize',
    'get_client',
    'log_event',
    'start_run',
    'end_run',
    'flush',
    'shutdown',
    # Payload classes
    'UserInputPayload',
    'UserFeedbackPayload',
    'AgentReasoningPayload',
    'AgentPlanningPayload',
    'AgentResponsePayload',
    'LLMRequestPayload',
    'LLMResponsePayload',
    'LLMStreamingChunkPayload',
    'ToolCallPayload',
    'ToolResponsePayload',
    'FunctionExecutionPayload',
    'APIRequestPayload',
    'APIResponsePayload',
    'ErrorPayload',
    'ValidationErrorPayload',
    'TimeoutErrorPayload',
    'SessionStartPayload',
    'SessionEndPayload',
    'PerformanceMetricPayload',
    # Exceptions
    'InsideLLMError',
    'ConfigurationError',
    'ValidationError',
    'NetworkError',
    'QueueError'
]
