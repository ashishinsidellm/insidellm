"""
InsideLLM Models - Event models and payload definitions
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator

from .utils import generate_uuid, get_iso_timestamp


class EventType(str, Enum):
    """Enumeration of all supported event types."""
    
    # User Interaction
    USER_INPUT = "user_input"
    USER_FEEDBACK = "user_feedback"
    
    # Agent Processing
    AGENT_REASONING = "agent_reasoning"
    AGENT_PLANNING = "agent_planning"
    AGENT_RESPONSE = "agent_response"
    
    # LLM Operations
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    LLM_STREAMING_CHUNK = "llm_streaming_chunk"
    
    # Tool/Function Calls
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    FUNCTION_EXECUTION = "function_execution"
    
    # External APIs
    API_REQUEST = "api_request"
    API_RESPONSE = "api_response"
    
    # Error Handling
    ERROR = "error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    
    # System Events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    PERFORMANCE_METRIC = "performance_metric"


# Payload Models for each event type

class BasePayload(BaseModel):
    """Base class for all event payloads."""
    pass


class UserInputPayload(BasePayload):
    """Payload for user input events."""
    input_text: str
    input_type: str = "text"  # text, voice, image, etc.
    channel: Optional[str] = None  # web, mobile, api, etc.
    session_context: Optional[Dict[str, Any]] = None


class UserFeedbackPayload(BasePayload):
    """Payload for user feedback events."""
    feedback_type: str  # rating, thumbs_up_down, text, etc.
    feedback_value: Union[str, int, float, bool]
    target_event_id: Optional[str] = None  # Event being rated
    feedback_text: Optional[str] = None


class AgentReasoningPayload(BasePayload):
    """Payload for agent reasoning events."""
    reasoning_type: str  # chain_of_thought, tree_of_thought, etc.
    reasoning_steps: List[str]
    confidence_score: Optional[float] = None
    reasoning_time_ms: Optional[int] = None


class AgentPlanningPayload(BasePayload):
    """Payload for agent planning events."""
    plan_type: str  # sequential, parallel, hierarchical, etc.
    planned_actions: List[Dict[str, Any]]
    planning_time_ms: Optional[int] = None
    plan_confidence: Optional[float] = None


class AgentResponsePayload(BasePayload):
    """Payload for agent response events."""
    response_text: str
    response_type: str = "text"  # text, action, function_call, etc.
    response_confidence: Optional[float] = None
    response_metadata: Optional[Dict[str, Any]] = None


class LLMRequestPayload(BasePayload):
    """Payload for LLM request events."""
    model_name: str
    provider: str  # openai, anthropic, huggingface, etc.
    prompt: str
    parameters: Optional[Dict[str, Any]] = None  # temperature, max_tokens, etc.
    system_message: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None  # For chat models


class LLMResponsePayload(BasePayload):
    """Payload for LLM response events."""
    model_name: str
    provider: str
    response_text: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None  # token counts, etc.
    response_time_ms: Optional[int] = None
    cost: Optional[float] = None


class LLMStreamingChunkPayload(BasePayload):
    """Payload for LLM streaming chunk events."""
    model_name: str
    provider: str
    chunk_text: str
    chunk_index: int
    is_final: bool = False
    chunk_metadata: Optional[Dict[str, Any]] = None


class ToolCallPayload(BasePayload):
    """Payload for tool call events."""
    tool_name: str
    tool_type: str  # function, api, database, file_system, etc.
    parameters: Dict[str, Any]
    call_id: Optional[str] = None


class ToolResponsePayload(BasePayload):
    """Payload for tool response events."""
    tool_name: str
    tool_type: str
    call_id: Optional[str] = None
    response_data: Any
    execution_time_ms: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None


class FunctionExecutionPayload(BasePayload):
    """Payload for function execution events."""
    function_name: str
    function_args: Dict[str, Any]
    return_value: Any
    execution_time_ms: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None


class APIRequestPayload(BasePayload):
    """Payload for external API request events."""
    api_name: str
    endpoint: str
    method: str  # GET, POST, PUT, DELETE, etc.
    headers: Optional[Dict[str, str]] = None
    parameters: Optional[Dict[str, Any]] = None
    request_body: Optional[Any] = None


class APIResponsePayload(BasePayload):
    """Payload for external API response events."""
    api_name: str
    endpoint: str
    status_code: int
    response_headers: Optional[Dict[str, str]] = None
    response_body: Optional[Any] = None
    response_time_ms: Optional[int] = None


class ErrorPayload(BasePayload):
    """Payload for error events."""
    error_type: str
    error_message: str
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    severity: str = "error"  # info, warning, error, critical


class ValidationErrorPayload(BasePayload):
    """Payload for validation error events."""
    validation_target: str  # input, output, parameter, etc.
    validation_rules: List[str]
    failed_rules: List[str]
    validation_details: Optional[Dict[str, Any]] = None


class TimeoutErrorPayload(BasePayload):
    """Payload for timeout error events."""
    timeout_type: str  # llm_request, api_call, function_execution, etc.
    timeout_duration_ms: int
    expected_duration_ms: Optional[int] = None
    operation_context: Optional[Dict[str, Any]] = None


class SessionStartPayload(BasePayload):
    """Payload for session start events."""
    session_id: str
    session_type: Optional[str] = None  # chat, task, workflow, etc.
    initial_context: Optional[Dict[str, Any]] = None


class SessionEndPayload(BasePayload):
    """Payload for session end events."""
    session_id: str
    session_duration_ms: Optional[int] = None
    session_summary: Optional[Dict[str, Any]] = None
    exit_reason: Optional[str] = None  # completed, timeout, error, user_exit


class PerformanceMetricPayload(BasePayload):
    """Payload for performance metric events."""
    metric_name: str
    metric_value: Union[int, float]
    metric_unit: Optional[str] = None  # ms, tokens, bytes, etc.
    metric_type: str = "gauge"  # gauge, counter, histogram, etc.
    additional_metrics: Optional[Dict[str, Union[int, float]]] = None


# Main Event Model

class Event(BaseModel):
    """
    Main event model representing a single event in the InsideLLM system.
    """
    
    event_id: str = Field(default_factory=generate_uuid)
    run_id: str
    timestamp: str = Field(default_factory=get_iso_timestamp)
    event_type: EventType
    user_id: str
    parent_event_id: Optional[str] = None
    metadata: Optional[Dict[str, Union[str, int, float, bool]]] = None
    payload: Dict[str, Any]
    
    @validator('event_id', 'run_id', 'parent_event_id')
    def validate_uuid(cls, v):
        """Validate UUID format."""
        if v is not None:
            try:
                uuid.UUID(v)
            except ValueError:
                raise ValueError(f"Invalid UUID format: {v}")
        return v
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate ISO 8601 timestamp format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError(f"Invalid ISO 8601 timestamp format: {v}")
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """Validate metadata values are primitive types."""
        if v is not None:
            for key, value in v.items():
                if not isinstance(value, (str, int, float, bool)):
                    raise ValueError(f"Metadata value for key '{key}' must be string, number, or boolean")
        return v
    
    def validate(self) -> None:
        """Additional validation for the complete event."""
        # Validate payload structure based on event type
        expected_payload_class = self._get_payload_class()
        if expected_payload_class:
            try:
                expected_payload_class(**self.payload)
            except Exception as e:
                raise ValueError(f"Invalid payload for event type {self.event_type}: {e}")
    
    def _get_payload_class(self) -> Optional[type]:
        """Get the expected payload class for this event type."""
        payload_mapping = {
            EventType.USER_INPUT: UserInputPayload,
            EventType.USER_FEEDBACK: UserFeedbackPayload,
            EventType.AGENT_REASONING: AgentReasoningPayload,
            EventType.AGENT_PLANNING: AgentPlanningPayload,
            EventType.AGENT_RESPONSE: AgentResponsePayload,
            EventType.LLM_REQUEST: LLMRequestPayload,
            EventType.LLM_RESPONSE: LLMResponsePayload,
            EventType.LLM_STREAMING_CHUNK: LLMStreamingChunkPayload,
            EventType.TOOL_CALL: ToolCallPayload,
            EventType.TOOL_RESPONSE: ToolResponsePayload,
            EventType.FUNCTION_EXECUTION: FunctionExecutionPayload,
            EventType.API_REQUEST: APIRequestPayload,
            EventType.API_RESPONSE: APIResponsePayload,
            EventType.ERROR: ErrorPayload,
            EventType.VALIDATION_ERROR: ValidationErrorPayload,
            EventType.TIMEOUT_ERROR: TimeoutErrorPayload,
            EventType.SESSION_START: SessionStartPayload,
            EventType.SESSION_END: SessionEndPayload,
            EventType.PERFORMANCE_METRIC: PerformanceMetricPayload,
        }
        return payload_mapping.get(self.event_type)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for API submission."""
        return self.dict(exclude_none=True)
    
    @classmethod
    def create_user_input(
        cls,
        run_id: str,
        user_id: str,
        input_text: str,
        input_type: str = "text",
        **kwargs
    ) -> 'Event':
        """Create a user input event."""
        payload = UserInputPayload(
            input_text=input_text,
            input_type=input_type,
            **{k: v for k, v in kwargs.items() if k in UserInputPayload.__fields__}
        ).dict()
        
        return cls(
            run_id=run_id,
            user_id=user_id,
            event_type=EventType.USER_INPUT,
            payload=payload,
            **{k: v for k, v in kwargs.items() if k in cls.__fields__}
        )
    
    @classmethod
    def create_llm_request(
        cls,
        run_id: str,
        user_id: str,
        model_name: str,
        provider: str,
        prompt: str,
        **kwargs
    ) -> 'Event':
        """Create an LLM request event."""
        payload = LLMRequestPayload(
            model_name=model_name,
            provider=provider,
            prompt=prompt,
            **{k: v for k, v in kwargs.items() if k in LLMRequestPayload.__fields__}
        ).dict()
        
        return cls(
            run_id=run_id,
            user_id=user_id,
            event_type=EventType.LLM_REQUEST,
            payload=payload,
            **{k: v for k, v in kwargs.items() if k in cls.__fields__}
        )
    
    @classmethod
    def create_llm_response(
        cls,
        run_id: str,
        user_id: str,
        model_name: str,
        provider: str,
        response_text: str,
        parent_event_id: Optional[str] = None,
        **kwargs
    ) -> 'Event':
        """Create an LLM response event."""
        payload = LLMResponsePayload(
            model_name=model_name,
            provider=provider,
            response_text=response_text,
            **{k: v for k, v in kwargs.items() if k in LLMResponsePayload.__fields__}
        ).dict()
        
        return cls(
            run_id=run_id,
            user_id=user_id,
            event_type=EventType.LLM_RESPONSE,
            payload=payload,
            parent_event_id=parent_event_id,
            **{k: v for k, v in kwargs.items() if k in cls.__fields__}
        )
    
    @classmethod
    def create_error(
        cls,
        run_id: str,
        user_id: str,
        error_type: str,
        error_message: str,
        parent_event_id: Optional[str] = None,
        **kwargs
    ) -> 'Event':
        """Create an error event."""
        payload = ErrorPayload(
            error_type=error_type,
            error_message=error_message,
            **{k: v for k, v in kwargs.items() if k in ErrorPayload.__fields__}
        ).dict()
        
        return cls(
            run_id=run_id,
            user_id=user_id,
            event_type=EventType.ERROR,
            payload=payload,
            parent_event_id=parent_event_id,
            **{k: v for k, v in kwargs.items() if k in cls.__fields__}
        )
