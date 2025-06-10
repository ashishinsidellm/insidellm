import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
import json # For serializing complex objects in metadata
import traceback # For logging full tracebacks in ErrorPayload

# Assuming InsideLLMClient and insidellm.models are importable
# from insidellm import InsideLLMClient
from insidellm.models import (
    ErrorPayload,
    Event,
    EventType,
    LLMRequestPayload,
    LLMResponsePayload,
    # ToolCallPayload, # Still keeping these out as LiteLLM doesn't directly give this context
    # ToolResponsePayload,
)

# LiteLLM's CustomLogger is the new base class
from litellm.integrations.custom_logger import CustomLogger as LiteLLMCustomLogger
# from litellm.types.utils import ModelResponse # For type hinting response_obj

# Placeholder for InsideLLMClient for standalone development
class InsideLLMClient:
    def log_event(self, event: Event):
        # A more detailed placeholder log
        log_message = (
            f"InsideLLMClient (Placeholder) Logging event:\n"
            f"  ID: {event.id}\n"
            f"  Type: {event.type.value}\n"
            f"  Timestamp: {event.timestamp}\n"
            f"  Parent ID: {event.parent_id}\n"
            f"  Payload: {event.payload}\n"
            f"  Metadata: {json.dumps(event.metadata, indent=2)}\n" # Pretty print metadata
        )
        print(log_message)


logger = logging.getLogger(__name__)

# Helper to safely serialize metadata
def safe_serialize(data: Any) -> Optional[str]:
    if data is None:
        return None
    try:
        # Attempt to serialize complex objects with default=str for robustness
        return json.dumps(data, default=str)
    except TypeError:
        return str(data) # Fallback to string representation

class CrewAIInsideLLMCallbackHandler(LiteLLMCustomLogger):
    """
    A callback handler for CrewAI that leverages LiteLLM's CustomLogger
    to capture LLM events and log them to InsideLLM.
    """

    def __init__(self, client: InsideLLMClient, debug: bool = False):
        super().__init__()
        self.client = client
        self.debug = debug
        self._event_stack: List[Event] = []
        # Add run_id and user_id, assuming they are provided or can be defaulted
        # These are required by the insidellm.models.Event
        self.current_run_id: str = str(uuid4()) # Example: generate a new run_id per handler instance
        self.current_user_id: str = "crewai_system" # Example default

        if self.debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        logger.info(f"CrewAIInsideLLMCallbackHandler (LiteLLM based) initialized for run_id: {self.current_run_id}")

    def _get_provider_from_model(self, model_name: str) -> str:
        """Infers provider from model name string."""
        if not model_name:
            return "unknown"
        if "/" in model_name:
            return model_name.split('/', 1)[0]
        # Add more specific heuristics if needed
        if model_name.startswith("gpt-"): return "openai"
        if model_name.startswith("claude-"): return "anthropic"
        if model_name.startswith("gemini-"): return "google" # or vertex_ai
        if model_name.startswith("cohere."): return "cohere" # Check actual cohere model name format
        if model_name.startswith("mistral"): return "mistral"
        # Fallback or raise error if provider cannot be determined
        return "unknown_provider"


    def _get_current_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _get_current_parent_id(self) -> Optional[str]:
        if self._event_stack:
            return self._event_stack[-1].id
        return None

    def _log_event(
        self,
        event_type: EventType,
        payload: Any, # Should be a Pydantic model (e.g., LLMRequestPayload)
        metadata: Optional[Dict[str, Any]] = None,
        parent_id_override: Optional[str] = None,
    ) -> Event:
        event_id = str(uuid4())
        parent_id = parent_id_override if parent_id_override is not None else self._get_current_parent_id()

        # Ensure metadata values are suitable for JSON (str, int, float, bool)
        processed_metadata = {}
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    processed_metadata[k] = v
                else:
                    processed_metadata[k] = safe_serialize(v)


        event = Event(
            event_id=event_id,       # Corrected field name
            event_type=event_type,   # Corrected field name
            timestamp=self._get_current_timestamp(),
            run_id=self.current_run_id,
            user_id=self.current_user_id,
            payload=payload.dict() if hasattr(payload, 'dict') else payload, # Convert Pydantic model to dict
            parent_event_id=parent_id, # Corrected field name
            metadata=processed_metadata,
        )
        try:
            self.client.log_event(event)
            if self.debug:
                logger.debug(f"Logged event: {event_type.value} (ID: {event_id}, Parent: {parent_id})")
        except Exception as e:
            logger.error(f"Error logging event {event_type.value} (ID: {event_id}): {e}")
            error_payload_dict = ErrorPayload(
                error_type=e.__class__.__name__, # Renamed from 'message' to 'error_type' in models.py
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                source="CrewAIInsideLLMCallbackHandler",
            ).dict()
            error_event = Event(
                event_id=str(uuid4()),   # Corrected field name
                event_type=EventType.ERROR, # Corrected field name
                timestamp=self._get_current_timestamp(),
                run_id=self.current_run_id,
                user_id=self.current_user_id,
                payload=error_payload_dict,
                parent_event_id=parent_id, # Corrected field name
                metadata={"original_event_type": event_type.value}
            )
            try:
                self.client.log_event(error_event)
            except Exception as critical_e:
                logger.critical(f"CRITICAL: Failed to log error event to InsideLLM: {critical_e}")
        return event

    # --- LiteLLM CustomLogger Methods ---

    async def async_log_pre_api_call(self, model: str, messages: List[Dict[str, Any]], kwargs: Dict[str, Any]):
        if self.debug:
            # Avoid serializing potentially large 'messages' or 'kwargs' directly in debug log line
            logger.debug(f"async_log_pre_api_call: Model: {model}, num_messages: {len(messages)}, kwargs keys: {list(kwargs.keys())}")

        litellm_params = kwargs.get("litellm_params", {})
        crewai_metadata = litellm_params.get("metadata", {})
        provider = self._get_provider_from_model(model_name=model)

        # Ensure messages are properly formatted if they are part of the payload directly
        # The insidellm.models.LLMRequestPayload takes 'prompt' (string) and 'messages' (list of dicts)
        # LiteLLM 'messages' is already a list of dicts. 'prompt' could be a string representation.

        payload = LLMRequestPayload(
            model_name=model,
            provider=provider,
            prompt=safe_serialize(messages), # Main prompt content as serialized messages
            messages=messages, # Actual messages list
            parameters=litellm_params.get("model_info", litellm_params), # Pass relevant params, avoid full duplication if possible
        )

        event_metadata = {"litellm_call_kwargs_keys": list(kwargs.keys())} # Log only keys to avoid large data
        if "stream" in kwargs: # Add stream info
             event_metadata["streaming"] = kwargs["stream"]
        if crewai_metadata:
            event_metadata["crewai_metadata"] = crewai_metadata # This is already a dict
            if "tool_name" in crewai_metadata:
                event_metadata["associated_tool"] = crewai_metadata["tool_name"]

        event = self._log_event(EventType.LLM_REQUEST, payload, metadata=event_metadata)
        self._event_stack.append(event)

    async def async_log_success_event(self, kwargs: Dict[str, Any], response_obj: Any, start_time: float, end_time: float):
        if self.debug:
            logger.debug(f"async_log_success_event: Model: {kwargs.get('model')}, response_obj type: {type(response_obj)}")

        parent_event = self._event_stack.pop() if self._event_stack and self._event_stack[-1].event_type == EventType.LLM_REQUEST else None # Corrected: .event_type
        parent_id = parent_event.event_id if parent_event else None # Corrected: .event_id

        litellm_params = kwargs.get("litellm_params", {})
        crewai_metadata = litellm_params.get("metadata", {})

        model_name_from_kwargs = kwargs.get("model", "unknown_model")
        provider = self._get_provider_from_model(model_name=model_name_from_kwargs)

        response_text = ""
        actual_model_name = model_name_from_kwargs # Default to model from request
        input_tokens = None
        output_tokens = None
        cost = None
        finish_reason = None

        if hasattr(response_obj, "choices") and response_obj.choices:
            first_choice = response_obj.choices[0]
            if hasattr(first_choice, "message") and hasattr(first_choice.message, "content"):
                response_text = first_choice.message.content
            elif hasattr(first_choice, "text"):
                 response_text = first_choice.text
            if hasattr(first_choice, "finish_reason"):
                finish_reason = str(first_choice.finish_reason)


        if hasattr(response_obj, "model"): # More accurate model name from response
            actual_model_name = response_obj.model
            provider = self._get_provider_from_model(actual_model_name) # Re-evaluate provider if model changed

        if hasattr(response_obj, "usage"): # Standard LiteLLM usage object
            usage_data = response_obj.usage
            if hasattr(usage_data, "prompt_tokens"): input_tokens = usage_data.prompt_tokens
            if hasattr(usage_data, "completion_tokens"): output_tokens = usage_data.completion_tokens

        # LiteLLM might also provide cost directly, or it can be calculated
        if "cost" in kwargs: # Cost might be pre-calculated by LiteLLM and passed in kwargs
            cost = kwargs["cost"]

        payload = LLMResponsePayload(
            model_name=actual_model_name,
            provider=provider,
            response_text=response_text,
            finish_reason=finish_reason,
            usage={"input_tokens": input_tokens, "output_tokens": output_tokens} if input_tokens is not None else None,
            response_time_ms=int((end_time - start_time) * 1000),
            cost=cost,
        )

        event_metadata = {
            "start_time": str(start_time), # Keep as string for metadata
            "end_time": str(end_time),
            # "litellm_kwargs_keys": list(kwargs.keys()), # Avoid full kwargs
            # "raw_response_type": str(type(response_obj)) # For debugging response structure
        }
        if crewai_metadata:
            event_metadata["crewai_metadata"] = crewai_metadata
            if "tool_name" in crewai_metadata:
                event_metadata["associated_tool"] = crewai_metadata["tool_name"]

        self._log_event(EventType.LLM_RESPONSE, payload, metadata=event_metadata, parent_id_override=parent_id)

    async def async_log_failure_event(self, kwargs: Dict[str, Any], response_obj: Any, start_time: float, end_time: float):
        if self.debug:
            logger.error(f"async_log_failure_event: Model: {kwargs.get('model')}, response_obj type: {type(response_obj)}")

        parent_event = self._event_stack.pop() if self._event_stack and self._event_stack[-1].event_type == EventType.LLM_REQUEST else None # Corrected: .event_type
        parent_id = parent_event.event_id if parent_event else None # Corrected: .event_id

        litellm_params = kwargs.get("litellm_params", {})
        crewai_metadata = litellm_params.get("metadata", {})
        model_name = kwargs.get("model", "unknown_model") # Get model from request context

        error_message = str(response_obj)
        error_type_str = response_obj.__class__.__name__ if isinstance(response_obj, Exception) else "LiteLLMError"

        details_str = ""
        if isinstance(response_obj, Exception):
            # More robust way to get traceback from an exception object
            details_str = "".join(traceback.format_exception(type(response_obj), response_obj, response_obj.__traceback__))
        elif isinstance(response_obj, dict) and "traceback" in response_obj: # LiteLLM might pass traceback string
            details_str = response_obj["traceback"]
        elif hasattr(response_obj, '__str__'): # Fallback for other error types
            details_str = str(response_obj)


        payload = ErrorPayload(
            error_type=error_type_str,
            error_message=error_message,
            stack_trace=details_str,
            context={"model_name": model_name, "error_source": "LiteLLMCall"} # Moved "source" into context
        )

        event_metadata = {
            "start_time": str(start_time),
            "end_time": str(end_time),
            # "litellm_kwargs_keys": list(kwargs.keys()),
            # "raw_error_response_type": str(type(response_obj))
        }
        if crewai_metadata:
            event_metadata["crewai_metadata"] = crewai_metadata
            if "tool_name" in crewai_metadata:
                event_metadata["associated_tool"] = crewai_metadata["tool_name"]

        self._log_event(EventType.ERROR, payload, metadata=event_metadata, parent_id_override=parent_id)

    # Synchronous versions (optional, can be pass-through or raise NotImplementedError if not used by CrewAI)
    def log_pre_api_call(self, model: str, messages: List[Dict[str, Any]], kwargs: Dict[str, Any]):
        logger.warning("Synchronous log_pre_api_call invoked. Consider ensuring CrewAI uses async LiteLLM calls.")
        # Example: asyncio.run(self.async_log_pre_api_call(model, messages, kwargs)) # Careful with nested asyncio.run

    def log_success_event(self, kwargs: Dict[str, Any], response_obj: Any, start_time: float, end_time: float):
        logger.warning("Synchronous log_success_event invoked.")

    def log_failure_event(self, kwargs: Dict[str, Any], response_obj: Any, start_time: float, end_time: float):
        logger.warning("Synchronous log_failure_event invoked.")


# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    inside_llm_client = InsideLLMClient()
    handler = CrewAIInsideLLMCallbackHandler(client=inside_llm_client, debug=True)
    # In a real scenario, this handler is registered with LiteLLM:
    # import litellm
    # litellm.callbacks = [handler]
    # litellm.set_verbose = True # Useful for seeing LiteLLM's own logs

    print("CrewAIInsideLLMCallbackHandler (LiteLLM based) is defined.")
    print("To test, this handler needs to be registered with LiteLLM.")
    print("CrewAI would then make calls through LiteLLM, triggering these callbacks.")
    print("\n--- Simulating LiteLLM direct callback calls ---")

    import asyncio

    async def simulate_calls():
        # Simulate an LLM call for a general agent task
        print("\nSimulating Agent LLM Call (OpenAI):")
        kwargs_agent_llm_openai = {
            "model": "gpt-3.5-turbo", # OpenAI model
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "litellm_params": {
                "metadata": { # This is where CrewAI should inject its context
                    "crew_name": "TravelCrew",
                    "task_id": "task_123",
                    "agent_name": "ResearcherAgent"
                },
                "model_info": {"temperature": 0.5} # Example model parameters
            }
        }
        await handler.async_log_pre_api_call(
            model=kwargs_agent_llm_openai["model"],
            messages=kwargs_agent_llm_openai["messages"],
            kwargs=kwargs_agent_llm_openai
        )
        # Mock response_obj for success
        mock_response_openai_success = type('ModelResponse', (), {})()
        mock_response_openai_success.choices = [type('Choice', (), {})()]
        mock_response_openai_success.choices[0].message = type('Message', (), {})()
        mock_response_openai_success.choices[0].message.content = "Paris"
        mock_response_openai_success.choices[0].finish_reason = "stop"
        mock_response_openai_success.model = "gpt-3.5-turbo-0125" # More specific model name
        mock_response_openai_success.usage = type('Usage', (), {'prompt_tokens': 10, 'completion_tokens': 2})()

        await handler.async_log_success_event(
            kwargs=kwargs_agent_llm_openai,
            response_obj=mock_response_openai_success,
            start_time=datetime.now().timestamp(),
            end_time=datetime.now().timestamp() + 1.0 # 1 second duration
        )

        # Simulate an LLM call made by a Tool (e.g. using Anthropic model)
        print("\nSimulating Tool LLM Call (Anthropic):")
        kwargs_tool_llm_anthropic = {
            "model": "claude-2", # Anthropic model
            "messages": [{"role": "user", "content": "Search for recent AI conferences."}],
            "litellm_params": {
                "metadata": {
                    "crew_name": "ResearchCrew",
                    "task_id": "task_456",
                    "agent_name": "SearchAgent",
                    "tool_name": "AIConferenceSearchTool", # Tool context
                    "tool_id": "tool_xyz"
                }
            },
            "cost": 0.00123 # Example cost passed by LiteLLM
        }
        await handler.async_log_pre_api_call(
            model=kwargs_tool_llm_anthropic["model"],
            messages=kwargs_tool_llm_anthropic["messages"],
            kwargs=kwargs_tool_llm_anthropic
        )
        # Mock response_obj for failure
        mock_response_anthropic_failure = type('ModelResponse', (), {})() # Can also be an Exception object
        # LiteLLM might return a ModelResponse even on failure, or an Exception object
        # For this test, let's simulate an Exception directly for failure_event
        simulated_exception = ConnectionError("Anthropic API connection timed out.")

        await handler.async_log_failure_event(
            kwargs=kwargs_tool_llm_anthropic,
            response_obj=simulated_exception, # Pass the exception object
            start_time=datetime.now().timestamp(),
            end_time=datetime.now().timestamp() + 2.0 # 2 seconds duration
        )

        print(f"\nFinal event stack size: {len(handler._event_stack)}")
        assert len(handler._event_stack) == 0, "Event stack should be empty after simulation"

    asyncio.run(simulate_calls())
    print("--- Simulation finished ---")
