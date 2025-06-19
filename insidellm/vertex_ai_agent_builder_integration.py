import logging
import json
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, AsyncGenerator # Removed Generator as we're using AsyncGenerator
from uuid import uuid4
import inspect # For dynamically wrapping methods

# Google Cloud Discovery Engine Client Library
from google.cloud import discoveryengine_v1 as discoveryengine
# Import specific request/response types for type hinting if needed later
from google.cloud.discoveryengine_v1.types import (
    ConverseConversationRequest,
    ConverseConversationResponse,
    TextInput,
    Reply,
    SearchResponse, # Corrected from Answer, Summary is a sub-message of SearchResponse
    Answer, # Keep for potential use if response.answer is populated
)

# InsideLLM models
from insidellm.models import (
    Event,
    EventType,
    UserInputPayload,
    AgentResponsePayload,
    ToolCallPayload,
    ToolResponsePayload,
    ErrorPayload,
    # LLMRequestPayload,     # Currently abstracting LLM calls as part of agent interaction
    # LLMResponsePayload,
)

# Placeholder for the actual InsideLLMClient - replace with actual import if available
# from insidellm import InsideLLMClient
class InsideLLMClient:
    def log_event(self, event: Event):
        log_message = (
            f"InsideLLMClient (Placeholder) Logging event:\n"
            f"  ID: {event.id}\n"
            f"  Type: {event.type.value}\n"
            f"  Timestamp: {event.timestamp}\n"
            f"  Run ID: {event.run_id}\n"
            f"  User ID: {event.user_id}\n"
            f"  Parent ID: {event.parent_id}\n"
            f"  Payload: {json.dumps(event.payload, indent=2, default=str)}\n"
            f"  Metadata: {json.dumps(event.metadata, indent=2, default=str)}\n"
        )
        print(log_message)

logger = logging.getLogger(__name__)

def _safe_serialize(data: Any) -> str:
    """Safely serialize data to JSON string, handling complex types."""
    try:
        return json.dumps(data, default=lambda o: f"<unserializable_{type(o).__name__}>")
    except Exception:
        return str(data)

class InsideLLMVertexAgentBuilderConversationalSearchClient:
    """
    A wrapper around google.cloud.discoveryengine_v1.ConversationalSearchServiceAsyncClient
    to log events to InsideLLM.
    """

    def __init__(
        self,
        client: discoveryengine.ConversationalSearchServiceAsyncClient,
        insidellm_client: InsideLLMClient,
        user_id: str,
        run_id: str,
        debug: bool = False,
    ):
        self._client = client
        self._insidellm_client = insidellm_client
        self._user_id = user_id
        self._run_id = run_id
        self._debug = debug

        if self._debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        logger.info(f"InsideLLMVertexAgentBuilderConversationalSearchClient initialized for run_id: {self._run_id}, user_id: {self._user_id}")

        # Dynamically wrap methods of the underlying client for pass-through if not explicitly overridden.
        # This makes the wrapper behave more like the original client for non-intercepted methods.
        for name, method in inspect.getmembers(self._client, inspect.isroutine): # isroutine covers methods and functions
            if not hasattr(self, name) and name != "__init__": # Avoid overriding own init or private/dunder methods
                setattr(self, name, method)


    def _log_event(
        self,
        event_type: EventType,
        payload: Any,
        metadata: Optional[Dict[str, Any]] = None,
        parent_event_id: Optional[str] = None,
    ) -> Event:
        event_id = str(uuid4())
        processed_metadata = {}
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    processed_metadata[k] = v
                else:
                    processed_metadata[k] = _safe_serialize(v) # Use helper for complex objects

        event_payload_dict = payload.dict() if hasattr(payload, 'dict') else payload

        event = Event(
            id=event_id, # Changed from event_id to id
            type=event_type, # Changed from event_type to type
            timestamp=datetime.now(timezone.utc).isoformat(),
            run_id=self._run_id,
            user_id=self._user_id,
            payload=event_payload_dict, # Ensure payload is a dict
            parent_id=parent_event_id, # Changed from parent_event_id to parent_id
            metadata=processed_metadata,
        )
        try:
            self._insidellm_client.log_event(event)
            if self._debug:
                logger.debug(f"Logged event: {event.type.value} (ID: {event.id}, Parent: {event.parent_id})")
        except Exception as e:
            logger.error(f"Error logging event {event.type.value} (ID: {event.id}): {e}", exc_info=True)
            # Log an ERROR event to InsideLLM itself
            error_payload = ErrorPayload(
                error_type=e.__class__.__name__, # As per insidellm.models.ErrorPayload
                error_message=str(e),
                stack_trace=traceback.format_exc(), # Renamed from details
                source="InsideLLMVertexAgentBuilderConversationalSearchClient",
            )
            # Create and log this secondary error event carefully
            error_event_obj = Event(
                id=str(uuid4()),
                type=EventType.ERROR,
                timestamp=datetime.now(timezone.utc).isoformat(),
                run_id=self._run_id,
                user_id=self._user_id, # Assuming system or related user for this internal error
                payload=error_payload.dict(),
                parent_id=event.id, # Link to the event that failed to log initially
                metadata={"description": "Failed to log an event to InsideLLM."}
            )
            try:
                self._insidellm_client.log_event(error_event_obj)
            except Exception as critical_e:
                logger.critical(f"CRITICAL: Failed to log error reporting event to InsideLLM: {critical_e}")

        return event

    async def converse_conversation(
        self,
        request: ConverseConversationRequest, # Use the specific type
        *args,
        **kwargs,
    ) -> ConverseConversationResponse: # Use the specific type
        parent_event_id_for_agent_response = None
        user_query = ""

        if request.query and hasattr(request.query, "text"):
            user_query = request.query.text
        elif request.query and hasattr(request.query, "query_entry") and hasattr(request.query.query_entry, "text"): # Check alternative path if query is QueryEntry
             user_query = request.query.query_entry.text


        user_input_payload = UserInputPayload(input_text=user_query if user_query else "N/A")
        user_input_event = self._log_event(EventType.USER_INPUT, user_input_payload)
        parent_event_id_for_agent_response = user_input_event.id

        try:
            response: ConverseConversationResponse = await self._client.converse_conversation(
                request=request, *args, **kwargs
            )

            agent_response_text = ""
            if response.reply and response.reply.summary: # summary is of type SearchResponse.Summary
                 agent_response_text = response.reply.summary.summary_text

            agent_response_payload = AgentResponsePayload(response_text=agent_response_text)
            agent_response_event = self._log_event(
                EventType.AGENT_RESPONSE,
                agent_response_payload,
                parent_event_id=parent_event_id_for_agent_response
            )
            current_parent_for_tools = agent_response_event.id # Tools are children of the agent's response

            # Parse and Log Tool Calls and Responses
            # Vertex AI Agent Builder's `ConverseConversationResponse` can include `Answer` objects,
            # which have `steps`. Each step can represent an action (like a tool call) and its observation.
            if response.answer and response.answer.steps:
                for step in response.answer.steps:
                    tool_name = "UnknownTool"
                    tool_input = "{}"
                    action_type = "unknown_action"

                    if step.action and step.action.search_action:
                        action_type = "search_action"
                        tool_name = "VertexAISearch" # Or derive from search_action details
                        tool_input = _safe_serialize({"query": step.action.search_action.query})
                    elif step.action and step.action.tool_action: # Assuming a generic 'tool_action' field
                        action_type = "tool_action"
                        tool_name = step.action.tool_action.tool or "UnnamedTool"
                        tool_input = step.action.tool_action.tool_input or "{}" # Assuming tool_input is a string or serializable
                    elif step.description: # Fallback if specific actions aren't found
                        tool_name = "AgentStep"
                        tool_input = _safe_serialize({"description": step.description})

                    # Log Tool Call
                    tool_call_payload = ToolCallPayload(
                        tool_name=tool_name,
                        tool_type=action_type,
                        parameters=json.loads(tool_input) if isinstance(tool_input, str) else tool_input,
                    )
                    tool_call_event = self._log_event(
                        EventType.TOOL_CALL,
                        tool_call_payload,
                        parent_event_id=current_parent_for_tools,
                        metadata={"step_description": step.description, "step_thought": step.thought}
                    )

                    # Log Tool Response (from observation)
                    tool_output_data = "{}"
                    success = step.state == Answer.Step.State.SUCCEEDED # Check step state for success

                    if step.action and step.action.observation and step.action.observation.search_results:
                        tool_output_data = _safe_serialize([sr.document for sr in step.action.observation.search_results])
                    elif step.action and step.action.observation and hasattr(step.action.observation, 'tool_output'):
                        tool_output_data = step.action.observation.tool_output # Assuming a direct output field
                    elif step.thought and not success : # If it failed, thought might contain error info
                        tool_output_data = _safe_serialize({"error_in_thought": step.thought})


                    tool_response_payload = ToolResponsePayload(
                        tool_name=tool_name,
                        tool_type=action_type,
                        response_data=json.loads(tool_output_data) if isinstance(tool_output_data, str) else tool_output_data,
                        success=success,
                        error_message=step.thought if not success else None
                    )
                    self._log_event(
                        EventType.TOOL_RESPONSE,
                        tool_response_payload,
                        parent_event_id=tool_call_event.id
                    )
            # Also log search results directly attached to the response if not part of steps
            elif response.search_results: # Fallback if no steps, but search_results exist
                 for sr in response.search_results:
                    tool_call_event = self._log_event(EventType.TOOL_CALL, ToolCallPayload(tool_name="VertexAISearchDirect", tool_type="search", parameters={"query": user_query}), parent_event_id=current_parent_for_tools)
                    self._log_event(EventType.TOOL_RESPONSE, ToolResponsePayload(tool_name="VertexAISearchDirect", tool_type="search", response_data=_safe_serialize(sr.document), success=True), parent_event_id=tool_call_event.id)


            return response

        except Exception as e:
            logger.error(f"Error during converse_conversation: {e}", exc_info=True)
            error_payload = ErrorPayload(
                error_type=e.__class__.__name__,
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                source="converse_conversation_sdk_call",
            )
            self._log_event(EventType.ERROR, error_payload, parent_event_id=parent_event_id_for_agent_response)
            raise

# Example Usage (conceptual)
async def example_usage():
    logging.basicConfig(level=logging.DEBUG if True else logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    class MockStepActionObservationSearchResult:
        def __init__(self, doc_id, content):
            self.document = {"id": doc_id, "content": content, "uri": f"http://example.com/{doc_id}"}
            self.id = doc_id # for search_result.id access

    class MockStepActionObservation:
        def __init__(self):
            self.search_results: List[MockStepActionObservationSearchResult] = []
            # self.tool_output = None # for generic tool output

    class MockStepAction:
        def __init__(self):
            self.search_action: Optional[Any] = None # discoveryengine.Answer.Step.Action.SearchAction
            self.tool_action: Optional[Any] = None # Placeholder for generic tool
            self.observation: Optional[MockStepActionObservation] = None

    class MockStep:
        def __init__(self, description, thought, state, action=None):
            self.description = description
            self.thought = thought
            self.state = state # discoveryengine.Answer.Step.State.SUCCEEDED or .FAILED
            self.action = action # MockStepAction

    class MockAnswer:
        def __init__(self):
            self.steps: List[MockStep] = []

    class MockGoogleConversationalSearchAsyncClient:
        async def converse_conversation(self, request: ConverseConversationRequest, *args, **kwargs):
            logger.info(f"MockGoogleClient: converse_conversation called with request query: '{request.query.text if request.query else 'N/A'}'")
            response = ConverseConversationResponse()
            response.reply = Reply()
            response.reply.summary = SearchResponse.Summary(summary_text=f"Mock reply to '{request.query.text if request.query else 'N/A'}'.")

            # Simulate an Answer with steps for tool usage
            response.answer = MockAnswer() # Use our mock Answer

            # Step 1: Search Action
            search_action_step = MockStep(
                description="Performing a search for relevant documents.",
                thought="I need to find documents about topic X.",
                state=Answer.Step.State.SUCCEEDED
            )
            search_action_step.action = MockStepAction()
            search_action_step.action.search_action = discoveryengine.Answer.Step.Action.SearchAction(query="topic X")
            search_action_step.action.observation = MockStepActionObservation()
            search_action_step.action.observation.search_results.append(MockStepActionObservationSearchResult("doc_alpha", "Content of document Alpha about X."))
            response.answer.steps.append(search_action_step)

            # Step 2: (Simulated) Tool Action that failed
            tool_action_step_failed = MockStep(
                description="Attempting to use custom summarizer tool.",
                thought="The summarizer tool failed due to an API key error.",
                state=Answer.Step.State.FAILED # Simulate failure
            )
            tool_action_step_failed.action = MockStepAction()
            tool_action_step_failed.action.tool_action = type('ToolAction', (), {'tool': 'CustomSummarizer', 'tool_input': '{"text": "long document..."}'})()
            # No observation for failure, or observation might contain error details
            response.answer.steps.append(tool_action_step_failed)

            return response

    mock_google_client = MockGoogleConversationalSearchAsyncClient()
    inside_llm_placeholder_client = InsideLLMClient()
    run_id_for_example = str(uuid4())

    wrapped_client = InsideLLMVertexAgentBuilderConversationalSearchClient(
        client=mock_google_client,
        insidellm_client=inside_llm_placeholder_client,
        user_id="vertex_example_user",
        run_id=run_id_for_example,
        debug=True
    )

    try:
        request_payload = ConverseConversationRequest(
            name="projects/your-project/locations/global/dataStores/your-datastore/conversations/-",
            query=TextInput(text="Tell me about topic X and summarize it."),
        )
        print("\n--- Making call via wrapped client (Vertex AI Agent Builder) ---")
        response_from_wrapper = await wrapped_client.converse_conversation(request=request_payload)
        print(f"Agent's raw response summary from wrapper: {response_from_wrapper.reply.summary.summary_text}")
    except Exception as e:
        print(f"An error occurred during example_usage: {e}")

if __name__ == "__main__":
    # Setup basic logging for the example
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    import asyncio
    asyncio.run(example_usage())
