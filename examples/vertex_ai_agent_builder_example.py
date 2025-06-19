import asyncio
import logging
import json
from uuid import uuid4
from datetime import datetime, timezone
from typing import Optional, Any, List # Added missing imports

# Google Cloud Discovery Engine Client Library - for types
from google.cloud.discoveryengine_v1 import (
    ConversationalSearchServiceAsyncClient, # For type hinting the real client
    ConverseConversationRequest,
    ConverseConversationResponse,
    Query, # Corrected from TextInput for request.query
    TextInput, # May be used elsewhere, but Query is for ConverseConversationRequest.query
    Reply,
    SearchResponse, # Summary is a sub-message of SearchResponse
    Answer,
    Document as DiscoveryEngineDocument, # Alias to avoid confusion if we have other Document types
)

# InsideLLM components
# Assuming insidellm is installed and __init__.py is set up correctly
from insidellm import InsideLLMClient, InsideLLMVertexAgentBuilderConversationalSearchClient

# Configure basic logging for the example
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Mock InsideLLMClient for example output ---
class PlaceholderInsideLLMClient:
    def log_event(self, event):
        log_message = (
            f"INSIDELLM_EVENT (PlaceholderClient):\n"
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

# --- Mock Google Cloud ConversationalSearchServiceAsyncClient ---
class MockStepActionObservationSearchResult:
    def __init__(self, doc_id, content_text, uri=""):
        self.document = DiscoveryEngineDocument( # Use the aliased Document type
            name=f"projects/p/locations/l/dataStores/d/branches/b/documents/{doc_id}",
            id=doc_id,
            content={"text": content_text, "mime_type": "text/plain"}, # Simplified content
            struct_data={"uri": uri if uri else f"http://example.com/docs/{doc_id}"}
        )

class MockStepActionObservation:
    def __init__(self):
        self.search_results: list[MockStepActionObservationSearchResult] = []
        self.tool_output: Optional[str] = None

class MockStepAction:
    def __init__(self):
        self.search_action: Optional[Answer.Step.Action.SearchAction] = None
        self.tool_action: Optional[Any] = None # Using Any for simplicity for now
        self.observation: Optional[MockStepActionObservation] = None

class MockStep:
    def __init__(self, description: str, thought: str, state: Answer.Step.State, action: Optional[MockStepAction] = None):
        self.description = description
        self.thought = thought
        self.state = state
        self.action = action if action else MockStepAction()

class MockAnswer: # Mimicking discoveryengine.Answer
    def __init__(self):
        self.steps: list[MockStep] = []
        self.name = "mock_answer_name" # Add other fields as needed by the wrapper

class MockConversationalSearchServiceAsyncClient:
    async def converse_conversation(
        self,
        request: ConverseConversationRequest,
        *args, **kwargs
    ) -> ConverseConversationResponse:
        logger.info(
            f"MockConversationalSearchServiceAsyncClient: converse_conversation called with query: "
            f"'{request.query.text if request.query and request.query.text else 'N/A'}'"
        )

        response = ConverseConversationResponse()

        # Populate Reply
        response.reply = Reply()
        response.reply.summary = SearchResponse.Summary() # Summary is part of SearchResponse
        response.reply.summary.summary_text = f"This is a mock Vertex AI Agent Builder reply to: '{request.query.text if request.query else ''}'"

        # Populate Answer with Steps (simulating tool calls)
        mock_answer = MockAnswer() # Use our mock Answer

        # Step 1: Successful Search Action
        search_step = MockStep(
            description="Performing a search for relevant information.",
            thought="I should search for documents related to the user's query.",
            state=Answer.Step.State.SUCCEEDED
        )
        search_step.action.search_action = Answer.Step.Action.SearchAction(query=f"search query for: {request.query.text if request.query else ''}")
        search_step.action.observation = MockStepActionObservation()
        search_step.action.observation.search_results.append(
            MockStepActionObservationSearchResult("doc-001", "Content of document 1 related to the query.", "http://example.com/doc-001")
        )
        search_step.action.observation.search_results.append(
            MockStepActionObservationSearchResult("doc-002", "More content in document 2.", "http://example.com/doc-002")
        )
        mock_answer.steps.append(search_step)

        # Step 2: Successful Custom Tool Action
        custom_tool_step_success = MockStep(
            description="Using a custom tool to process data.",
            thought="The data looks good, I will use CustomProcessTool.",
            state=Answer.Step.State.SUCCEEDED
        )
        # Mocking a generic tool_action structure; the actual SDK might have specific fields
        custom_tool_step_success.action.tool_action = type('ToolAction', (), {
            'tool': 'CustomProcessTool',
            'tool_input': json.dumps({"data_id": "doc-001", "params": {"detail_level": "high"}})
        })()
        custom_tool_step_success.action.observation = MockStepActionObservation()
        custom_tool_step_success.action.observation.tool_output = json.dumps({"status": "completed", "processed_items": 1, "summary": "Data processed successfully."})
        mock_answer.steps.append(custom_tool_step_success)

        # Step 3: Failed Tool Action
        custom_tool_step_failure = MockStep(
            description="Attempting to use another custom tool.",
            thought="This tool might fail due to invalid input.",
            state=Answer.Step.State.FAILED # Simulate failure
        )
        custom_tool_step_failure.action.tool_action = type('ToolAction', (), {
            'tool': 'RiskyTool',
            'tool_input': json.dumps({"input_value": -1})
        })()
        custom_tool_step_failure.action.observation = MockStepActionObservation()
        custom_tool_step_failure.action.observation.tool_output = json.dumps({"error": "Invalid input value", "error_code": "VAL-002"})
        # In case of failure, the 'thought' field might also contain error details from the agent's perspective
        # custom_tool_step_failure.thought = "RiskyTool failed: Input value must be positive."
        mock_answer.steps.append(custom_tool_step_failure)

        response.answer = mock_answer # Assign the mock_answer to response.answer

        # Simulate some top-level search results as well (if the wrapper checks this separately)
        response.search_results = [
            SearchResponse.SearchResult(id="top_doc_1"),
            SearchResponse.SearchResult(id="top_doc_2")
        ]

        return response

async def run_example():
    logger.info("Starting Vertex AI Agent Builder Example with InsideLLM Integration...")

    # 1. Instantiate a placeholder InsideLLMClient
    # In a real application, this would be initialized with API keys, etc.
    # from insidellm import initialize
    # inside_llm_client = initialize(api_key="YOUR_INSIDELLM_API_KEY")
    inside_llm_client = PlaceholderInsideLLMClient()
    logger.info("PlaceholderInsideLLMClient instantiated.")

    # 2. Instantiate the Mock Google SDK Client
    mock_sdk_client = MockConversationalSearchServiceAsyncClient()
    logger.info("MockConversationalSearchServiceAsyncClient instantiated.")

    # 3. Instantiate the InsideLLM Wrapper Client
    run_id = str(uuid4())
    user_id = "vertex_example_user_001"
    wrapped_vertex_client = InsideLLMVertexAgentBuilderConversationalSearchClient(
        client=mock_sdk_client,
        insidellm_client=inside_llm_client,
        user_id=user_id,
        run_id=run_id,
        debug=True # Enable debug logging from the wrapper
    )
    logger.info("InsideLLMVertexAgentBuilderConversationalSearchClient wrapper instantiated.")

    # 4. Prepare a sample ConverseConversationRequest
    # Note: The `name` parameter should be the resource name of the conversation or agent.
    # projects/{project_number}/locations/{location}/dataStores/{data_store_id}/conversations/- (for new conv)
    # projects/{project_number}/locations/{location}/dataStores/{data_store_id}/conversations/{conversation_id}
    # projects/{project_number}/locations/{location}/agents/{agent_id}/sessions/- (for new session with agent)
    sample_request = ConverseConversationRequest(
        name="projects/your-gcp-project/locations/global/dataStores/your-data-store-id/conversations/-",
        query=Query(text="Hello Vertex AI Agent! Tell me about Large Language Models and search for some papers."),
        # session = "projects/.../sessions/your-session-id" # Optionally specify a session
    )
    logger.info(f"Sample ConverseConversationRequest created for run_id: {run_id}")

    # 5. Make a call using the wrapped client
    print("\n--- Making call to wrapped_vertex_client.converse_conversation ---")
    try:
        response = await wrapped_vertex_client.converse_conversation(request=sample_request)
        logger.info("Call to wrapped client finished.")
        print("\n--- Mocked SDK Response Summary ---")
        if response.reply and response.reply.summary:
            print(f"Agent Reply: {response.reply.summary.summary_text}")
        if response.answer and response.answer.steps:
            print(f"Number of steps in answer: {len(response.answer.steps)}")

    except Exception as e:
        logger.error(f"An error occurred during the example run: {e}", exc_info=True)
        print(f"\n--- Example Execution Failed ---")
        print(f"Error: {e}")

    logger.info("Vertex AI Agent Builder Example finished.")

if __name__ == "__main__":
    asyncio.run(run_example())
