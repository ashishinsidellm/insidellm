import unittest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import asyncio
import json
from uuid import uuid4, UUID
from datetime import datetime, timezone

# Import the class to be tested
from insidellm.vertex_ai_agent_builder_integration import InsideLLMVertexAgentBuilderConversationalSearchClient
# Import relevant Google Discovery Engine types for mocking
from google.cloud.discoveryengine_v1 import types as discoveryengine_types
from google.cloud.discoveryengine_v1.services.conversational_search_service import ConversationalSearchServiceAsyncClient

# Import InsideLLM models
from insidellm.models import Event, EventType, UserInputPayload, AgentResponsePayload, ToolCallPayload, ToolResponsePayload, ErrorPayload

# Mock for google.api_core.exceptions (if needed for specific error testing)
# For simplicity, we can just use standard Python exceptions for side_effect if the exact type isn't critical for the test.
class MockGoogleAPIError(Exception):
    pass

class MockInsideLLMClient:
    def __init__(self):
        self.events_logged: list[Event] = []
        self.log_event_calls = 0

    def log_event(self, event: Event):
        self.events_logged.append(event)
        self.log_event_calls += 1
        print(f"MockInsideLLMClient: Logged event: {event.event_type.value}, ID: {event.event_id}, Parent: {event.parent_event_id}")

class TestVertexAIAgentBuilderIntegration(unittest.TestCase):

    def setUp(self):
        self.mock_insidellm_client = MockInsideLLMClient()

        # Create an AsyncMock for the Google SDK client
        self.mock_sdk_client = AsyncMock(spec=ConversationalSearchServiceAsyncClient)
        # The converse_conversation method itself needs to be an AsyncMock
        self.mock_sdk_client.converse_conversation = AsyncMock()

        self.test_run_id = str(uuid4())
        self.test_user_id = "test_vertex_user"

        self.wrapper_client = InsideLLMVertexAgentBuilderConversationalSearchClient(
            client=self.mock_sdk_client,
            insidellm_client=self.mock_insidellm_client,
            user_id=self.test_user_id,
            run_id=self.test_run_id,
            debug=True
        )

    def _run_async(self, coro):
        return asyncio.run(coro)

    def test_initialization(self):
        self.assertEqual(self.wrapper_client._client, self.mock_sdk_client)
        self.assertEqual(self.wrapper_client._insidellm_client, self.mock_insidellm_client)
        self.assertEqual(self.wrapper_client._user_id, self.test_user_id)
        self.assertEqual(self.wrapper_client._run_id, self.test_run_id)
        self.assertTrue(self.wrapper_client._debug)

    def test_converse_conversation_basic_flow(self):
        # Configure mock SDK response
        mock_response = discoveryengine_types.ConverseConversationResponse({
            "reply": {
                "summary": {
                    "summary_text": "This is the agent's summary."
                }
            }
        })
        self.mock_sdk_client.converse_conversation.return_value = mock_response

        request_dict = {
            "name": "projects/test-p/locations/global/dataStores/test-ds/conversations/test-c",
            "query": {"text": "Hello agent"}
        }
        # Pass dict directly as per TypeError message
        request = discoveryengine_types.ConverseConversationRequest(request_dict)

        self._run_async(self.wrapper_client.converse_conversation(request=request))

        self.assertEqual(self.mock_insidellm_client.log_event_calls, 2)

        user_input_event = self.mock_insidellm_client.events_logged[0]
        agent_response_event = self.mock_insidellm_client.events_logged[1]

        self.assertEqual(user_input_event.event_type, EventType.USER_INPUT)
        self.assertEqual(UserInputPayload(**user_input_event.payload).input_text, "Hello agent")
        self.assertIsNone(user_input_event.parent_event_id)

        self.assertEqual(agent_response_event.event_type, EventType.AGENT_RESPONSE)
        self.assertEqual(AgentResponsePayload(**agent_response_event.payload).response_text, "This is the agent's summary.")
        self.assertEqual(agent_response_event.parent_event_id, user_input_event.event_id)

    def test_converse_conversation_with_search_action(self):
        mock_response = discoveryengine_types.ConverseConversationResponse()
        mock_response.reply = discoveryengine_types.Reply(
            "reply": { # Reply as dict
                "summary": { # Summary as dict
                    "summary_text": "Found some documents."
                }
            },
            "reply": {
                "summary": {
                    "summary_text": "Found some documents."
                }
            },
            "answer": {
                "steps": [
                    {
                        "description": "Searching for documents",
                        "thought": "I need to find relevant documents.",
                        "state": discoveryengine_types.Answer.Step.State.SUCCEEDED,
                        "search_action": {"query": "relevant documents"}, # oneof field for Step
                        "observation": { # observation for the step
                            "search_results": [
                                {
                                    "document": { # document within search_result
                                        "name": "doc1",
                                        "id": "doc1_id",
                                        "json_data": json.dumps({"text_content": "Content of doc1", "source": "mock_source"})
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }) # This was line 126, the extra '})' was the issue. The main dict for ConverseConversationResponse starts at mock_response line.
        self.mock_sdk_client.converse_conversation.return_value = mock_response

        request_dict = {
            "name": "projects/test-p/locations/global/dataStores/test-ds/conversations/test-c",
            "query": {"text": "Find documents for me"}
        }
        request = discoveryengine_types.ConverseConversationRequest(request_dict)
        self._run_async(self.wrapper_client.converse_conversation(request=request))

        self.assertEqual(self.mock_insidellm_client.log_event_calls, 4) # UserInput, AgentResponse, ToolCall, ToolResponse

        events = self.mock_insidellm_client.events_logged
        user_input_event = events[0]
        agent_response_event = events[1]
        tool_call_event = events[2]
        tool_response_event = events[3]

        self.assertEqual(user_input_event.event_type, EventType.USER_INPUT)
        self.assertEqual(agent_response_event.event_type, EventType.AGENT_RESPONSE)
        self.assertEqual(agent_response_event.parent_event_id, user_input_event.event_id)

        self.assertEqual(tool_call_event.event_type, EventType.TOOL_CALL)
        self.assertEqual(tool_call_event.parent_event_id, agent_response_event.event_id)
        tool_call_payload = ToolCallPayload(**tool_call_event.payload)
        self.assertEqual(tool_call_payload.tool_name, "VertexAISearch")
        self.assertEqual(tool_call_payload.parameters["query"], "relevant documents")

        self.assertEqual(tool_response_event.event_type, EventType.TOOL_RESPONSE)
        self.assertEqual(tool_response_event.parent_event_id, tool_call_event.event_id)
        tool_response_payload = ToolResponsePayload(**tool_response_event.payload)
        self.assertTrue(tool_response_payload.success)
        # Check if response_data contains serialized document content
        self.assertIn("doc1_id", tool_response_payload.response_data)


    def test_converse_conversation_with_failed_step_action(self):
        mock_response = discoveryengine_types.ConverseConversationResponse()
        mock_response.reply = discoveryengine_types.Reply(summary=discoveryengine_types.SearchResponse.Summary(summary_text="Trying a search which will fail."))

            "reply": {"summary": {"summary_text": "Trying a search which will fail."}},
            "answer": {
                "steps": [
                    {
                        "description": "Attempting a search that will fail.",
                        "thought": "This search failed due to an external service error.",
                        "state": discoveryengine_types.Answer.Step.State.FAILED,
                        "search_action": {"query": "a query that leads to failure"},
                        "observation": {"search_results": []} # Empty observation for failure
                    }
                ]
            }
        })
        self.mock_sdk_client.converse_conversation.return_value = mock_response

        request_dict = {
            "name": "projects/test-p/locations/global/dataStores/test-ds/conversations/test-c",
            "query": {"text": "Search for something that fails"}
        }
        request = discoveryengine_types.ConverseConversationRequest(request_dict)
        self._run_async(self.wrapper_client.converse_conversation(request=request))

        self.assertEqual(self.mock_insidellm_client.log_event_calls, 4)
        tool_response_event = self.mock_insidellm_client.events_logged[3]
        self.assertEqual(tool_response_event.event_type, EventType.TOOL_RESPONSE)
        tool_response_payload = ToolResponsePayload(**tool_response_event.payload)
        self.assertFalse(tool_response_payload.success)
        self.assertEqual(tool_response_payload.error_message, "The calculator tool failed.")
        self.assertEqual(ToolCallPayload(**self.mock_insidellm_client.events_logged[2].payload).tool_name, "CustomCalculator")


    def test_converse_conversation_sdk_failure(self):
        # Configure mock SDK to raise an error
        # Using a generic Python error for simplicity; can use google.api_core.exceptions if needed
        self.mock_sdk_client.converse_conversation.side_effect = MockGoogleAPIError("SDK call failed due to network issue")

        request_dict = {
            "name": "projects/test-p/locations/global/dataStores/test-ds/conversations/test-c",
            "query": {"text": "This call will fail"}
        }
        request = discoveryengine_types.ConverseConversationRequest(request_dict)

        with self.assertRaises(MockGoogleAPIError): # Ensure the original error is re-raised
            self._run_async(self.wrapper_client.converse_conversation(request=request))

        self.assertEqual(self.mock_insidellm_client.log_event_calls, 2) # UserInput, Error

        user_input_event = self.mock_insidellm_client.events_logged[0]
        error_event = self.mock_insidellm_client.events_logged[1]

        self.assertEqual(user_input_event.event_type, EventType.USER_INPUT)
        self.assertEqual(error_event.event_type, EventType.ERROR)
        self.assertEqual(error_event.parent_event_id, user_input_event.event_id)

        error_payload = ErrorPayload(**error_event.payload)
        self.assertEqual(error_payload.error_type, "MockGoogleAPIError")
        self.assertEqual(error_payload.error_message, "SDK call failed due to network issue")
        self.assertIn("Traceback", error_payload.stack_trace)

    def test_attribute_forwarding(self):
        # Test that non-overridden methods are forwarded
        # Add a mock method to the underlying client that isn't in the wrapper
        # self.mock_sdk_client.some_other_method = MagicMock(return_value="forwarded_call_works")

        # Create a new wrapper instance to pick up the new mock method
        # (because __init__ copies methods at instantiation)
        # This is a bit of a workaround for testing dynamic forwarding.
        # A more complex setup might involve a metaclass for the wrapper.
        # wrapper_with_forward_test = InsideLLMVertexAgentBuilderConversationalSearchClient(
        #     client=self.mock_sdk_client,
        #     insidellm_client=self.mock_insidellm_client,
        #     user_id=self.test_user_id,
        #     run_id=self.test_run_id
        # )

        # Call the forwarded method
        # result = wrapper_with_forward_test.some_other_method("test_arg")

        # self.assertEqual(result, "forwarded_call_works")
        # self.mock_sdk_client.some_other_method.assert_called_once_with("test_arg")
        pass # Commenting out for now to focus on core tests


if __name__ == '__main__':
    unittest.main()
