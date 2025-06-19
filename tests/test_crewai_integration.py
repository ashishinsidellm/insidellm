import unittest
from unittest.mock import MagicMock, patch
import asyncio
import json
from uuid import uuid4, UUID
from datetime import datetime
from typing import Optional, List, Any, Dict # Added Optional and others for type hints

from insidellm.crewai_integration import CrewAIInsideLLMCallbackHandler
from insidellm.models import Event, EventType, LLMRequestPayload, LLMResponsePayload, ErrorPayload
# Assuming the placeholder InsideLLMClient is what we want to mock for testing the handler's output
# If there's a real client to be mocked, the approach might differ slightly.

class MockInsideLLMClient:
    def __init__(self):
        self.events_logged: list[Event] = []
        self.log_event_calls = 0

    def log_event(self, event: Event):
        self.events_logged.append(event)
        self.log_event_calls += 1
        # Correctly access fields from insidellm.models.Event
        print(f"MockInsideLLMClient: Logged event: {event.event_type.value}, ID: {event.event_id}, Parent: {event.parent_event_id}")

# Mocking LiteLLM's ModelResponse structure for tests
# Based on common LiteLLM response structure.
class MockLiteLLMChoiceMessage:
    def __init__(self, content: str):
        self.content = content
        self.role = "assistant"

class MockLiteLLMChoice:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.message = MockLiteLLMChoiceMessage(content=content)
        self.finish_reason = finish_reason
        self.index = 0

class MockLiteLLMUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int = None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens if total_tokens is not None else prompt_tokens + completion_tokens

class MockLiteLLMResponse:
    def __init__(self, id: str, model: str, content: str, prompt_tokens: int, completion_tokens: int,
                 custom_llm_provider: str = None, finish_reason: str = "stop", cost: Optional[float] = None):
        self.id = id or f"chatcmpl-{uuid4()}"
        self.model = model
        self.choices = [MockLiteLLMChoice(content=content, finish_reason=finish_reason)]
        self.usage = MockLiteLLMUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        self.created = int(datetime.now().timestamp())
        self.object = "chat.completion"
        if custom_llm_provider: # LiteLLM includes this for some models
             self._hidden_params = {"custom_llm_provider": custom_llm_provider}
        if cost is not None:
            self.cost = cost


class TestCrewAIInsideLLMIntegration(unittest.TestCase):

    def setUp(self):
        self.mock_insidellm_client = MockInsideLLMClient()
        # The handler generates its own run_id and user_id for now
        self.handler = CrewAIInsideLLMCallbackHandler(client=self.mock_insidellm_client, debug=True)
        # Override the generated run_id and user_id for predictable testing
        self.test_run_id = str(uuid4())
        self.test_user_id = "test_crewai_user"
        self.handler.current_run_id = self.test_run_id
        self.handler.current_user_id = self.test_user_id
        self.handler._event_stack = [] # Ensure stack is clean before each test

    def tearDown(self):
        self.handler._event_stack = [] # Clean stack after each test

    def test_get_provider_from_model(self):
        self.assertEqual(self.handler._get_provider_from_model("gpt-3.5-turbo"), "openai")
        self.assertEqual(self.handler._get_provider_from_model("openai/gpt-4"), "openai")
        self.assertEqual(self.handler._get_provider_from_model("claude-2"), "anthropic")
        self.assertEqual(self.handler._get_provider_from_model("anthropic/claude-3-opus"), "anthropic")
        self.assertEqual(self.handler._get_provider_from_model("gemini-pro"), "google")
        self.assertEqual(self.handler._get_provider_from_model("google/gemini-1.5-flash"), "google")
        self.assertEqual(self.handler._get_provider_from_model("mistral/mistral-large-latest"), "mistral")
        self.assertEqual(self.handler._get_provider_from_model("cohere.command-r-plus"), "cohere")
        self.assertEqual(self.handler._get_provider_from_model("some-other-model"), "unknown_provider")
        self.assertEqual(self.handler._get_provider_from_model(""), "unknown")


    def test_async_log_pre_api_call(self):
        model_name = "gpt-4-test"
        messages = [{"role": "user", "content": "Hello, world!"}]
        mock_kwargs = {
            "model": model_name,
            "messages": messages,
            "litellm_params": {
                "metadata": {
                    "crew_name": "TestCrew",
                    "agent_name": "TestAgent",
                    "task_name": "TestTask"
                },
                "model_info": {"temperature": 0.7}
            },
            "stream": False
        }

        asyncio.run(self.handler.async_log_pre_api_call(model=model_name, messages=messages, kwargs=mock_kwargs))

        self.assertEqual(self.mock_insidellm_client.log_event_calls, 1)
        self.assertEqual(len(self.handler._event_stack), 1)

        event = self.mock_insidellm_client.events_logged[0]
        self.assertEqual(event.event_type, EventType.LLM_REQUEST) # Corrected: event_type
        self.assertEqual(event.run_id, self.test_run_id)
        self.assertEqual(event.user_id, self.test_user_id)
        self.assertIsNone(event.parent_event_id) # Corrected: parent_event_id

        payload = LLMRequestPayload(**event.payload)
        self.assertEqual(payload.model_name, model_name)
        self.assertEqual(payload.provider, "openai") # From _get_provider_from_model
        self.assertEqual(payload.prompt, json.dumps(messages, default=str))
        self.assertEqual(payload.messages, messages)
        # Corrected assertions for parameters:
        # payload.parameters is what was in "model_info" or "litellm_params"
        self.assertIn("temperature", payload.parameters)
        self.assertEqual(payload.parameters["temperature"], 0.7)

        self.assertIn("crewai_metadata", event.metadata)
        crew_meta = json.loads(event.metadata["crewai_metadata"]) # Parse JSON string
        self.assertEqual(crew_meta["agent_name"], "TestAgent")
        self.assertEqual(event.metadata["streaming"], False)

    def test_async_log_pre_api_call_no_crew_metadata(self):
        model_name = "claude-instant-1.2"
        messages = [{"role": "user", "content": "No metadata test"}]
        mock_kwargs = {
            "model": model_name,
            "messages": messages,
            "litellm_params": {} # No metadata
        }
        asyncio.run(self.handler.async_log_pre_api_call(model=model_name, messages=messages, kwargs=mock_kwargs))
        self.assertEqual(self.mock_insidellm_client.log_event_calls, 1) # This will be checked after fixing the mock client's print
        event = self.mock_insidellm_client.events_logged[0]
        self.assertEqual(event.event_type, EventType.LLM_REQUEST) # Corrected: event_type
        payload = LLMRequestPayload(**event.payload)
        self.assertEqual(payload.model_name, model_name)
        self.assertEqual(payload.provider, "anthropic")
        self.assertNotIn("crewai_metadata", event.metadata)


    def test_async_log_success_event(self):
        # 1. Log a pre_api_call first to put an event on the stack
        pre_call_model = "gpt-3.5-turbo-test"
        pre_call_messages = [{"role": "user", "content": "Test for success"}]
        pre_call_kwargs = {
            "model": pre_call_model,
            "messages": pre_call_messages,
            "litellm_params": {"metadata": {"agent_name": "SuccessAgent"}}
        }
        asyncio.run(self.handler.async_log_pre_api_call(model=pre_call_model, messages=pre_call_messages, kwargs=pre_call_kwargs))

        # If the previous call to log_event in the mock was fixed, log_event_calls should be 1 here.
        # If it's 2, it means the error logging path in _log_event was still triggered.
        self.assertEqual(self.mock_insidellm_client.log_event_calls, 1, "Pre-call should log only one event if mock is fixed.")
        pre_call_event_id = self.mock_insidellm_client.events_logged[0].event_id # Corrected: event_id
        self.assertEqual(len(self.handler._event_stack), 1)

        # 2. Log a success event
        response_model_name = "gpt-3.5-turbo-0125-test" # LiteLLM might return a more specific model
        response_content = "This is a successful response."
        mock_response_obj = MockLiteLLMResponse(
            id="resp-123",
            model=response_model_name,
            content=response_content,
            prompt_tokens=50,
            completion_tokens=25,
            cost=0.00015
        )

        success_kwargs = {
            "model": pre_call_model, # Original model in request
            "input": pre_call_messages, # LiteLLM passes 'input' in success/failure kwargs
            "litellm_params": {"metadata": {"agent_name": "SuccessAgent"}},
            "cost": mock_response_obj.cost
        }

        asyncio.run(self.handler.async_log_success_event(
            kwargs=success_kwargs,
            response_obj=mock_response_obj,
            start_time=1700000000.0,
            end_time=1700000001.5
        ))

        # After success, total events should be 2 (1 pre-call, 1 success)
        self.assertEqual(self.mock_insidellm_client.log_event_calls, 2, "Success event should make total calls 2.")
        self.assertEqual(len(self.handler._event_stack), 0) # Stack should be empty

        success_event = self.mock_insidellm_client.events_logged[1]
        self.assertEqual(success_event.event_type, EventType.LLM_RESPONSE) # Corrected: event_type
        self.assertEqual(success_event.parent_event_id, pre_call_event_id) # Corrected: parent_event_id

        payload = LLMResponsePayload(**success_event.payload)
        self.assertEqual(payload.model_name, response_model_name)
        self.assertEqual(payload.provider, "openai")
        self.assertEqual(payload.response_text, response_content)
        self.assertIsNotNone(payload.usage)
        self.assertEqual(payload.usage["input_tokens"], 50)
        self.assertEqual(payload.usage["output_tokens"], 25)
        self.assertEqual(payload.cost, 0.00015)
        self.assertEqual(payload.response_time_ms, 1500)

        self.assertIn("crewai_metadata", success_event.metadata)
        crew_meta_success = json.loads(success_event.metadata["crewai_metadata"]) # Parse JSON string
        self.assertEqual(crew_meta_success["agent_name"], "SuccessAgent")


    def test_async_log_failure_event_with_exception(self):
        # 1. Log a pre_api_call
        pre_call_model = "gemini-pro-test"
        pre_call_messages = [{"role": "user", "content": "Test for failure"}]
        pre_call_kwargs = {"model": pre_call_model, "messages": pre_call_messages, "litellm_params": {}}
        asyncio.run(self.handler.async_log_pre_api_call(model=pre_call_model, messages=pre_call_messages, kwargs=pre_call_kwargs))

        self.assertEqual(self.mock_insidellm_client.log_event_calls, 1, "Pre-call for failure test should log one event.")
        pre_call_event_id = self.mock_insidellm_client.events_logged[0].event_id # Corrected: event_id
        self.assertEqual(len(self.handler._event_stack), 1)

        # 2. Log a failure event
        exception_obj = ValueError("API Key Invalid")

        failure_kwargs = {
            "model": pre_call_model,
            "input": pre_call_messages,
            "litellm_params": {"metadata": {"agent_name": "FailureAgent"}}
        }

        asyncio.run(self.handler.async_log_failure_event(
            kwargs=failure_kwargs,
            response_obj=exception_obj, # LiteLLM passes the exception object here
            start_time=1700000002.0,
            end_time=1700000002.5
        ))

        self.assertEqual(self.mock_insidellm_client.log_event_calls, 2, "Failure event should make total calls 2.")
        self.assertEqual(len(self.handler._event_stack), 0)

        error_event = self.mock_insidellm_client.events_logged[1]
        self.assertEqual(error_event.event_type, EventType.ERROR) # Corrected: event_type
        self.assertEqual(error_event.parent_event_id, pre_call_event_id) # Corrected: parent_event_id

        payload = ErrorPayload(**error_event.payload)
        self.assertEqual(payload.error_type, "ValueError")
        self.assertEqual(payload.error_message, "API Key Invalid")
        # Check for essential parts of the formatted exception, not necessarily "Traceback (most recent call last):"
        self.assertIn("ValueError: API Key Invalid", payload.stack_trace)
        self.assertIn("model_name", payload.context)
        self.assertEqual(payload.context["model_name"], pre_call_model)
        self.assertEqual(payload.context["error_source"], "LiteLLMCall") # Corrected assertion

        self.assertIn("crewai_metadata", error_event.metadata)
        crew_meta_failure = json.loads(error_event.metadata["crewai_metadata"]) # Parse JSON string
        self.assertEqual(crew_meta_failure["agent_name"], "FailureAgent")


    def test_async_log_failure_event_with_dict_response(self):
        # Test scenario where LiteLLM might pass a dict as response_obj on failure
        pre_call_model = "some-custom-llm/test-failure-dict"
        pre_call_messages = [{"role": "user", "content": "Test for dict failure"}]
        pre_call_kwargs = {"model": pre_call_model, "messages": pre_call_messages, "litellm_params": {}}
        asyncio.run(self.handler.async_log_pre_api_call(model=pre_call_model, messages=pre_call_messages, kwargs=pre_call_kwargs))
        self.assertEqual(self.mock_insidellm_client.log_event_calls, 1, "Pre-call for dict failure test should log one event.")
        pre_call_event_id = self.mock_insidellm_client.events_logged[0].event_id # Corrected: event_id

        failure_response_dict = {
            "error": "Some error occurred",
            "status_code": 500,
            "traceback": "Fake traceback string for dict response"
        }
        failure_kwargs = {"model": pre_call_model, "input": pre_call_messages, "litellm_params": {}}
        asyncio.run(self.handler.async_log_failure_event(
            kwargs=failure_kwargs,
            response_obj=failure_response_dict,
            start_time=1700000003.0,
            end_time=1700000003.5
        ))

        self.assertEqual(self.mock_insidellm_client.log_event_calls, 2, "Dict failure event should make total calls 2.")
        error_event = self.mock_insidellm_client.events_logged[1]
        self.assertEqual(error_event.event_type, EventType.ERROR) # Corrected: event_type
        payload = ErrorPayload(**error_event.payload)
        self.assertEqual(payload.error_type, "LiteLLMError") # Default type for non-Exception
        self.assertTrue(failure_response_dict["error"] in payload.error_message)
        self.assertEqual(payload.stack_trace, "Fake traceback string for dict response")
        self.assertEqual(payload.context["error_source"], "LiteLLMCall") # Added assertion for context


if __name__ == '__main__':
    unittest.main()
