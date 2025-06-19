# InsideLLM Python SDK

A comprehensive Python SDK for LLM/Agent analytics that provides asynchronous event ingestion with LangChain integration and custom agent support.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/insidellm.svg)](https://badge.fury.io/py/insidellm)

## Features

- **Asynchronous Event Processing**: Non-blocking event queuing with configurable batch processing
- **LangChain Integration**: Automatic tracking for LangChain agents, tools, and LLM calls
- **Custom Agent Support**: Decorators and context managers for easy integration with any agent framework
- **Comprehensive Event Types**: Support for 15+ essential event types covering the full LLM/Agent lifecycle
- **Robust Error Handling**: Retry mechanisms with exponential backoff and graceful failure handling
- **Performance Monitoring**: Built-in metrics and queue statistics
- **Thread-Safe Operations**: Designed for concurrent usage in multi-threaded environments

## Installation

### From PyPI

```bash
pip install insidellm
```

### With LangChain support

```bash
pip install insidellm[langchain]
```

### Development installation

```bash
git clone https://github.com/insidellm/python-sdk.git
cd python-sdk
pip install -e .[dev]
```

## Quick Start

### Basic Usage

```python
import insidellm

# Initialize the SDK
insidellm.initialize(api_key="your-api-key")

# Start a session
client = insidellm.get_client()
run_id = client.start_run(user_id="user-123")

# Log events
user_event = insidellm.Event.create_user_input(
    run_id=run_id,
    user_id="user-123",
    input_text="Hello, AI assistant!"
)
client.log_event(user_event)

# Events are automatically batched and sent asynchronously
# End the session
client.end_run(run_id)
insidellm.shutdown()
```

### LangChain Integration

```python
import insidellm
from langchain.llms import OpenAI
from langchain.agents import initialize_agent

# Initialize InsideLLM
insidellm.initialize(api_key="your-api-key")

# Create LangChain callback
callback = insidellm.InsideLLMCallback(
    client=insidellm.get_client(),
    user_id="user-123"
)

# Use with any LangChain component
llm = OpenAI(callbacks=[callback])
agent = initialize_agent(tools, llm, callbacks=[callback])

# All LLM calls, tool usage, and agent actions are automatically tracked
response = agent.run("What's the weather like today?")
```

## Supported Frameworks / Integrations

Beyond LangChain, the InsideLLM SDK provides or is developing integrations for other popular agent and LLM frameworks.

### CrewAI Integration

The `CrewAIInsideLLMCallbackHandler` allows you to capture LLM events from your CrewAI agent executions. Since CrewAI uses LiteLLM for its LLM interactions, this handler is based on LiteLLM's `CustomLogger` interface.

**Usage:**

1.  **Initialize `InsideLLMClient`** as shown in the basic usage.
2.  **Create `CrewAIInsideLLMCallbackHandler`**: Pass your `InsideLLMClient` instance to it.
3.  **Register with LiteLLM**: Append the handler to `litellm.callbacks` list *before* your CrewAI crew starts making LLM calls.

```python
from insidellm import InsideLLMClient, CrewAIInsideLLMCallbackHandler
import litellm # CrewAI uses LiteLLM for its LLM interactions
# from crewai import Agent, Task, Crew # Your CrewAI imports

# 1. Initialize InsideLLMClient
# insidellm.initialize(api_key="YOUR_INSIDELLM_API_KEY") # If using global client
# client = insidellm.get_client()
# For direct client instantiation:
client = InsideLLMClient(api_key="YOUR_INSIDELLM_API_KEY")

# 2. Create the callback handler
# The handler manages its own run_id and user_id if not provided by a broader context.
# You can pass specific run_id and user_id if your application manages them.
crewai_callback_handler = CrewAIInsideLLMCallbackHandler(
    client=client,
    # run_id="your_specific_run_id", # Optional: if you want to tie to an existing run
    # user_id="your_specific_user_id", # Optional
    debug=True
)

# 3. Register with LiteLLM's global callback list
if not hasattr(litellm, 'callbacks') or not litellm.callbacks:
    litellm.callbacks = []
if not any(isinstance(cb, CrewAIInsideLLMCallbackHandler) for cb in litellm.callbacks):
    litellm.callbacks.append(crewai_callback_handler)
    print("CrewAIInsideLLMCallbackHandler registered with LiteLLM.")

# 4. Set up and run your CrewAI agents and crew as usual
# Example:
# research_agent = Agent(role='Researcher', goal='...', llm=...)
# research_task = Task(description='...', agent=research_agent)
# my_crew = Crew(agents=[research_agent], tasks=[research_task])
# result = my_crew.kickoff()
# Events related to LLM calls made by CrewAI agents will be captured.
```

*Note: Effective capturing of events depends on CrewAI's internal usage of LiteLLM and the timing of callback registration. Ensure registration happens before LLM calls are made by agents.*

### Google Vertex AI Agent Builder Integration

The `InsideLLMVertexAgentBuilderConversationalSearchClient` acts as a wrapper around the official `google.cloud.discoveryengine_v1.ConversationalSearchServiceAsyncClient`. It intercepts calls to methods like `converse_conversation` to log events.

**Captured Events:**
- User input (`user.input`)
- Agent's final response (`agent.response`)
- Tool calls and responses (`tool.call`, `tool.response`) parsed from the conversation flow, particularly from `Answer.Step` structures (e.g., `SearchAction` and observations).

**Usage:**

1.  **Initialize `InsideLLMClient`**.
2.  **Initialize the Google SDK's `ConversationalSearchServiceAsyncClient`**.
3.  **Wrap the SDK client**: Pass the SDK client, your `InsideLLMClient` instance, a `user_id`, and a `run_id` to the `InsideLLMVertexAgentBuilderConversationalSearchClient` constructor.
4.  Use the wrapped client in place of the original SDK client.

```python
from google.cloud.discoveryengine_v1 import ConversationalSearchServiceAsyncClient
from insidellm import InsideLLMClient, InsideLLMVertexAgentBuilderConversationalSearchClient
# from insidellm import initialize # If using global client
# from google.cloud.discoveryengine_v1.types import ConverseConversationRequest, Query # For making requests
# import asyncio

# async def run_vertex_example():
#     # 1. Initialize InsideLLMClient
#     # initialize(api_key="YOUR_INSIDELLM_API_KEY")
#     # inside_llm_client = get_client()
#     inside_llm_client = InsideLLMClient(api_key="YOUR_INSIDELLM_API_KEY")

#     # 2. Initialize the original Google SDK client (ensure GOOGLE_APPLICATION_CREDENTIALS is set)
#     google_sdk_client = ConversationalSearchServiceAsyncClient() # Add your client config if needed

#     # 3. Define user_id and run_id for the session/interaction
#     user_id = "example_vertex_user_456"
#     run_id = "vertex_interaction_run_001" # Manage this ID for your application's session

#     # 4. Wrap the Google SDK client
#     wrapped_client = InsideLLMVertexAgentBuilderConversationalSearchClient(
#         client=google_sdk_client,
#         insidellm_client=inside_llm_client,
#         user_id=user_id,
#         run_id=run_id,
#         debug=True
#     )

#     # 5. Use the wrapped client as you would the original SDK client
#     # request = ConverseConversationRequest(
#     #     name="projects/YOUR_PROJECT/locations/global/dataStores/YOUR_DATASTORE/conversations/-",
#     #     query=Query(text="Hello Vertex AI Agent!")
#     # )
#     # try:
#     #     response = await wrapped_client.converse_conversation(request=request)
#     #     if response.reply and response.reply.summary:
#     #         print(f"Agent response: {response.reply.summary.summary_text}")
#     # except Exception as e:
#     #     print(f"An error occurred: {e}")
#     # finally:
#     #     await google_sdk_client.close() # Close the original client when done
#     #     # Consider if wrapper needs a close method or if original client closure is sufficient

# # if __name__ == "__main__":
# # asyncio.run(run_vertex_example())
```

*Note: The wrapper intercepts specific methods like `converse_conversation`. Other methods of the SDK client are forwarded dynamically but won't have specific event logging unless added to the wrapper.*

### Custom Agent Integration

#### Using Decorators

```python
import insidellm

@insidellm.track_llm_call("gpt-4", "openai")
def call_llm(prompt):
    # Your LLM call logic
    return llm_response

@insidellm.track_tool_use("web_search", "api")
def search_web(query):
    # Your tool logic
    return search_results

@insidellm.track_agent_step("planning")
def plan_task(task):
    # Your planning logic
    return plan
```

#### Using Context Managers

```python
import insidellm

with insidellm.InsideLLMTracker(user_id="user-123") as tracker:
    # Log user input
    input_id = tracker.log_user_input("Hello, how can you help?")
    
    # Track LLM calls
    with tracker.track_llm_call("gpt-4", "openai", "User greeting") as log_response:
        response = call_llm("User greeting")
        log_response(response)
    
    # Track tool usage
    with tracker.track_tool_call("calculator", {"expression": "2+2"}) as log_response:
        result = calculator("2+2")
        log_response(result)
    
    # Log agent response
    tracker.log_agent_response("Hello! I can help with many tasks.", parent_event_id=input_id)
```

## Event Types

The SDK supports 15 comprehensive event types:

### User Interaction
- `user_input` - User inputs and queries
- `user_feedback` - User feedback and ratings

### Agent Processing
- `agent_reasoning` - Agent reasoning steps
- `agent_planning` - Agent planning processes
- `agent_response` - Agent responses

### LLM Operations
- `llm_request` - LLM API requests
- `llm_response` - LLM API responses
- `llm_streaming_chunk` - Streaming response chunks

### Tool/Function Calls
- `tool_call` - Tool invocations
- `tool_response` - Tool results
- `function_execution` - Function executions

### External APIs
- `api_request` - External API calls
- `api_response` - External API responses

### Error Handling
- `error` - General errors
- `validation_error` - Validation failures
- `timeout_error` - Timeout events

### System Events
- `session_start` - Session initiation
- `session_end` - Session completion
- `performance_metric` - Performance measurements

## Configuration

### Environment Variables

```bash
export INSIDELLM_API_KEY="your-api-key"
export INSIDELLM_BATCH_SIZE=50
export INSIDELLM_AUTO_FLUSH_INTERVAL=30.0
export INSIDELLM_MAX_RETRIES=3
```

### Programmatic Configuration

```python
import insidellm

config = insidellm.InsideLLMConfig(
    max_queue_size=10000,
    batch_size=50,
    auto_flush_interval=30.0,
    request_timeout=30.0,
    max_retries=3,
    raise_on_error=False
)

insidellm.initialize(api_key="your-api-key", config=config)
```

## Advanced Usage

### Manual Event Creation

```python
import insidellm
from insidellm import Event, EventType

# Create custom events
event = Event(
    run_id="your-run-id",
    user_id="user-123",
    event_type=EventType.PERFORMANCE_METRIC,
    payload={
        "metric_name": "response_time",
        "metric_value": 250,
        "metric_unit": "ms"
    }
)

client = insidellm.get_client()
client.log_event(event)
```

### Error Handling

```python
import insidellm

try:
    # Your agent logic
    result = process_user_request()
except Exception as e:
    # Log errors automatically
    error_event = insidellm.Event.create_error(
        run_id=current_run_id,
        user_id=current_user_id,
        error_type="processing_error",
        error_message=str(e),
        error_code=type(e).__name__
    )
    client.log_event(error_event)
```

### Performance Monitoring

```python
import insidellm

client = insidellm.get_client()

# Get queue statistics
stats = client.queue_manager.get_statistics()
print(f"Events queued: {stats['events_queued']}")
print(f"Success rate: {stats['success_rate']:.1f}%")

# Check client health
if client.is_healthy():
    print("Client is healthy")
```

## Examples

The repository includes comprehensive examples:

- [`examples/basic_usage.py`](examples/basic_usage.py) - Basic SDK functionality
- [`examples/langchain_example.py`](examples/langchain_example.py) - LangChain integration
- [`examples/custom_agent_example.py`](examples/custom_agent_example.py) - Custom agent integration

## API Reference

### Core Classes

- `InsideLLMClient` - Main client for event logging
- `Event` - Event data model
- `InsideLLMTracker` - Context manager for workflow tracking
- `InsideLLMCallback` - LangChain callback handler (from `.langchain_integration`)
- `CrewAIInsideLLMCallbackHandler` - Callback handler for CrewAI (via LiteLLM)
- `InsideLLMVertexAgentBuilderConversationalSearchClient` - Wrapper for Vertex AI Agent Builder SDK client

### Decorators

- `@track_llm_call()` - Automatic LLM call tracking
- `@track_tool_use()` - Automatic tool usage tracking
- `@track_agent_step()` - Automatic agent step tracking

### Configuration

- `InsideLLMConfig` - Configuration management
- Environment variable support for all settings

## Requirements

- Python 3.8+
- `pydantic >= 2.0.0`
- `requests >= 2.25.0`
- `langchain >= 0.1.0` (optional, for LangChain integration)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Documentation: https://docs.insidellm.com/python-sdk
- Issues: https://github.com/insidellm/python-sdk/issues
- Email: support@insidellm.com
