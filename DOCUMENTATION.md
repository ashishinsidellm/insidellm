# InsideLLM SDK Documentation

## Introduction

The InsideLLM SDK provides a comprehensive toolkit for developers to integrate powerful Large Language Model (LLM) observability and analytics into their applications. This SDK allows you to track various events related to your LLM interactions, agent behavior, tool usage, and user interactions, sending this data to the InsideLLM platform for monitoring, debugging, and performance analysis.

By using the InsideLLM SDK, you can:
- Gain deep insights into your LLM application's behavior.
- Monitor requests, responses, and errors from LLM providers.
- Track the execution flow of agents and tools.
- Log custom events and metadata tailored to your application's needs.
- Efficiently batch and send event data asynchronously.
- Utilize convenient decorators for automatic event tracking with minimal code changes.

## Installation

To get started with the InsideLLM SDK, you can install it using pip:

```bash
pip install insidellm
```

Make sure you have Python 3.8 or higher installed.

## Initialization and Configuration

Before you can start logging events, you need to initialize the InsideLLM SDK. This is typically done once when your application starts.

### Basic Initialization

The primary way to initialize the SDK is by calling the `insidellm.initialize()` function:

```python
import insidellm
import os

insidellm.initialize(
    api_key="YOUR_API_KEY" # Alternatively, set INSIDELLM_API_KEY environment variable
    # base_url="https://api.insidellm.com" # Optional: Defaults to the production URL
)
```

**Parameters for `insidellm.initialize()`:**

*   `api_key` (str, required): Your InsideLLM API key. You can pass this directly or set the `INSIDELLM_API_KEY` environment variable. The function will automatically pick it up if the environment variable is set and `api_key` is not provided.
*   `base_url` (str, optional): The base URL for the InsideLLM API. Defaults to `https://api.insidellm.com`. You might change this for development or testing environments.
*   `config` (InsideLLMConfig, optional): An instance of `InsideLLMConfig` for advanced configuration. If not provided, a default configuration is used.
*   `**kwargs`: You can also pass individual configuration parameters directly to `initialize()`, which will be used to create an `InsideLLMConfig` object internally. For example: `insidellm.initialize(api_key="YOUR_API_KEY", batch_size=100, auto_flush_interval=60)`.

### Advanced Configuration with `InsideLLMConfig`

For more granular control over the SDK's behavior, you can create an instance of `InsideLLMConfig` and pass it to `insidellm.initialize()`.

```python
from insidellm import InsideLLMConfig

custom_config = InsideLLMConfig(
    max_queue_size=5000,
    batch_size=100,
    auto_flush_interval=60.0,  # seconds
    request_timeout=45.0,     # seconds
    max_retries=5,
    backoff_factor=2.0,
    raise_on_error=False,     # If True, SDK might raise exceptions on network errors
    strict_validation=True,   # If True, event validation failures will raise errors
    log_level="INFO",
    enable_debug_logging=False
)

insidellm.initialize(
    api_key="YOUR_API_KEY",
    config=custom_config
)
```

**`InsideLLMConfig` Parameters:**

*   **Queue Management:**
    *   `max_queue_size` (int, default: 10000): Maximum number of events to hold in the internal queue before potential dropping.
    *   `batch_size` (int, default: 50): Number of events to group together in a single API request.
    *   `auto_flush_interval` (float, default: 30.0): Time in seconds after which the queue will be automatically flushed, even if `batch_size` is not reached.
*   **Network Settings:**
    *   `request_timeout` (float, default: 30.0): Timeout in seconds for HTTP requests to the InsideLLM API.
    *   `max_retries` (int, default: 3): Maximum number of times to retry sending a batch of events if a network error occurs.
    *   `backoff_factor` (float, default: 2.0): Factor by which the delay increases between retries (exponential backoff).
*   **Error Handling:**
    *   `raise_on_error` (bool, default: False): If `True`, network errors during event sending will be raised as exceptions (e.g., `NetworkError`). If `False`, errors are logged, and events might be dropped after retries are exhausted.
    *   `strict_validation` (bool, default: True): If `True`, events that fail Pydantic validation will raise a `ValueError`. If `False`, invalid events are logged and dropped.
*   **Logging:**
    *   `log_level` (str, default: "INFO"): Logging level for the SDK's internal logger (e.g., "DEBUG", "INFO", "WARNING", "ERROR").
    *   `enable_debug_logging` (bool, default: False): A shortcut to set the log level to "DEBUG".

### Configuration via Environment Variables

You can also configure many `InsideLLMConfig` parameters using environment variables. These are checked if you don't provide explicit values in code. The SDK uses `InsideLLMConfig.from_env()` internally if no direct config is supplied.

*   `INSIDELLM_API_KEY`: Your API key.
*   `INSIDELLM_BASE_URL`: The API base URL.
*   `INSIDELLM_MAX_QUEUE_SIZE`: Maximum queue size.
*   `INSIDELLM_BATCH_SIZE`: Batch size for event processing.
*   `INSIDELLM_AUTO_FLUSH_INTERVAL`: Auto flush interval in seconds.
*   `INSIDELLM_REQUEST_TIMEOUT`: Request timeout in seconds.
*   `INSIDELLM_MAX_RETRIES`: Maximum number of retries.
*   `INSIDELLM_BACKOFF_FACTOR`: Backoff factor for retries.
*   `INSIDELLM_RAISE_ON_ERROR`: Whether to raise on errors (true/false).
*   `INSIDELLM_STRICT_VALIDATION`: Whether to use strict validation (true/false).
*   `INSIDELLM_LOG_LEVEL`: Log level (DEBUG, INFO, WARNING, ERROR).
*   `INSIDELLM_ENABLE_DEBUG_LOGGING`: Enable debug logging (true/false).

If both environment variables and direct parameters (or `InsideLLMConfig` object) are provided, the direct parameters take precedence.

## Core Concepts

Understanding these core concepts is key to effectively using the InsideLLM SDK.

### Runs

A **Run** represents a distinct session or execution flow within your application that you want to track. All events logged during a run are associated with a unique `run_id`. This allows you to group related events together in the InsideLLM platform for analysis.

*   **`run_id` (str):** A unique identifier for the run. You can provide your own or let the SDK generate one (UUID).
*   **`user_id` (str):** An identifier for the user associated with this run. This helps in segmenting data by user.
*   **`metadata` (dict, optional):** You can attach key-value metadata to a run to provide additional context (e.g., session type, environment).

You typically start a run when a relevant process begins and end it when the process concludes.

```python
client = insidellm.get_client() # Assuming SDK is initialized

# Start a new run
current_run_id = client.start_run(
    user_id="user_12345",
    metadata={"application_version": "1.2.0", "mode": "production"}
)
print(f"Started run with ID: {current_run_id}")

# ... log events related to this run ...

# End the run
client.end_run(run_id=current_run_id) # You can omit run_id to end the current active run
print(f"Ended run: {current_run_id}")
```

If you start a new run while another is active, the new run becomes the "current" one for subsequent event logging if those events don't explicitly specify a `run_id`.

### Events

An **Event** is a single piece of telemetry data that you log using the SDK. Each event captures a specific occurrence in your application at a point in time.

The `Event` model has several key fields:

*   `event_id` (str): A unique UUID automatically generated for each event.
*   `run_id` (str): The ID of the run this event belongs to. If not explicitly set when creating an event, it defaults to the currently active `run_id` on the client.
*   `timestamp` (str): An ISO 8601 timestamp automatically generated when the event object is created, indicating when the event occurred.
*   `event_type` (EventType): The type of the event (see "Supported Event Types and Payloads" below). This is a crucial field that categorizes the event.
*   `user_id` (str): The ID of the user associated with this event. Defaults to the `user_id` of the current run if not set.
*   `parent_event_id` (str, optional): The `event_id` of a related parent event. This is useful for creating traces and understanding causal relationships between events (e.g., an `LLM_RESPONSE` event might have an `LLM_REQUEST` event as its parent).
*   `metadata` (dict, optional): A dictionary of custom key-value pairs (strings, numbers, booleans) to add extra information to the event. This metadata is merged with any run-level metadata.
*   `payload` (dict): A dictionary containing data specific to the `event_type`. The structure of the payload varies depending on the event type.

### EventType Enum

The `EventType` (available as `insidellm.EventType`) is an enumeration that defines all categories of events supported by the SDK. Examples include `USER_INPUT`, `LLM_REQUEST`, `TOOL_CALL`, `ERROR`, etc. Each `EventType` has an expected structure for its `payload`.

### Asynchronous Nature and Flushing

The InsideLLM SDK is designed to be non-blocking. When you call `client.log_event(event)`, the event is added to an internal in-memory queue. A background worker thread then processes this queue, batching events and sending them to the InsideLLM API.

*   **Automatic Flushing:** The SDK automatically flushes (sends) the queue based on `batch_size` (number of events) or `auto_flush_interval` (time elapsed), whichever condition is met first.
*   **Manual Flushing:** You can force the SDK to send all pending events immediately using `client.flush()`:
    ```python
    client.flush()
    ```
    This can be useful before your application exits or at critical points where you want to ensure data is sent.
*   **Shutdown:** It's important to properly shut down the SDK to ensure all pending events are flushed and resources are released. You can do this explicitly with `client.shutdown()` or `insidellm.shutdown()` (if using the global client).
    ```python
    client.shutdown(timeout=10.0) # Optional: wait up to 10 seconds for flush
    ```
    The `InsideLLMClient` also supports being used as a context manager, which handles shutdown automatically:
    ```python
    with insidellm.InsideLLMClient(api_key="YOUR_API_KEY") as client:
        # Use client to log events
        client.start_run(user_id="user_ctx_mgr")
        # ...
    # client.shutdown() is automatically called here
    ```

This asynchronous design minimizes the impact on your application's performance. However, be aware that if the application exits abruptly without a proper shutdown, some events in the queue might be lost.

## Using the SDK

This section describes how to obtain a client instance and log events manually. For automatic event tracking, see the "Using Decorators" section.

### Getting the Client Instance

Once the SDK is initialized using `insidellm.initialize()`, you can get the default client instance:

```python
import insidellm

# Ensure SDK is initialized first (see Initialization section)
# insidellm.initialize(api_key="YOUR_API_KEY")

client = insidellm.get_client()
```

Alternatively, you can instantiate `InsideLLMClient` directly if you need to manage multiple client instances or prefer explicit instantiation (though using the global client via `get_client()` is common for most use cases):

```python
from insidellm import InsideLLMClient, InsideLLMConfig

config = InsideLLMConfig(batch_size=10) # Example custom config
my_client = InsideLLMClient(api_key="YOUR_API_KEY", config=config)

# Remember to manage its lifecycle (e.g., my_client.shutdown())
```

### Logging Events Manually

The core method for logging events is `client.log_event(event)`. You first need to create an `Event` object.

**Creating an `Event` Object:**

You can create an `Event` object by providing its properties. The `insidellm.Event` model and `insidellm.EventType` enum are essential here.

```python
from insidellm import Event, EventType

# Example: Creating a custom event
custom_event = Event(
    run_id=client.get_current_run_id(), # Or a specific run_id
    user_id=client.get_current_user_id(), # Or a specific user_id
    event_type=EventType.PERFORMANCE_METRIC, # Refer to EventType enum
    payload={
        "metric_name": "database_query_time",
        "metric_value": 150.7,
        "metric_unit": "ms",
        "metric_type": "gauge",
        "additional_metrics": {
            "query_complexity": 5
        }
    },
    metadata={"db_host": "prod-db-1", "region": "us-east-1"}
)
```

**Helper Factory Methods for Common Events:**

The `Event` class also provides convenient factory methods for creating common event types, which can simplify event creation by pre-filling some fields and ensuring correct payload structures:

*   `Event.create_user_input(...)`
*   `Event.create_llm_request(...)`
*   `Event.create_llm_response(...)`
*   `Event.create_error(...)`

Example using a factory method:

```python
# Assumes client.start_run() has been called
llm_req_event = Event.create_llm_request(
    run_id=client.get_current_run_id(),
    user_id=client.get_current_user_id(),
    model_name="gpt-4o",
    provider="openai",
    prompt="What is the capital of France?",
    parameters={"temperature": 0.5},
    metadata={"request_source": "chatbot_ui"}
)
```

**Logging the Event:**

Once you have an `Event` object, pass it to `client.log_event()`:

```python
client.log_event(custom_event)
client.log_event(llm_req_event)
```

The event will be added to the internal queue for asynchronous processing and sending to the InsideLLM API.

**Key fields to consider when creating events:**

*   `run_id`: Essential for associating the event with the correct session. Often obtained from `client.get_current_run_id()`.
*   `user_id`: Important for user-centric analytics. Often obtained from `client.get_current_user_id()`.
*   `event_type`: Must be one of the values from `insidellm.EventType`.
*   `payload`: A dictionary whose structure depends on the `event_type`. See the next section for details on payloads for each event type.
*   `parent_event_id` (optional): Set this to the `event_id` of a preceding event to establish a causal link. For example, an `LLM_RESPONSE` event should have the `event_id` of its corresponding `LLM_REQUEST` as its `parent_event_id`.
*   `metadata` (optional): Add any custom contextual information relevant to the event.

The next section, "Supported Event Types and Payloads," provides detailed information on each `EventType` and the expected structure and fields for their respective `payload` objects.

### Supported Event Types and Payloads

The `insidellm.EventType` enum defines the various types of events you can log. Each event type has a specific `payload` structure, defined by Pydantic models in `insidellm.models`. Below is a detailed list of supported event types and their corresponding payload fields.

---

**1. `EventType.USER_INPUT` (`UserInputPayload`)**

*   **Purpose:** Logs input provided by a user to the system.
*   **Payload Fields:**
    *   `input_text` (str, required): The actual text or content of the user's input.
    *   `input_type` (str, default: "text"): The type of input (e.g., "text", "voice", "image", "button_click").
    *   `channel` (str, optional): The channel through which the input was received (e.g., "web", "mobile_app", "slack", "api").
    *   `session_context` (dict, optional): Any contextual information about the user's session at the time of input.

*   **Example Payload:**
    ```json
    {
        "input_text": "Tell me a joke",
        "input_type": "text",
        "channel": "web_chatbot",
        "session_context": {"page": "/chat", "language": "en"}
    }
    ```

---

**2. `EventType.USER_FEEDBACK` (`UserFeedbackPayload`)**

*   **Purpose:** Logs feedback provided by a user regarding a previous interaction or response.
*   **Payload Fields:**
    *   `feedback_type` (str, required): The type of feedback (e.g., "rating", "thumbs_up_down", "text_comment", "correction").
    *   `feedback_value` (str | int | float | bool, required): The actual value of the feedback (e.g., 5 for a 5-star rating, `True` for thumbs up, "The answer was incorrect." for text).
    *   `target_event_id` (str, optional): The `event_id` of the event being rated or commented on (e.g., the `event_id` of an `AGENT_RESPONSE` or `LLM_RESPONSE`).
    *   `feedback_text` (str, optional): Additional textual comment if the feedback type doesn't fully capture it.

*   **Example Payload:**
    ```json
    {
        "feedback_type": "thumbs_up_down",
        "feedback_value": true,
        "target_event_id": "event_id_of_agent_response_xyz",
        "feedback_text": "Very helpful!"
    }
    ```

---

**3. `EventType.AGENT_REASONING` (`AgentReasoningPayload`)**

*   **Purpose:** Logs internal reasoning steps or thoughts of an AI agent. Useful for understanding the agent's decision-making process.
*   **Payload Fields:**
    *   `reasoning_type` (str, required): The type or method of reasoning (e.g., "chain_of_thought", "tree_of_thought", "self_reflection", "planning_step").
    *   `reasoning_steps` (list[str], required): A list of strings, where each string describes a step in the reasoning process.
    *   `confidence_score` (float, optional): A score indicating the agent's confidence in its current reasoning path or conclusion.
    *   `reasoning_time_ms` (int, optional): Time taken for this reasoning step in milliseconds.

*   **Example Payload:**
    ```json
    {
        "reasoning_type": "chain_of_thought",
        "reasoning_steps": [
            "User asked for a flight to Paris.",
            "Identified intent: book_flight.",
            "Required parameters: origin, destination, date.",
            "Destination is Paris. Need origin and date."
        ],
        "confidence_score": 0.85,
        "reasoning_time_ms": 120
    }
    ```

---

**4. `EventType.AGENT_PLANNING` (`AgentPlanningPayload`)**

*   **Purpose:** Logs the plan generated by an agent, outlining future actions.
*   **Payload Fields:**
    *   `plan_type` (str, required): The type of plan (e.g., "sequential", "parallel", "hierarchical", "goal_oriented").
    *   `planned_actions` (list[dict], required): A list of dictionaries, where each dictionary describes a planned action (e.g., `{"action_name": "search_flights", "parameters": {...}}`).
    *   `planning_time_ms` (int, optional): Time taken to generate this plan in milliseconds.
    *   `plan_confidence` (float, optional): The agent's confidence in the generated plan.

*   **Example Payload:**
    ```json
    {
        "plan_type": "sequential",
        "planned_actions": [
            {"action_name": "ask_for_origin_city", "tool_to_use": null},
            {"action_name": "ask_for_departure_date", "tool_to_use": null},
            {"action_name": "search_flights_api", "tool_to_use": "flight_search_tool"}
        ],
        "planning_time_ms": 250
    }
    ```

---

**5. `EventType.AGENT_RESPONSE` (`AgentResponsePayload`)**

*   **Purpose:** Logs the final response generated by an agent to be presented to the user.
*   **Payload Fields:**
    *   `response_text` (str, required): The textual content of the agent's response.
    *   `response_type` (str, default: "text"): The type of response (e.g., "text", "action_request", "image_url", "function_call_suggestion").
    *   `response_confidence` (float, optional): The agent's confidence in its response.
    *   `response_metadata` (dict, optional): Any additional metadata associated with the response (e.g., sources used, suggested follow-up questions).

*   **Example Payload:**
    ```json
    {
        "response_text": "I can help you book a flight. What is your origin city?",
        "response_type": "text_with_question",
        "response_confidence": 0.92,
        "response_metadata": {"requires_follow_up": true}
    }
    ```

---

**6. `EventType.LLM_REQUEST` (`LLMRequestPayload`)**

*   **Purpose:** Logs a request made to a Large Language Model (LLM).
*   **Payload Fields:**
    *   `model_name` (str, required): The name or identifier of the LLM being called (e.g., "gpt-4o", "claude-3-opus-20240229").
    *   `provider` (str, required): The provider of the LLM (e.g., "openai", "anthropic", "google", "huggingface", "custom").
    *   `prompt` (str, required): The full prompt sent to the LLM. For chat models, this might be a concatenation or representation of the message history if `messages` is not used.
    *   `parameters` (dict, optional): A dictionary of parameters sent with the request (e.g., `{"temperature": 0.7, "max_tokens": 500}`).
    *   `system_message` (str, optional): The system message used, if applicable.
    *   `messages` (list[dict], optional): For chat models, a list of message objects (e.g., `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`).

*   **Example Payload (for a completion model):**
    ```json
    {
        "model_name": "text-davinci-003",
        "provider": "openai",
        "prompt": "Translate 'hello world' to French.",
        "parameters": {"temperature": 0.5, "max_tokens": 50}
    }
    ```
*   **Example Payload (for a chat model):**
    ```json
    {
        "model_name": "gpt-4o",
        "provider": "openai",
        "prompt": "User: What is the capital of France?", # Often a summary or last user message
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "parameters": {"temperature": 0.7}
    }
    ```

---

**7. `EventType.LLM_RESPONSE` (`LLMResponsePayload`)**

*   **Purpose:** Logs a response received from an LLM.
*   **Payload Fields:**
    *   `model_name` (str, required): The name of the LLM that responded.
    *   `provider` (str, required): The provider of the LLM.
    *   `response_text` (str, required): The main textual content of the LLM's response.
    *   `finish_reason` (str, optional): The reason the LLM finished generating the response (e.g., "stop", "length", "tool_calls", "content_filter").
    *   `usage` (dict, optional): Token usage information (e.g., `{"prompt_tokens": 10, "completion_tokens": 25, "total_tokens": 35}`).
    *   `response_time_ms` (int, optional): Time taken to receive the response from the LLM in milliseconds (measured by the client).
    *   `cost` (float, optional): Estimated cost for this LLM call.

*   **Example Payload:**
    ```json
    {
        "model_name": "gpt-4o",
        "provider": "openai",
        "response_text": "The capital of France is Paris.",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 15, "completion_tokens": 7, "total_tokens": 22},
        "response_time_ms": 850,
        "cost": 0.00015
    }
    ```

---

**8. `EventType.LLM_STREAMING_CHUNK` (`LLMStreamingChunkPayload`)**

*   **Purpose:** Logs a single chunk of data received during an LLM streaming operation.
*   **Payload Fields:**
    *   `model_name` (str, required): Name of the LLM.
    *   `provider` (str, required): Provider of the LLM.
    *   `chunk_text` (str, required): The text content of this specific chunk.
    *   `chunk_index` (int, required): The sequential index of this chunk in the stream.
    *   `is_final` (bool, default: False): `True` if this is the last chunk in the stream.
    *   `chunk_metadata` (dict, optional): Any metadata specific to this chunk (e.g., token IDs if available).

*   **Example Payload:**
    ```json
    {
        "model_name": "claude-3-sonnet",
        "provider": "anthropic",
        "chunk_text": "Paris",
        "chunk_index": 2,
        "is_final": false
    }
    ```

---

**9. `EventType.TOOL_CALL` (`ToolCallPayload`)**

*   **Purpose:** Logs a call made to a tool or function by the agent or system.
*   **Payload Fields:**
    *   `tool_name` (str, required): The name of the tool being called (e.g., "get_weather", "database_lookup").
    *   `tool_type` (str, required): The type of tool (e.g., "function", "api", "database", "file_system").
    *   `parameters` (dict, required): A dictionary of parameters passed to the tool.
    *   `call_id` (str, optional): A unique identifier for this specific tool call, useful for correlating with `TOOL_RESPONSE`. If not provided, one might be generated by decorators.

*   **Example Payload:**
    ```json
    {
        "tool_name": "get_current_weather",
        "tool_type": "api",
        "parameters": {"location": "New York, NY", "unit": "celsius"},
        "call_id": "tool_call_abc_123"
    }
    ```

---

**10. `EventType.TOOL_RESPONSE` (`ToolResponsePayload`)**

*   **Purpose:** Logs the response received from a tool call.
*   **Payload Fields:**
    *   `tool_name` (str, required): The name of the tool that responded.
    *   `tool_type` (str, required): The type of the tool.
    *   `call_id` (str, optional): The `call_id` from the corresponding `TOOL_CALL` event.
    *   `response_data` (any, required): The data returned by the tool. Can be a string, number, dict, list, etc.
    *   `execution_time_ms` (int, optional): Time taken for the tool to execute in milliseconds.
    *   `success` (bool, default: True): Whether the tool execution was successful.
    *   `error_message` (str, optional): If `success` is `False`, the error message from the tool.

*   **Example Payload (Success):**
    ```json
    {
        "tool_name": "get_current_weather",
        "tool_type": "api",
        "call_id": "tool_call_abc_123",
        "response_data": {"temperature": 22, "condition": "Cloudy"},
        "execution_time_ms": 450,
        "success": true
    }
    ```
*   **Example Payload (Failure):**
    ```json
    {
        "tool_name": "database_lookup",
        "tool_type": "database",
        "call_id": "tool_call_def_456",
        "response_data": null,
        "execution_time_ms": 300,
        "success": false,
        "error_message": "Connection timed out"
    }
    ```

---

**11. `EventType.FUNCTION_EXECUTION` (`FunctionExecutionPayload`)**

*   **Purpose:** Logs the execution of an arbitrary tracked function within your application (often used by the `@track_function_execution` decorator).
*   **Payload Fields:**
    *   `function_name` (str, required): The name of the executed function.
    *   `function_args` (dict, required): Arguments passed to the function (e.g., `{"args": [...], "kwargs": {...}}`).
    *   `return_value` (any): The value returned by the function.
    *   `execution_time_ms` (int, optional): Time taken for the function to execute in milliseconds.
    *   `success` (bool, default: True): Whether the function executed without raising an exception.
    *   `error_message` (str, optional): If `success` is `False`, the string representation of the exception.

*   **Example Payload:**
    ```json
    {
        "function_name": "process_user_data",
        "function_args": {"args": ["user_id_123"], "kwargs": {"deep_clean": true}},
        "return_value": {"status": "processed", "items_cleaned": 10},
        "execution_time_ms": 75,
        "success": true
    }
    ```

---

**12. `EventType.API_REQUEST` (`APIRequestPayload`)**

*   **Purpose:** Logs an outgoing request made to an external API (other than LLMs, which have their own types).
*   **Payload Fields:**
    *   `api_name` (str, required): A descriptive name for the API being called (e.g., "PaymentGateway", "GeoLocationService").
    *   `endpoint` (str, required): The specific endpoint URL or path.
    *   `method` (str, required): The HTTP method used (e.g., "GET", "POST", "PUT", "DELETE").
    *   `headers` (dict, optional): Request headers (sensitive information should be redacted).
    *   `parameters` (dict, optional): Query parameters or URL parameters.
    *   `request_body` (any, optional): The body of the request (for POST, PUT, etc.).

*   **Example Payload:**
    ```json
    {
        "api_name": "WeatherService",
        "endpoint": "https://api.weather.com/v1/current",
        "method": "GET",
        "parameters": {"city": "London", "units": "metric"}
    }
    ```

---

**13. `EventType.API_RESPONSE` (`APIResponsePayload`)**

*   **Purpose:** Logs a response received from an external API call.
*   **Payload Fields:**
    *   `api_name` (str, required): Name of the API that responded.
    *   `endpoint` (str, required): The endpoint that was called.
    *   `status_code` (int, required): The HTTP status code of the response.
    *   `response_headers` (dict, optional): Response headers (sensitive information should be redacted).
    *   `response_body` (any, optional): The body of the response.
    *   `response_time_ms` (int, optional): Time taken to receive the response in milliseconds.

*   **Example Payload:**
    ```json
    {
        "api_name": "WeatherService",
        "endpoint": "https://api.weather.com/v1/current",
        "status_code": 200,
        "response_body": {"temp": 15, "description": "Cloudy"},
        "response_time_ms": 600
    }
    ```

---

**14. `EventType.ERROR` (`ErrorPayload`)**

*   **Purpose:** Logs an error or exception that occurred within the application.
*   **Payload Fields:**
    *   `error_type` (str, required): The type or class name of the error (e.g., "ValueError", "NetworkError", "APIError").
    *   `error_message` (str, required): The error message.
    *   `error_code` (str, optional): A specific error code, if applicable.
    *   `stack_trace` (str, optional): The stack trace of the error.
    *   `context` (dict, optional): Additional context about where or why the error occurred (e.g., function name, operation being performed).
    *   `severity` (str, default: "error"): Severity of the error (e.g., "info", "warning", "error", "critical").

*   **Example Payload:**
    ```json
    {
        "error_type": "TypeError",
        "error_message": "Cannot read property 'name' of undefined",
        "stack_trace": "TypeError: Cannot read property 'name' of undefined\n    at processData (file:///app/utils.js:10:25)...",
        "context": {"function": "processData", "module": "data_processor"},
        "severity": "error"
    }
    ```

---

**15. `EventType.VALIDATION_ERROR` (`ValidationErrorPayload`)**

*   **Purpose:** Logs an error specifically related to data validation failure.
*   **Payload Fields:**
    *   `validation_target` (str, required): What was being validated (e.g., "user_input", "api_response", "configuration").
    *   `validation_rules` (list[str], required): The rules that were supposed to be met.
    *   `failed_rules` (list[str], required): The specific rules that failed.
    *   `validation_details` (dict, optional): More detailed information about the validation failure (e.g., field names and their invalid values).

*   **Example Payload:**
    ```json
    {
        "validation_target": "user_registration_form",
        "validation_rules": ["email_format", "password_strength_min_8_chars"],
        "failed_rules": ["password_strength_min_8_chars"],
        "validation_details": {"field": "password", "value": "short", "reason": "Password too short"}
    }
    ```

---

**16. `EventType.TIMEOUT_ERROR` (`TimeoutErrorPayload`)**

*   **Purpose:** Logs an error that occurred due to an operation timing out.
*   **Payload Fields:**
    *   `timeout_type` (str, required): Description of the operation that timed out (e.g., "llm_request", "api_call", "database_query", "function_execution").
    *   `timeout_duration_ms` (int, required): The duration in milliseconds after which the timeout occurred.
    *   `expected_duration_ms` (int, optional): The expected or configured maximum duration for the operation.
    *   `operation_context` (dict, optional): Context about the timed-out operation.

*   **Example Payload:**
    ```json
    {
        "timeout_type": "llm_request",
        "timeout_duration_ms": 30000,
        "expected_duration_ms": 30000,
        "operation_context": {"model_name": "some_large_model", "provider": "custom_provider"}
    }
    ```

---

**17. `EventType.SESSION_START` (`SessionStartPayload`)** (Note: `run.start` in SDK)

*   **Purpose:** Marks the beginning of a user session or a run. Typically logged automatically by `client.start_run()`.
*   **Payload Fields:**
    *   `session_id` (str, required): The ID of the session/run being started (this is the `run_id`).
    *   `session_type` (str, optional): The type of session (e.g., "chat", "task_execution", "workflow_processing", "user_visit").
    *   `initial_context` (dict, optional): Any context available at the start of the session (e.g., user agent, device info, starting parameters).

*   **Example Payload (as logged by `client.start_run`):**
    ```json
    {
        "session_id": "generated_run_id_guid"
        // session_type and initial_context might be part of the Event's metadata
        // when using client.start_run()
    }
    ```

---

**18. `EventType.SESSION_END` (`SessionEndPayload`)** (Note: `run.end` in SDK)

*   **Purpose:** Marks the end of a user session or a run. Typically logged automatically by `client.end_run()`.
*   **Payload Fields:**
    *   `session_id` (str, required): The ID of the session/run being ended.
    *   `session_duration_ms` (int, optional): Total duration of the session in milliseconds.
    *   `session_summary` (dict, optional): A summary of the session (e.g., number of interactions, tasks completed).
    *   `exit_reason` (str, optional): Reason for the session ending (e.g., "user_logout", "completed", "timeout", "error").

*   **Example Payload (as logged by `client.end_run`):**
    ```json
    {
        "session_id": "generated_run_id_guid"
        // session_duration_ms and other fields might be calculated and added by the backend
        // or could be added if provided explicitly.
    }
    ```

---

**19. `EventType.PERFORMANCE_METRIC` (`PerformanceMetricPayload`)**

*   **Purpose:** Logs a custom performance metric.
*   **Payload Fields:**
    *   `metric_name` (str, required): The name of the metric (e.g., "cpu_usage", "response_latency", "token_throughput").
    *   `metric_value` (int | float, required): The value of the metric.
    *   `metric_unit` (str, optional): The unit of the metric (e.g., "ms", "percentage", "tokens_per_second", "MB").
    *   `metric_type` (str, default: "gauge"): The type of metric (e.g., "gauge", "counter", "histogram", "timer").
    *   `additional_metrics` (dict, optional): A dictionary for logging multiple related sub-metrics under a single event.

*   **Example Payload:**
    ```json
    {
        "metric_name": "image_processing_time",
        "metric_value": 750.5,
        "metric_unit": "ms",
        "metric_type": "gauge",
        "additional_metrics": {"image_size_kb": 2048}
    }
    ```

---

This comprehensive list should help you understand what data to log for each event type and how to structure the payloads for effective tracking and analysis within the InsideLLM platform. Remember to also leverage the `metadata` field on the `Event` object for any additional custom context not covered by the specific payload fields.

## Using Decorators for Automatic Tracking

The InsideLLM SDK provides several decorators to simplify the process of logging common events. These decorators wrap your existing functions and automatically handle the creation and logging of relevant `Event` objects.

**Prerequisites for using decorators:**

1.  The SDK must be initialized (`insidellm.initialize(...)`).
2.  A run must be active (`client.start_run(...)`). Decorators rely on the currently active run on the default client (`insidellm.get_client()`) to get `run_id` and `user_id`. If no run is active, decorators will typically log a warning and not track events.

---

### 1. `@track_llm_call`

*   **Purpose:** Automatically logs `LLM_REQUEST` and `LLM_RESPONSE` events when a decorated function is called. It also handles error logging by creating an `ERROR` event if the decorated function raises an exception.
*   **Decorator Parameters:**
    *   `model_name` (str, required): The name of the LLM model being called by the decorated function.
    *   `provider` (str, required): The provider of the LLM (e.g., "openai", "anthropic").
    *   `client` (InsideLLMClient, optional): An specific `InsideLLMClient` instance to use. Defaults to `insidellm.get_client()`.
    *   `extract_prompt` (Callable, optional): A function that takes the same arguments as the decorated function (`*args`, `**kwargs`) and returns the prompt string. If not provided, the decorator attempts to find common prompt arguments like `prompt` or `messages`.
    *   `extract_response` (Callable, optional): A function that takes the result of the decorated function and returns the response text string. If not provided, `str(result)` is used.
    *   `metadata` (dict, optional): Additional metadata to include in the logged events.

*   **How it works:**
    1.  Before the decorated function executes, an `LLM_REQUEST` event is logged. The `payload` includes `model_name`, `provider`, extracted prompt, and parameters.
    2.  The decorated function is executed.
    3.  After the function completes successfully, an `LLM_RESPONSE` event is logged. The `payload` includes `model_name`, `provider`, extracted response, and response time.
    4.  If the function raises an exception, an `ERROR` event is logged with details about the error.

*   **Example:**

    ```python
    import insidellm
    from insidellm import track_llm_call

    # Assume client is initialized and a run is started
    # client = insidellm.get_client()
    # client.start_run(user_id="decorator_user")

    @track_llm_call(model_name="gpt-4o", provider="openai")
    def call_my_llm(user_query: str, system_prompt: str = "You are a helpful assistant."):
        # In a real scenario, this function would call the OpenAI API
        # For example:
        # response = openai.chat.completions.create(
        # model="gpt-4o",
        # messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]
        # )
        # return response.choices[0].message.content
        print(f"Simulating LLM call for: {user_query}")
        return f"This is a simulated LLM response to: {user_query}"

    # When you call this function:
    # response = call_my_llm("What's the weather in London?", system_prompt="Be concise.")
    # - An LLM_REQUEST event is logged before execution.
    # - An LLM_RESPONSE event is logged after execution.
    ```

    To customize prompt/response extraction:

    ```python
    def custom_prompt_extractor(*args, **kwargs):
        return kwargs.get("my_custom_prompt_arg")

    def custom_response_extractor(result_object):
        return result_object.get("text_output")

    @track_llm_call(
        model_name="custom_model",
        provider="local_provider",
        extract_prompt=custom_prompt_extractor,
        extract_response=custom_response_extractor
    )
    def call_custom_model_api(my_custom_prompt_arg):
        # ... API call ...
        return {"text_output": "...", "tokens": 0} # Example complex return
    ```

---

### 2. `@track_tool_use`

*   **Purpose:** Automatically logs `TOOL_CALL` and `TOOL_RESPONSE` events when a decorated function representing a "tool" is executed. It also logs an `ERROR` event if the tool function fails.
*   **Decorator Parameters:**
    *   `tool_name` (str, required): The name of the tool.
    *   `tool_type` (str, default: "function"): The type of tool (e.g., "function", "api", "database").
    *   `client` (InsideLLMClient, optional): Specific client instance. Defaults to global client.
    *   `extract_parameters` (Callable, optional): A function that takes `*args, **kwargs` of the decorated tool function and returns a dictionary of parameters. Defaults to including all `args` and `kwargs`.
    *   `extract_response` (Callable, optional): A function that takes the result of the decorated tool function and returns the data to be logged as `response_data`. Defaults to the entire result.
    *   `metadata` (dict, optional): Additional metadata for the events.

*   **How it works:**
    1.  Before the tool function executes, a `TOOL_CALL` event is logged.
    2.  The tool function is executed.
    3.  After successful execution, a `TOOL_RESPONSE` event is logged with `success: true` and the extracted response.
    4.  If the tool function raises an exception, a `TOOL_RESPONSE` event is logged with `success: false` and the error message. (Note: Earlier versions might log `ERROR` type, check SDK specifics if behavior differs).

*   **Example:**

    ```python
    from insidellm import track_tool_use

    @track_tool_use(tool_name="get_stock_price", tool_type="api_call")
    def fetch_stock_price(ticker_symbol: str):
        # Simulate calling a stock API
        if ticker_symbol == "GOOD":
            return {"symbol": ticker_symbol, "price": 150.75}
        elif ticker_symbol == "BAD":
            raise ValueError("Invalid ticker symbol for simulation")
        return {"symbol": ticker_symbol, "price": 0}


    # stock_data = fetch_stock_price("GOOD")
    # - TOOL_CALL event logged for "get_stock_price".
    # - TOOL_RESPONSE event logged with price data and success:true.

    # try:
    #     fetch_stock_price("BAD")
    # except ValueError:
    #     pass
    # - TOOL_CALL event logged.
    # - TOOL_RESPONSE event logged with success:false and error message.
    ```

---

### 3. `@track_agent_step`

*   **Purpose:** Logs an `AGENT_REASONING` event when the decorated function (representing a step in an agent's thought process or an internal action) is executed.
*   **Decorator Parameters:**
    *   `step_name` (str, required): A descriptive name for this agent step. This will be used as the `reasoning_type` in the `AgentReasoningPayload`.
    *   `client` (InsideLLMClient, optional): Specific client instance. Defaults to global client.
    *   `metadata` (dict, optional): Additional metadata.

*   **How it works:**
    1.  The decorated function executes.
    2.  After successful execution, an `AGENT_REASONING` event is logged. The `payload` includes `reasoning_type` (from `step_name`), execution time, and a default reasoning step message.
    3.  If the function raises an exception, an `ERROR` event is logged.

*   **Example:**

    ```python
    from insidellm import track_agent_step

    @track_agent_step(step_name="user_intent_classification")
    def classify_intent(user_query: str):
        # Simulate intent classification logic
        if "weather" in user_query:
            return "get_weather_intent"
        return "general_query_intent"

    # intent = classify_intent("What's the weather like?")
    # - AGENT_REASONING event logged with reasoning_type: "user_intent_classification".
    ```

---

### 4. `@track_function_execution`

*   **Purpose:** A generic decorator to log the execution of any function. It logs a `FUNCTION_EXECUTION` event.
*   **Decorator Parameters:**
    *   `function_name` (str, optional): A name to log for the function. Defaults to the actual name of the decorated function.
    *   `client` (InsideLLMClient, optional): Specific client instance. Defaults to global client.
    *   `metadata` (dict, optional): Additional metadata.

*   **How it works:**
    1.  The decorated function is executed.
    2.  After execution (successful or not), a `FUNCTION_EXECUTION` event is logged. The `payload` includes the function name, arguments, return value (or error message if an exception occurred), execution time, and success status.

*   **Example:**

    ```python
    from insidellm import track_function_execution

    @track_function_execution(function_name="data_processor_v1", metadata={"version": "1.0"})
    def process_my_data(data: dict, options: list):
        # Simulate data processing
        processed_data = {key: len(value) for key, value in data.items()}
        if "fail" in options:
            raise RuntimeError("Simulated processing failure")
        return processed_data

    # result = process_my_data({"a": [1,2,3], "b": [4,5]}, options=["fast_mode"])
    # - FUNCTION_EXECUTION event logged with success:true.

    # try:
    #     process_my_data({"c": [6]}, options=["fail"])
    # except RuntimeError:
    #     pass
    # - FUNCTION_EXECUTION event logged with success:false and error message.
    ```

---

**Managing Event Hierarchy with Decorators (`parent_event_id`)**

Decorators automatically attempt to manage a simple parent-child relationship for events logged within the same synchronous call chain using Python's `contextvars`.

*   When a decorator logs its "start" event (e.g., `LLM_REQUEST`, `TOOL_CALL`), it sets this event's ID as the current parent ID in a context variable.
*   If another decorated function is called *synchronously within* the first decorated function, its "start" event will pick up this parent ID.
*   The "end" or "response" event from a decorator (e.g., `LLM_RESPONSE`, `TOOL_RESPONSE`) will also use the same parent ID as its corresponding "start" event.

This helps in creating simple traces. For more complex asynchronous scenarios or explicit parent-child relationships, you might need to manage `parent_event_id` manually when creating events.

## REST API Interaction (for advanced users or direct integration)

While the SDK provides the most convenient way to send event data to InsideLLM, you can also interact directly with the REST API. This might be useful for non-Python environments, custom batching solutions, or if you need to send data from systems where integrating the full SDK is not feasible.

### Endpoint

The primary endpoint for ingesting events is:

*   **URL:** `https://api.insidellm.com/api/v1/ingest` (or your configured `base_url` if different)
*   **Method:** `POST`

### Authentication

Authentication is performed using a Bearer Token in the `Authorization` header. The token is your InsideLLM API key.

*   **Header:** `Authorization: Bearer YOUR_API_KEY`

Replace `YOUR_API_KEY` with your actual API key.

### Request Body

The request body must be JSON. You can send:

1.  **A Single Event Object:** A JSON object representing a single `Event`.
2.  **An Array of Event Objects:** A JSON array where each element is an `Event` object, for batch sending.

The structure of the `Event` object in JSON should match the Pydantic model defined in `insidellm.models.Event`. Key fields include:

*   `event_id` (string, UUID): Unique identifier for the event. If not provided, the backend might generate one, but providing a client-generated UUID is recommended for idempotency and client-side linking.
*   `run_id` (string, UUID, required): Identifier for the run this event belongs to.
*   `timestamp` (string, ISO 8601, required): Timestamp of when the event occurred (e.g., `2023-10-26T10:30:00.123Z`).
*   `event_type` (string, required): One of the supported event type strings (e.g., "user.input", "llm.request"). Refer to the `EventType` enum in `insidellm.models` for all valid string values.
*   `user_id` (string, required): Identifier for the user associated with the event.
*   `parent_event_id` (string, UUID, optional): The `event_id` of a parent event, for tracing.
*   `metadata` (object, optional): A JSON object of key-value pairs (string, number, boolean) for custom metadata.
*   `payload` (object, required): A JSON object containing data specific to the `event_type`. The structure of this object must match the corresponding payload model for the given `event_type` (e.g., `UserInputPayload` for `user.input`).

**Example: Single Event JSON**

```json
{
    "event_id": "e7a4f3c2-3b1d-4e8a-9c5f-0a1b2c3d4e5f",
    "run_id": "r1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d",
    "timestamp": "2024-03-15T14:30:00.500Z",
    "event_type": "llm.request",
    "user_id": "user_johndoe_123",
    "parent_event_id": "p9f8e7d6-c5b4-a3b2-1c0d-9f8e7d6c5b4a",
    "metadata": {
        "environment": "production",
        "app_version": "2.1.0"
    },
    "payload": {
        "model_name": "gpt-4o",
        "provider": "openai",
        "prompt": "What is the weather in London today?",
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 100
        }
    }
}
```

**Example: Batch of Events JSON**

```json
[
    {
        "event_id": "...",
        "run_id": "...",
        "timestamp": "...",
        "event_type": "user.input",
        "user_id": "...",
        "payload": {
            "input_text": "Hello"
        }
    },
    {
        "event_id": "...",
        "run_id": "...",
        "timestamp": "...",
        "event_type": "llm.request",
        "user_id": "...",
        "payload": {
            "model_name": "gpt-3.5-turbo",
            "provider": "openai",
            "prompt": "User: Hello"
        }
    }
]
```

### Headers

*   `Content-Type: application/json`
*   `Authorization: Bearer YOUR_API_KEY`
*   `User-Agent` (optional but recommended): Helps identify the client (e.g., `MyCustomApp/1.0`). The SDK uses `InsideLLM-Python-SDK/VERSION`.

### Responses

*   **`202 Accepted`:** Successfully accepted the event(s) for processing. The body of the response may be empty or contain a simple success message. This indicates the request was well-formed and queued; it doesn't guarantee events are fully processed and stored yet.
*   **`400 Bad Request`:** The request was malformed (e.g., invalid JSON, missing required fields, incorrect data types in payload). The response body should contain details about the error.
*   **`401 Unauthorized`:** The API key is missing, invalid, or expired.
*   **`403 Forbidden`:** The API key is valid but does not have permission to perform the action.
*   **`429 Too Many Requests`:** You have exceeded the rate limits.
*   **`5xx Server Error`:** An error occurred on the InsideLLM server side.

### Health Check Endpoint

There is a health check endpoint that can be used to verify connectivity to the API:

*   **URL:** `https://api.insidellm.com/health` (or your `base_url` + `/health`)
*   **Method:** `GET`
*   **Authentication:** Not typically required for the health endpoint, but check specific InsideLLM instance details if applicable.
*   **Success Response:** `200 OK` (body might be simple like `{"status": "healthy"}` or empty).

This direct API interaction provides flexibility but requires careful implementation of event formatting, batching, retries, and error handling, which the SDK normally manages for you.
