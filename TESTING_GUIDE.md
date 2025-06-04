# InsideLLM SDK - Testing and Examples Guide

This guide shows you how to run and test the SDK examples with your actual API.

## Prerequisites

1. **API Key**: Get your InsideLLM API key from your dashboard
2. **Python Environment**: Python 3.8+ with required dependencies
3. **SDK Installation**: Install the SDK locally or from PyPI

## Quick Setup

### 1. Install the SDK

**Option A: Install locally (for development)**
```bash
pip install -e .
```

**Option B: Install from PyPI (when published)**
```bash
pip install insidellm
```

### 2. Set your API key

**Environment Variable (Recommended)**
```bash
export INSIDELLM_API_KEY="your-actual-api-key-here"
```

**Or create a .env file**
```bash
echo "INSIDELLM_API_KEY=your-actual-api-key-here" > .env
```

## Running the Examples

### 1. Basic Usage Example

This demonstrates core SDK functionality with manual event creation.

```bash
python examples/basic_usage.py
```

**What it does:**
- Initializes the SDK with your API key
- Creates a session and logs various event types
- Demonstrates user input, LLM calls, tool usage, errors, and metrics
- Shows queue statistics and performance monitoring

**Expected output:**
```
InsideLLM Basic Usage Example
==================================================
✓ SDK initialized successfully
✓ Started run: [run-id]
✓ User input event logged: [event-id]
✓ LLM request event logged: [event-id]
✓ LLM response event logged: [event-id]
✓ Tool call event logged: [event-id]
✓ Tool response event logged: [event-id]
✓ Error event logged: [event-id]
✓ Performance metric logged: [event-id]
```

### 2. Custom Agent Example

This shows how to integrate the SDK with your custom agent using decorators and context managers.

```bash
python examples/custom_agent_example.py
```

**What it does:**
- Creates a custom agent class with SDK integration
- Uses decorators to automatically track LLM calls and tool usage
- Demonstrates context manager patterns for workflow tracking
- Shows agent planning, execution, and response tracking

**Key features demonstrated:**
- `@track_llm_call` decorator
- `@track_tool_use` decorator
- `@track_agent_step` decorator
- `InsideLLMTracker` context manager

### 3. LangChain Integration Example

**Note: Requires LangChain installation**
```bash
pip install langchain openai  # Install LangChain and your LLM provider
python examples/langchain_example.py
```

**Environment variables needed:**
```bash
export INSIDELLM_API_KEY="your-insidellm-api-key"
export OPENAI_API_KEY="your-openai-api-key"  # Or your LLM provider key
```

**What it does:**
- Sets up LangChain with InsideLLM callback
- Demonstrates automatic tracking of LangChain operations
- Shows agent workflow tracking with tools
- Captures all LLM calls, tool usage, and agent actions automatically

## Testing with Your Own Code

### Basic Integration Test

Create a simple test to verify your API connection:

```python
import os
import insidellm

# Initialize with your API key
insidellm.initialize(api_key=os.getenv("INSIDELLM_API_KEY"))

# Test basic functionality
client = insidellm.get_client()
run_id = client.start_run(user_id="test-user")

# Log a simple event
event = insidellm.Event.create_user_input(
    run_id=run_id,
    user_id="test-user",
    input_text="Hello, this is a test!"
)
client.log_event(event)

# Flush to send immediately
client.flush()

# Check if healthy
if client.is_healthy():
    print("✓ API connection successful")
else:
    print("✗ API connection failed")

client.end_run(run_id)
insidellm.shutdown()
```

### Custom Agent Integration

```python
import insidellm

# Initialize SDK
insidellm.initialize(api_key="your-api-key")

# Use decorators for automatic tracking
@insidellm.track_llm_call("your-model", "your-provider")
def call_your_llm(prompt):
    # Your LLM call logic here
    response = your_llm_service.generate(prompt)
    return response

@insidellm.track_tool_use("your-tool", "function")
def use_your_tool(parameters):
    # Your tool logic here
    result = your_tool_service.execute(parameters)
    return result

# Use context manager for workflows
with insidellm.InsideLLMTracker(user_id="user-123") as tracker:
    # Your agent workflow here
    input_id = tracker.log_user_input("User's question")
    
    # These calls are automatically tracked
    llm_response = call_your_llm("Process user question")
    tool_result = use_your_tool({"param": "value"})
    
    tracker.log_agent_response(
        response_text="Agent's final response",
        parent_event_id=input_id
    )
```

### LangChain Integration

```python
import insidellm
from langchain.llms import YourLLM
from langchain.agents import initialize_agent

# Initialize InsideLLM
insidellm.initialize(api_key="your-api-key")

# Create callback
callback = insidellm.InsideLLMCallback(
    client=insidellm.get_client(),
    user_id="user-123"
)

# Use with any LangChain component
llm = YourLLM(callbacks=[callback])
agent = initialize_agent(tools, llm, callbacks=[callback])

# All operations automatically tracked
response = agent.run("Your user's question")
```

## Troubleshooting

### Common Issues

**1. API Key Issues**
```bash
# Check if API key is set
echo $INSIDELLM_API_KEY

# Test API connection
python -c "
import insidellm
insidellm.initialize(api_key='your-key')
print('✓ API key valid' if insidellm.get_client().is_healthy() else '✗ API key invalid')
"
```

**2. Import Errors**
```bash
# Check SDK installation
python -c "import insidellm; print(f'SDK version: {insidellm.__version__}')"

# Reinstall if needed
pip uninstall insidellm
pip install insidellm
```

**3. Network Issues**
- Check your internet connection
- Verify firewall settings
- Test API endpoint accessibility

**4. Event Not Appearing**
- Events are batched by default (check after flush)
- Verify API key has proper permissions
- Check for validation errors in logs

### Debug Mode

Enable debug logging to see detailed information:

```python
import logging
import insidellm

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Initialize with debug config
config = insidellm.InsideLLMConfig(
    enable_debug_logging=True,
    raise_on_error=True  # Raise errors instead of silent failure
)

insidellm.initialize(
    api_key="your-api-key",
    config=config
)
```

### Performance Testing

Test the SDK's performance with high event volumes:

```python
import time
import insidellm

insidellm.initialize(api_key="your-api-key")
client = insidellm.get_client()
run_id = client.start_run(user_id="perf-test")

# Log many events quickly
start_time = time.time()
for i in range(1000):
    event = insidellm.Event.create_user_input(
        run_id=run_id,
        user_id="perf-test",
        input_text=f"Test message {i}"
    )
    client.log_event(event)

# Check queue statistics
stats = client.queue_manager.get_statistics()
print(f"Events queued: {stats['events_queued']}")
print(f"Queue size: {stats['queue_size']}")

# Flush and measure
flush_start = time.time()
client.flush()
flush_time = time.time() - flush_start

print(f"Total time: {time.time() - start_time:.2f}s")
print(f"Flush time: {flush_time:.2f}s")
print(f"Events/sec: {1000 / (time.time() - start_time):.1f}")

client.end_run(run_id)
insidellm.shutdown()
```

## Next Steps

1. **Run the examples** with your API key to verify everything works
2. **Integrate with your existing agent** using decorators or context managers
3. **Monitor your analytics** in the InsideLLM dashboard
4. **Optimize configuration** based on your usage patterns
5. **Scale up** to production workloads

## Support

If you encounter issues:
1. Check this troubleshooting guide
2. Review the API documentation
3. Check SDK logs for error details
4. Contact support with specific error messages and configurations