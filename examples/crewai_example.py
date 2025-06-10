import os
import logging
from uuid import uuid4

# CrewAI components
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# InsideLLM components
# Assuming insidellm is installed and __init__.py is set up correctly
from insidellm import InsideLLMClient, CrewAIInsideLLMCallbackHandler

# Configure basic logging for the example
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Define a Simple Custom Tool for the Agent ---
class MySimpleTool(BaseTool):
    name: str = "My Simple Tool"
    description: str = "A simple tool that returns a fixed message."

    def _run(self, argument: str) -> str:
        return f"MySimpleTool processed: '{argument}' and returned this fixed message."

def main():
    logger.info("Starting CrewAI Example with InsideLLM Integration...")

    # --- 1. Set up API Keys (Optional - for LLMs that require them) ---
    # For this example, we'll rely on CrewAI's default LLM or one configured via LiteLLM
    # which might not strictly require an API key if it's a local model or using a free tier.
    # If using OpenAI, you would typically set:
    # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
    # If using a tool that needs an API key (e.g., Serper):
    # os.environ["SERPER_API_KEY"] = "YOUR_SERPER_API_KEY"

    # For this example, we mock the LLM to avoid actual API calls and ensure runnability.
    # CrewAI uses LiteLLM. We'll specify a mock model and set its cost to zero.
    # This helps LiteLLM process it without needing external API keys.
    os.environ["LITELLM_MODEL_COST_MAP"] = json.dumps({
        "mock-llm": {
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0
        }
    })
    # To make LiteLLM "successfully" return a mock response for "mock-llm" without an API call:
    # This is an advanced LiteLLM feature. If it doesn't work, we still expect failure callbacks.
    def mock_llm_completion(model, messages, **kwargs):
        print(f"\n\nDEBUG: mock_llm_completion called for model: {model}\n\n")
        # Simulate a LiteLLM ModelResponse object structure
        return {
            "id": "chatcmpl-mock-" + str(uuid4()),
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "This is a mock LLM response to the query: " + messages[-1]['content'],
                    "role": "assistant"
                }
            }],
            "created": datetime.now().timestamp(),
            "model": model, # Or "mock-llm"
            "object": "chat.completion",
            "usage": {
                "prompt_tokens": len(str(messages)), # Simple token approximation
                "completion_tokens": 20,
                "total_tokens": len(str(messages)) + 20
            }
        }
    litellm.register_model({
        "mock-llm": {
            "litellm_params": { # This is how you specify it's a custom litellm model
                "model": "custom_llm", # custom_llm tells LiteLLM to use the 'completion' function below
                "completion": mock_llm_completion,
                "custom_llm_provider": "openai", # Added provider, common requirement for custom models
                "input_cost_per_token": 0, # Optional
                "output_cost_per_token": 0, # Optional
                "mode": "chat" # Optional
            },
            "model_info":{
                "description": "Mock LLM for testing purposes"
            }
        }
    })

    logger.info("API Keys / LLM Configuration: Mock LLM registered with LiteLLM.")

    # --- 2. Instantiate InsideLLMClient ---
    # In a real application, you would initialize this with your API key and other settings.
    # For this example, we use the default behavior (which might be a local/test client if configured in insidellm.initialize).
    # We will use the placeholder client defined in the callback handler for direct output.

    # If insidellm.initialize() is part of your library's public API:
    # from insidellm import initialize
    # client = initialize(local_testing=True) # or with API key

    # For this example, we'll use the placeholder client from the callback handler's perspective
    # The handler itself instantiates a placeholder if a real one isn't passed.
    # For a more robust example, we'd show proper client initialization from insidellm.

    # Let's assume a basic client for now. The handler itself has a placeholder.
    # The key is that the handler *receives* a client.
    class PlaceholderInsideLLMClient: # Re-defining for clarity in example context
        def __init__(self, api_key: str = None, **kwargs):
            self.api_key = api_key
            self.run_id_stack = []
            logger.info(f"PlaceholderInsideLLMClient initialized. API Key: {'Set' if api_key else 'Not Set'}")

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
            print(log_message) # Print to console for example visibility

        def start_run(self, run_id: str = None, user_id: str = None, metadata: dict = None):
            run_id = run_id or str(uuid4())
            self.run_id_stack.append(run_id)
            logger.info(f"PlaceholderInsideLLMClient: Run started - ID: {run_id}, User: {user_id}, Metadata: {metadata}")
            return run_id

        def end_run(self, run_id: str):
            if self.run_id_stack and self.run_id_stack[-1] == run_id:
                self.run_id_stack.pop()
            logger.info(f"PlaceholderInsideLLMClient: Run ended - ID: {run_id}")

        def flush(self):
            logger.info("PlaceholderInsideLLMClient: Flush called.")

        def shutdown(self):
            logger.info("PlaceholderInsideLLMClient: Shutdown called.")

    # Instantiate the client
    # In a real scenario, this would be `from insidellm import InsideLLMClient`
    # and then `client = InsideLLMClient(api_key="YOUR_API_KEY")` or `initialize()`
    inside_llm_client = PlaceholderInsideLLMClient(api_key="example_api_key")
    logger.info("InsideLLMClient instantiated.")

    # --- 3. Instantiate CrewAIInsideLLMCallbackHandler ---
    # The handler takes the InsideLLMClient instance.
    # We also pass a default run_id and user_id to the handler,
    # as required by the handler's __init__ method (based on previous implementation).
    # Note: The handler's __init__ was modified to generate its own run_id/user_id if not provided.
    # Let's rely on that, or pass them explicitly.
    # The handler itself will generate a run_id if not passed via client context.
    # For this example, we'll let the handler manage its own run_id for simplicity here.

    # The handler itself will manage its own current_run_id and current_user_id
    # as per the last version of the handler.
    inside_llm_crew_callback = CrewAIInsideLLMCallbackHandler(
        client=inside_llm_client,
        debug=True # Enable debug logging from the handler
    )
    logger.info("CrewAIInsideLLMCallbackHandler instantiated.")

    # --- 4. Define Agent(s) and Task(s) ---
    # Define a simple agent
    researcher_agent = Agent(
        role='Example Researcher',
        goal='Research a simple topic and use a tool.',
        backstory='You are an example agent designed to showcase callback integration. You use a simple tool for your tasks.',
        verbose=True, # Enables agent's own logging, which can be helpful
        allow_delegation=False,
        # LLM configuration: CrewAI uses LiteLLM.
        # To make this runnable without real API keys, we specify a mock model.
        # This assumes LiteLLM is configured to handle "mock-llm".
        # In a real scenario, you'd configure a real model like "gpt-3.5-turbo", "claude-3-opus", etc.
        llm={"model": "mock-llm"}, # Specify the mock LLM for the agent
        tools=[MySimpleTool()]
    )
    logger.info("Agent defined.")

    # Define a simple task
    research_task = Task(
        description='Research the topic "AI in 2024" and use My Simple Tool with "AI trends" as input.',
        expected_output='A short summary based on the tool\'s output for "AI trends".',
        agent=researcher_agent
    )
    logger.info("Task defined.")

    # --- 5. Create the Crew ---
    # Register the callback handler with the Crew.
    # Note: CrewAI's documentation for callbacks might vary.
    # As of recent versions, callbacks are often passed to the Crew object.
    # The handler we built is based on LiteLLM's CustomLogger, so it's registered via litellm.callbacks.
    # However, CrewAI might also have its own callback list for higher-level events if our handler
    # was designed for that (which was the initial plan that changed).

    # Given our handler is a LiteLLM CustomLogger, it should be registered with LiteLLM.
    # CrewAI initializes LiteLLM. We need to ensure our handler is in litellm.callbacks.
    import litellm
    if not hasattr(litellm, 'callbacks') or not litellm.callbacks:
        litellm.callbacks = []

    # Add our handler to LiteLLM's global callback list.
    # This is the primary way LiteLLM discovers custom loggers.
    if not any(isinstance(cb, CrewAIInsideLLMCallbackHandler) for cb in litellm.callbacks):
        litellm.callbacks.append(inside_llm_crew_callback)
        logger.info("CrewAIInsideLLMCallbackHandler registered with litellm.callbacks global list.")
    else:
        logger.info("CrewAIInsideLLMCallbackHandler already in litellm.callbacks global list.")

    # Forcing LiteLLM to be verbose to see its internal logging.
    litellm.set_verbose = True # Deprecated but often still works for quick checks
    # os.environ['LITELLM_LOG'] = 'DEBUG' # The newer way

    # If the above doesn't work, one could also try setting specific callback lists directly,
    # though `litellm.callbacks` should be sufficient for CustomLogger classes.
    # litellm.success_callback.append(inside_llm_crew_callback)
    # litellm.failure_callback.append(inside_llm_crew_callback)

    example_crew = Crew(
        agents=[researcher_agent],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True
        # callbacks=[inside_llm_crew_callback] # Not using this for LiteLLM CustomLoggers; this is for CrewAI's own system.
    )
    logger.info("Crew created.")

    # --- 6. Execute the Crew ---
    logger.info("Kicking off the Crew...")
    try:
        # Kickoff the crew's process. This will trigger agent actions, LLM calls, tool uses, etc.
        # which should be captured by the CrewAIInsideLLMCallbackHandler (via LiteLLM).
        result = example_crew.kickoff()
        logger.info("Crew kickoff finished.")
        print("\n--- Crew Execution Result ---")
        print(result)
    except Exception as e:
        logger.error(f"An error occurred during crew kickoff: {e}", exc_info=True)
        print(f"\n--- Crew Execution Failed ---")
        print(f"Error: {e}")

    logger.info("CrewAI Example finished.")

if __name__ == "__main__":
    # The main function needs to be async if we are to use `await` directly for async handler methods,
    # but CrewAI's kickoff is synchronous. The handler's async methods are called by LiteLLM's async infrastructure.
    main()
