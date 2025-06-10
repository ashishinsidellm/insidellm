"""
InsideLLM SDK: Multimodal Event Tracking Example
"""

from insidellm import (
    InsideLLMClient,
    InsideLLMTracker,
    MultimodalContent,
    ContentMetadata,
)

# Placeholder for a base64 encoded image string (1x1 blue pixel GIF)
# In a real scenario, this would be the actual base64 content of an image.
BASE64_PLACEHOLDER_IMAGE = "R0lGODlhAQABAIAAAAUEBAAAACwAAAAAAQABAAACAkQBADs="

def main():
    """
    Demonstrates multimodal event tracking using the InsideLLM SDK.
    """
    print("Initializing InsideLLM Client and Tracker...")
    # Initialize the client (e.g., with a dummy API key and local settings if needed)
    # For this example, API calls won't actually be made.
    client = InsideLLMClient(api_key="test-key-multimodal-example")

    # Initialize the tracker
    # User ID should be set to the actual user identifier in a real application
    tracker = InsideLLMTracker(client=client, user_id="user-123-multimodal")

    try:
        # Start a session/run
        tracker.start_session()
        print(f"Session started with run_id: {tracker.run_id}")

        # 1. Log Text-Only User Input (Testing Backward Compatibility)
        print("\nLogging text-only user input...")
        text_input_event_id = tracker.log_user_input(
            content="Hello, what is the weather today?",
            content_type="text",
            channel="web"
        )
        print(f"Logged text input event: {text_input_event_id}")

        # As per the prompt, log_user_input in context_manager.py requires content_type.
        # Event.create_user_input handles defaulting if content is string and content_type is not "text" (with a warning),
        # but the tracker method itself has `content_type: str` as a required parameter in its signature.
        another_text_input_event_id = tracker.log_user_input(
            content="Can you show me some multimodal examples?",
            content_type="text", # content_type is required by log_user_input method
            channel="mobile"
        )
        print(f"Logged another text input event: {another_text_input_event_id}")

        # 2. Log Multimodal User Input (Image URI + Text)
        print("\nLogging multimodal user input (image + text)...")
        multimodal_user_input = MultimodalContent(
            text="Generate a website that looks like this, but for a coffee shop.",
            uri=f"data:image/gif;base64,{BASE64_PLACEHOLDER_IMAGE}", # Using the placeholder
            metadata=ContentMetadata(description="Layout sketch for a website.")
        )
        image_input_event_id = tracker.log_user_input(
            content=multimodal_user_input,
            content_type="image", # Could also be "mixed" or a custom type if defined
            session_context={"app_version": "1.2.0"}
        )
        print(f"Logged image input event: {image_input_event_id}")

        # 3. Log Multimodal LLM Request (Text + Image URI)
        print("\nLogging multimodal LLM request (text + image URI)...")
        llm_input_parts = [
            MultimodalContent(text="What’s in this image?"),
            MultimodalContent(
                uri="https://example.com/images/cat_on_sofa.png", # Fictional URI
                metadata=ContentMetadata(
                    description="A cute cat relaxing.",
                    format="png",
                    # width=1024, # Optional metadata
                    # height=768   # Optional metadata
                )
            )
        ]
        llm_request_id = tracker.log_llm_request(
            model_name="gpt-4o",
            provider="openai",
            input=llm_input_parts,
            parameters={"temperature": 0.7, "max_tokens": 300}
        )
        print(f"Logged LLM request event: {llm_request_id}")

        # 4. Log Text-Only LLM Response
        # The SDK's log_llm_response (and underlying Event.create_llm_response)
        # will automatically wrap this string into a MultimodalContent object.
        print("\nLogging text-only LLM response...")
        llm_text_response_id = tracker.log_llm_response(
            model_name="gpt-4o",
            provider="openai",
            output="The image contains a black cat sitting on a comfortable sofa, looking curious.",
            parent_event_id=llm_request_id, # Linking response to the request
            usage={"prompt_tokens": 50, "completion_tokens": 35, "total_tokens": 85}
        )
        print(f"Logged LLM text response event: {llm_text_response_id}")

        # 5. Log Multimodal LLM Response (e.g., HTML content)
        print("\nLogging multimodal LLM response (HTML)...")
        html_response_content = MultimodalContent(
            text="""<!DOCTYPE html>
<html>
<head><title>Coffee Shop</title></head>
<body><h1>Welcome to Our Coffee Shop!</h1><p>Enjoy our finest brews and pastries.</p></body>
</html>""",
            metadata=ContentMetadata(
                format="html",
                description="A simple HTML page for a coffee shop."
            )
        )
        llm_html_response_id = tracker.log_llm_response(
            model_name="gpt-4o",
            provider="openai",
            output=html_response_content,
            parent_event_id=llm_request_id, # Also linked to the same request
                                            # Or could be a response to a different request
            finish_reason="stop"
        )
        print(f"Logged LLM HTML response event: {llm_html_response_id}")

        print("\nAll example events logged.")

    except Exception as e:
        print(f"An error occurred: {e}")
        # Optionally log the error using the tracker
        if tracker and tracker._session_started: # Check if session started
             tracker.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                # stack_trace=traceback.format_exc() # If you want to include stack trace
             )
    finally:
        if tracker and tracker._session_started:
            print("\nEnding session...")
            tracker.end_session()
            print("Session ended.")
        # In a real application, you might also want to flush the client
        # client.flush()

if __name__ == "__main__":
    main()
    print("\nMultimodal example script finished.")
