"""
InsideLLM LangChain Integration Example

This example demonstrates how to use InsideLLM with LangChain
to automatically track LLM calls, tool usage, and agent actions.
"""

import os
import insidellm

# Optional: Only run if LangChain is available
try:
    from langchain.llms import OpenAI
    from langchain.agents import initialize_agent, AgentType
    from langchain.tools import Tool
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available. Install with: pip install langchain")


def simple_calculator(expression: str) -> str:
    """Simple calculator tool for demonstration."""
    try:
        # Safe evaluation of simple math expressions
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def web_search_mock(query: str) -> str:
    """Mock web search tool for demonstration."""
    return f"Mock search results for: {query}"


def main():
    """Main example function."""
    if not LANGCHAIN_AVAILABLE:
        print("This example requires LangChain. Please install it first.")
        return
    
    # Initialize InsideLLM
    insidellm.initialize(
        api_key=os.getenv("INSIDELLM_API_KEY", "demo-key"),
        auto_flush_interval=5.0,  # Flush every 5 seconds for demo
        batch_size=10
    )
    
    print("InsideLLM LangChain Integration Example")
    print("=" * 50)
    
    # Create LangChain components
    llm = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "demo-key"),
        temperature=0.7
    )
    
    # Create tools
    tools = [
        Tool(
            name="Calculator",
            func=simple_calculator,
            description="Useful for mathematical calculations"
        ),
        Tool(
            name="WebSearch",
            func=web_search_mock,
            description="Useful for searching the web"
        )
    ]
    
    # Create InsideLLM callback
    callback = insidellm.InsideLLMCallback(
        client=insidellm.get_client(),
        user_id="demo-user",
        metadata={
            "environment": "demo",
            "agent_type": "langchain_agent"
        }
    )
    
    print(f"Started tracking run: {callback.run_id}")
    
    # Example 1: Simple LLM Chain
    print("\n1. Simple LLM Chain Example")
    print("-" * 30)
    
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Answer the following question: {question}"
    )
    
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        callbacks=[callback]
    )
    
    try:
        response = chain.run(
            question="What is the capital of France?",
            callbacks=[callback]
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error in LLM chain: {e}")
    
    # Example 2: Agent with Tools
    print("\n2. Agent with Tools Example")
    print("-" * 30)
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        callbacks=[callback],
        verbose=True
    )
    
    try:
        response = agent.run(
            "Calculate 15 * 24 and then search for information about that number"
        )
        print(f"Agent Response: {response}")
    except Exception as e:
        print(f"Error in agent execution: {e}")
    
    # Example 3: Manual Event Logging
    print("\n3. Manual Event Logging")
    print("-" * 30)
    
    client = insidellm.get_client()
    
    # Log user input
    user_input_event = insidellm.Event.create_user_input(
        run_id=callback.run_id,
        user_id="demo-user",
        input_text="What's 2+2?",
        input_type="text",
        metadata={"channel": "demo"}
    )
    client.log_event(user_input_event)
    
    # Log agent response
    agent_response_event = insidellm.Event(
        run_id=callback.run_id,
        user_id="demo-user",
        event_type=insidellm.EventType.AGENT_RESPONSE,
        parent_event_id=user_input_event.event_id,
        payload={
            "response_text": "The answer is 4",
            "response_type": "calculation",
            "response_confidence": 1.0
        }
    )
    client.log_event(agent_response_event)
    
    print("Manual events logged successfully")
    
    # Flush and display statistics
    print("\n4. Flushing Events and Statistics")
    print("-" * 30)
    
    client.flush()
    
    # Get queue statistics
    stats = client.queue_manager.get_statistics()
    print(f"Queue Statistics:")
    print(f"  Events Queued: {stats['events_queued']}")
    print(f"  Events Sent: {stats['events_sent']}")
    print(f"  Events Failed: {stats['events_failed']}")
    print(f"  Success Rate: {stats['success_rate']:.1f}%")
    print(f"  Current Queue Size: {stats['queue_size']}")
    
    # Shutdown
    print("\n5. Shutting Down")
    print("-" * 30)
    
    insidellm.shutdown()
    print("InsideLLM client shutdown complete")


if __name__ == "__main__":
    main()
