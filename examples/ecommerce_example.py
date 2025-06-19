"""
E-commerce Example - Demonstrates all InsideLLM event types through a shopping scenario
"""

import uuid,os
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from pydantic import BaseModel

from insidellm.models import (
    Event, EventType,
    UserInputPayload, UserFeedbackPayload,
    AgentReasoningPayload, AgentPlanningPayload, AgentResponsePayload,
    ChainStartPayload, ChainEndPayload,
    LLMRequestPayload, LLMResponsePayload, LLMStreamingChunkPayload,
    ToolCallPayload, ToolResponsePayload, FunctionExecutionPayload,
    APIRequestPayload, APIResponsePayload,
    ErrorPayload, ValidationErrorPayload, TimeoutErrorPayload,
    SessionStartPayload, SessionEndPayload,
    PerformanceMetricPayload
)
from insidellm.utils import generate_uuid, get_iso_timestamp
import insidellm

client = insidellm.initialize(
        api_key=os.getenv("INSIDELLM_API_KEY", "iilmn-sample-key"),
        local_testing=True,
    )



def format_payload(payload: Dict[str, Any]) -> str:
    """Format payload for better readability."""
    return json.dumps(payload, indent=2)

def print_event_details(console: Console, event: Event, index: int):
    """Print event details in a formatted way."""
    # Create a table for the event
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    # Add basic event information
    table.add_row("Event Number", str(index))
    table.add_row("Event Type", str(event.event_type))
    table.add_row("Event ID", event.event_id)
    if event.parent_event_id:
        table.add_row("Parent Event ID", event.parent_event_id)
    
    # Print the table
    console.print(table)
    
    # Print payload in a panel with syntax highlighting
    payload_str = format_payload(event.payload)
    console.print(Panel(
        Syntax(payload_str, "json", theme="monokai", line_numbers=True),
        title="Event Payload",
        border_style="blue"
    ))
    
    # Add a separator
    console.print("\n" + "="*80 + "\n")

def print_session_summary(console: Console, events: List[Event]):
    """Print a summary of the session."""
    # Count events by type
    event_counts = {}
    for event in events:
        event_type = str(event.event_type)
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    # Create summary table
    summary_table = Table(show_header=True, header_style="bold yellow")
    summary_table.add_column("Event Type", style="cyan")
    summary_table.add_column("Count", style="green")
    
    for event_type, count in sorted(event_counts.items()):
        summary_table.add_row(event_type, str(count))
    
    # Print summary
    console.print("\n[bold yellow]Session Summary[/bold yellow]")
    console.print(summary_table)
    
    # Print session duration
    session_start = next(e for e in events if e.event_type == EventType.SESSION_START)
    session_end = next(e for e in events if e.event_type == EventType.SESSION_END)
    duration_ms = int(session_end.payload.get("session_duration_ms", 0))
    
    console.print(f"\n[bold]Session Duration:[/bold] {duration_ms}ms")
    
    # Print success rate
    total_events = len(events)
    error_events = sum(1 for e in events if e.event_type in [EventType.ERROR, EventType.VALIDATION_ERROR, EventType.TIMEOUT_ERROR])
    success_rate = ((total_events - error_events) / total_events) * 100
    
    console.print(f"[bold]Success Rate:[/bold] {success_rate:.1f}%")
    console.print(f"[bold]Total Events:[/bold] {total_events}")
    console.print(f"[bold]Error Events:[/bold] {error_events}")

def create_ecommerce_session():
    """Create a complete e-commerce session demonstrating all event types."""
    
    # Generate session IDs
    run_id = generate_uuid()
    user_id = generate_uuid()
    session_id = generate_uuid()
    
    # Start a new session
    client.start_run(run_id=run_id, user_id=user_id)
    
    # 1. Session Start
    session_start = Event(
        run_id=run_id,
        user_id=user_id,
        event_type=EventType.SESSION_START,
        payload=SessionStartPayload(
            session_id=session_id,
            session_type="shopping",
            initial_context={"platform": "web", "device": "desktop"}
        ).dict()
    )
    client.log_event(session_start)
    
    # 2. User Input - Search Query
    user_search = Event(
        run_id=run_id,
        user_id=user_id,
        event_type=EventType.USER_INPUT,
        parent_event_id=session_start.event_id,
        payload=UserInputPayload(
            input_text="I'm looking for a wireless gaming mouse under $50",
            input_type="text",
            channel="web",
            session_context={"cart_items": 0}
        ).dict()
    )
    client.log_event(user_search)
    
    # 3. Agent Planning
    agent_plan = Event(
        run_id=run_id,
        user_id=user_id,
        event_type=EventType.AGENT_PLANNING,
        parent_event_id=user_search.event_id,
        payload=AgentPlanningPayload(
            plan_type="sequential",
            planned_actions=[
                {"action": "search_products", "parameters": {"category": "gaming_mice", "max_price": 50}},
                {"action": "filter_results", "parameters": {"wireless": True}},
                {"action": "rank_by_rating", "parameters": {"min_rating": 4.0}}
            ],
            planning_time_ms=150,
            plan_confidence=0.95
        ).dict()
    )
    client.log_event(agent_plan)
    
    # 4. Agent Reasoning
    agent_reasoning = Event(
        run_id=run_id,
        user_id=user_id,
        event_type=EventType.AGENT_REASONING,
        parent_event_id=agent_plan.event_id,
        payload=AgentReasoningPayload(
            reasoning_type="chain_of_thought",
            reasoning_steps=[
                "User wants a wireless gaming mouse under $50",
                "Need to search gaming mice category",
                "Filter for wireless options",
                "Sort by price and rating",
                "Present top 3 options"
            ],
            confidence_score=0.9,
            reasoning_time_ms=200
        ).dict()
    )
    client.log_event(agent_reasoning)
    
    # 5. Tool Call - Search Products
    search_tool = Event(
        run_id=run_id,
        user_id=user_id,
        event_type=EventType.TOOL_CALL,
        parent_event_id=agent_reasoning.event_id,
        payload=ToolCallPayload(
            tool_name="product_search",
            tool_type="api",
            parameters={
                "category": "gaming_mice",
                "max_price": 50,
                "wireless": True
            },
            call_id=generate_uuid()
        ).dict()
    )
    client.log_event(search_tool)
    
    # 6. API Request
    api_request = Event(
        run_id=run_id,
        user_id=user_id,
        event_type=EventType.API_REQUEST,
        parent_event_id=search_tool.event_id,
        payload=APIRequestPayload(
            api_name="product_catalog",
            endpoint="/api/v1/products/search",
            method="GET",
            parameters={
                "category": "gaming_mice",
                "max_price": 50,
                "wireless": True
            }
        ).dict()
    )
    client.log_event(api_request)
    
    # 7. API Response
    api_response = Event(
        run_id=run_id,
        user_id=user_id,
        event_type=EventType.API_RESPONSE,
        parent_event_id=api_request.event_id,
        payload=APIResponsePayload(
            api_name="product_catalog",
            endpoint="/api/v1/products/search",
            status_code=200,
            response_body={
                "products": [
                    {"id": "mouse1", "name": "Pro Gaming Mouse", "price": 45.99},
                    {"id": "mouse2", "name": "Wireless Elite", "price": 49.99}
                ]
            },
            response_time_ms=150
        ).dict()
    )
    client.log_event(api_response)
    
    # 8. Tool Response
    tool_response = Event(
        run_id=run_id,
        user_id=user_id,
        event_type=EventType.TOOL_RESPONSE,
        parent_event_id=search_tool.event_id,
        payload=ToolResponsePayload(
            tool_name="product_search",
            tool_type="api",
            call_id=search_tool.payload["call_id"],
            response_data={
                "products": [
                    {"id": "mouse1", "name": "Pro Gaming Mouse", "price": 45.99},
                    {"id": "mouse2", "name": "Wireless Elite", "price": 49.99}
                ]
            },
            execution_time_ms=200,
            success=True
        ).dict()
    )
    client.log_event(tool_response)
    
    # 9. LLM Request - Generate Product Recommendations
    llm_request = Event(
        run_id=run_id,
        user_id=user_id,
        event_type=EventType.LLM_REQUEST,
        parent_event_id=tool_response.event_id,
        payload=LLMRequestPayload(
            model_name="gpt-4",
            provider="openai",
            prompt="Compare these gaming mice and recommend the best one: [product details]",
            parameters={"temperature": 0.7, "max_tokens": 150}
        ).dict()
    )
    client.log_event(llm_request)
    
    # 10. LLM Streaming Chunk
    llm_chunk = Event(
        run_id=run_id,
        user_id=user_id,
        event_type=EventType.LLM_STREAMING_CHUNK,
        parent_event_id=llm_request.event_id,
        payload=LLMStreamingChunkPayload(
            model_name="gpt-4",
            provider="openai",
            chunk_text="Based on the specifications, I recommend the Pro Gaming Mouse",
            chunk_index=0,
            is_final=False
        ).dict()
    )
    client.log_event(llm_chunk)
    
    # 11. LLM Response
    llm_response = Event(
        run_id=run_id,
        user_id=user_id,
        event_type=EventType.LLM_RESPONSE,
        parent_event_id=llm_request.event_id,
        payload=LLMResponsePayload(
            model_name="gpt-4",
            provider="openai",
            response_text="I recommend the Pro Gaming Mouse. It offers the best value with its high DPI sensor and long battery life.",
            finish_reason="stop",
            usage={"prompt_tokens": 50, "completion_tokens": 30},
            response_time_ms=2500,
            cost=0.002
        ).dict()
    )
    client.log_event(llm_response)
    
    # 12. Agent Response
    agent_response = Event(
        run_id=run_id,
        user_id=user_id,
        event_type=EventType.AGENT_RESPONSE,
        parent_event_id=llm_response.event_id,
        payload=AgentResponsePayload(
            response_text="I found two great options for you. I recommend the Pro Gaming Mouse at $45.99. It has excellent reviews and features a high DPI sensor and long battery life. Would you like to see more details or add it to your cart?",
            response_type="text",
            response_confidence=0.95,
            response_metadata={"recommended_product_id": "mouse1"}
        ).dict()
    )
    client.log_event(agent_response)
    
    # 13. User Input - Add to Cart
    add_to_cart = Event(
        run_id=run_id,
        user_id=user_id,
        event_type=EventType.USER_INPUT,
        parent_event_id=agent_response.event_id,
        payload=UserInputPayload(
            input_text="Yes, add it to my cart",
            input_type="text",
            channel="web",
            session_context={"cart_items": 1}
        ).dict()
    )
    client.log_event(add_to_cart)
    
    # 14. Function Execution - Add to Cart
    cart_function = Event(
        run_id=run_id,
        user_id=user_id,
        event_type=EventType.FUNCTION_EXECUTION,
        parent_event_id=add_to_cart.event_id,
        payload=FunctionExecutionPayload(
            function_name="add_to_cart",
            function_args={"product_id": "mouse1", "quantity": 1},
            return_value={"success": True, "cart_total": 45.99},
            execution_time_ms=100,
            success=True
        ).dict()
    )
    client.log_event(cart_function)
    
    # 15. Performance Metric
    performance = Event(
        run_id=run_id,
        user_id=user_id,
        event_type=EventType.PERFORMANCE_METRIC,
        parent_event_id=cart_function.event_id,
        payload=PerformanceMetricPayload(
            metric_name="response_time",
            metric_value=3500,
            metric_unit="ms",
            metric_type="gauge",
            additional_metrics={
                "search_time": 200,
                "llm_time": 2500,
                "cart_time": 100
            }
        ).dict()
    )
    client.log_event(performance)
    
    # 16. User Feedback
    user_feedback = Event(
        run_id=run_id,
        user_id=user_id,
        event_type=EventType.USER_FEEDBACK,
        parent_event_id=agent_response.event_id,
        payload=UserFeedbackPayload(
            feedback_type="thumbs_up_down",
            feedback_value=True,
            target_event_id=agent_response.event_id,
            feedback_text="Great recommendation!"
        ).dict()
    )
    client.log_event(user_feedback)
    
    # 17. Session End
    session_end = Event(
        run_id=run_id,
        user_id=user_id,
        event_type=EventType.SESSION_END,
        parent_event_id=session_start.event_id,
        payload=SessionEndPayload(
            session_id=session_id,
            session_duration_ms=5000,
            session_summary={
                "products_viewed": 2,
                "products_added": 1,
                "total_spent": 45.99
            },
            exit_reason="completed"
        ).dict()
    )
    client.log_event(session_end)
    
    # Edge Cases and Additional Events
    
    # 18. Validation Error - Invalid Product ID
    validation_error = Event(
        run_id=run_id,
        user_id=user_id,
        event_type=EventType.VALIDATION_ERROR,
        parent_event_id=add_to_cart.event_id,
        payload=ValidationErrorPayload(
            validation_target="product_id",
            validation_rules=["exists", "in_stock"],
            failed_rules=["exists"],
            validation_details={"provided_id": "invalid_id", "available_ids": ["mouse1", "mouse2"]}
        ).dict()
    )
    client.log_event(validation_error)
    
    # 19. Timeout Error - API Call
    timeout_error = Event(
        run_id=run_id,
        user_id=user_id,
        event_type=EventType.TIMEOUT_ERROR,
        parent_event_id=api_request.event_id,
        payload=TimeoutErrorPayload(
            timeout_type="api_call",
            timeout_duration_ms=5000,
            expected_duration_ms=1000,
            operation_context={"api_name": "product_catalog", "endpoint": "/api/v1/products/search"}
        ).dict()
    )
    client.log_event(timeout_error)
    
    # 20. Chain Start - Product Comparison
    chain_start = Event(
        run_id=run_id,
        user_id=user_id,
        event_type=EventType.CHAIN_START,
        parent_event_id=agent_reasoning.event_id,
        payload=ChainStartPayload(
            chain_name="product_comparison",
            inputs={"products": ["mouse1", "mouse2"]},
            parameters={"comparison_type": "specifications"},
            chain_type="sequential",
            chain_metadata={"purpose": "help_user_decide"}
        ).dict()
    )
    client.log_event(chain_start)
    
    # 21. Chain End - Product Comparison
    chain_end = Event(
        run_id=run_id,
        user_id=user_id,
        event_type=EventType.CHAIN_END,
        parent_event_id=chain_start.event_id,
        payload=ChainEndPayload(
            chain_name="product_comparison",
            outputs={"recommended_product": "mouse1", "comparison_summary": "mouse1 has better value"},
            execution_time_ms=300,
            chain_type="sequential",
            chain_metadata={"purpose": "help_user_decide"},
            success=True
        ).dict()
    )
    client.log_event(chain_end)
    
    # 22. Error - Payment Processing
    payment_error = Event(
        run_id=run_id,
        user_id=user_id,
        event_type=EventType.ERROR,
        parent_event_id=cart_function.event_id,
        payload=ErrorPayload(
            error_type="PaymentProcessingError",
            error_message="Failed to process payment: Insufficient funds",
            error_code="PAYMENT_001",
            stack_trace="Traceback (most recent call last):\n  File 'payment_processor.py', line 45, in process_payment\n    raise InsufficientFundsError()",
            context={"amount": 45.99, "payment_method": "credit_card"},
            severity="error"
        ).dict()
    )
    client.log_event(payment_error)
    
    # End the session
    client.end_run(run_id=run_id)
    
    # Return all events in chronological order
    return [
        session_start,
        user_search,
        agent_plan,
        agent_reasoning,
        search_tool,
        api_request,
        api_response,
        tool_response,
        llm_request,
        llm_chunk,
        llm_response,
        agent_response,
        add_to_cart,
        cart_function,
        performance,
        user_feedback,
        validation_error,
        timeout_error,
        chain_start,
        chain_end,
        payment_error,
        session_end
    ]

if __name__ == "__main__":
    # Initialize rich console
    console = Console()
    
    # Print header
    console.print(Panel.fit(
        "[bold blue]E-commerce Session Demo[/bold blue]\n"
        "Demonstrating all InsideLLM event types through a shopping scenario",
        border_style="blue"
    ))
    
    # Create and print all events
    events = create_ecommerce_session()
    
    # Print each event with formatting
    for i, event in enumerate(events, 1):
        print_event_details(console, event, i)
    
    # Print session summary
    print_session_summary(console, events) 