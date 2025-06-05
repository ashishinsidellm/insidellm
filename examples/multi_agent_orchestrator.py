"""
InsideLLM Multi-Agent Orchestrator Example

This example demonstrates a multi-agent system where an orchestrator agent
delegates tasks to multiple specialized agents running on different servers.
All agents share the same session (run_id) with proper parent_event_id 
context for maintaining event ordering and traceability.
"""

import os
import sys
import time
import random
import asyncio
import json
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import threading

# Add parent directory to path for local testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import insidellm
from insidellm import Event, EventType


class RemoteAgent:
    """
    Simulates a remote agent running on a different server.
    In real implementation, this would be HTTP calls to remote agents.
    """
    
    def __init__(self, agent_name: str, specialization: str, server_url: str = None):
        self.agent_name = agent_name
        self.specialization = specialization
        self.server_url = server_url or f"http://agent-{agent_name.lower()}.example.com"
        self.processing_time = random.uniform(0.5, 2.0)  # Simulate network latency
    
    def process_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate remote agent processing"""
        print(f"[{self.agent_name}] Processing task on {self.server_url}")
        
        # Simulate network delay and processing
        time.sleep(self.processing_time)
        
        # Mock different agent responses based on specialization
        if self.specialization == "research":
            result = {
                "agent": self.agent_name,
                "task_type": "research",
                "findings": [
                    f"Research finding 1 for: {task[:30]}...",
                    f"Research finding 2 for: {task[:30]}...",
                    "Relevant data points identified"
                ],
                "confidence": random.uniform(0.8, 0.95),
                "sources": ["source1.com", "source2.com"],
                "processing_time_ms": int(self.processing_time * 1000)
            }
        elif self.specialization == "analysis":
            result = {
                "agent": self.agent_name,
                "task_type": "analysis", 
                "analysis": f"Detailed analysis of: {task[:50]}...",
                "insights": [
                    "Key insight 1 from analysis",
                    "Key insight 2 from analysis"
                ],
                "confidence": random.uniform(0.85, 0.98),
                "metrics": {"complexity": "medium", "accuracy": 0.92},
                "processing_time_ms": int(self.processing_time * 1000)
            }
        elif self.specialization == "synthesis":
            result = {
                "agent": self.agent_name,
                "task_type": "synthesis",
                "synthesis": f"Synthesized response for: {task[:40]}...",
                "combined_insights": [
                    "Synthesis point 1",
                    "Synthesis point 2", 
                    "Final recommendation"
                ],
                "confidence": random.uniform(0.9, 0.99),
                "quality_score": 0.94,
                "processing_time_ms": int(self.processing_time * 1000)
            }
        else:
            result = {
                "agent": self.agent_name,
                "task_type": "general",
                "response": f"Processed: {task}",
                "confidence": random.uniform(0.7, 0.9),
                "processing_time_ms": int(self.processing_time * 1000)
            }
        
        return result


class AgentOrchestrator:
    """
    Orchestrator that manages multiple remote agents and maintains
    proper event tracking with parent_event_id context.
    """
    
    def __init__(self, run_id: str, user_id: str):
        self.run_id = run_id
        self.user_id = user_id
        self.client = insidellm.get_client()
        
        # Initialize remote agents (simulating different servers)
        self.agents = {
            "research_agent": RemoteAgent("ResearchAgent", "research", "http://research-server:8001"),
            "analysis_agent": RemoteAgent("AnalysisAgent", "analysis", "http://analysis-server:8002"), 
            "synthesis_agent": RemoteAgent("SynthesisAgent", "synthesis", "http://synthesis-server:8003"),
            "validation_agent": RemoteAgent("ValidationAgent", "validation", "http://validation-server:8004")
        }
        
        print(f"Orchestrator initialized with {len(self.agents)} remote agents")
    
    def process_user_query(self, user_query: str) -> str:
        """
        Process user query through multi-agent orchestration.
        Maintains proper event hierarchy with parent_event_id context.
        """
        print(f"\n{'='*60}")
        print(f"ORCHESTRATOR: Processing user query")
        print(f"Query: {user_query}")
        print(f"Run ID: {self.run_id}")
        print(f"{'='*60}")
        
        # 1. Log user input event
        user_input_event = Event.create_user_input(
            run_id=self.run_id,
            user_id=self.user_id,
            input_text=user_query,
            input_type="text",
            metadata={
                "orchestrator": "multi_agent",
                "agent_count": len(self.agents),
                "session_type": "collaborative"
            }
        )
        self.client.log_event(user_input_event)
        print(f"✓ User input logged: {user_input_event.event_id}")
        
        # 2. Orchestrator planning phase
        planning_event_id = self._log_orchestrator_planning(user_query, user_input_event.event_id)
        
        # 3. Parallel agent execution with proper event tracking
        agent_results = self._execute_agents_parallel(user_query, planning_event_id)
        
        # 4. Result synthesis
        final_response = self._synthesize_results(agent_results, planning_event_id)
        
        # 5. Log final agent response
        response_event = Event(
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.AGENT_RESPONSE,
            parent_event_id=user_input_event.event_id,
            metadata={
                "orchestrator": "multi_agent",
                "agents_involved": ",".join(self.agents.keys()),
                "synthesis_complete": True
            },
            payload={
                "response_text": final_response,
                "response_type": "orchestrated_multi_agent",
                "response_confidence": 0.95,
                "agents_consulted": len(self.agents),
                "collaboration_mode": "parallel_execution",
                "synthesis_method": "weighted_consensus"
            }
        )
        self.client.log_event(response_event)
        print(f"✓ Final response logged: {response_event.event_id}")
        
        return final_response
    
    def _log_orchestrator_planning(self, query: str, parent_event_id: str) -> str:
        """Log orchestrator planning phase"""
        print(f"\n[ORCHESTRATOR] Planning agent delegation...")
        
        # Simulate planning logic
        time.sleep(0.2)
        
        plan = {
            "strategy": "parallel_multi_agent",
            "agents_to_engage": list(self.agents.keys()),
            "execution_order": "parallel",
            "synthesis_method": "weighted_consensus",
            "estimated_time_ms": 3000
        }
        
        planning_event = Event(
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.AGENT_PLANNING,
            parent_event_id=parent_event_id,
            metadata={
                "orchestrator": "multi_agent",
                "planning_phase": "agent_delegation"
            },
            payload={
                "plan_type": "multi_agent_orchestration",
                "planned_actions": f"agents:{','.join(self.agents.keys())}",
                "planning_time_ms": 200,
                "plan_confidence": 0.9,
                "execution_strategy": plan
            }
        )
        self.client.log_event(planning_event)
        print(f"✓ Planning logged: {planning_event.event_id}")
        
        return planning_event.event_id
    
    def _execute_agents_parallel(self, query: str, parent_event_id: str) -> Dict[str, Any]:
        """Execute multiple agents in parallel with proper event tracking"""
        print(f"\n[ORCHESTRATOR] Executing {len(self.agents)} agents in parallel...")
        
        agent_results = {}
        
        def execute_single_agent(agent_name: str, agent: RemoteAgent) -> None:
            """Execute a single agent and track its events"""
            thread_id = threading.get_ident()
            print(f"[{agent_name}] Starting execution on thread {thread_id}")
            
            # 1. Log agent reasoning start
            reasoning_event_id = self._log_agent_reasoning(agent_name, query, parent_event_id)
            
            # 2. Log LLM request (simulating agent's internal LLM call)
            llm_request_event_id = self._log_agent_llm_request(agent_name, query, reasoning_event_id)
            
            # 3. Execute agent (simulate remote call)
            try:
                start_time = time.time()
                result = agent.process_task(query, {"run_id": self.run_id, "parent_event_id": reasoning_event_id})
                execution_time_ms = int((time.time() - start_time) * 1000)
                
                # 4. Log LLM response
                self._log_agent_llm_response(agent_name, result, llm_request_event_id, execution_time_ms)
                
                # 5. Log tool usage (if agent used tools)
                if agent.specialization in ["research", "analysis"]:
                    tool_event_id = self._log_agent_tool_usage(agent_name, agent.specialization, reasoning_event_id)
                
                # 6. Log agent completion
                self._log_agent_completion(agent_name, result, reasoning_event_id, execution_time_ms)
                
                agent_results[agent_name] = result
                print(f"✓ [{agent_name}] Completed successfully")
                
            except Exception as e:
                # Log error if agent fails
                error_event = Event.create_error(
                    run_id=self.run_id,
                    user_id=self.user_id,
                    error_type="agent_execution_error",
                    error_message=f"Agent {agent_name} failed: {str(e)}",
                    error_code="AGENT_FAILURE",
                    parent_event_id=reasoning_event_id,
                    context={
                        "agent_name": agent_name,
                        "specialization": agent.specialization,
                        "server_url": agent.server_url
                    }
                )
                self.client.log_event(error_event)
                print(f"✗ [{agent_name}] Failed: {e}")
        
        # Execute all agents in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = [
                executor.submit(execute_single_agent, agent_name, agent)
                for agent_name, agent in self.agents.items()
            ]
            
            # Wait for all agents to complete
            for future in futures:
                future.result()
        
        print(f"✓ All {len(agent_results)} agents completed")
        return agent_results
    
    def _log_agent_reasoning(self, agent_name: str, query: str, parent_event_id: str) -> str:
        """Log agent reasoning start"""
        reasoning_event = Event(
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.AGENT_REASONING,
            parent_event_id=parent_event_id,
            metadata={
                "agent_name": agent_name,
                "orchestrator": "multi_agent",
                "reasoning_phase": "task_analysis"
            },
            payload={
                "reasoning_type": "specialized_processing",
                "reasoning_steps": [
                    f"Analyzing query for {self.agents[agent_name].specialization}",
                    f"Determining optimal processing approach",
                    f"Preparing {agent_name} execution context"
                ],
                "confidence_score": 0.9,
                "reasoning_time_ms": 100
            }
        )
        self.client.log_event(reasoning_event)
        return reasoning_event.event_id
    
    def _log_agent_llm_request(self, agent_name: str, query: str, parent_event_id: str) -> str:
        """Log agent's internal LLM request"""
        agent = self.agents[agent_name]
        
        llm_request = Event.create_llm_request(
            run_id=self.run_id,
            user_id=self.user_id,
            model_name=f"{agent_name.lower()}_model",
            provider="agent_internal",
            prompt=f"As a {agent.specialization} specialist, process: {query}",
            parent_event_id=parent_event_id,
            parameters={
                "temperature": 0.7,
                "max_tokens": 500,
                "agent_context": agent.specialization
            },
            metadata={
                "agent_name": agent_name,
                "server_url": agent.server_url,
                "orchestrator": "multi_agent"
            }
        )
        self.client.log_event(llm_request)
        return llm_request.event_id
    
    def _log_agent_llm_response(self, agent_name: str, result: Dict[str, Any], parent_event_id: str, execution_time_ms: int) -> str:
        """Log agent's LLM response"""
        agent = self.agents[agent_name]
        
        response_text = json.dumps(result, indent=2)
        
        llm_response = Event.create_llm_response(
            run_id=self.run_id,
            user_id=self.user_id,
            model_name=f"{agent_name.lower()}_model",
            provider="agent_internal",
            response_text=response_text,
            parent_event_id=parent_event_id,
            response_time_ms=execution_time_ms,
            usage={
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150
            },
            metadata={
                "agent_name": agent_name,
                "server_url": agent.server_url,
                "orchestrator": "multi_agent"
            }
        )
        self.client.log_event(llm_response)
        return llm_response.event_id
    
    def _log_agent_tool_usage(self, agent_name: str, tool_type: str, parent_event_id: str) -> str:
        """Log agent tool usage"""
        tool_call = Event(
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.TOOL_CALL,
            parent_event_id=parent_event_id,
            metadata={
                "agent_name": agent_name,
                "orchestrator": "multi_agent"
            },
            payload={
                "tool_name": f"{tool_type}_toolkit",
                "tool_type": "specialized_service",
                "parameters": {
                    "agent_specialization": tool_type,
                    "processing_mode": "optimized"
                },
                "call_id": f"{agent_name}_tool_001"
            }
        )
        self.client.log_event(tool_call)
        
        # Log tool response
        tool_response = Event(
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.TOOL_RESPONSE,
            parent_event_id=tool_call.event_id,
            metadata={
                "agent_name": agent_name,
                "orchestrator": "multi_agent"
            },
            payload={
                "tool_name": f"{tool_type}_toolkit",
                "tool_type": "specialized_service",
                "call_id": f"{agent_name}_tool_001",
                "response_data": f"{tool_type} processing completed successfully",
                "execution_time_ms": 200,
                "success": True
            }
        )
        self.client.log_event(tool_response)
        return tool_call.event_id
    
    def _log_agent_completion(self, agent_name: str, result: Dict[str, Any], parent_event_id: str, execution_time_ms: int) -> str:
        """Log agent completion"""
        completion_event = Event(
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.AGENT_RESPONSE,
            parent_event_id=parent_event_id,
            metadata={
                "agent_name": agent_name,
                "orchestrator": "multi_agent",
                "completion_status": "success"
            },
            payload={
                "response_text": f"Agent {agent_name} completed specialized processing",
                "response_type": "agent_completion",
                "response_confidence": result.get("confidence", 0.8),
                "execution_time_ms": execution_time_ms,
                "response_metadata": f"agent:{agent_name},specialization:{self.agents[agent_name].specialization}"
            }
        )
        self.client.log_event(completion_event)
        return completion_event.event_id
    
    def _synthesize_results(self, agent_results: Dict[str, Any], parent_event_id: str) -> str:
        """Synthesize results from all agents"""
        print(f"\n[ORCHESTRATOR] Synthesizing results from {len(agent_results)} agents...")
        
        # Log synthesis reasoning
        synthesis_reasoning = Event(
            run_id=self.run_id,
            user_id=self.user_id,
            event_type=EventType.AGENT_REASONING,
            parent_event_id=parent_event_id,
            metadata={
                "orchestrator": "multi_agent",
                "reasoning_phase": "result_synthesis"
            },
            payload={
                "reasoning_type": "multi_agent_synthesis",
                "reasoning_steps": [
                    "Collecting results from all specialized agents",
                    "Analyzing consistency and confidence scores",
                    "Weighting contributions based on agent expertise",
                    "Synthesizing coherent final response"
                ],
                "confidence_score": 0.95,
                "reasoning_time_ms": 300
            }
        )
        self.client.log_event(synthesis_reasoning)
        
        # Simulate synthesis processing
        time.sleep(0.3)
        
        # Create synthesized response
        total_confidence = sum(result.get("confidence", 0.8) for result in agent_results.values())
        avg_confidence = total_confidence / len(agent_results) if agent_results else 0.8
        
        synthesis = f"""
Multi-Agent Collaborative Analysis Complete

Based on specialized processing from {len(agent_results)} expert agents:

🔬 Research Findings:
{agent_results.get('research_agent', {}).get('findings', ['No research data available'])}

📊 Analysis Insights: 
{agent_results.get('analysis_agent', {}).get('insights', ['No analysis available'])}

🔗 Synthesis Results:
{agent_results.get('synthesis_agent', {}).get('combined_insights', ['No synthesis available'])}

✅ Validation Status: Completed with {avg_confidence:.1%} confidence

The collaborative analysis leveraged multiple specialized agents running on distributed 
servers to provide comprehensive coverage of your query. Each agent contributed their 
domain expertise while maintaining full traceability through the orchestration layer.
        """.strip()
        
        # Log performance metrics
        self._log_orchestration_metrics(agent_results, synthesis_reasoning.event_id)
        
        print(f"✓ Synthesis completed with {avg_confidence:.1%} confidence")
        return synthesis
    
    def _log_orchestration_metrics(self, agent_results: Dict[str, Any], parent_event_id: str) -> None:
        """Log orchestration performance metrics"""
        total_processing_time = sum(
            result.get("processing_time_ms", 0) for result in agent_results.values()
        )
        
        metrics = [
            ("agent_count", len(agent_results), "count"),
            ("total_processing_time", total_processing_time, "ms"),
            ("average_confidence", sum(r.get("confidence", 0.8) for r in agent_results.values()) / len(agent_results), "percentage"),
            ("orchestration_efficiency", 0.92, "percentage")
        ]
        
        for metric_name, value, unit in metrics:
            metric_event = Event(
                run_id=self.run_id,
                user_id=self.user_id,
                event_type=EventType.PERFORMANCE_METRIC,
                parent_event_id=parent_event_id,
                metadata={
                    "orchestrator": "multi_agent",
                    "metric_category": "orchestration"
                },
                payload={
                    "metric_name": metric_name,
                    "metric_value": value,
                    "metric_unit": unit,
                    "metric_type": "gauge"
                }
            )
            self.client.log_event(metric_event)


def run_multi_agent_example():
    """Run the multi-agent orchestrator example"""
    print("InsideLLM Multi-Agent Orchestrator Example")
    print("=" * 60)
    print("Demonstrating multi-agent system with proper event tracking")
    print("All agents share the same session with parent_event_id context\n")
    
    # Initialize SDK for local testing
    insidellm.initialize(local_testing=True, log_directory="multi_agent_logs")
    
    client = insidellm.get_client()
    
    # Start a shared session for all agents
    run_id = client.start_run(
        user_id="multi-agent-user", 
        metadata={
            "session_type": "multi_agent_orchestration",
            "architecture": "distributed_agents",
            "orchestrator_version": "1.0"
        }
    )
    
    # Create orchestrator
    orchestrator = AgentOrchestrator(run_id, "multi-agent-user")
    
    # Test queries for multi-agent processing
    test_queries = [
        "Analyze the impact of artificial intelligence on modern healthcare systems",
        "Research and synthesize information about renewable energy adoption trends",
        "Evaluate the effectiveness of remote work policies in technology companies"
    ]
    
    # Process each query through the multi-agent system
    for i, query in enumerate(test_queries, 1):
        print(f"\n🚀 QUERY {i}/{len(test_queries)}")
        response = orchestrator.process_user_query(query)
        print(f"\n📋 FINAL RESPONSE:")
        print(response)
        print(f"\n{'='*60}")
        
        # Brief pause between queries
        time.sleep(1)
    
    # End session and show statistics
    client.end_run(run_id)
    
    # Show session summary (for local testing mode)
    print(f"\n📊 SESSION SUMMARY:")
    print(f"Run ID: {run_id}")
    print(f"Multi-agent orchestration completed successfully")
    
    insidellm.shutdown()
    
    print(f"\n✅ Multi-agent orchestration example completed!")
    print(f"Check 'multi_agent_logs/' directory for detailed event logs")
    print(f"\nKey Features Demonstrated:")
    print(f"• Shared session (run_id) across multiple agents")
    print(f"• Proper parent_event_id context for event ordering")
    print(f"• Parallel agent execution with individual event tracking")
    print(f"• Distributed agent architecture simulation")
    print(f"• Complete traceability of multi-agent workflows")
    print(f"• Performance metrics and error handling")


if __name__ == "__main__":
    run_multi_agent_example()