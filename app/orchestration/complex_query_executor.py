"""
Complex Query Executor - Executes multi-step execution plans using LangGraph.

This module implements a LangGraph-based executor that:
1. Takes execution plans from planner_agent
2. Executes each step sequentially with state management
3. Uses LLM for intelligent tool selection in query_analytics steps
4. Supports multi-tenant data isolation via org_id
5. Handles comparison logic and chart generation
6. Formats final results with natural language

Architecture:
- ExecutionState: TypedDict for state management (org_id, results, errors)
- Action Handlers: Functions for each action type (query, compare, chart, format)
- LangGraph Workflow: Orchestrates execution flow with conditional routing
- Multi-tenant Support: org_id passed through entire execution pipeline
"""
import logging
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from app.config import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger("complex_query_executor")


# ============================================================================
# State Management
# ============================================================================

class ExecutionState(TypedDict):
    """
    State for complex query execution workflow.
    
    This state is used internally by LangGraph for orchestration.
    The org_id is passed through Python state but NOT sent to LLM prompts.
    """
    plan: dict  # ExecutionPlan from planner_agent
    org_id: str  # Organization ID for multi-tenant data isolation
    user_query: str  # Original user question
    current_step_index: int  # Current step being executed
    step_results: Dict[str, Any]  # Results from each step (step_id -> result)
    errors: List[str]  # Accumulated errors during execution
    final_result: Optional[dict]  # Final formatted response


# ============================================================================
# Action Handler: Query Analytics (with LLM Tool Selection)
# ============================================================================

async def execute_query_analytics(
    state: ExecutionState, 
    step: dict
) -> Dict[str, Any]:
    """
    Execute analytics query with LLM-based intelligent tool selection.
    
    This handler:
    1. Analyzes the target (domain/file) from step params
    2. Uses LLM to select appropriate analytics tool
    3. Passes org_id to the selected tool for data isolation
    4. Returns raw analytics data
    
    Args:
        state: Current execution state (contains org_id)
        step: Current step with params {target, metric_type}
    
    Returns:
        Analytics data dictionary with metrics
    """
    params = step.get("params", {})
    target = params.get("target")  # e.g., "product.csv" or "customer"
    metric_type = params.get("metric_type", "success_rate")  # success_rate or failure_rate
    org_id = state["org_id"]
    
    logger.info(f"Executing query_analytics: target={target}, metric={metric_type}, org_id={org_id}")
    
    # Determine if target is domain or file
    target_type = "file_name" if target and "." in target else "domain_name"
    
    # Get available analytics tools
    from app.tools.analytics_tools import get_analytics_tools
    tools = get_analytics_tools()
    
    # Create LLM with tool calling capability
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    llm_with_tools = llm.bind_tools(tools)
    
    # System prompt for tool selection
    system_prompt = """You are an analytics tool selector. Your job is to:

1. Analyze the requested metric_type
2. Select the appropriate analytics tool
3. Format the tool call with correct parameters

Available Tools:
- generate_success_rate_report: For success rate analysis
- generate_failure_rate_report: For failure rate analysis

Rules:
- metric_type = "success_rate" → Use generate_success_rate_report
- metric_type = "failure_rate" → Use generate_failure_rate_report
- Provide EXACTLY ONE of domain_name or file_name (never both)
- Set the unused parameter to null

Example:
metric_type: "success_rate"
target_type: "file_name"
target: "product.csv"

→ Tool: generate_success_rate_report
→ Args: {"file_name": "product.csv", "domain_name": null}
"""
    
    # User prompt with parameters
    user_prompt = f"""Select the analytics tool for:

metric_type: "{metric_type}"
target_type: "{target_type}"
target: "{target}"

Call the appropriate tool with the correct parameters."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        # Let LLM select and call the tool
        response = llm_with_tools.invoke(messages)
        
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            logger.info(f"LLM selected tool: {tool_name} with args: {tool_args}")
            
            # Add org_id to tool arguments for multi-tenant support
            tool_args["org_id"] = org_id
            
            # Execute the selected tool
            for tool in tools:
                if tool.name == tool_name:
                    result = tool.invoke(tool_args)
                    
                    if result.get("success"):
                        logger.info(f"Tool {tool_name} executed successfully")
                        return result.get("data", {})
                    else:
                        error_msg = result.get("error", "Tool execution failed")
                        logger.error(f"Tool {tool_name} failed: {error_msg}")
                        raise Exception(error_msg)
            
            raise Exception(f"Tool {tool_name} not found")
        else:
            raise Exception("LLM did not call any tool")
            
    except Exception as e:
        logger.exception(f"Query analytics failed: {e}")
        raise


# ============================================================================
# Action Handler: Compare Results
# ============================================================================

async def execute_compare_results(
    state: ExecutionState,
    step: dict
) -> Dict[str, Any]:
    """
    Compare analytics results from multiple targets.
    
    This handler:
    1. Retrieves results from previous query_analytics steps
    2. Compares metrics (success_rate or failure_rate)
    3. Determines winner and calculates differences
    4. Returns comparison data structure
    
    Args:
        state: Current execution state with step_results
        step: Current step with params {compare_steps, metric}
    
    Returns:
        Comparison data: {targets, metric, winner, comparison_data}
    """
    params = step.get("params", {})
    compare_steps = params.get("compare_steps", [])  # List of step IDs to compare
    metric = params.get("metric", "success_rate")  # Metric to compare
    
    logger.info(f"Executing compare_results: steps={compare_steps}, metric={metric}")
    
    # Collect results from specified steps
    targets_data = []
    for step_id in compare_steps:
        if step_id in state["step_results"]:
            result = state["step_results"][step_id]
            targets_data.append({
                "target": result.get("target_value"),
                "metric_value": result.get(metric, 0),
                "total_requests": result.get("total_requests", 0),
                "successful_requests": result.get("successful_requests", 0),
                "failed_requests": result.get("failed_requests", 0)
            })
    
    if len(targets_data) < 2:
        raise Exception(f"Need at least 2 targets to compare, got {len(targets_data)}")
    
    # Find winner (highest metric value)
    winner = max(targets_data, key=lambda x: x["metric_value"])
    
    # Calculate differences
    comparison_data = {
        "targets": [t["target"] for t in targets_data],
        "metric": metric,
        "winner": winner["target"],
        "comparison_details": targets_data,
        "metric_type": metric
    }
    
    logger.info(f"Comparison complete: winner={winner['target']} with {winner['metric_value']}%")
    
    return comparison_data


# ============================================================================
# Action Handler: Generate Chart
# ============================================================================

async def execute_generate_chart(
    state: ExecutionState,
    step: dict
) -> Dict[str, str]:
    """
    Generate comparison chart from comparison data.
    
    This handler:
    1. Retrieves comparison data from previous step
    2. Calls chart_service to generate visualization
    3. Returns base64-encoded chart image
    
    Args:
        state: Current execution state with step_results
        step: Current step with params {comparison_step_id}
    
    Returns:
        Chart data: {chart_image: base64_string}
    """
    params = step.get("params", {})
    comparison_step_id = params.get("comparison_step_id")
    
    logger.info(f"Executing generate_chart: using data from step {comparison_step_id}")
    
    # Get comparison data from previous step
    if comparison_step_id not in state["step_results"]:
        raise Exception(f"Comparison step {comparison_step_id} not found in results")
    
    comparison_data = state["step_results"][comparison_step_id]
    
    # Generate chart using chart_service
    from app.services.chart_service import generate_comparison_chart
    
    try:
        chart_base64 = generate_comparison_chart(comparison_data)
        
        if chart_base64:
            logger.info(f"Chart generated successfully ({len(chart_base64)} bytes)")
            return {"chart_image": chart_base64}
        else:
            logger.warning("Chart generation returned None")
            return {"chart_image": None}
            
    except Exception as e:
        logger.exception(f"Chart generation failed: {e}")
        return {"chart_image": None}


# ============================================================================
# Action Handler: Format Response
# ============================================================================

async def execute_format_response(
    state: ExecutionState,
    step: dict
) -> Dict[str, Any]:
    """
    Format final response using LLM for natural language generation.
    
    This handler:
    1. Retrieves comparison and chart data
    2. Uses LLM to generate conversational response
    3. Returns structured response with message and chart
    
    Args:
        state: Current execution state with all step_results
        step: Current step with params {comparison_step_id, chart_step_id}
    
    Returns:
        Final response: {success, message, chart_image}
    """
    params = step.get("params", {})
    comparison_step_id = params.get("comparison_step_id")
    chart_step_id = params.get("chart_step_id")
    user_query = state["user_query"]
    
    logger.info(f"Executing format_response: comparison={comparison_step_id}, chart={chart_step_id}")
    
    # Get comparison data
    comparison_data = state["step_results"].get(comparison_step_id, {})
    
    # Get chart image
    chart_image = None
    if chart_step_id and chart_step_id in state["step_results"]:
        chart_image = state["step_results"][chart_step_id].get("chart_image")
    
    # Build comparison data
    targets = comparison_data.get("targets", [])
    winner = comparison_data.get("winner")
    metric = comparison_data.get("metric", "success_rate")
    details = comparison_data.get("comparison_details", [])
    
    # System prompt - defines role and output requirements
    system_prompt = """You are a helpful analytics assistant specialized in presenting comparison results.

Your role:
- Analyze comparison data and present it in a conversational, easy-to-understand format
- Highlight key insights and differences between targets
- Use a friendly, professional tone
- Make the data accessible to non-technical users

Output requirements:
1. Directly answer the user's question
2. Clearly identify the winner and explain why
3. Highlight key differences between targets
4. Provide actionable insights about the comparison
5. Use appropriate emojis (🏆 for winner, 📊 for stats, etc.) for visual clarity
6. Keep the tone conversational but professional
7. Mention when a chart visualization is available
8. Return ONLY the message text (not JSON)

Format guidelines:
- Start with a direct answer to the user's question
- Use bullet points or structured paragraphs for clarity
- Include specific numbers and percentages
- End with a summary or recommendation if appropriate
"""
    
    # User prompt - provides the actual data
    user_prompt = f"""The user asked: "{user_query}"

I compared {metric.replace('_', ' ')} across {len(targets)} targets and found:

Comparison Results:
"""
    
    for detail in details:
        is_winner = detail["target"] == winner
        user_prompt += f"\n{'🏆 ' if is_winner else ''}**{detail['target']}**:"
        user_prompt += f"\n  - {metric.replace('_', ' ').title()}: {detail['metric_value']}%"
        user_prompt += f"\n  - Total Requests: {detail['total_requests']}"
        user_prompt += f"\n  - Successful: {detail['successful_requests']}"
        user_prompt += f"\n  - Failed: {detail['failed_requests']}"
    
    user_prompt += f"\n\nWinner: {winner}"
    user_prompt += f"\n\nChart visualization: {'Available' if chart_image else 'Not available'}"
    user_prompt += "\n\nPlease format this into a natural, conversational response following the guidelines."
    
    # Build messages array with proper role separation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    logger.info("Generating LLM-formatted response...")
    
    try:
        # Use LLM to generate natural response
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.7, api_key=OPENAI_API_KEY)
        response = llm.invoke(messages)
        
        message_text = response.content
        logger.info(f"LLM response generated ({len(message_text)} chars)")
        
        # Build structured response
        final_response = {
            "success": True,
            "message": message_text,
            "chart_image": chart_image
        }
        
        return final_response
        
    except Exception as e:
        logger.exception(f"LLM formatting failed: {e}")
        
        # Fallback to basic formatting if LLM fails
        fallback_message = f"📊 Comparison Results for: {user_query}\n\n"
        fallback_message += f"Winner: 🏆 {winner}\n\n"
        fallback_message += "Details:\n"
        
        for detail in details:
            is_winner = detail["target"] == winner
            fallback_message += f"\n{'🏆 ' if is_winner else '•'} {detail['target']}:\n"
            fallback_message += f"  - {metric.replace('_', ' ').title()}: {detail['metric_value']}%\n"
            fallback_message += f"  - Total: {detail['total_requests']}, Success: {detail['successful_requests']}, Failed: {detail['failed_requests']}\n"
        
        if chart_image:
            fallback_message += "\n Chart visualization is included below."
        
        logger.info("Using fallback formatting due to LLM error")
        
        return {
            "success": True,
            "message": fallback_message,
            "chart_image": chart_image
        }


# ============================================================================
# Action Handlers Registry
# ============================================================================

ACTION_HANDLERS = {
    "query_analytics": execute_query_analytics,
    "compare_results": execute_compare_results,
    "generate_chart": execute_generate_chart,
    "format_response": execute_format_response
}


# ============================================================================
# LangGraph Workflow Nodes
# ============================================================================

async def execute_step_node(state: ExecutionState) -> dict:
    """
    Execute current step using appropriate action handler.
    
    This node:
    1. Gets current step from plan
    2. Dispatches to appropriate action handler
    3. Stores result in step_results
    4. Increments step index
    
    Returns:
        Updated state dict
    """
    plan = state["plan"]
    steps = plan["steps"]
    current_index = state["current_step_index"]
    
    # Check if we're done
    if current_index >= len(steps):
        return {"current_step_index": current_index}
    
    step = steps[current_index]
    step_id = step["step_id"]
    action = step["action"]
    
    logger.info(f"Executing step {current_index + 1}/{len(steps)}: {step_id} ({action})")
    
    try:
        # Get action handler
        if action not in ACTION_HANDLERS:
            raise Exception(f"Unknown action: {action}")
        
        handler = ACTION_HANDLERS[action]
        
        # Execute action
        result = await handler(state, step)
        
        # Store result
        step_results = state["step_results"].copy()
        step_results[step_id] = result
        
        # Special handling for format_response (final step)
        if action == "format_response":
            return {
                "step_results": step_results,
                "current_step_index": current_index + 1,
                "final_result": result
            }
        
        return {
            "step_results": step_results,
            "current_step_index": current_index + 1
        }
        
    except Exception as e:
        logger.exception(f"Step {step_id} failed: {e}")
        errors = state["errors"].copy()
        errors.append(f"Step {step_id} failed: {str(e)}")
        
        return {
            "errors": errors,
            "current_step_index": current_index + 1
        }


def should_continue(state: ExecutionState) -> str:
    """
    Determine if execution should continue or end.
    
    Returns:
        "continue" if more steps to execute, "end" otherwise
    """
    plan = state["plan"]
    current_index = state["current_step_index"]
    
    # Check for errors
    if state["errors"]:
        logger.error(f"Execution stopped due to errors: {state['errors']}")
        return "end"
    
    # Check if more steps
    if current_index < len(plan["steps"]):
        return "continue"
    else:
        logger.info("All steps completed successfully")
        return "end"


# ============================================================================
# LangGraph Workflow Builder
# ============================================================================

def build_execution_graph() -> StateGraph:
    """
    Build LangGraph workflow for complex query execution.
    
    Workflow:
    1. execute_step_node - Execute current step
    2. should_continue - Check if done or continue
    3. Loop back to execute_step_node or END
    
    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(ExecutionState)
    
    # Add nodes
    workflow.add_node("execute_step", execute_step_node)
    
    # Set entry point
    workflow.set_entry_point("execute_step")
    
    # Add conditional edge
    workflow.add_conditional_edges(
        "execute_step",
        should_continue,
        {
            "continue": "execute_step",
            "end": END
        }
    )
    
    return workflow.compile()


# ============================================================================
# Main Execution Function
# ============================================================================

async def execute_plan(
    plan: dict,
    org_id: str,
    user_query: str
) -> dict:
    """
    Execute a complex query execution plan.
    
    This is the main entry point for executing plans generated by planner_agent.
    
    Args:
        plan: ExecutionPlan dict from planner_agent.create_execution_plan()
        org_id: Organization ID for multi-tenant data isolation
        user_query: Original user question
    
    Returns:
        Final response dict: {success, message, chart_image}
    """
    logger.info(f"Starting execution for plan: {plan['plan_id']}")
    logger.info(f"Organization: {org_id}")
    logger.info(f"Steps to execute: {len(plan['steps'])}")
    
    # Initialize state
    initial_state: ExecutionState = {
        "plan": plan,
        "org_id": org_id,
        "user_query": user_query,
        "current_step_index": 0,
        "step_results": {},
        "errors": [],
        "final_result": None
    }
    
    # Build and execute workflow
    graph = build_execution_graph()
    
    try:
        final_state = await graph.ainvoke(initial_state)
        
        # Check for errors
        if final_state["errors"]:
            logger.error(f"Execution completed with errors: {final_state['errors']}")
            return {
                "success": False,
                "message": f"Execution failed: {'; '.join(final_state['errors'])}",
                "chart_image": None
            }
        
        # Return final result
        if final_state["final_result"]:
            logger.info("Execution completed successfully")
            return final_state["final_result"]
        else:
            logger.error("Execution completed but no final result found")
            return {
                "success": False,
                "message": "Execution completed but no result was generated",
                "chart_image": None
            }
            
    except Exception as e:
        logger.exception(f"Execution failed: {e}")
        return {
            "success": False,
            "message": f"Execution error: {str(e)}",
            "chart_image": None
        }
