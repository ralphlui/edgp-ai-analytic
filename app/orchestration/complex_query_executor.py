import logging
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.prompts.complex_executor_prompts import (
    ComplexExecutorToolSelectionPrompt,
    ComplexExecutorResponseFormattingPrompt
)

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
    chart_type: Optional[str]  # User's preferred chart type (e.g., 'bar', 'pie', 'line')
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
    
    # Initialize secure prompt template for tool selection
    tool_selection_prompt = ComplexExecutorToolSelectionPrompt()
    
    # Get secure system prompt with leakage prevention
    system_prompt = tool_selection_prompt.get_system_prompt()
    
    # Format user message with security validation and structural isolation
    user_prompt = tool_selection_prompt.format_user_message(
        metric_type=metric_type,
        target_type=target_type,
        target=target
    )
    
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
# LLM Chart Type Suggestion
# ============================================================================

async def suggest_chart_type_with_llm(
    comparison_data: Dict[str, Any],
    user_query: str
) -> str:
    """
    Use LLM to suggest the best chart type based on comparison data characteristics.
    
    This function analyzes:
    - Number of targets being compared
    - Metric type (success_rate vs failure_rate)
    - Data distribution patterns
    - User's phrasing in the query
    
    Args:
        comparison_data: The comparison result from execute_compare_results
        user_query: Original user question for context
    
    Returns:
        Suggested chart type: 'bar', 'horizontal_bar', 'line', 'pie', or 'grouped_bar'
    """
    logger.info("Invoking LLM to suggest chart type")
    
    # Extract relevant information from comparison data
    num_targets = len(comparison_data.get("comparison_details", []))
    metric = comparison_data.get("metric_type", "unknown")
    winner = comparison_data.get("winner", {}).get("target", "unknown")
    
    # Build prompt for LLM
    suggestion_prompt = f"""
You are a data visualization expert. Based on the comparison data, suggest the MOST APPROPRIATE chart type.

**User Query**: {user_query}

**Comparison Data**:
- Metric Type: {metric}
- Number of Targets: {num_targets}
- Winner: {winner}

**Available Chart Types**:
1. **bar** - Vertical bar chart (BEST for comparing success/failure rates across targets)
2. **horizontal_bar** - Horizontal bar chart (better for long target names)
3. **line** - Line chart (best for showing trends over time or ordered categories)
4. **grouped_bar** - Grouped bar chart (best for comparing success AND failure rates side-by-side)

**CRITICAL RULES**:
- **NEVER use 'pie' for comparing success/failure rates** - Pie charts are for showing parts of a whole, NOT for comparing independent percentages
- For success_rate or failure_rate comparisons → Use 'bar' (vertical) or 'horizontal_bar'
- For comparing BOTH success AND failure rates → Use 'grouped_bar'
- For time-series or sequential data → Use 'line'

**Selection Guidelines**:
- **Comparing success rates or failure rates (2-5 targets)** → 'bar' (DEFAULT)
- **Target names are long (>15 chars)** → 'horizontal_bar'
- **Want to see success AND failure side-by-side** → 'grouped_bar'
- **Time-series or trend data** → 'line'

**Your Task**: Respond with ONLY ONE WORD - the chart type name (bar/horizontal_bar/line/grouped_bar).
No explanation, just the type name.
"""
    
    try:
        # Initialize LLM
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            temperature=0.0  # Deterministic output
        )
        
        # Get LLM suggestion
        response = await llm.ainvoke(suggestion_prompt)
        suggested_type = response.content.strip().lower()
        
        # Validate against allowed types (pie removed - not suitable for rate comparisons)
        allowed_types = ['bar', 'horizontal_bar', 'line', 'grouped_bar']
        if suggested_type not in allowed_types:
            logger.warning(f"LLM suggested invalid type '{suggested_type}', defaulting to 'bar'")
            suggested_type = 'bar'
        
        logger.info(f"LLM suggested chart type: {suggested_type}")
        return suggested_type
        
    except Exception as e:
        logger.error(f"LLM chart type suggestion failed: {e}, defaulting to 'bar'")
        return 'bar'  # Safe fallback


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
    2. Checks if user specified a chart_type preference
    3. If not, uses LLM to suggest the best chart type
    4. Calls chart_service to generate visualization
    5. Returns base64-encoded chart image
    
    Args:
        state: Current execution state with step_results and chart_type
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
    
    # Determine chart type: use user preference or ask LLM
    chart_type = state.get("chart_type")
    
    if chart_type:
        logger.info(f"Using user-specified chart type: {chart_type}")
    else:
        # logger.info("No user preference, asking LLM for chart type suggestion")
        # chart_type = await suggest_chart_type_with_llm(
        #     comparison_data=comparison_data,
        #     user_query=state["user_query"]
        # )
        # logger.info(f"LLM suggested chart type: {chart_type}")
        chart_type = 'bar'  # Default to 'bar' for simplicity
    
    # Generate chart using chart_service
    from app.services.chart_service import generate_comparison_chart
    
    try:
        chart_base64 = generate_comparison_chart(
            comparison_data=comparison_data,
            chart_type=chart_type
        )
        
        if chart_base64:
            logger.info(f"Chart generated successfully ({len(chart_base64)} bytes)")
            logger.info(f"Chart type used: {chart_type}")
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
    
    # Initialize secure prompt template for response formatting
    response_formatting_prompt = ComplexExecutorResponseFormattingPrompt()
    
    # Get secure system prompt with leakage prevention
    system_prompt = response_formatting_prompt.get_system_prompt()
    
    # Format user message with security validation and structural isolation
    user_prompt = response_formatting_prompt.format_user_message(
        user_query=user_query,
        targets=targets,
        winner=winner,
        metric=metric,
        details=details,
        has_chart=bool(chart_image)
    )
    
    # Build messages array with secure prompts
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
        
        # Add warning if pie/donut chart was used for comparison
        chart_type = state.get("chart_type")
        if chart_type in ["pie", "donut"]:
            warning_message = (
                "\n\n💡 Chart Type Note: Pie/donut charts show proportions (parts of a whole), "
                "which may be misleading when comparing independent success/failure rates. "
                "The percentages displayed represent relative proportions, not actual rate values.\n\n"
                "📊 For more accurate comparisons, consider using:\n"
                "• bar - Shows actual rate values clearly (recommended)\n"
                "• horizontal_bar - Better for long target names\n"
                "• grouped_bar - Shows success AND failure rates side-by-side"
            )
            message_text += warning_message
            logger.info(f"Added chart warning for {chart_type} chart in comparison")
        
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
        
        # Add warning if pie/donut chart was used for comparison
        chart_type = state.get("chart_type")
        if chart_type in ["pie", "donut"]:
            warning_message = (
                "\n\n💡 Chart Type Note: Pie/donut charts show proportions (parts of a whole), "
                "which may be misleading when comparing independent success/failure rates. "
                "The percentages displayed represent relative proportions, not actual rate values.\n\n"
                "📊 For more accurate comparisons, consider using:\n"
                "• bar - Shows actual rate values clearly (recommended)\n"
                "• horizontal_bar - Better for long target names\n"
                "• grouped_bar - Shows success AND failure rates side-by-side"
            )
            fallback_message += warning_message
            logger.info(f"Added chart warning for {chart_type} chart in comparison (fallback)")
        
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


async def should_continue(state: ExecutionState) -> str:
    """
    LLM-based decision to determine if execution should continue or end.
    
    The LLM analyzes the current execution state and decides whether to:
    - CONTINUE: Proceed to next step
    - END: Stop execution (completed or unrecoverable errors)
    
    Returns:
        "continue" if should proceed, "end" otherwise
    """
    import json
    
    plan = state["plan"]
    current_index = state["current_step_index"]
    step_results = state.get("step_results", {})
    errors = state.get("errors", [])
    
    # Build context for LLM decision
    context = {
        "total_steps": len(plan["steps"]),
        "current_step": current_index,
        "remaining_steps": len(plan["steps"]) - current_index,
        "completed_steps": list(step_results.keys()),
        "errors": errors,
        "next_step": plan["steps"][current_index] if current_index < len(plan["steps"]) else None,
        "last_result": list(step_results.values())[-1] if step_results else None
    }
    
    # Create decision prompt
    decision_prompt = f"""You are an execution controller analyzing whether to continue or end execution.

**Current Execution State:**
{json.dumps(context, indent=2)}

**Decision Criteria:**
1. If there are CRITICAL errors that block further execution → END
2. If all planned steps are completed successfully → END
3. If current step failed but subsequent steps can still proceed → CONTINUE
4. If dependencies for next step are missing → END
5. If within normal execution flow with steps remaining → CONTINUE

**Important:**
- Non-critical errors may allow continuation if remaining steps are independent
- Consider if partial results are still valuable
- Evaluate if next step can execute despite previous issues

Analyze the execution state and respond with ONLY one word: "CONTINUE" or "END"
"""
    
    try:
        # Call LLM for decision
        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0
        )
        
        response = await llm.ainvoke(decision_prompt)
        decision = response.content.strip().upper()
        logger.info("LLM Decision Response: %s", decision)
        
        # Validate LLM response
        if decision not in ["CONTINUE", "END"]:
            logger.warning(f"Invalid LLM decision: '{decision}', applying deterministic fallback logic")
            # Fallback to previous deterministic logic
            if errors:
                logger.error(f"Deterministic decision: Execution stopped due to errors: {errors}")
                return "end"
            
            if current_index < len(plan["steps"]):
                logger.info(f"Deterministic decision: CONTINUE (step {current_index + 1}/{len(plan['steps'])})")
                return "continue"
            else:
                logger.info("Deterministic decision: All steps completed successfully")
                return "end"
        
        # Log LLM decision
        if decision == "END":
            logger.info(f"LLM Decision: END execution at step {current_index}/{len(plan['steps'])}")
        else:
            logger.info(f"LLM Decision: CONTINUE to step {current_index + 1}/{len(plan['steps'])}")
        
        # Map to graph edge names
        return "end" if decision == "END" else "continue"
        
    except Exception as e:
        logger.error(f"Error in LLM decision-making: {e}, using deterministic fallback logic")
        # Fallback to previous deterministic logic on exception
        if errors:
            logger.error(f"Deterministic decision: Execution stopped due to errors: {errors}")
            return "end"
        
        if current_index < len(plan["steps"]):
            logger.info(f"Deterministic decision: CONTINUE (step {current_index + 1}/{len(plan['steps'])})")
            return "continue"
        else:
            logger.info("Deterministic decision: All steps completed successfully")
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
    user_query: str,
    chart_type: Optional[str] = None
) -> dict:
    """
    Execute a complex query execution plan.
    
    This is the main entry point for executing plans generated by planner_agent.
    
    Args:
        plan: ExecutionPlan dict from planner_agent.create_execution_plan()
        org_id: Organization ID for multi-tenant data isolation
        user_query: Original user question
        chart_type: User's preferred chart type (e.g., 'bar', 'pie', 'line'). 
                   If None, LLM will suggest the best chart type based on data.
    
    Returns:
        Final response dict: {success, message, chart_image}
    """
    logger.info(f"Starting execution for plan: {plan['plan_id']}")
    logger.info(f"Organization: {org_id}")
    logger.info(f"Steps to execute: {len(plan['steps'])}")
    logger.info(f"Chart type: {chart_type or 'LLM will suggest'}")
    
    # Initialize state
    initial_state: ExecutionState = {
        "plan": plan,
        "org_id": org_id,
        "user_query": user_query,
        "chart_type": chart_type,
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
