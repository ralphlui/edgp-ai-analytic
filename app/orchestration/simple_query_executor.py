import logging
from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from config.app_config import OPENAI_API_KEY, OPENAI_MODEL
from app.prompts.simple_executor_prompts import (
    SimpleExecutorToolSelectionPrompt,
    SimpleExecutorResponseFormattingPrompt
)
from app.security.pii_redactor import PIIRedactionFilter, redact_pii


logger = logging.getLogger("analytic_agent")

# Add PII redaction filter to this logger
pii_filter = PIIRedactionFilter()
logger.addFilter(pii_filter)


class AnalyticsState(TypedDict):
    """State for analytics workflow."""
    user_query: str
    extracted_data: dict  # {report_type, domain_name, file_name}
    org_id: str  # Organization ID for multi-tenant data isolation
    tool_result: dict  # Raw data from tools
    chart_image: str  # Base64 chart image (plain base64 string or None)
    final_response: dict  # Structured response: {success, message, chart_image}


def execute_analytics_tool(state: AnalyticsState) -> dict:
    """
    Intelligent tool selection using report_type priority with LLM fallback.
    
    Two-tier selection strategy:
    1. If report_type provided (from multi-turn context) → Use it directly
    2. If report_type is None → LLM analyzes user query keywords
    
    This provides:
    - Accuracy for multi-turn conversations (explicit intent)
    - Flexibility for single-turn queries (keyword analysis)
    
    Returns raw data only - no message formatting.
    """
    user_query = state["user_query"]
    extracted_data = state["extracted_data"]
    report_type = extracted_data.get("report_type")
    domain_name = extracted_data.get("domain_name")
    file_name = extracted_data.get("file_name")
    
    logger.info(f"Tool selection for query: '{user_query}'")
    logger.info(f"Report type: {report_type}, Domain: {domain_name}, File: {file_name}")
    
    # Get analytics tools
    from app.tools.analytics_tools import get_analytics_tools
    tools = get_analytics_tools()
    
    # HYBRID APPROACH: LLM-first with deterministic fallback
    # Strategy 1: Always try LLM first (most flexible)
    # Strategy 2: If LLM fails, use deterministic fallback (most reliable)
    
    logger.info(f"Attempting LLM tool selection first...")
    
    # Create LLM with tool calling capability
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    llm_with_tools = llm.bind_tools(tools)
    
    # Initialize secure prompt template
    tool_selection_prompt = SimpleExecutorToolSelectionPrompt()
    
    # Get secure system prompt with leakage prevention
    system_prompt = tool_selection_prompt.get_system_prompt()
    
    # Format user message with security validation and structural isolation
    user_prompt = tool_selection_prompt.format_user_message(
        user_query=user_query,
        report_type=report_type or "",
        domain_name=domain_name or "",
        file_name=file_name or ""
    )

    # Combine system and user prompts
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Let LLM decide which tool to invoke
    try:
        response = llm_with_tools.invoke(messages)
        
        # Check if LLM called a tool
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            logger.info(f"LLM selected tool: {tool_name}")
            logger.info(f"Tool arguments: {tool_args}")
            
            # Add org_id to tool arguments for multi-tenant support
            org_id = state.get("org_id")
            if org_id:
                tool_args["org_id"] = org_id
                logger.info(f"Added org_id to tool args: {org_id}")
            
            # Execute the selected tool
            for tool in tools:
                if tool.name == tool_name:
                    result = tool.invoke(tool_args)
                    logger.info(f"Tool execution complete: success={result.get('success')}")
                    
                    # Store which tool was called for accurate chart filtering
                    if "data" in result and isinstance(result["data"], dict):
                        # Determine report_type from the tool that was called
                        if "success_rate" in tool_name:
                            result["data"]["_report_type"] = "success_rate"
                        elif "failure_rate" in tool_name:
                            result["data"]["_report_type"] = "failure_rate"
                    
                    return {"tool_result": result}
            
            # Tool not found (shouldn't happen)
            logger.error(f"Tool '{tool_name}' not found in available tools")
            return {
                "tool_result": {
                    "success": False,
                    "error": f"Tool {tool_name} not found"
                }
            }
        else:
            # LLM didn't call any tool - use deterministic fallback
            logger.warning("LLM did not call any tool, falling back to deterministic selection")
            logger.warning(f"LLM response: {response.content}")
            
            # FALLBACK: Use deterministic selection
            return _deterministic_fallback(state, tools, report_type, domain_name, file_name)
            
    except Exception as e:
        logger.exception(f"Error in LLM tool selection: {e}")
        logger.info("Falling back to deterministic selection due to LLM error")
        
        # FALLBACK: Use deterministic selection
        return _deterministic_fallback(state, tools, report_type, domain_name, file_name)


def _deterministic_fallback(state: AnalyticsState, tools: list, report_type: str, domain_name: str, file_name: str) -> dict:
    """
    Deterministic fallback when LLM fails to select a tool.
    
    Priority:
    1. If report_type provided → Use it directly
    2. If target provided → Default to success_rate
    3. Otherwise → Return error
    """
    logger.info("🔧 Using deterministic fallback selection")
    
    # Fallback 1: If report_type is explicitly provided → Use it directly
    if report_type and (domain_name or file_name):
        logger.info(f"Fallback: Using report_type={report_type}")
        
        # Map report_type to tool
        tool_map = {
            "success_rate": "generate_success_rate_report",
            "failure_rate": "generate_failure_rate_report"
        }
        
        tool_name = tool_map.get(report_type)
        if tool_name:
            # Build tool arguments
            tool_args = {}
            if domain_name:
                tool_args["domain_name"] = domain_name
            if file_name:
                tool_args["file_name"] = file_name
            
            # Add org_id
            org_id = state.get("org_id")
            if org_id:
                tool_args["org_id"] = org_id
            
            logger.info(f"Fallback selection: {tool_name} with args {tool_args}")
            
            # Execute tool directly
            for tool in tools:
                if tool.name == tool_name:
                    result = tool.invoke(tool_args)
                    logger.info(f"Tool execution complete: success={result.get('success')}")
                    
                    # Store report_type in result
                    if "data" in result and isinstance(result["data"], dict):
                        result["data"]["_report_type"] = report_type
                    
                    return {"tool_result": result}
    
    # Fallback 2: If target provided but no report_type → Ask for clarification
    elif domain_name or file_name:
        logger.warning(f"Fallback: No report_type specified, asking user for clarification")
        
        target = domain_name or file_name
        target_type = "domain" if domain_name else "file"

        return {
        "tool_result": {
            "success": False,
            "error": f"I found the {target_type} '{target}', but I need to know what you'd like to analyze. Would you like to see:\n\n"
                           f"**Success rate** \n"
                           f" or **Failure rate**"
        }
        }
        
    
    # Fallback 3: No valid parameters → Return error
    logger.error("Fallback failed: No valid parameters for tool selection")
    return {
        "tool_result": {
            "success": False,
            "error": "I couldn't determine what you'd like to analyze. Please provide more details about what analytics you need."
        }
    }


async def generate_chart_node(state: AnalyticsState) -> dict:
    """
    Generate chart from raw analytics data, filtered by report_type.
    
    This node runs after tool execution and creates a base64-encoded
    chart visualization that will be included in the final response.
    
    Report Type Filtering:
    - "success_rate": Show only success data in chart
    - "failure_rate": Show only failure data in chart
    """
    tool_result = state["tool_result"]
    logger.info(f"Generating chart from tool result... {tool_result}")

    # Skip chart if tool failed
    if not tool_result.get("success"):
        logger.warning("Skipping chart generation (tool failed)")
        return {"chart_image": None}
    
    # Get data and report type
    data = tool_result.get("data", {})
    
    # Determine report_type from the actual tool that was called (stored in data)
    # This is more reliable than using extracted_data which might be incorrect
    report_type = data.get("report_type")
    logger.info(f"Using report type from tool result: {report_type}")
    
    
    # Skip chart if no data
    if data.get("total_requests", 0) == 0:
        logger.warning("Skipping chart generation (no data available)")
        return {"chart_image": None}
    
    # Use complete data for charts - always show both success and failure
    # This ensures accurate visualization regardless of report type
    logger.info(f"Generating chart for report_type: {report_type}")
    
    # Use the full, unfiltered data
    filtered_data = data.copy()
    
    # Log the chart data for debugging
    success_rate = data.get("success_rate", 0)
    
    # Get user-specified chart type from extracted_data (if provided)
    extracted_data = state["extracted_data"]
    user_chart_type = extracted_data.get("chart_type")
    
    # PRIORITY 1: User explicitly specified chart type
    if user_chart_type:
        chart_style = user_chart_type
        logger.info(f"Using user-specified chart type: {chart_style}")
    else:
        chart_style = "bar"  # Default chart type
        # PRIORITY 2: LLM recommendation (intelligent selection)
        # logger.info("No user-specified chart type, requesting LLM recommendation...")
        # from app.services.chart_service import get_chart_type_recommendation
        
        # chart_style = await get_chart_type_recommendation(
        #     user_query=state["user_query"],
        #     report_type=report_type,
        #     data=filtered_data
        # )
        # logger.info(f"Chart type determined: {chart_style}")
    
    # Generate chart with determined style
    from app.services.chart_service import generate_analytics_chart
    
    logger.info(f"Generating {report_type} chart with style '{chart_style}'...")
    
    try:
        chart_base64 = generate_analytics_chart(
            data=filtered_data,
            chart_type=report_type,
            style=chart_style
        )
        
        # Return just the base64 string (or None if generation failed)
        if chart_base64:
            logger.info(f"Chart generated successfully ({len(chart_base64)} bytes)")
        else:
            logger.warning("Chart generation returned None")
        
        return {"chart_image": chart_base64}
        
    except Exception as e:
        logger.exception(f"Chart generation error: {e}")
        return {"chart_image": None}


def format_response_with_llm(state: AnalyticsState) -> dict:
    """
    Use LLM to format raw analytics data into natural language response.
    
    Returns structured JSON with success, message, and chart_image.
    The LLM generates the conversational message text, and we wrap it
    in the structured response format.
    """
    tool_result = state["tool_result"]
    user_query = state["user_query"]
    chart_image = state.get("chart_image")  # Get chart from previous node
    
    # Handle tool errors
    if not tool_result.get("success"):
        error_msg = tool_result.get("error", "Unknown error occurred")
        logger.error(f"Tool error: {error_msg}")
        return {
            "final_response": {
                "success": False,
                "message": f"I encountered an error: {error_msg}",
                "chart_image": None
            }
        }
    
    # Get raw data
    data = tool_result.get("data", {})
    
    # Initialize secure prompt template for response formatting
    response_formatting_prompt = SimpleExecutorResponseFormattingPrompt()
    
    # Get secure system prompt with leakage prevention
    system_prompt = response_formatting_prompt.get_system_prompt()
    
    # Format user message with security validation and structural isolation
    user_prompt = response_formatting_prompt.format_user_message(
        user_query=user_query,
        data=data,
        has_chart=bool(chart_image)
    )
    
    logger.info("Generating LLM-formatted message...")
    
    # Use LLM to generate natural response with secure prompts
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=OPENAI_API_KEY)
    
    # Create messages with secure prompts
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = llm.invoke(messages)
    
    message_text = response.content
    logger.info(f"LLM message generated ({len(message_text)} chars)")
    
    # Build structured response
    structured_response = {
        "success": True,
        "message": message_text,
        "chart_image": chart_image  # Include chart from state (or None)
    }
    
    return {"final_response": structured_response}


def build_analytics_orchestrator() -> StateGraph:
    """
    Build the analytics orchestrator for Pattern B with chart generation.
    
    Orchestration Flow:
    1. execute_analytics_tool - Calls tool, gets raw data
    2. generate_chart_node - Creates chart visualization from data
    3. format_response_with_llm - LLM formats message, wraps in JSON with chart
    4. END - Return structured response: {success, message, chart_image}
    """
    workflow = StateGraph(AnalyticsState)
    
    # Add nodes
    workflow.add_node("execute_tool", execute_analytics_tool)
    workflow.add_node("generate_chart", generate_chart_node)
    workflow.add_node("format_response", format_response_with_llm)
    
    # Define edges
    workflow.set_entry_point("execute_tool")
    workflow.add_edge("execute_tool", "generate_chart")
    workflow.add_edge("generate_chart", "format_response")
    workflow.add_edge("format_response", END)
    
    return workflow.compile()


# Example usage
async def run_analytics_query(user_query: str, extracted_data: dict, org_id: Optional[str] = None) -> dict:
    orchestrator = build_analytics_orchestrator()
    
    initial_state = {
        "user_query": user_query,
        "extracted_data": extracted_data,
        "org_id": org_id or "",  # Pass org_id through state
        "tool_result": {},
        "chart_image": None,
        "final_response": {}
    }
    
    logger.info(f"Starting orchestrator for: {user_query}")
    logger.info(f"Extracted parameters: {extracted_data}")
    if org_id:
        logger.info(f"Organization ID: {org_id}")
    
    # Run orchestrator - LLM will select appropriate tool
    final_state = await orchestrator.ainvoke(initial_state)
    
    logger.info(f"Orchestrator complete")
    
    return final_state["final_response"]


# Example test cases
if __name__ == "__main__":
    import asyncio
    
    # Test case 1: Success rate query - LLM decides the tool
    async def test_success_rate():
        print("\n" + "="*60)
        print("TEST 1: Success Rate Query (LLM analyzes keywords)")
        print("="*60)
        
        response = await run_analytics_query(
            user_query="What's the success rate for customer domain?",
            extracted_data={
                "report_type": None,  # Not provided - LLM analyzes query
                "domain_name": "customer",
                "file_name": None
            }
        )
        
        print(f"\nSuccess: {response['success']}")
        print(f"Message:\n{response['message']}\n")
        print(f"Chart: {'Available ✓' if response['chart_image'] else 'Not available ✗'}")
        if response['chart_image']:
            print(f"   Chart size: {len(response['chart_image'])} chars")
        print()
    
    # Test case 2: Failure rate query - LLM decides the tool
    async def test_failure_rate():
        print("\n" + "="*60)
        print("TEST 2: Failure Rate Query (LLM analyzes keywords)")
        print("="*60)
        
        response = await run_analytics_query(
            user_query="Show me failures for transactions.csv",
            extracted_data={
                "report_type": None,  # Not provided - LLM analyzes query
                "domain_name": None,
                "file_name": "transactions.csv"
            }
        )
        
        print(f"\nSuccess: {response['success']}")
        print(f"Message:\n{response['message']}\n")
        print(f"Chart: {'Available ✓' if response['chart_image'] else 'Not available ✗'}")
        if response['chart_image']:
            print(f"   Chart size: {len(response['chart_image'])} chars")
        print()
    
    # Test case 3: Multi-turn with report_type (PRIORITY)
    async def test_multiturn():
        print("\n" + "="*60)
        print("TEST 3: Multi-turn - report_type provided (PRIORITY)")
        print("="*60)
        
        response = await run_analytics_query(
            user_query="payment domain",  # Ambiguous - no intent keywords
            extracted_data={
                "report_type": "failure_rate",  # From Turn 1 - LLM uses this!
                "domain_name": "payment",
                "file_name": None
            }
        )
        
        print(f"\nSuccess: {response['success']}")
        print(f"Message:\n{response['message']}\n")
        print(f"Chart: {'Available ✓' if response['chart_image'] else 'Not available ✗'}")
        if response['chart_image']:
            print(f"   Chart size: {len(response['chart_image'])} chars")
        print()
    
    # Test case 4: Natural language - "wins" maps to success_rate
    async def test_natural():
        print("\n" + "="*60)
        print("TEST 4: Natural Language - 'wins' (LLM maps to success)")
        print("="*60)
        
        response = await run_analytics_query(
            user_query="Show me all the wins for data.json",
            extracted_data={
                "report_type": None,  # Not provided - LLM analyzes "wins"
                "domain_name": None,
                "file_name": "data.json"
            }
        )
        
        print(f"\nSuccess: {response['success']}")
        print(f"Message:\n{response['message']}\n")
        print(f"Chart: {'Available ✓' if response['chart_image'] else 'Not available ✗'}")
        if response['chart_image']:
            print(f"   Chart size: {len(response['chart_image'])} chars")
        print()
    
    # Run all tests
    print("\n" + " Hybrid Tool Selection Tests (Priority + Fallback)".center(60, "="))
    asyncio.run(test_success_rate())
    asyncio.run(test_failure_rate())
    asyncio.run(test_multiturn())
    asyncio.run(test_natural())
    print("\n" + "All Tests Complete".center(60, "=") + "\n")
