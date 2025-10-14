"""
Analytics Orchestrator - Coordinates analytics query execution using Pattern B.

Pattern B: Tools return raw data, LLM generates natural language responses.

This orchestrator coordinates:
- Tool execution (analytics data retrieval)
- Chart generation (visualization)
- LLM response formatting (natural language)

Benefits:
- Better conversational flow with context-aware responses
- Flexibility for multi-turn conversations
- Natural language adaptation to user's query style
- Easier maintenance (clear separation of concerns)
"""
import logging
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from app.config import OPENAI_API_KEY, OPENAI_MODEL

from app.tools.analytics_tools import (
    generate_success_rate_report,
    generate_failure_rate_report
)

logger = logging.getLogger("analytic_agent")


class AnalyticsState(TypedDict):
    """State for analytics workflow."""
    user_query: str
    extracted_data: dict  # {report_type, domain_name, file_name}
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
    
    # Create LLM with tool calling capability
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    llm_with_tools = llm.bind_tools(tools)
    
    # Comprehensive system prompt explaining the task
    system_prompt = """You are an analytics tool selector. Your job is to:

1. **Check if report_type is provided** (from previous conversation context)
2. **If yes**: Use report_type to select the tool directly (HIGHEST PRIORITY)
3. **If no**: Analyze the user's query to determine intent from keywords
4. **Format the tool call** with the correct parameters in JSON format

## PRIORITY RULES (Follow in order):

### Rule 1: Use report_type if provided (HIGHEST PRIORITY)
If report_type is given in the available parameters:
- report_type = "success_rate" → MUST call generate_success_rate_report
- report_type = "failure_rate" → MUST call generate_failure_rate_report
- Do NOT analyze the user query for intent - trust the report_type

### Rule 2: Analyze user query if report_type is missing (FALLBACK)
If report_type is null/not provided:
- Analyze user query for intent keywords:
  - "success", "wins", "passed", "uptime", "completion" → generate_success_rate_report
  - "failure", "errors", "issues", "problems", "fail" → generate_failure_rate_report

## Available Tools:

### generate_success_rate_report
- **Purpose**: Analyze how often requests succeed
- **When to use**: 
  - report_type = "success_rate" (explicit - PRIORITY), OR
  - User asks about success, wins, completion, passed requests, uptime (inferred)
- **Parameters**: 
  - domain_name (optional): The domain to analyze (e.g., "customer", "payment")
  - file_name (optional): The file to analyze (e.g., "customer.csv", "data.json")
- **JSON Format**: {"domain_name": "value"} OR {"file_name": "value"}
- **Constraint**: Provide EXACTLY ONE of domain_name or file_name, never both

### generate_failure_rate_report
- **Purpose**: Analyze how often requests fail
- **When to use**: 
  - report_type = "failure_rate" (explicit - PRIORITY), OR
  - User asks about failures, errors, issues, problems, failed requests (inferred)
- **Parameters**:
  - domain_name (optional): The domain to analyze
  - file_name (optional): The file to analyze
- **JSON Format**: {"domain_name": "value"} OR {"file_name": "value"}
- **Constraint**: Provide EXACTLY ONE of domain_name or file_name, never both

## Examples:

**Example 1: Multi-turn with report_type (USE PRIORITY RULE)**
Available Parameters:
- report_type: "success_rate"  ← USE THIS FIRST!
- domain_name: "customer"
- file_name: null

User Query: "customer domain"  ← Ignore ambiguous query, use report_type

→ Tool: generate_success_rate_report (from report_type, not query)
→ Args: {"domain_name": "customer", "file_name": null}

**Example 2: Single-turn, no report_type (USE FALLBACK RULE)**
Available Parameters:
- report_type: null  ← Not provided, analyze query
- domain_name: null
- file_name: "data.csv"

User Query: "Show me failures in data.csv"  ← Analyze "failures" keyword

→ Tool: generate_failure_rate_report (from "failures" keyword)
→ Args: {"domain_name": null, "file_name": "data.csv"}

**Example 3: Multi-turn continuation**
Available Parameters:
- report_type: "failure_rate"  ← USE THIS!
- domain_name: "payment"
- file_name: null

User Query: "payment domain"  ← Ignore query, use report_type

→ Tool: generate_failure_rate_report (from report_type)
→ Args: {"domain_name": "payment", "file_name": null}

**Example 4: Natural language with report_type**
Available Parameters:
- report_type: "success_rate"  ← USE THIS!
- domain_name: null
- file_name: "data.json"

User Query: "data.json"  ← Ignore query, use report_type

→ Tool: generate_success_rate_report (from report_type)
→ Args: {"domain_name": null, "file_name": "data.json"}

## Important Rules:

1. **report_type Priority**: ALWAYS use report_type if provided (takes precedence over query analysis)
2. **XOR Constraint**: NEVER provide both domain_name AND file_name - choose ONE
3. **Null values**: Set the unused parameter to null/None
4. **Case sensitivity**: Tool names are case-sensitive
5. **JSON structure**: Arguments must be valid JSON with quoted keys

Now analyze the available parameters and select the appropriate tool."""

    # Build the prompt with available parameters
    user_prompt = f"""User Query: "{user_query}"

Available Parameters (extracted from conversation):
- report_type: {report_type if report_type else "null"}  ← CHECK THIS FIRST!
- domain_name: {domain_name if domain_name else "null"}
- file_name: {file_name if file_name else "null"}

IMPORTANT: If report_type is provided, use it to select the tool (PRIORITY). Otherwise, analyze the user query.

Select the appropriate analytics tool and call it with the correct parameters."""

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
            # LLM didn't call any tool - this shouldn't happen with proper tools
            logger.warning("LLM did not call any tool")
            logger.warning(f"LLM response: {response.content}")
            return {
                "tool_result": {
                    "success": False,
                    "error": "Could not determine which analytics tool to use"
                }
            }
            
    except Exception as e:
        logger.exception(f"Error in LLM tool selection: {e}")
        return {
            "tool_result": {
                "success": False,
                "error": f"Tool selection failed: {str(e)}"
            }
        }


def generate_chart_node(state: AnalyticsState) -> dict:
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
    
    # Generate chart with filtered data
    from app.services.chart_service import generate_analytics_chart
    
    logger.info(f"Generating {report_type} chart...")
    
    try:
        chart_base64 = generate_analytics_chart(
            data=filtered_data,
            chart_type=report_type,
            style="bar"
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
    
    # Build prompt for LLM to generate ONLY the message text
    prompt = f"""You are a helpful analytics assistant. The user asked: "{user_query}"

I retrieved the following analytics data:
- Target Type: {data.get('target_type')}
- Target Value: {data.get('target_value')}
- Total Requests: {data.get('total_requests')}
- Successful Requests: {data.get('successful_requests')}
- Failed Requests: {data.get('failed_requests')}
- {"Success" if "success_rate" in data else "Failure"} Rate: {data.get('success_rate', data.get('failure_rate'))}%

A chart visualization {"is" if chart_image else "is NOT"} available.

Please format this data into a natural, conversational response that:
1. Directly answers the user's question
2. Highlights key insights based on the metrics
3. Provides context-appropriate recommendations if the rate is concerning
4. Uses appropriate emojis for visual clarity
5. Keeps a friendly, professional tone
{"6. Mention that a chart is included for visualization" if chart_image else ""}

If there are no requests (total_requests = 0), explain that no data is available yet.

Return ONLY the message text, not JSON. I will wrap it in the response structure.
"""
    
    logger.info("Generating LLM-formatted message...")
    
    # Use LLM to generate natural response
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=OPENAI_API_KEY)
    response = llm.invoke(prompt)
    
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
async def run_analytics_query(user_query: str, extracted_data: dict) -> dict:
    """
    Run analytics query through orchestrator with hybrid tool selection.
    
    The orchestrator uses a two-tier selection strategy:
    1. **Priority**: If report_type is provided → Use it directly (multi-turn context)
    2. **Fallback**: If report_type is None → LLM analyzes query keywords (single-turn)
    
    This approach provides:
    - Accuracy for multi-turn conversations (explicit intent from context)
    - Flexibility for single-turn queries (intelligent keyword analysis)
    - Robustness (handles both scenarios seamlessly)
    
    Args:
        user_query: Original user question (e.g., "Show me success rate for customer domain")
        extracted_data: {
            report_type?: "success_rate" | "failure_rate" | None,  # Explicit intent (priority)
            domain_name?: str | None,
            file_name?: str | None
        }
    
    Returns:
        Structured response dictionary:
        {
            "success": bool,
            "message": str (natural language response from LLM),
            "chart_image": str or None (base64 format)
        }
    
    Examples:
        # Multi-turn: report_type provided from previous context (PRIORITY)
        >>> await run_analytics_query(
        ...     user_query="customer domain",  # Turn 2 (ambiguous)
        ...     extracted_data={
        ...         "report_type": "success_rate",  # From Turn 1
        ...         "domain_name": "customer",
        ...         "file_name": None
        ...     }
        ... )
        # LLM uses report_type → generate_success_rate_report ✓
        
        # Single-turn: LLM decides from query keywords (FALLBACK)
        >>> await run_analytics_query(
        ...     user_query="Show me failures in data.csv",
        ...     extracted_data={
        ...         "report_type": None,  # Not provided
        ...         "domain_name": None,
        ...         "file_name": "data.csv"
        ...     }
        ... )
        # LLM analyzes "failures" → generate_failure_rate_report ✓
    """
    orchestrator = build_analytics_orchestrator()
    
    initial_state = {
        "user_query": user_query,
        "extracted_data": extracted_data,
        "tool_result": {},
        "chart_image": None,
        "final_response": {}
    }
    
    logger.info(f"Starting orchestrator for: {user_query}")
    logger.info(f"Extracted parameters: {extracted_data}")
    
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
        
        print(f"\n✅ Success: {response['success']}")
        print(f"📝 Message:\n{response['message']}\n")
        print(f"📊 Chart: {'Available ✓' if response['chart_image'] else 'Not available ✗'}")
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
        
        print(f"\n✅ Success: {response['success']}")
        print(f"📝 Message:\n{response['message']}\n")
        print(f"📊 Chart: {'Available ✓' if response['chart_image'] else 'Not available ✗'}")
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
        
        print(f"\n✅ Success: {response['success']}")
        print(f"📝 Message:\n{response['message']}\n")
        print(f"📊 Chart: {'Available ✓' if response['chart_image'] else 'Not available ✗'}")
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
        
        print(f"\n✅ Success: {response['success']}")
        print(f"📝 Message:\n{response['message']}\n")
        print(f"📊 Chart: {'Available ✓' if response['chart_image'] else 'Not available ✗'}")
        if response['chart_image']:
            print(f"   Chart size: {len(response['chart_image'])} chars")
        print()
    
    # Run all tests
    print("\n" + "🚀 Hybrid Tool Selection Tests (Priority + Fallback)".center(60, "="))
    asyncio.run(test_success_rate())
    asyncio.run(test_failure_rate())
    asyncio.run(test_multiturn())
    asyncio.run(test_natural())
    print("\n" + "✅ All Tests Complete".center(60, "=") + "\n")
