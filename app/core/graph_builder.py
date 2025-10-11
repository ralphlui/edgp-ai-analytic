"""
Refactored LangGraph builder with typed state and proper node separation.
"""
import json
import logging
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.config import OPENAI_MODEL, USE_LLM, SYSTEM, MAX_AGENT_LOOPS, OPENAI_API_KEY
from app.utils.formatting import format_error_message, format_basic_message
from app.utils.context import extract_context_insights
from app.core.agent_state import AnalyticsAgentState
from app.core.message_compression import (
    compress_tool_messages,
    build_interpretation_summary,
    TOKEN_BUDGETS
)

logger = logging.getLogger(__name__)


def clean_message_sequence(messages):
    """
    Clean message sequence to remove incomplete tool call sequences.
    Removes any AIMessage with tool_calls that don't have corresponding ToolMessages.
    """
    cleaned_messages = []
    i = 0
    
    while i < len(messages):
        msg = messages[i]
        
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_call_ids = set()
            for tc in msg.tool_calls:
                if isinstance(tc, dict) and 'id' in tc:
                    tool_call_ids.add(tc['id'])
                elif hasattr(tc, 'id'):
                    tool_call_ids.add(tc.id)
            
            # Look ahead for tool responses
            j = i + 1
            found_tool_responses = set()
            
            while j < len(messages) and isinstance(messages[j], ToolMessage):
                tool_msg = messages[j]
                if hasattr(tool_msg, 'tool_call_id'):
                    found_tool_responses.add(tool_msg.tool_call_id)
                j += 1
            
            # Only include if all tool calls have responses
            if tool_call_ids.issubset(found_tool_responses):
                cleaned_messages.append(msg)
                for k in range(i + 1, j):
                    if isinstance(messages[k], ToolMessage):
                        cleaned_messages.append(messages[k])
                i = j
            else:
                logger.warning(f"Removing incomplete tool calls: {tool_call_ids - found_tool_responses}")
                i += 1
        else:
            cleaned_messages.append(msg)
            i += 1
    
    return cleaned_messages


def assistant_node(state: AnalyticsAgentState) -> AnalyticsAgentState:
    """
    Assistant node: enforces loop cap, compresses messages, calls tool-bound LLM.
    """
    messages = state["messages"]
    loop_count = state.get("loop_count", 0)
    
    # Enforce loop cap
    if loop_count >= MAX_AGENT_LOOPS:
        error_msg = format_error_message(
            "Maximum tool call cycles reached",
            "Query too complex. Try breaking it into simpler parts.",
            f"Attempted {loop_count} tool calls"
        )
        return {
            **state,
            "messages": [AIMessage(content=error_msg)],
            "loop_count": loop_count
        }
    
    # Clean message sequence
    messages = clean_message_sequence(list(messages))
    logger.info(f"Assistant node: {len(messages)} messages, loop {loop_count}")
    
    # Compress tool messages to fit budget
    compressed_messages, compression_stats = compress_tool_messages(
        messages,
        budget_per_msg=TOKEN_BUDGETS['tool_message'],
        total_budget=TOKEN_BUDGETS['total_limit']
    )
    
    if compression_stats['compressed_count'] > 0:
        logger.info(f"Compressed {compression_stats['compressed_count']} tool messages, "
                   f"saved ~{compression_stats['tokens_saved']} tokens")
    
    if compression_stats['pii_detected']:
        logger.warning("PII detected and redacted in conversation")
    
    # Import here to avoid circular dependency
    from app.core.analytic_service import AnalyticService
    
    # Create tool-bound LLM
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY
    ).bind_tools(AnalyticService.TOOLS)
    
    # Add system message and invoke
    enhanced_messages = [SystemMessage(content=SYSTEM)] + compressed_messages
    
    try:
        response = llm.invoke(enhanced_messages)
        logger.info(f"Assistant response: tool_calls={bool(getattr(response, 'tool_calls', None))}")
        
        return {
            **state,
            "messages": [response],
            "loop_count": loop_count + 1,
            "compression_applied": compression_stats['compressed_count'] > 0,
            "total_tokens_estimate": compression_stats.get('total_tokens', 0)
        }
    except Exception as e:
        logger.error(f"Assistant node error: {e}")
        error_content = format_error_message(
            "Processing Error",
            "An unexpected error occurred. Please try again.",
            str(e)
        )
        return {
            **state,
            "messages": [AIMessage(content=error_content)],
            "loop_count": loop_count
        }


def interpretation_node(state: AnalyticsAgentState) -> AnalyticsAgentState:
    """
    Interpretation node: calls no-tools LLM with summary-only context.
    """
    messages = state["messages"]
    tool_results = state.get("tool_results", [])
    context_insights = state.get("context_insights", [])
    
    logger.info(f"Interpretation node: {len(tool_results)} tool results")
    
    # Extract user query
    user_query = ""
    for msg in reversed(list(messages)):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break
    
    # Build summary-only interpretation prompt
    interpretation_summary = build_interpretation_summary(
        user_query=user_query,
        tool_results=tool_results,
        context_insights=context_insights,
        max_tokens=2000
    )
    
    logger.info(f"Interpretation summary: {len(interpretation_summary)} chars")
    
    # Create interpretation messages (no tools bound)
    interpretation_messages = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=interpretation_summary)
    ]
    
    # Call LLM without tool binding
    interpretation_llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY
    )
    
    try:
        interpretation_response = interpretation_llm.invoke(interpretation_messages)
        logger.info(f"Interpretation complete: {len(interpretation_response.content)} chars")
        
        return {
            **state,
            "messages": [interpretation_response]
        }
    except Exception as e:
        logger.error(f"Interpretation failed: {e}")
        
        # Fallback to basic formatting
        file_name = None
        row_count = 0
        
        for result in tool_results:
            if isinstance(result, dict):
                file_name = file_name or result.get('file_name')
                row_count = row_count or result.get('row_count', 0)
        
        fallback_content = format_basic_message(
            chart_data=[],
            file_name=file_name,
            row_count=row_count,
            chart_type="bar",
            report_type="both",
            date_filter_used=None,
            original_chart_data=[]
        )
        
        return {
            **state,
            "messages": [AIMessage(content=fallback_content)]
        }



def tools_node_wrapper(state: AnalyticsAgentState) -> AnalyticsAgentState:
    """
    Tools node wrapper: executes tools and extracts results/insights.
    
    SECURITY: Sanitizes all tool outputs to prevent injection attacks.
    """
    from app.core.analytic_service import AnalyticService
    from app.utils.sanitization import sanitize_tool_output
    
    # Execute tools using LangGraph's ToolNode
    tool_node = ToolNode(AnalyticService.TOOLS)
    result = tool_node.invoke(state)
    
    # Extract tool results and insights from new messages
    messages = result.get("messages", [])
    tool_results = list(state.get("tool_results", []))
    context_insights = list(state.get("context_insights", []))
    
    for msg in messages:
        if isinstance(msg, ToolMessage):
            try:
                tool_data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                
                # SECURITY: Sanitize tool output before adding to state
                # This prevents injection through compromised tools
                tool_data = sanitize_tool_output(tool_data, max_length=10000)
                
                tool_results.append(tool_data)
                
                # Extract insights if data present
                if isinstance(tool_data, dict) and "data" in tool_data:
                    insights = extract_context_insights(tool_data["data"])
                    context_insights.extend(insights)
                    
            except Exception as e:
                logger.warning(f"Failed to parse tool result: {e}")
    
    return {
        **result,
        "tool_results": tool_results,
        "context_insights": context_insights
    }


def should_use_tools(state: AnalyticsAgentState) -> Literal["tools", "interpretation", "end"]:
    """
    Router: decides whether to call tools, interpret, or end.
    
    Logic:
    - If last AI message has tool_calls AND loop_count < MAX → "tools"
    - If we have tool_results → "interpretation"
    - Otherwise → "end"
    """
    messages = state.get("messages", [])
    loop_count = state.get("loop_count", 0)
    tool_results = state.get("tool_results", [])
    
    if not messages:
        return "end"
    
    last_message = messages[-1]
    
    # Check if last message has tool calls
    if isinstance(last_message, AIMessage):
        has_tool_calls = hasattr(last_message, 'tool_calls') and last_message.tool_calls
        
        if has_tool_calls and loop_count < MAX_AGENT_LOOPS:
            logger.info(f"Router: tools (loop {loop_count})")
            return "tools"
        
        # If we have tool results but no more tool calls, interpret
        if tool_results:
            logger.info(f"Router: interpretation ({len(tool_results)} results)")
            return "interpretation"
    
    # Default: end
    logger.info("Router: end")
    return "end"


def build_analytics_graph(
    base_llm: ChatOpenAI = None,
    tools: list = None,
    max_loops: int = MAX_AGENT_LOOPS
) -> StateGraph:
    """
    Factory function to build analytics graph with typed state.
    
    Args:
        base_llm: Base ChatOpenAI instance (optional, uses default if None)
        tools: List of tools to bind (optional, uses AnalyticService.TOOLS if None)
        max_loops: Maximum tool call loops
        
    Returns:
        Compiled LangGraph
    """
    if not USE_LLM:
        raise SystemExit("OPENAI_API_KEY missing. Add it to .env to run the agent.")
    
    # Create graph with typed state
    graph = StateGraph(AnalyticsAgentState)
    
    # Add nodes
    graph.add_node("assistant", assistant_node)
    graph.add_node("tools", tools_node_wrapper)
    graph.add_node("interpretation", interpretation_node)
    
    # Set entry point
    graph.set_entry_point("assistant")
    
    # Add conditional edges from assistant
    graph.add_conditional_edges(
        "assistant",
        should_use_tools,
        {
            "tools": "tools",
            "interpretation": "interpretation",
            "end": END
        }
    )
    
    # Tools always route back to assistant
    graph.add_edge("tools", "assistant")
    
    # Interpretation always ends
    graph.add_edge("interpretation", END)
    
    return graph.compile()


"""
LangGraph builder for analytic processing workflow.
"""
import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from app.config import OPENAI_MODEL, USE_LLM, SYSTEM, MAX_AGENT_LOOPS, OPENAI_API_KEY
from app.utils.formatting import format_error_message, format_basic_message
from app.utils.context import extract_context_insights

logger = logging.getLogger(__name__)


def clean_message_sequence(messages):
    """
    Clean message sequence to remove incomplete tool call sequences that cause API errors.
    Removes any AIMessage with tool_calls that don't have corresponding ToolMessages.
    """
    cleaned_messages = []
    i = 0
    
    while i < len(messages):
        msg = messages[i]
        
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            # This AI message has tool calls - check if they have responses
            tool_call_ids = set()
            for tc in msg.tool_calls:
                if isinstance(tc, dict) and 'id' in tc:
                    tool_call_ids.add(tc['id'])
                elif hasattr(tc, 'id'):
                    tool_call_ids.add(tc.id)
            
            # Look ahead to find corresponding tool messages
            j = i + 1
            found_tool_responses = set()
            
            while j < len(messages) and isinstance(messages[j], ToolMessage):
                tool_msg = messages[j]
                if hasattr(tool_msg, 'tool_call_id'):
                    found_tool_responses.add(tool_msg.tool_call_id)
                j += 1
            
            # Only include this AI message if all tool calls have responses
            if tool_call_ids.issubset(found_tool_responses):
                cleaned_messages.append(msg)
                # Add the corresponding tool messages
                for k in range(i + 1, j):
                    if isinstance(messages[k], ToolMessage):
                        cleaned_messages.append(messages[k])
                i = j
            else:
                logger.warning(f"Removing AI message with incomplete tool calls: {tool_call_ids - found_tool_responses}")
                i += 1
        else:
            cleaned_messages.append(msg)
            i += 1
    
    return cleaned_messages


# Legacy V1 implementation removed
# The build_app() function has been replaced by build_analytics_graph() which uses:
# - Typed state (AnalyticsAgentState)
# - Message compression with PII protection
# - Token budgets and optimization
# - 3-node architecture (assistant, tools, interpretation)
# See build_analytics_graph() above for the active implementation