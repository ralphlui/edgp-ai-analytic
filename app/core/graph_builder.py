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
from app.utils.context import extract_context_insights, get_conversation_context, create_interpretation_prompt

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


def build_app():
    """
    Build the LangGraph application for analytic processing.
    """
    if not USE_LLM:
        raise SystemExit("OPENAI_API_KEY missing. Add it to .env to run the chat agent.")

    # Import AnalyticsService here to avoid circular imports
    from app.core.analytic_service import AnalyticService

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY).bind_tools(AnalyticService.TOOLS)

    graph = StateGraph(MessagesState)

    def assistant(state: MessagesState):
        messages = state["messages"]
        
        # Clean message sequence to prevent tool_call mismatch errors
        messages = clean_message_sequence(messages)
        logger.info(f"Cleaned message sequence: {len(messages)} messages")

        # Enhanced loop protection with context awareness
        loop_count = state.get("_loop_count", 0)
        if loop_count >= MAX_AGENT_LOOPS:
            error_msg = format_error_message(
                "Maximum tool call cycles reached",
                "The query is too complex or requires too many data operations. Try breaking it into simpler parts.",
                f"Query attempted {loop_count} tool calls"
            )
            return {"messages": [AIMessage(content=error_msg)]}

        state["_loop_count"] = loop_count + 1

        try:
            # Validate message sequence to prevent tool_call mismatch errors
            logger.info(f"Processing {len(messages)} messages")
            for i, msg in enumerate(messages):
                logger.debug(f"Message {i}: {type(msg).__name__} - {'has tool_calls' if hasattr(msg, 'tool_calls') and msg.tool_calls else 'no tool_calls'}")

            # Process tool results with enhanced interpretation
            # Only use tool results from the current turn (after the latest user message)
            has_tool_results = False
            tool_results = []
            context_insights = []

            # Find the index of the latest HumanMessage
            latest_human_index = -1
            for i in reversed(range(len(messages))):
                if isinstance(messages[i], HumanMessage):
                    latest_human_index = i
                    break

            # Only process ToolMessages that come after the latest HumanMessage
            for i, msg in enumerate(messages):
                if isinstance(msg, ToolMessage) and i > latest_human_index:
                    has_tool_results = True
                    try:
                        logger.debug(f"Processing tool message: {msg.tool_call_id}")
                        tool_data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                        tool_results.append(tool_data)

                        # Extract context insights from tool results
                        if isinstance(tool_data, dict) and "data" in tool_data:
                            insights = extract_context_insights(tool_data["data"])
                            context_insights.extend(insights)
                    except Exception as e:
                        logger.warning(f"Failed to parse tool result: {e}")
                        tool_results.append({"content": msg.content, "error": str(e)})

            # If we have tool results, use enhanced interpretation
            if has_tool_results and tool_results:
                try:
                    # Get conversation history for context
                    logger.info(f"Messages to assistant tool_result: {tool_results}")
                    conversation_context = get_conversation_context(messages)

                    # Extract user query and report type for context-aware interpretation
                    user_query = ""
                    report_type = None
                    
                    # Find the most recent human message (user query)
                    for msg in reversed(messages):
                        if isinstance(msg, HumanMessage):
                            user_query = msg.content
                            break
                    
                    # Extract report type from tool results if available
                    for result in tool_results:
                        if isinstance(result, dict) and result.get("report_type"):
                            report_type = result["report_type"]
                            break

                    # Validate inputs before calling LLM
                    if not tool_results:
                        raise ValueError("Empty tool_results")
                    
                    # Create enhanced interpretation prompt with user query context
                    try:
                        interpretation_prompt = create_interpretation_prompt(
                            user_query=user_query,
                            tool_results=tool_results, 
                            report_type=report_type,
                            context_insights=context_insights
                        )
                        logger.info(f"Context-aware interpretation for query: '{user_query[:50]}...' with report_type: '{report_type}'")
                        logger.info(f"Interpretation prompt length: {len(interpretation_prompt) if interpretation_prompt else 0}")
                        if not interpretation_prompt or len(interpretation_prompt.strip()) == 0:
                            raise ValueError("Empty interpretation prompt generated")
                    except Exception as prompt_error:
                        logger.error(f"Failed to create interpretation prompt: {prompt_error}")
                        raise prompt_error

                    # Create fresh message sequence for interpretation to avoid tool_call mismatch
                    # Only include the system message and the interpretation prompt
                    interpretation_messages = [
                        SystemMessage(content=SYSTEM),
                        HumanMessage(content=interpretation_prompt)
                    ]
                    
                    # Validate message structure
                    total_length = sum(len(str(msg.content)) for msg in interpretation_messages)
                    logger.info(f"Total message length: {total_length} characters")
                    
                    if total_length > 100000:  # ~25k tokens rough estimate
                        logger.warning(f"Message might be too long for API: {total_length} chars")

                    # Use LLM for intelligent interpretation without tool binding to avoid tool_call issues
                    interpretation_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)
                    interpretation_response = interpretation_llm.invoke(interpretation_messages)
                    logger.info(f"Interpretation response: {interpretation_response}")
                    return {"messages": [interpretation_response]}

                except Exception as e:
                    logger.error(f"Interpretation failed: {e}")
                    logger.error(f"Error type: {type(e)}")
                    
                    # Log more details for debugging
                    if hasattr(e, 'response'):
                        logger.error(f"API response status: {getattr(e.response, 'status_code', 'Unknown')}")
                    if hasattr(e, 'body'):
                        logger.error(f"API error body: {e.body}")
                    
                    # Log the data that caused the failure
                    logger.error(f"Tool results count: {len(tool_results)}")
                    logger.error(f"Context insights count: {len(context_insights)}")
                    
                    # Fallback to basic formatting with correct parameters
                    # Extract parameters from tool_results for fallback
                    file_name = None
                    row_count = 0
                    chart_type = "bar"
                    report_type = "both"

                    # Try to extract parameters from tool results
                    for result in tool_results:
                        if isinstance(result, dict):
                            if "file_name" in result:
                                file_name = result["file_name"]
                            if "row_count" in result:
                                row_count = result["row_count"]
                            if "chart_type" in result:
                                chart_type = result["chart_type"]
                            if "report_type" in result:
                                report_type = result["report_type"]

                    fallback_content = format_basic_message(
                        chart_data=[],  # No chart data in fallback
                        file_name=file_name,
                        row_count=row_count,
                        chart_type=chart_type,
                        report_type=report_type,
                        date_filter_used=None,
                        original_chart_data=[]
                    )
                    return {"messages": [AIMessage(content=fallback_content)]}

            # Handle initial queries and follow-ups without tool results
            elif not has_tool_results:
                # Add system prompt to ensure consistent behavior and use cleaned messages
                enhanced_messages = [SystemMessage(content=SYSTEM)] + list(messages)
                response = llm.invoke(enhanced_messages)
                logger.info(f"First initial call llm response: {response}")
                return {"messages": [response]}

            # Fallback for edge cases
            else:
                fallback_msg = format_basic_message(
                    chart_data=[],
                    file_name=None,
                    row_count=0,
                    chart_type="bar",
                    report_type="both",
                    date_filter_used=None,
                    original_chart_data=[]
                )
                return {"messages": [AIMessage(content=fallback_msg)]}

        except Exception as e:
            logger.error(f"Assistant function error: {e}")
            error_content = format_error_message(
                "Processing Error",
                "An unexpected error occurred while processing your request. Please try again.",
                f"Technical details: {str(e)}"
            )
            return {"messages": [AIMessage(content=error_content)]}

    tool_node = ToolNode(AnalyticService.TOOLS)

    graph.add_node("assistant", assistant)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")

    return graph.compile()