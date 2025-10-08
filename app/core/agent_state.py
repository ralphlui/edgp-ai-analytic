"""
Typed state management for analytics agent graph.
"""
from typing import TypedDict, Annotated, Sequence, Optional, List, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AnalyticsAgentState(TypedDict):
    """
    Typed state for analytics agent workflow.
    
    Note: user_id is NOT stored in graph state for security reasons.
    Multi-tenant isolation uses contextvars (see app.utils.request_context.USER_ID_CTX)
    set at the API layer, accessible via get_current_user_id().
    
    Attributes:
        messages: Conversation message history (auto-merged via add_messages)
        loop_count: Number of tool call cycles executed
        tool_results: Accumulated tool execution results
        context_insights: Extracted insights from tool data
        compression_applied: Flag indicating if message compression was applied
        total_tokens_estimate: Rough token count estimate for monitoring
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    loop_count: int
    tool_results: List[Dict[str, Any]]
    context_insights: List[str]
    compression_applied: bool
    total_tokens_estimate: int
