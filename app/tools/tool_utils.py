"""
Shared utilities for LangGraph tools to reduce code duplication and improve performance.
"""
import asyncio
import json
import logging
import concurrent.futures
from typing import Optional, Dict, Any, Callable, Awaitable
from app.utils.request_context import get_current_org_id, get_current_session_id
import contextvars

logger = logging.getLogger("tool_utils")

# Reusable thread pool for async execution
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool_async")

# Import session bindings - avoid circular imports
_session_tenant_bindings: Optional[Dict[str, Dict[str, str]]] = None

def get_session_bindings():
    """Lazy import to avoid circular dependency"""
    global _session_tenant_bindings
    if _session_tenant_bindings is None:
        from .session_manager import _session_tenant_bindings as bindings
        _session_tenant_bindings = bindings
    return _session_tenant_bindings

def validate_chart_type(chart_type: Optional[str]) -> str:
    """Validate and normalize chart type parameter."""
    valid_chart_types = ['bar', 'pie', 'donut', 'line', 'stacked']
    if chart_type not in valid_chart_types:
        return 'bar'  # Default to bar if invalid type provided
    return chart_type

def validate_report_type(report_type: Optional[str]) -> str:
    """Validate and normalize report type parameter."""
    valid_report_types = ['success', 'failure', 'both']
    # Handle None, empty string, or invalid values
    if not report_type or report_type not in valid_report_types:
        return ''  # Default to empty string if invalid type provided
    return report_type

def get_org_id_for_tool() -> Optional[str]:
    """
    Get org_id from multiple sources with fallback strategy.
    Returns the first available org_id or None if not found.
    """
    # Try context variables first
    captured_org_id = get_current_org_id()
    logger.debug(f"Captured org_id from contextvars: {captured_org_id}")
    
    if captured_org_id is None:
        # Try session bindings as fallback, but only for the current session id
        session_id = get_current_session_id()
        if session_id:
            session_bindings = get_session_bindings()
            binding = session_bindings.get(session_id)
            if binding and binding.get("org_id"):
                captured_org_id = binding["org_id"]
                logger.info(f"Using org_id from session binding for session {session_id[:8]}...")
    
    return captured_org_id

async def execute_db_method_async(
    method_name: str,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute a database service method asynchronously.
    Reuses connection and avoids creating new event loops.
    """
    from app.services.database_service import DatabaseService
    
    db_service = DatabaseService()
    method = getattr(db_service, method_name)
    
    return await method(*args, **kwargs)

def execute_async_tool_call(
    db_method_name: str,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute async database calls with proper event loop handling.
    Reuses thread pool for better performance.
    """
    # Capture current contextvars so org_id/user_id/session_id propagate into thread
    _ctx = contextvars.copy_context()

    def run_async_db_call():
        """Run the async database call in a new event loop"""
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(
                execute_db_method_async(db_method_name, *args, **kwargs)
            )
        finally:
            new_loop.close()

    try:
        # Try to get the current event loop
        current_loop = asyncio.get_running_loop()
        # If we're in an async context, run in a thread to avoid blocking
        result = _thread_pool.submit(lambda: _ctx.run(run_async_db_call)).result()
    except RuntimeError:
        # No event loop running, we can call the async function directly
        result = _ctx.run(run_async_db_call)
    
    return result

def format_tool_response(
    result: Dict[str, Any],
    chart_type: str,
    tool_name: str,
    return_as_json: bool = False,
    report_type: str = "both"
) -> Any:
    """
    Format tool response with consistent structure and error handling.
    
    Args:
        result: Database result dictionary
        chart_type: Validated chart type
        tool_name: Name of the calling tool for error messages
        return_as_json: Whether to return JSON string or dict
        report_type: Type of report to generate ('success', 'failure', 'both')
    
    Returns:
        Formatted response as JSON string or dict
    """
    try:
        if result.get("success"):
            result["chart_type"] = chart_type
            result["chart_type_requested"] = True
            result["report_type"] = report_type
            result["report_type_requested"] = True
            result.setdefault("stop", True)
        
        if return_as_json:
            return json.dumps(result)
        else:
            return result
            
    except Exception as e:
        logger.exception(f"Error formatting response for {tool_name}: {e}")
        error_result = {
            "success": False,
            "error": str(e),
            "message": f"An unexpected error occurred in {tool_name}. Please try again.",
            "chart_type": chart_type,
            "report_type": report_type,
            "stop": True
        }
        
        if return_as_json:
            return json.dumps(error_result)
        else:
            return error_result

def create_auth_error_response(return_as_json: bool = False) -> Any:
    """Create standardized authentication error response."""
    error_result = {
        "success": False,
        "error": "Authentication context lost",
        "message": "Unable to determine organization context. Please try again."
    }
    
    if return_as_json:
        return json.dumps(error_result)
    else:
        return error_result

def add_performance_assessment(result: Dict[str, Any], failure_rate: float, assessment_type: str = "general") -> None:
    """
    Add performance assessment based on failure rate.
    
    Args:
        result: Result dictionary to modify
        failure_rate: Failure rate percentage
        assessment_type: Type of assessment ('rule', 'data_quality', 'general')
    """
    if not result.get("success"):
        return
    
    if assessment_type == "rule":
        if failure_rate == 0:
            result["performance_assessment"] = "Excellent - no rule failures detected"
        elif failure_rate <= 5:
            result["performance_assessment"] = "Good - low rule failure rate"
        elif failure_rate <= 15:
            result["performance_assessment"] = "Moderate - room for improvement in rule compliance"
        else:
            result["performance_assessment"] = "Poor - high rule failure rate requires attention"
    
    elif assessment_type == "data_quality":
        if failure_rate == 0:
            result["performance_assessment"] = "Excellent - no data quality issues detected"
        elif failure_rate <= 5:
            result["performance_assessment"] = "Good - low data quality failure rate"
        elif failure_rate <= 15:
            result["performance_assessment"] = "Moderate - some data quality issues need attention"
        else:
            result["performance_assessment"] = "Poor - significant data quality issues require immediate action"
    
    else:  # general
        if failure_rate == 0:
            result["performance_assessment"] = "Excellent - no issues detected"
        elif failure_rate <= 5:
            result["performance_assessment"] = "Good - low failure rate"
        elif failure_rate <= 15:
            result["performance_assessment"] = "Moderate - some issues need attention"
        else:
            result["performance_assessment"] = "Poor - high failure rate requires immediate action"