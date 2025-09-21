from langchain_core.tools import tool
from typing import Optional, Dict, Any
import asyncio
import json
import logging
from app.utils.request_context import get_current_org_id

logger = logging.getLogger("tools_agent")

# Module-level storage for current request org_id (fallback)
_current_org_id: Optional[str] = None

# Session to tenant binding storage
_session_tenant_bindings: Dict[str, Dict[str, str]] = {}

def bind_session_to_tenant(session_id: str, user_id: str, org_id: str) -> bool:
    """Bind a session to tenant context for cross-thread access."""
    try:
        _session_tenant_bindings[session_id] = {
            "user_id": user_id,
            "org_id": org_id
        }
        logger.info(f"Bound session {session_id} to org {org_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to bind session {session_id}: {e}")
        return False

def unbind_session(session_id: str) -> None:
    """Remove session binding."""
    try:
        if session_id in _session_tenant_bindings:
            del _session_tenant_bindings[session_id]
            logger.info(f"Unbound session {session_id}")
    except Exception as e:
        logger.error(f"Failed to unbind session {session_id}: {e}")

def get_session_context(session_id: str) -> Optional[Dict[str, str]]:
    """Get session context for tools."""
    return _session_tenant_bindings.get(session_id)


@tool("get_success_rate_by_file_name", return_direct=False)
def get_success_rate_by_file_name_tool(
    file_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    chart_type: Optional[str] = "bar"  # Allow org_id to be passed as parameter
) -> Dict[str, Any]:
    """Calculate success/failure percentage for a file and specify visualization type.

    
    Args:
        file_name: Name of the file to analyze (e.g., 'customer_sample_values.csv')
        start_date: Start date for filtering (YYYY-MM-DD format)
        end_date: End date for filtering (YYYY-MM-DD format)
        chart_type: Type of chart to generate ('bar', 'pie', 'donut', 'line', 'stacked'). Default is 'bar'.
    """
    # Validate chart_type
    valid_chart_types = ['bar', 'pie', 'donut', 'line', 'stacked']
    if chart_type not in valid_chart_types:
        chart_type = 'bar'  # Default to bar if invalid type provided
    
    # Capture org_id in closure to preserve across async boundary
    captured_org_id = get_current_org_id()  # Try context variables first
    
    # If context variables don't work, try to get org_id from session bindings
    if captured_org_id is None:
        # Try to find session binding by looking through all active sessions
        for session_id, binding in _session_tenant_bindings.items():
            if binding and binding.get("org_id"):
                captured_org_id = binding["org_id"]
                logger.info(f"Using org_id from session binding: {captured_org_id}")
                break
    
    if captured_org_id is None:
        logger.error("No org_id available from context variables or session bindings")
        return json.dumps({
            "success": False,
            "error": "Authentication context lost",
            "message": "Unable to determine organization context. Please try again."
        })
    
    # Handle the async database call - pass org_id explicitly to preserve it
    def run_async_db_call():
        """Run the async database call in a new event loop"""
        from app.services.database_service import DatabaseService
        db_service = DatabaseService()
        
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            # Use captured_org_id to avoid thread-local issues
            logger.info("Async execution using captured_org_id: %s", captured_org_id)
            return new_loop.run_until_complete(
                db_service.get_success_rate_by_file_name(file_name, captured_org_id, start_date, end_date)
            )
        finally:
            new_loop.close()

    try:
        # Try to get the current event loop
        current_loop = asyncio.get_running_loop()
        # If we're in an async context, run in a thread to avoid blocking
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = executor.submit(run_async_db_call).result()
    except RuntimeError:
        # No event loop running, safe to run directly
        result = run_async_db_call()
    
    # Ensure the result is properly formatted
    if isinstance(result, dict):
        # Add the chart_type to the result
        result["chart_type"] = chart_type
        result["chart_type_requested"] = True  # Flag to indicate LLM specified the type
        
        # Add stop flag to prevent infinite loops
        result.setdefault("stop", True)
        
        # Return as JSON string for LangGraph compatibility
        import json
        return json.dumps(result)
    else:
        # Handle unexpected result format
        error_result = {
            "success": False,
            "message": f"Unexpected result format from database: {type(result)}",
            "raw_result": str(result),
            "chart_type": chart_type,
            "stop": True
        }
        import json
        return json.dumps(error_result)


@tool("get_success_rate_by_domain_name", return_direct=False)
def get_success_rate_by_domain_name_tool(
    domain_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    chart_type: Optional[str] = "bar"
) -> Dict[str, Any]:
    """Calculate success/failure percentage for a domain and specify visualization type.
    
    Args:
        domain_name: Name of the domain to analyze (e.g., 'customer', 'product', 'sales')
        start_date: Start date for filtering (YYYY-MM-DD format)
        end_date: End date for filtering (YYYY-MM-DD format)
        chart_type: Type of chart to generate ('bar', 'pie', 'donut', 'line', 'stacked'). Default is 'bar'.
    """
    # Validate chart_type
    valid_chart_types = ['bar', 'pie', 'donut', 'line', 'stacked']
    if chart_type not in valid_chart_types:
        chart_type = 'bar'  # Default to bar if invalid type provided
    
    # Clean up domain_name - remove "_domain" suffix if present
    if domain_name and domain_name.endswith('_domain'):
        domain_name = domain_name[:-7]  # Remove "_domain" suffix
    
    # Get org_id from multiple sources (parameter, context variables, thread-local, module fallback)
    
    # Capture org_id in closure to preserve across async boundary
    captured_org_id = get_current_org_id()  # Try context variables first
    
    # If context variables don't work, try to get org_id from session bindings
    if captured_org_id is None:
        # Try to find session binding by looking through all active sessions
        for session_id, binding in _session_tenant_bindings.items():
            if binding and binding.get("org_id"):
                captured_org_id = binding["org_id"]
                logger.info(f"Using org_id from session binding: {captured_org_id}")
                break
    
    if captured_org_id is None:
        logger.error("No org_id available from context variables or session bindings")
        return json.dumps({
            "success": False,
            "error": "Authentication context lost",
            "message": "Unable to determine organization context. Please try again."
        })
    
    # Handle the async database call - pass org_id explicitly to preserve it
    def run_async_db_call():
        """Run the async database call in a new event loop"""
        from app.services.database_service import DatabaseService
        db_service = DatabaseService()
        
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            # Use captured_org_id to avoid thread-local issues
            logger.info("Async execution using captured_org_id: %s", captured_org_id)
            logger.info("Querying domain_name: %s", domain_name)
            return new_loop.run_until_complete(
                db_service.get_success_rate_by_domain_name(domain_name, captured_org_id, start_date, end_date)
            )
        finally:
            new_loop.close()

    try:
        # Try to get the current event loop
        current_loop = asyncio.get_running_loop()
        # If we're in an async context, run in a thread to avoid blocking
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = executor.submit(run_async_db_call).result()
    except RuntimeError:
        # No event loop running, safe to run directly
        result = run_async_db_call()
    
    # Ensure the result is properly formatted
    if isinstance(result, dict):
        # Add the chart_type to the result
        result["chart_type"] = chart_type
        result["chart_type_requested"] = True  # Flag to indicate LLM specified the type
        
        # Add stop flag to prevent infinite loops
        result.setdefault("stop", True)
        
        # Return as JSON string for LangGraph compatibility
        import json
        return json.dumps(result)
    else:
        # Handle unexpected result format
        error_result = {
            "success": False,
            "message": f"Unexpected result format from database: {type(result)}",
            "raw_result": str(result),
            "chart_type": chart_type,
            "stop": True
        }
        import json
        return json.dumps(error_result)