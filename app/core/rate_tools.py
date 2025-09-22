from langchain_core.tools import tool
from typing import Optional
import asyncio
import json
import logging
from app.utils.request_context import get_current_org_id

logger = logging.getLogger("rate_tools")

# Import session bindings from tools_agent
from app.core.tools_agent import _session_tenant_bindings


@tool("get_rule_failure_rate", return_direct=False)
def get_rule_failure_rate_tool(
    file_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    chart_type: Optional[str] = "bar"
) -> str:
    """Calculate the failure rate percentage for rule validation (rule_status=false) for a specific file.

    Args:
        file_name: Name of the file to analyze (e.g., 'customer_sample_values.csv')
        start_date: Start date for filtering (YYYY-MM-DD format)
        end_date: End date for filtering (YYYY-MM-DD format)
        chart_type: Type of chart to generate ('bar', 'pie', 'donut'). Default is 'bar'.

    Returns:
        JSON string with rule failure rate percentage, counts, and chart data
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
            logger.info("Calculating rule failure rate using captured_org_id: %s", captured_org_id)
            return new_loop.run_until_complete(
                db_service.get_rule_failure_rate(file_name, captured_org_id, start_date, end_date)
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
        # No event loop running, we can call the async function directly
        result = run_async_db_call()

    # Add chart_type to result for chart generation
    if result.get("success"):
        result["chart_type"] = chart_type
        result["chart_type_requested"] = True
        
        # Add performance assessment
        failure_rate = result.get("failure_rate", 0)
        if failure_rate == 0:
            result["performance_assessment"] = "Excellent - no rule failures detected"
        elif failure_rate <= 5:
            result["performance_assessment"] = "Good - low rule failure rate"
        elif failure_rate <= 15:
            result["performance_assessment"] = "Moderate - room for improvement in rule compliance"
        else:
            result["performance_assessment"] = "Poor - high rule failure rate requires attention"

    try:
        return json.dumps(result)
    except Exception as e:
        logger.exception(f"Unexpected error in get_rule_failure_rate_tool: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "message": f"An unexpected error occurred while analyzing rule failure rate for '{file_name}'. Please try again."
        })


@tool("get_data_quality_failure_rate", return_direct=False)
def get_data_quality_failure_rate_tool(
    file_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    chart_type: Optional[str] = "bar"
) -> str:
    """Calculate the failure rate percentage for data quality validation (data_quality_status=false) for a specific file.

    Args:
        file_name: Name of the file to analyze (e.g., 'customer_sample_values.csv')
        start_date: Start date for filtering (YYYY-MM-DD format)
        end_date: End date for filtering (YYYY-MM-DD format)
        chart_type: Type of chart to generate ('bar', 'pie', 'donut'). Default is 'bar'.

    Returns:
        JSON string with data quality failure rate percentage, counts, and chart data
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
            logger.info("Calculating data quality failure rate using captured_org_id: %s", captured_org_id)
            return new_loop.run_until_complete(
                db_service.get_data_quality_failure_rate(file_name, captured_org_id, start_date, end_date)
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
        # No event loop running, we can call the async function directly
        result = run_async_db_call()

    # Add chart_type to result for chart generation
    if result.get("success"):
        result["chart_type"] = chart_type
        result["chart_type_requested"] = True
        
        # Add performance assessment
        failure_rate = result.get("failure_rate", 0)
        if failure_rate == 0:
            result["performance_assessment"] = "Excellent - no data quality issues detected"
        elif failure_rate <= 5:
            result["performance_assessment"] = "Good - low data quality failure rate"
        elif failure_rate <= 15:
            result["performance_assessment"] = "Moderate - some data quality issues need attention"
        else:
            result["performance_assessment"] = "Poor - significant data quality issues require immediate action"

    try:
        return json.dumps(result)
    except Exception as e:
        logger.exception(f"Unexpected error in get_data_quality_failure_rate_tool: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "message": f"An unexpected error occurred while analyzing data quality failure rate for '{file_name}'. Please try again."
        })