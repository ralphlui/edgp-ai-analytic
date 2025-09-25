"""
Customer analytics tools for analyzing customer data from the tracker table.
These tools work with the existing DynamoDB tracker table where domain = 'customer'.
"""
from langchain_core.tools import tool
from typing import Optional
import logging
from .tool_utils import (
    validate_chart_type,
    validate_report_type,
    get_org_id_for_tool,
    execute_async_tool_call,
    format_tool_response,
    create_auth_error_response
)

logger = logging.getLogger("customer_analytics_tools")


@tool("get_customers_per_country", return_direct=False)
def get_customers_per_country_tool(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    chart_type: Optional[str] = "bar",
    report_type: Optional[str] = "both"
) -> str:
    """Get customer count by country from the tracker table where domain = 'customer'.
    
    This tool analyzes customer distribution across countries by querying the tracker table
    for records where domain_name = 'customer', filtered by organization ID.
    
    Args:
        start_date: Start date for filtering (YYYY-MM-DD format)
        end_date: End date for filtering (YYYY-MM-DD format) 
        chart_type: Type of chart to generate ('bar', 'pie', 'donut'). Default is 'bar'.
        report_type: Type of report to generate ('success', 'failure', 'both'). Default is 'both'.
        
    Returns:
        JSON string with customer counts per country and chart data
    """
    # Validate parameters
    chart_type = validate_chart_type(chart_type)
    #report_type = validate_report_type(report_type)
    
    # Get org_id from context using shared utility
    captured_org_id = get_org_id_for_tool()
    if captured_org_id is None:
        logger.error("No org_id available from context variables or session bindings")
        return create_auth_error_response(return_as_json=True)
    
    try:
        # Execute database call using shared utility
        logger.info("Getting customers per country using org_id: %s", captured_org_id)
        result = execute_async_tool_call(
            "get_customers_per_country",
            captured_org_id, start_date, end_date
        )
        
        return format_tool_response(result, chart_type, "get_customers_per_country_tool", return_as_json=True, report_type=report_type)
        
    except Exception as e:
        logger.exception(f"Unexpected error in get_customers_per_country_tool: {e}")
        return format_tool_response({
            "success": False,
            "error": str(e),
            "message": "An unexpected error occurred while analyzing customer distribution by country. Please try again."
        }, chart_type, "get_customers_per_country_tool", return_as_json=True, report_type=report_type)