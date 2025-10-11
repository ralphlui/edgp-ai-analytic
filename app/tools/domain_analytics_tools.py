"""
Generic domain analytics tools for analyzing any domain data from the tracker table.
These tools work with the DynamoDB tracker table and can handle any domain (customer, product, etc.)
with flexible grouping fields (country, category, etc.).
"""
from langchain_core.tools import tool
from typing import Optional
import logging
import re
from .tool_utils import (
    validate_chart_type,
    validate_report_type,
    get_org_id_for_tool,
    execute_async_tool_call,
    format_tool_response,
    create_auth_error_response
)

logger = logging.getLogger("domain_analytics_tools")


@tool("get_domain_analytics_by_field", return_direct=False)
def get_domain_analytics_by_field_tool(
    domain_name: str,
    group_by_field: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    chart_type: Optional[str] = "bar",
    report_type: Optional[str] = "both"
) -> str:
    """Get analytics for any domain grouped by any field from the tracker table.
    
    This is a flexible tool that can analyze any domain (customer, product, order, etc.)
    and group results by any field (country, category, status, region, etc.).
    
    Examples:
    - domain_name="customer", group_by_field="country" → Customer distribution by country
    - domain_name="product", group_by_field="category" → Product distribution by category  
    - domain_name="order", group_by_field="region" → Order distribution by region
    
    Args:
        domain_name: Domain to analyze (e.g., 'customer', 'product', 'order')
        group_by_field: Field to group by (e.g., 'country', 'category', 'region')
        start_date: Start date for filtering (YYYY-MM-DD format)
        end_date: End date for filtering (YYYY-MM-DD format)
        chart_type: Type of chart to generate ('bar', 'pie', 'donut'). Default is 'bar'.
        report_type: Type of report to generate ('success', 'failure', 'both'). Default is 'both'.
        
    Returns:
        JSON string with analytics data grouped by the specified field
    """
    # Validate parameters
    chart_type = validate_chart_type(chart_type)
    
    if not domain_name or not group_by_field:
        return format_tool_response({
            "success": False,
            "error": "Both domain_name and group_by_field are required",
            "message": "Please specify both the domain to analyze and the field to group by."
        }, chart_type, "get_domain_analytics_by_field_tool", return_as_json=True, report_type=report_type)
    
    # Get org_id from context using shared utility
    captured_org_id = get_org_id_for_tool()
    # if captured_org_id is None:
    #     logger.error("No org_id available from context variables or session bindings")
    #     return create_auth_error_response(return_as_json=True)
    
    try:
        # Execute database call using shared utility
        logger.info("Getting %s analytics grouped by %s using org_id: %s", domain_name, group_by_field, captured_org_id)
        result = execute_async_tool_call(
            "get_domain_analytics_by_field",
            domain_name, group_by_field, captured_org_id, start_date, end_date
        )
        
        return format_tool_response(result, chart_type, "get_domain_analytics_by_field_tool", return_as_json=True, report_type=report_type)
        
    except Exception as e:
        logger.exception(f"Unexpected error in get_domain_analytics_by_field_tool: {e}")
        return format_tool_response({
            "success": False,
            "error": str(e),
            "message": f"An unexpected error occurred while analyzing {domain_name} distribution by {group_by_field}. Please try again."
        }, chart_type, "get_domain_analytics_by_field_tool", return_as_json=True, report_type=report_type)

