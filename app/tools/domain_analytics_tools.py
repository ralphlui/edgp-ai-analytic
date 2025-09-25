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
    if captured_org_id is None:
        logger.error("No org_id available from context variables or session bindings")
        return create_auth_error_response(return_as_json=True)
    
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


@tool("analyze_query_for_domain_analytics", return_direct=False) 
def analyze_query_for_domain_analytics_tool(
    user_query: str,
    chart_type: Optional[str] = "bar"
) -> str:
    """Analyze a user query to extract domain and grouping field, then get analytics.
    
    This tool parses natural language queries to identify:
    1. The domain to analyze (customer, product, order, etc.)
    2. The field to group by (country, category, region, etc.)
    3. The chart type requested (pie, bar, etc.)
    
    Examples:
    - "How many customers per country using pie chart?" → domain="customer", field="country", chart="pie"
    - "Show products by category" → domain="product", field="category", chart="bar"
    - "Order distribution by region as donut chart" → domain="order", field="region", chart="donut"
    
    Args:
        user_query: Natural language query to analyze
        chart_type: Default chart type if not specified in query
        
    Returns:
        JSON string with analytics results based on parsed query
    """
    try:
        # Parse the user query to extract domain and grouping field
        domain_name, group_by_field, extracted_chart_type = _parse_analytics_query(user_query)
        
        # Use extracted chart type if found, otherwise use provided default
        if extracted_chart_type:
            chart_type = extracted_chart_type
        
        chart_type = validate_chart_type(chart_type)
        
        if not domain_name or not group_by_field:
            return format_tool_response({
                "success": False,
                "error": "Could not parse query",
                "message": f"Could not identify domain and grouping field from query: '{user_query}'. Please be more specific (e.g., 'customers per country', 'products by category')."
            }, chart_type, "analyze_query_for_domain_analytics_tool", return_as_json=True)
        
        # Get org_id from context
        captured_org_id = get_org_id_for_tool()
        if captured_org_id is None:
            logger.error("No org_id available from context variables or session bindings")
            return create_auth_error_response(return_as_json=True)
        
        logger.info("Parsed query - domain: %s, group_by: %s, chart: %s", domain_name, group_by_field, chart_type)
        
        # Execute the analytics query
        result = execute_async_tool_call(
            "get_domain_analytics_by_field", 
            domain_name, group_by_field, captured_org_id, None, None
        )
        
        # Add parsing info to result for transparency
        if result.get("success"):
            result["parsed_query"] = {
                "original_query": user_query,
                "extracted_domain": domain_name,
                "extracted_field": group_by_field,
                "extracted_chart_type": extracted_chart_type or "default"
            }
        
        return format_tool_response(result, chart_type, "analyze_query_for_domain_analytics_tool", return_as_json=True)
        
    except Exception as e:
        logger.exception(f"Error analyzing query: {user_query}")
        return format_tool_response({
            "success": False,
            "error": str(e),
            "message": f"An error occurred while analyzing the query: '{user_query}'. Please try rephrasing your request."
        }, chart_type, "analyze_query_for_domain_analytics_tool", return_as_json=True)


def _parse_analytics_query(query: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse a natural language query to extract domain, grouping field, and chart type.
    
    Returns:
        tuple: (domain_name, group_by_field, chart_type)
    """
    query_lower = query.lower().strip()
    
    # Extract chart type
    chart_type = None
    chart_patterns = {
        r'\bpie\s+chart\b|\bpie\b': 'pie',
        r'\bbar\s+chart\b|\bbar\b': 'bar', 
        r'\bdonut\s+chart\b|\bdonut\b': 'donut',
        r'\bline\s+chart\b|\bline\b': 'line'
    }
    
    for pattern, chart in chart_patterns.items():
        if re.search(pattern, query_lower):
            chart_type = chart
            break
    
    # Extract domain and field combinations
    patterns = [
        # "customers per country", "customer by country"
        (r'\b(customers?)\s+(?:per|by)\s+(\w+)', lambda m: (m.group(1).rstrip('s'), m.group(2))),
        # "products by category", "product per category"  
        (r'\b(products?)\s+(?:per|by)\s+(\w+)', lambda m: (m.group(1).rstrip('s'), m.group(2))),
        # "orders by region", "order per region"
        (r'\b(orders?)\s+(?:per|by)\s+(\w+)', lambda m: (m.group(1).rstrip('s'), m.group(2))),
        # "users by status", "user per status"
        (r'\b(users?)\s+(?:per|by)\s+(\w+)', lambda m: (m.group(1).rstrip('s'), m.group(2))),
        # "how many X per Y"
        (r'how\s+many\s+(\w+)\s+(?:per|by)\s+(\w+)', lambda m: (m.group(1).rstrip('s'), m.group(2))),
        # "show X by Y"
        (r'show\s+(\w+)\s+(?:per|by)\s+(\w+)', lambda m: (m.group(1).rstrip('s'), m.group(2))),
        # "X distribution by Y"
        (r'(\w+)\s+distribution\s+(?:per|by)\s+(\w+)', lambda m: (m.group(1).rstrip('s'), m.group(2))),
        # "breakdown of X by Y" 
        (r'breakdown\s+of\s+(\w+)\s+(?:per|by)\s+(\w+)', lambda m: (m.group(1).rstrip('s'), m.group(2))),
        # "X breakdown by Y" (handles "Users breakdown by status")
        (r'(\w+)\s+breakdown\s+(?:per|by)\s+(\w+)', lambda m: (m.group(1).rstrip('s'), m.group(2)))
    ]
    
    for pattern, extractor in patterns:
        match = re.search(pattern, query_lower)
        if match:
            domain, field = extractor(match)
            return domain, field, chart_type
    
    return None, None, chart_type