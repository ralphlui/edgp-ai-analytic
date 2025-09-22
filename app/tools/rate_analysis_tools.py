"""
Comprehensive rate analysis tools for the analytic system.
Consolidated module containing all success/failure rate calculation tools.
"""
from langchain_core.tools import tool
from typing import Optional
import logging
from .tool_utils import (
    validate_chart_type,
    get_org_id_for_tool,
    execute_async_tool_call,
    format_tool_response,
    create_auth_error_response,
    add_performance_assessment
)

logger = logging.getLogger("rate_analysis_tools")


@tool("get_file_analysis_rates", return_direct=False)
def get_file_analysis_rates_tool(
    file_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    chart_type: Optional[str] = "bar"
) -> str:
    """Calculate comprehensive success/failure rates for a file with visualization options.

    This tool analyzes overall processing results for a specific file, calculating both 
    success and failure percentages based on the general processing status.

    Args:
        file_name: Name of the file to analyze (e.g., 'customer_sample_values.csv')
        start_date: Start date for filtering (YYYY-MM-DD format)
        end_date: End date for filtering (YYYY-MM-DD format)
        chart_type: Type of chart to generate ('bar', 'pie', 'donut', 'line', 'stacked'). Default is 'bar'.
    """
    # Validate chart_type
    chart_type = validate_chart_type(chart_type)
    
    # Get org_id using shared utility
    captured_org_id = get_org_id_for_tool()
    
    if captured_org_id is None:
        logger.error("No org_id available from context variables or session bindings")
        return create_auth_error_response(return_as_json=True)
    
    try:
        # Execute database call using shared utility
        logger.info("Getting file analysis rates using org_id: %s", captured_org_id)
        result = execute_async_tool_call(
            "get_success_rate_by_file_name",
            file_name, captured_org_id, start_date, end_date
        )

        return format_tool_response(result, chart_type, "get_file_analysis_rates_tool", return_as_json=True)

    except Exception as e:
        logger.exception(f"Unexpected error in get_file_analysis_rates_tool: {e}")
        return format_tool_response({
            "success": False,
            "error": str(e),
            "message": f"An unexpected error occurred while analyzing file rates for '{file_name}'. Please try again."
        }, chart_type, "get_file_analysis_rates_tool", return_as_json=True)


@tool("get_domain_analysis_rates", return_direct=False)
def get_domain_analysis_rates_tool(
    domain_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    chart_type: Optional[str] = "bar"
) -> str:
    """Calculate comprehensive success/failure rates for a domain with visualization options.
    
    This tool analyzes overall processing results for a specific domain, calculating both 
    success and failure percentages based on the general processing status.
    
    Args:
        domain_name: Name of the domain to analyze (e.g., 'customer', 'product', 'sales')
        start_date: Start date for filtering (YYYY-MM-DD format)
        end_date: End date for filtering (YYYY-MM-DD format)
        chart_type: Type of chart to generate ('bar', 'pie', 'donut', 'line', 'stacked'). Default is 'bar'.
    """
    # Validate chart_type
    chart_type = validate_chart_type(chart_type)
    
    # Clean up domain_name - remove "_domain" suffix if present
    if domain_name and domain_name.endswith('_domain'):
        domain_name = domain_name[:-7]  # Remove "_domain" suffix
    
    # Get org_id using shared utility
    captured_org_id = get_org_id_for_tool()
    
    if captured_org_id is None:
        logger.error("No org_id available from context variables or session bindings")
        return create_auth_error_response(return_as_json=True)
    
    try:
        # Execute database call using shared utility
        logger.info("Getting domain analysis rates using org_id: %s", captured_org_id)
        logger.info("Querying domain_name: %s", domain_name)
        result = execute_async_tool_call(
            "get_success_rate_by_domain_name",
            domain_name, captured_org_id, start_date, end_date
        )

        return format_tool_response(result, chart_type, "get_domain_analysis_rates_tool", return_as_json=True)

    except Exception as e:
        logger.exception(f"Unexpected error in get_domain_analysis_rates_tool: {e}")
        return format_tool_response({
            "success": False,
            "error": str(e),
            "message": f"An unexpected error occurred while analyzing domain rates for '{domain_name}'. Please try again."
        }, chart_type, "get_domain_analysis_rates_tool", return_as_json=True)


@tool("get_rule_validation_rates", return_direct=False)
def get_rule_validation_rates_tool(
    file_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    chart_type: Optional[str] = "bar"
) -> str:
    """Calculate comprehensive success/failure rates for rule validation with detailed analysis.

    This tool analyzes rule validation results for a specific file, calculating both 
    success and failure percentages specifically for rule validation processes 
    (rule_status field analysis).

    Args:
        file_name: Name of the file to analyze (e.g., 'customer_sample_values.csv')
        start_date: Start date for filtering (YYYY-MM-DD format)
        end_date: End date for filtering (YYYY-MM-DD format)
        chart_type: Type of chart to generate ('bar', 'pie', 'donut'). Default is 'bar'.

    Returns:
        JSON string with rule validation rates, counts, performance assessment, and chart data
    """
    # Validate chart_type
    chart_type = validate_chart_type(chart_type)
    
    # Get org_id using shared utility
    captured_org_id = get_org_id_for_tool()
    
    if captured_org_id is None:
        logger.error("No org_id available from context variables or session bindings")
        return create_auth_error_response(return_as_json=True)

    try:
        # Execute database call using shared utility
        logger.info("Calculating rule validation rates using org_id: %s", captured_org_id)
        result = execute_async_tool_call(
            "get_rule_failure_rate",
            file_name, captured_org_id, start_date, end_date
        )

        # Add performance assessment for rule validation
        if result.get("success"):
            failure_rate = result.get("failure_rate", 0)
            add_performance_assessment(result, failure_rate, "rule")

        return format_tool_response(result, chart_type, "get_rule_validation_rates_tool", return_as_json=True)

    except Exception as e:
        logger.exception(f"Unexpected error in get_rule_validation_rates_tool: {e}")
        return format_tool_response({
            "success": False,
            "error": str(e),
            "message": f"An unexpected error occurred while analyzing rule validation rates for '{file_name}'. Please try again."
        }, chart_type, "get_rule_validation_rates_tool", return_as_json=True)


@tool("get_data_quality_validation_rates", return_direct=False)
def get_data_quality_validation_rates_tool(
    file_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    chart_type: Optional[str] = "bar"
) -> str:
    """Calculate comprehensive success/failure rates for data quality validation with detailed analysis.

    This tool analyzes data quality validation results for a specific file, calculating both 
    success and failure percentages specifically for data quality validation processes 
    (data_quality_status field analysis).

    Args:
        file_name: Name of the file to analyze (e.g., 'customer_sample_values.csv')
        start_date: Start date for filtering (YYYY-MM-DD format)
        end_date: End date for filtering (YYYY-MM-DD format)
        chart_type: Type of chart to generate ('bar', 'pie', 'donut'). Default is 'bar'.

    Returns:
        JSON string with data quality validation rates, counts, performance assessment, and chart data
    """
    # Validate chart_type
    chart_type = validate_chart_type(chart_type)
    
    # Get org_id using shared utility
    captured_org_id = get_org_id_for_tool()
    
    if captured_org_id is None:
        logger.error("No org_id available from context variables or session bindings")
        return create_auth_error_response(return_as_json=True)

    try:
        # Execute database call using shared utility
        logger.info("Calculating data quality validation rates using org_id: %s", captured_org_id)
        result = execute_async_tool_call(
            "get_data_quality_failure_rate",
            file_name, captured_org_id, start_date, end_date
        )

        # Add performance assessment for data quality validation
        if result.get("success"):
            failure_rate = result.get("failure_rate", 0)
            add_performance_assessment(result, failure_rate, "data_quality")

        return format_tool_response(result, chart_type, "get_data_quality_validation_rates_tool", return_as_json=True)

    except Exception as e:
        logger.exception(f"Unexpected error in get_data_quality_validation_rates_tool: {e}")
        return format_tool_response({
            "success": False,
            "error": str(e),
            "message": f"An unexpected error occurred while analyzing data quality validation rates for '{file_name}'. Please try again."
        }, chart_type, "get_data_quality_validation_rates_tool", return_as_json=True)