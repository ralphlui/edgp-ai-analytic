"""
Comprehensive rate analysis tools for the analytic system.

This module consolidates success/failure rate calculation tools for files, domains,
rule validation, and data quality validation. It supports 'auto' chart type selection
that recommends an appropriate visualization when the user doesn't specify one.
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
    create_auth_error_response,
    add_performance_assessment
)

logger = logging.getLogger("rate_analysis_tools")


def _recommend_chart_type(result: dict, report_type: Optional[str]) -> str:
    """Heuristic to pick a chart type when user doesn't specify one.

    - If timeseries-like data is present, prefer 'line' (or 'stacked' for both).
    - If exactly two categories (success/failure), prefer 'donut'.
    - Otherwise, default to 'bar'.
    """
    try:
        # Look for timeseries keys commonly returned by tools
        if any(k in result for k in ("by_date", "timeseries", "trend")):
            return "stacked" if (report_type or "both") == "both" else "line"

        chart_data = result.get("chart_data") or []
        # If two categories success/fail, donut/pie works well
        labels = set()
        for item in chart_data:
            status = item.get("status") or item.get("label")
            if status:
                labels.add(str(status).lower())
        if labels.issubset({"success", "fail", "failure"}) and 1 < len(labels) <= 2:
            return "donut"
    except Exception:
        pass
    return "bar"


@tool("get_file_analysis_rates", return_direct=False)
def get_file_analysis_rates_tool(
    file_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    chart_type: Optional[str] = "bar",
    report_type: Optional[str] = None
) -> str:
    """Calculate comprehensive success/failure rates for a file with visualization options.

    This tool analyzes overall processing results for a specific file, calculating both 
    success and failure percentages based on the general processing status.

    Args:
        file_name: Name of the file to analyze (e.g., 'customer_sample_values.csv')
        start_date: Start date for filtering (YYYY-MM-DD format)
        end_date: End date for filtering (YYYY-MM-DD format)
        chart_type: Type of chart to generate ('auto', 'bar', 'pie', 'donut', 'line', 'stacked'). Default is 'auto'.
        report_type: Type of report to generate ('success', 'failure', 'both'). REQUIRED - must be specified.
    """
    
    # Get org_id using shared utility
    captured_org_id = get_org_id_for_tool()
    
    try:
        # Execute database call using shared utility
        logger.info("Getting file analysis rates using org_id: %s", captured_org_id)
        result = execute_async_tool_call(
            "get_success_rate_by_file_name",
            file_name, captured_org_id, start_date, end_date
        )

        return format_tool_response(result, chart_type, "get_file_analysis_rates_tool", return_as_json=True, report_type=report_type)

    except Exception as e:
        logger.exception(f"Unexpected error in get_file_analysis_rates_tool: {e}")
        chart_type_fallback = chart_type or "bar"
        return format_tool_response({
            "success": False,
            "error": str(e),
            "message": f"An unexpected error occurred while analyzing file rates for '{file_name}'. Please try again."
        }, chart_type_fallback, "get_file_analysis_rates_tool", return_as_json=True, report_type=report_type)


@tool("get_domain_analysis_rates", return_direct=False)
def get_domain_analysis_rates_tool(
    domain_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    chart_type: Optional[str] = "bar",
    report_type: Optional[str] = None
) -> str:
    """Calculate comprehensive success/failure rates for a domain with visualization options.
    
    This tool analyzes overall processing results for a specific domain, calculating both 
    success and failure percentages based on the general processing status.
    
    Args:
        domain_name: Name of the domain to analyze (e.g., 'customer', 'product', 'sales')
        start_date: Start date for filtering (YYYY-MM-DD format)
        end_date: End date for filtering (YYYY-MM-DD format)
        chart_type: Type of chart to generate ('auto', 'bar', 'pie', 'donut', 'line', 'stacked'). Default is 'auto'.
        report_type: Type of report to generate ('success', 'failure', 'both'). REQUIRED - must be specified.
    """
    
    # Clean up domain_name - remove "_domain" suffix if present
    if domain_name and domain_name.endswith('_domain'):
        domain_name = domain_name[:-7]  # Remove "_domain" suffix
    
    # Get org_id using shared utility
    captured_org_id = get_org_id_for_tool()
    
    try:
        # Execute database call using shared utility
        logger.info("Getting domain analysis rates using org_id: %s", captured_org_id)
        logger.info("Querying domain_name: %s", domain_name)
        result = execute_async_tool_call(
            "get_success_rate_by_domain_name",
            domain_name, captured_org_id, start_date, end_date
        )

        return format_tool_response(result, chart_type, "get_domain_analysis_rates_tool", return_as_json=True, report_type=report_type)

    except Exception as e:
        logger.exception(f"Unexpected error in get_domain_analysis_rates_tool: {e}")
        chart_type_fallback = chart_type or "bar"
        return format_tool_response({
            "success": False,
            "error": str(e),
            "message": f"An unexpected error occurred while analyzing domain rates for '{domain_name}'. Please try again."
        }, chart_type_fallback, "get_domain_analysis_rates_tool", return_as_json=True, report_type=report_type)


@tool("get_rule_validation_rates", return_direct=False)
def get_rule_validation_rates_tool(
    file_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    chart_type: Optional[str] = "bar",
    report_type: Optional[str] = None
) -> str:
    """Calculate comprehensive success/failure rates for rule validation with detailed analysis.

    This tool analyzes rule validation results for a specific file, calculating both 
    success and failure percentages specifically for rule validation processes 
    (rule_status field analysis).

    Args:
        file_name: Name of the file to analyze (e.g., 'customer_sample_values.csv')
        start_date: Start date for filtering (YYYY-MM-DD format)
        end_date: End date for filtering (YYYY-MM-DD format)
        chart_type: Type of chart to generate ('auto', 'bar', 'pie', 'donut', 'line', 'stacked'). Default is 'auto'.
        report_type: Type of report to generate ('success', 'failure', 'both'). REQUIRED - must be specified.

    Returns:
        JSON string with rule validation rates, counts, performance assessment, and chart data
    """
    
    # Get org_id using shared utility
    captured_org_id = get_org_id_for_tool()
    
    # if captured_org_id is None:
    #     logger.error("No org_id available from context variables or session bindings")
    #     return create_auth_error_response(return_as_json=True)

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

        return format_tool_response(result, chart_type, "get_rule_validation_rates_tool", return_as_json=True, report_type=report_type)

    except Exception as e:
        logger.exception(f"Unexpected error in get_rule_validation_rates_tool: {e}")
        chart_type_fallback = chart_type or "bar"
        return format_tool_response({
            "success": False,
            "error": str(e),
            "message": f"An unexpected error occurred while analyzing rule validation rates for '{file_name}'. Please try again."
        }, chart_type_fallback, "get_rule_validation_rates_tool", return_as_json=True, report_type=report_type)


@tool("get_data_quality_validation_rates", return_direct=False)
def get_data_quality_validation_rates_tool(
    file_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    chart_type: Optional[str] = "bar",
    report_type: Optional[str] = None
) -> str:
    """Calculate comprehensive success/failure rates for data quality validation with detailed analysis.

    This tool analyzes data quality validation results for a specific file, calculating both 
    success and failure percentages specifically for data quality validation processes 
    (data_quality_status field analysis).

    Args:
        file_name: Name of the file to analyze (e.g., 'customer_sample_values.csv')
        start_date: Start date for filtering (YYYY-MM-DD format)
        end_date: End date for filtering (YYYY-MM-DD format)
        chart_type: Type of chart to generate ('auto', 'bar', 'pie', 'donut', 'line', 'stacked'). Default is 'auto'.
        report_type: Type of report to generate ('success', 'failure', 'both'). REQUIRED - must be specified.

    Returns:
        JSON string with data quality validation rates, counts, performance assessment, and chart data
    """
    
    # Get org_id using shared utility
    captured_org_id = get_org_id_for_tool()
    
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

        # Choose chart type if not provided or set to auto
        #chart_type_final = chart_type or _recommend_chart_type(result, report_type)

        return format_tool_response(result, chart_type, "get_data_quality_validation_rates_tool", return_as_json=True, report_type=report_type)

    except Exception as e:
        logger.exception(f"Unexpected error in get_data_quality_validation_rates_tool: {e}")
        chart_type_fallback = chart_type or "bar"
        return format_tool_response({
            "success": False,
            "error": str(e),
            "message": f"An unexpected error occurred while analyzing data quality validation rates for '{file_name}'. Please try again."
        }, chart_type_fallback, "get_data_quality_validation_rates_tool", return_as_json=True, report_type=report_type)