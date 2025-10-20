"""
LangChain tools for analytics report generation.

These tools integrate with the AnalyticsRepository to generate
success/failure rate reports with charts.
"""
import logging
from typing import Optional
from langchain.tools import tool

from app.repositories.analytics_repository import get_analytics_repository

logger = logging.getLogger("analytic_agent")


@tool
def generate_success_rate_report(
    domain_name: Optional[str] = None,
    file_name: Optional[str] = None,
    org_id: Optional[str] = None
) -> dict:
    """
    Retrieve success rate analytics data for a domain or file.
    
    This tool returns RAW DATA ONLY. The LLM will format the response naturally.
    Use this tool when the user asks for success rate analysis.
    Provide exactly ONE of domain_name or file_name.
    
    Args:
        domain_name: The domain to analyze (e.g., "customer", "payment")
        file_name: The file to analyze (e.g., "customer.csv", "transactions.csv")
        org_id: Organization ID for multi-tenant data isolation
    
    Returns:
        Dictionary with success status and raw analytics data:
        {
            "success": bool,
            "data": {
                "target_type": str,  # "domain" or "file"
                "target_value": str,  # actual domain/file name
                "total_requests": int,
                "successful_requests": int,
                "failed_requests": int,
                "success_rate": float  # percentage 0-100
            }
        }
    
    Examples:
        - generate_success_rate_report(domain_name="customer")
        - generate_success_rate_report(file_name="customer.csv")
    """
    logger.info(f"Tool called: generate_success_rate_report(domain={domain_name}, file={file_name}, org_id={org_id})")

    # Validate input

    if not org_id:
        logger.error("Organization ID not provided")
        return {
            "success": False,
            "error": "Your account is not associated with any organization"
        }
    
    if not domain_name and not file_name:
        logger.error("Neither domain_name nor file_name provided")
        return {
            "success": False,
            "error": "Must provide either domain_name or file_name"
        }
    
    if domain_name and file_name:
        logger.error("Both domain_name and file_name provided")
        return {
            "success": False,
            "error": "Provide only ONE of domain_name or file_name, not both"
        }
    
    try:
        # Initialize repository
        repo = get_analytics_repository()
        
        # Query based on target type
        if domain_name:
            logger.info(f"Querying success rate for domain: {domain_name}")
            data = repo.get_success_rate_by_domain(domain_name, org_id=org_id)
        else:
            logger.info(f"Querying success rate for file: {file_name}")
            data = repo.get_success_rate_by_file(file_name, org_id=org_id)
        
        # Return raw data only - LLM will generate natural language response
        logger.info(f"Success rate data retrieved: {data['total_requests']} requests, {data['success_rate']}% success")
        
        return {
            "success": True,
            "data": {
                "target_type": data["target_type"],
                "target_value": data["target_value"],
                "total_requests": data["total_requests"],
                "successful_requests": data["successful_requests"],
                "failed_requests": data["failed_requests"],
                "success_rate": data["success_rate"],
                "report_type": "success_rate"  # Report type for chart generation
            }
        }
        
    except Exception as e:
        logger.exception(f"Error generating success rate report: {e}")
        return {
            "success": False,
            "error": f"Error generating report: {str(e)}"
        }


@tool
def generate_failure_rate_report(
    domain_name: Optional[str] = None,
    file_name: Optional[str] = None,
    org_id: Optional[str] = None
) -> dict:
    """
    Retrieve failure rate analytics data.
    
    Analyzes failure rates for a specific domain or file by examining analytics events
    and calculating how often requests fail.
    
    RAW DATA ONLY. The LLM will format the response naturally.
    
    Args:
        domain_name: The domain to analyze (e.g., "example.com")
        file_name: The file to analyze (e.g., "upload.pdf")
        org_id: Organization ID for multi-tenant data isolation
    
    Returns:
        dict: {
            "success": bool,
            "data": {
                "target_type": str ("domain" or "file"),
                "target_value": str (actual domain/file name),
                "total_requests": int,
                "successful_requests": int,
                "failed_requests": int,
                "failure_rate": float (percentage)
            }
        } on success, or {
            "success": bool,
            "error": str (error description)
        } on failure
    
    Note: Must provide exactly one of domain_name or file_name, not both.
    """
    logger.info(f"Tool called: generate_failure_rate_report(domain={domain_name}, file={file_name}, org_id={org_id})")
    
    # Validate input - exactly one of domain_name or file_name required


    if not org_id:
        logger.error("Organization ID not provided")
        return {
            "success": False,
            "error": "Your account is not associated with any organization"
        }
        
    
    if not domain_name and not file_name:
        logger.error("Neither domain_name nor file_name provided")
        return {
            "success": False,
            "error": "Must provide either domain_name or file_name"
        }
    
    if domain_name and file_name:
        logger.error("Both domain_name and file_name provided")
        return {
            "success": False,
            "error": "Provide only ONE of domain_name or file_name, not both"
        }
    
    try:
        # Initialize repository
        repo = get_analytics_repository()
        
        # Query based on target type
        if domain_name:
            logger.info(f"Querying failure rate for domain: {domain_name}")
            data = repo.get_failure_rate_by_domain(domain_name, org_id=org_id)
        else:
            logger.info(f"Querying failure rate for file: {file_name}")
            data = repo.get_failure_rate_by_file(file_name, org_id=org_id)
        
        # Return raw data only - LLM will generate natural language response
        logger.info(f"Failure rate data retrieved: {data['total_requests']} requests, {data['failure_rate']}% failures")
        
        return {
            "success": True,
            "data": {
                "target_type": data["target_type"],
                "target_value": data["target_value"],
                "total_requests": data["total_requests"],
                "successful_requests": data["successful_requests"],
                "failed_requests": data["failed_requests"],
                "failure_rate": data["failure_rate"],
                "report_type": "failure_rate"  # Report type for chart generation
            }
        }
        
    except Exception as e:
        logger.exception(f"Error generating failure rate report: {e}")
        return {
            "success": False,
            "error": f"Error generating report: {str(e)}"
        }


def get_analytics_tools():
    """
    Get list of all analytics tools for LangGraph/LangChain.
    
    Returns:
        List of tool functions
    """
    return [
        generate_success_rate_report,
        generate_failure_rate_report
    ]
