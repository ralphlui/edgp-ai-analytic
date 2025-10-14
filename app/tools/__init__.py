"""
Tools package for the analytic system.

This package contains all LangGraph tools and utilities for:
- Rate analysis and validation reporting
- Session management and authentication context  
- Domain analytics and data insights

Modules:
- rate_analysis_tools: Comprehensive success/failure rate calculation tools
- domain_analytics_tools: Domain-based analytics and grouping tools
- session_manager: Session binding and authentication context management

Note: Conversation history uses TTL-based automatic cleanup via DynamoDB.
"""

from .analytics_tools import (
    generate_success_rate_report,
    generate_failure_rate_report,
    get_analytics_tools
)

__all__ = [
    # Rate analysis tools
    "get_file_analysis_rates_tool",
    "get_domain_analysis_rates_tool", 
    "get_rule_validation_rates_tool",
    "get_data_quality_validation_rates_tool",
    
    # Domain analytics tools
    "get_domain_analytics_by_field_tool",
    #"analyze_query_for_domain_analytics_tool",
    
    # Session management
    "bind_session_to_tenant",
    "unbind_session",
    "get_session_context",
    "cleanup_expired_sessions",
    
    # New analytics tools
    "generate_success_rate_report",
    "generate_failure_rate_report",
    "get_analytics_tools",
]


# New analytics tools (success/failure rate)
ANALYTICS_TOOLS = [
    generate_success_rate_report,
    generate_failure_rate_report
]