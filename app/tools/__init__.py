"""
Tools package for the analytic system.

This package contains all LangGraph tools and utilities for:
- Rate analysis and validation reporting
- Session management and authentication context
- Shared utilities for tool optimization

Modules:
- rate_analysis_tools: Comprehensive success/failure rate calculation tools
- tool_utils: Shared utilities for performance and consistency
- session_manager: Session binding and authentication context management
"""

from .rate_analysis_tools import (
    get_file_analysis_rates_tool,
    get_domain_analysis_rates_tool,
    get_rule_validation_rates_tool,
    get_data_quality_validation_rates_tool
)

from .domain_analytics_tools import (
    get_domain_analytics_by_field_tool,
    analyze_query_for_domain_analytics_tool
)

from .session_manager import (
    bind_session_to_tenant,
    unbind_session,
    get_session_context,
    cleanup_expired_sessions,
    get_active_sessions
)

__all__ = [
    # Rate analysis tools
    "get_file_analysis_rates_tool",
    "get_domain_analysis_rates_tool", 
    "get_rule_validation_rates_tool",
    "get_data_quality_validation_rates_tool",
    
    # Domain analytics tools
    "get_domain_analytics_by_field_tool",
    "analyze_query_for_domain_analytics_tool",
    
    # Session management
    "bind_session_to_tenant",
    "unbind_session",
    "get_session_context",
    "cleanup_expired_sessions",
    "get_active_sessions"
]

# Tool registry for easy access
ANALYSIS_TOOLS = [
    get_file_analysis_rates_tool,
    get_domain_analysis_rates_tool,
    get_rule_validation_rates_tool,
    get_data_quality_validation_rates_tool,
    get_domain_analytics_by_field_tool,
    analyze_query_for_domain_analytics_tool
]