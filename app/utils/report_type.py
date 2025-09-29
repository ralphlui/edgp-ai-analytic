"""
Report-type detection utilities for analytic processing.

Simple utility functions to detect if a user wants success-only, failure-only,
or both types of data in their analytics query. Uses lightweight regex patterns
for fast classification.
"""
import logging

logger = logging.getLogger(__name__)


async def is_analytics_related(prompt: str) -> bool:
    """
    Check if the user prompt is related to data analytics.

    Args:
        prompt: User's query string

    Returns:
        True if the query appears to be analytics-related, False otherwise
    """
    if not prompt or not prompt.strip():
        return False

    prompt_lower = prompt.lower().strip()

    # Analytics-related keywords and patterns
    analytics_keywords = [
        # Data analysis terms
        "analytics", "analysis", "analyze", "data", "statistics", "stats",
        "metrics", "report", "reporting", "dashboard", "chart", "graph",
        "visualization", "insights", "trends", "patterns",

        # Success/failure terms
        "success", "failure", "fail", "error", "successful", "failed",
        "pass", "passed", "rate", "percentage", "ratio",

        # File and domain terms
        "file", "domain", "upload", "dataset", "records", "entries",

        # Query terms
        "show me", "get", "retrieve", "fetch", "query", "search",
        "find", "lookup", "check", "view", "display", "list",

        # Time-based terms (be specific to avoid false positives)
        "date range", "time period", "date filter", "time filter", "between dates",
        "from date", "to date", "start date", "end date", "date from", "date to",

        # Aggregation terms
        "count", "total", "average", "sum", "group", "by"
    ]

    # Check for analytics keywords
    has_analytics_keywords = any(keyword in prompt_lower for keyword in analytics_keywords)

    # Check for question patterns that might be analytics-related
    question_patterns = [
        "what is", "how many", "how much", "what are",
        "can you show", "can you get", "can you tell me",
        "give me", "tell me about", "what", "how", "can you"
    ]

    has_question_pattern = any(pattern in prompt_lower for pattern in question_patterns)

    # If it has analytics keywords OR is a question that might be analytics-related
    if has_analytics_keywords:
        return True

    # For questions, be more lenient - assume analytics-related unless clearly not
    if has_question_pattern:
        # Exclude obviously non-analytics questions
        non_analytics_indicators = [
            "weather", "time is it", "current time", "what time", "day", "date", "calendar", "schedule",
            "joke", "funny", "laugh", "music", "song", "movie", "film",
            "food", "recipe", "cook", "restaurant", "travel", "vacation",
            "sports", "game", "play", "book", "read", "author",
            "health", "medical", "doctor", "exercise", "fitness",
            "shopping", "buy", "purchase", "price", "cost",
            "news", "politics", "government", "election",
            "help", "support", "contact", "phone", "email",
            "how are you", "how do you", "what are you", "who are you",
            "favorite", "color", "like", "prefer", "opinion", "think",
            "feel", "emotion", "mood", "happy", "sad"
        ]

        has_non_analytics = any(indicator in prompt_lower for indicator in non_analytics_indicators)

        # If it's a question but has non-analytics indicators, classify as non-analytics
        if has_non_analytics:
            return False

        # Otherwise, assume it's analytics-related (better to err on the side of processing)
        return True

    # For non-questions without analytics keywords, assume non-analytics
    return False


async def get_report_type(prompt: str) -> str:
    """
    Detect report type from user prompt.
    
    Args:
        prompt: User's query string
        
    Returns:
        "success" - only success data
        "failure" - only failure data  
        "both" - both success and failure data (default for ambiguous cases)
    """
    result = detect_report_type_simple(prompt)
    # Convert "uncertain" to "both" as the safe default
    return result if result != "uncertain" else "both"


def detect_report_type_simple(prompt: str) -> str:
    """
    Fast regex detection for clear-cut cases only.
    
    Args:
        prompt: User's query string
        
    Returns:
        "success", "failure", "both", or "uncertain"
    """
    if not prompt:
        return "uncertain"
        
    prompt_lower = prompt.lower().strip()
    
    # Very explicit success-only patterns
    success_only_patterns = [
        "only success", "success only", "just success",
        "successful only", "only successful",
        "show me success", "success records only",
        "success rate", "success percentage", "what is success rate",
        "get success rate", "show success rate"
    ]
    
    # Very explicit failure-only patterns  
    failure_only_patterns = [
        "only fail", "failure only", "only failure",
        "just fail", "just failure", "error only",
        "only error", "failed only", "only failed",
        "show me failures", "failure records only",
        "failure rate", "fail rate", "error rate",
        "what is failure rate", "get failure rate", "show failure rate"
    ]
    
    # Very explicit both/all patterns
    both_patterns = [
        "success and fail", "success and failure",
        "success and error", "pass and fail", 
        "fail and success", "failure and success",  # Order variations
        "all data", "complete analysis", "full report",
        "everything", "total analysis", "overall report",
        "both success and", "both types", "all records",
        "success rate and fail", "failure rate and success",
        "both success", "both failure", "success & fail",
        "success & failure", "pass & fail"
    ]
    
    # Check for explicit patterns - order matters (check "both" first)
    for pattern in both_patterns:
        if pattern in prompt_lower:
            logger.debug(f"Matched both pattern: '{pattern}' in prompt")
            return "both"
        
    for pattern in success_only_patterns:
        if pattern in prompt_lower:
            logger.debug(f"Matched success pattern: '{pattern}' in prompt")
            return "success"
    
    for pattern in failure_only_patterns:
        if pattern in prompt_lower:
            logger.debug(f"Matched failure pattern: '{pattern}' in prompt")
            return "failure"
    
    # Additional check for "success" AND "fail" words appearing together
    has_success = any(word in prompt_lower for word in ['success', 'successful', 'pass', 'passed'])
    has_failure = any(word in prompt_lower for word in ['fail', 'failure', 'failed', 'error'])
    
    if has_success and has_failure:
        logger.debug(f"Found both success and failure keywords in prompt")
        return "both"
    
    # For anything else, return uncertain (caller decides default)
    logger.debug(f"No clear pattern matched for prompt: '{prompt[:50]}...'")
    return "uncertain"