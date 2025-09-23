"""
LLM-based classification utilities for analytic processing.
"""
import asyncio
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import OPENAI_MODEL, USE_LLM
from app.utils.sanitization import sanitize_text_input

logger = logging.getLogger(__name__)


async def get_report_type(prompt: str) -> str:
    """
    Optimized hybrid report type detection.
    
    Uses fast regex for obvious cases, LLM for complex queries.
    
    Args:
        prompt: User's query string
        
    Returns:
        "success" - only success data
        "failure" - only failure data  
        "both" - both success and failure data (default)
    """
    # Quick regex check for obvious cases (handles ~80-90% of queries instantly)
    quick_result = detect_report_type_simple(prompt)
    if quick_result != "uncertain":
        logger.info(f"Quick classification: '{quick_result}' for prompt: '{prompt[:50]}...'")
        return quick_result
    
    # Use LLM for complex/ambiguous cases
    if USE_LLM:
        try:
            logger.info(f"Using LLM classification for complex prompt: '{prompt[:50]}...'")
            result = await get_report_type_from_llm(prompt)
            return result
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, falling back to default")
    else:
        logger.info("LLM disabled, using default for ambiguous prompt")
    
    # Safe fallback - show both types when unsure
    return "both"


def detect_report_type_simple(prompt: str) -> str:
    """
    Simplified regex detection for clear-cut cases only.
    
    Only checks for very obvious patterns to avoid false positives.
    Returns "uncertain" for anything ambiguous.
    
    Args:
        prompt: User's query string
        
    Returns:
        "success", "failure", "both", or "uncertain"
    """
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
    
    # Check for explicit patterns
    for pattern in both_patterns:
        if pattern in prompt_lower:
            logger.info(f"Matched both pattern: '{pattern}' in prompt")
            return "both"
        
    for pattern in success_only_patterns:
        if pattern in prompt_lower:
            logger.info(f"Matched success pattern: '{pattern}' in prompt")
            return "success"
    
    for pattern in failure_only_patterns:
        if pattern in prompt_lower:
            logger.info(f"Matched failure pattern: '{pattern}' in prompt")
            return "failure"
            
    
    # Additional check for "success" AND "fail" words appearing together
    has_success = any(word in prompt_lower for word in ['success', 'successful', 'pass', 'passed'])
    has_failure = any(word in prompt_lower for word in ['fail', 'failure', 'failed', 'error'])
    
    if has_success and has_failure:
        logger.info(f"Found both success and failure keywords in prompt, returning 'both'")
        return "both"
    
    # For anything else, let LLM decide or use default
    logger.info(f"No pattern matched for prompt: '{prompt[:50]}...', returning uncertain")
    return "uncertain"


async def get_report_type_from_llm(prompt: str) -> str:
    """
    Use LLM to determine report type for complex/ambiguous queries.
    
    Args:
        prompt: User's query string
        
    Returns:
        "success", "failure", or "both"
    """
    try:
        llm = ChatOpenAI(
            model=OPENAI_MODEL, 
            temperature=0.0,  # Zero temperature for consistent classification
            max_tokens=10     # We only need one word
        )

        # Clean and truncate the prompt for safety
        safe_prompt = sanitize_text_input(prompt, 200)
        
        classification_prompt = f"""Analyze this query and determine what type of report the user wants:

QUERY: "{safe_prompt}"

RULES:
- Return "success" if user wants ONLY success/passed/working data
- Return "failure" if user wants ONLY failure/error/broken data  
- Return "both" if user wants complete analysis or mentions both types

Examples:
- "show success rate" → "success"
- "what failed last week" → "failure"  
- "analyze the results" → "both"
- "performance report" → "both"

Respond with exactly one word: success, failure, or both"""

        messages = [
            SystemMessage(content="You are a query classifier. Respond with exactly one word: success, failure, or both."),
            HumanMessage(content=classification_prompt)
        ]

        # Run LLM in thread pool to avoid blocking
        response = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: llm.invoke(messages)
        )

        result = response.content.strip().lower()
        
        # Validate and sanitize the response
        if result in ["success", "failure", "both"]:
            logger.info(f"LLM classified '{safe_prompt[:30]}...' as '{result}'")
            return result
        else:
            logger.warning(f"LLM returned invalid classification: '{result}', defaulting to 'both'")
            return "both"

    except Exception as e:
        logger.error(f"LLM classification error: {e}")
        # Safe fallback
        return "both"