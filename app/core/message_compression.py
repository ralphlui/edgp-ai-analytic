"""
Message compression utilities with PII protection and token budgets.
"""
import re
import logging
from typing import List, Dict, Any, Tuple
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage

logger = logging.getLogger(__name__)

# Sensitivity patterns for PII detection
SENSITIVITY_PATTERNS = {
    'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    'phone': re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
    'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    'credit_card': re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
    'api_key': re.compile(r'\b[A-Za-z0-9]{32,}\b'),
    'jwt': re.compile(r'\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b'),
}

# Token budget constraints
TOKEN_BUDGETS = {
    'system_message': 2000,      # System prompt budget
    'user_message': 500,          # Per user message
    'tool_message': 1000,         # Per tool result (compressed)
    'ai_message': 300,            # AI responses (summary only)
    'total_limit': 25000,         # Total conversation limit (~6k tokens)
}


def _detect_sensitive_data(text: str) -> List[str]:
    """Detect PII/sensitive patterns in text."""
    detected = []
    for pattern_name, pattern in SENSITIVITY_PATTERNS.items():
        if pattern.search(text):
            detected.append(pattern_name)
    return detected


def _redact_sensitive_data(text: str) -> Tuple[str, List[str]]:
    """Redact sensitive data and return redacted text + detected types."""
    detected = []
    redacted = text
    
    for pattern_name, pattern in SENSITIVITY_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            detected.append(pattern_name)
            redacted = pattern.sub(f'[REDACTED_{pattern_name.upper()}]', redacted)
    
    return redacted, detected


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4


def _compress_tool_message(msg: ToolMessage, budget: int = TOKEN_BUDGETS['tool_message']) -> str:
    """
    Compress tool message to fit budget with PII protection.
    
    Returns summary string with key metrics and redacted sensitive data.
    """
    try:
        import json
        content = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
        
        # Extract key fields
        summary_parts = []
        
        if isinstance(content, dict):
            # Add success status
            if 'success' in content:
                summary_parts.append(f"Success: {content['success']}")
            
            # Add key metrics
            if 'row_count' in content:
                summary_parts.append(f"Rows: {content['row_count']}")
            if 'file_name' in content:
                summary_parts.append(f"File: {content['file_name']}")
            if 'domain_name' in content:
                summary_parts.append(f"Domain: {content['domain_name']}")
            if 'report_type' in content:
                summary_parts.append(f"Type: {content['report_type']}")
            
            # Compress chart_data if present
            if 'chart_data' in content and isinstance(content['chart_data'], list):
                data_count = len(content['chart_data'])
                summary_parts.append(f"Data points: {data_count}")
                
                # Include sample (first 3 items) with PII check
                if data_count > 0:
                    sample = content['chart_data'][:3]
                    sample_str = json.dumps(sample, indent=None)
                    
                    # Check for PII
                    sample_redacted, detected = _redact_sensitive_data(sample_str)
                    if detected:
                        logger.warning(f"PII detected in tool result: {detected}")
                    
                    # Truncate if still too long
                    if _estimate_tokens(sample_redacted) > budget // 2:
                        sample_redacted = sample_redacted[:budget * 2] + "..."
                    
                    summary_parts.append(f"Sample: {sample_redacted}")
        
        summary = " | ".join(summary_parts)
        
        # Final budget check
        if _estimate_tokens(summary) > budget:
            summary = summary[:budget * 4] + "... [truncated]"
        
        return summary
        
    except Exception as e:
        logger.warning(f"Failed to compress tool message: {e}")
        return f"[Tool result - compression failed: {str(e)[:100]}]"


def compress_tool_messages(
    messages: List[BaseMessage],
    budget_per_msg: int = TOKEN_BUDGETS['tool_message'],
    total_budget: int = TOKEN_BUDGETS['total_limit']
) -> Tuple[List[BaseMessage], Dict[str, Any]]:
    """
    Compress tool messages to fit token budget with PII protection.
    
    Returns:
        - List of compressed messages
        - Compression stats dict
    """
    compressed = []
    total_tokens = 0
    stats = {
        'original_count': len(messages),
        'compressed_count': 0,
        'tokens_saved': 0,
        'pii_detected': False,
        'truncated': False
    }
    
    for msg in messages:
        if isinstance(msg, ToolMessage):
            # Compress tool message
            original_size = _estimate_tokens(str(msg.content))
            compressed_content = _compress_tool_message(msg, budget_per_msg)
            compressed_size = _estimate_tokens(compressed_content)
            
            # Create compressed message
            compressed_msg = ToolMessage(
                content=compressed_content,
                tool_call_id=msg.tool_call_id
            )
            compressed.append(compressed_msg)
            
            stats['tokens_saved'] += (original_size - compressed_size)
            stats['compressed_count'] += 1
            total_tokens += compressed_size
            
        elif isinstance(msg, HumanMessage):
            # Check user messages for PII
            content_str = str(msg.content)
            redacted, detected = _redact_sensitive_data(content_str)
            
            if detected:
                stats['pii_detected'] = True
                logger.warning(f"PII detected in user message: {detected}")
                compressed.append(HumanMessage(content=redacted))
            else:
                compressed.append(msg)
            
            total_tokens += _estimate_tokens(redacted if detected else content_str)
            
        elif isinstance(msg, AIMessage):
            # Summarize AI messages (keep tool_calls, compress content)
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                compressed.append(msg)  # Keep tool calls intact
            else:
                # Compress non-tool AI responses
                content_str = str(msg.content)[:500] + "..." if len(str(msg.content)) > 500 else str(msg.content)
                compressed.append(AIMessage(content=content_str))
            
            total_tokens += _estimate_tokens(str(msg.content))
            
        else:
            compressed.append(msg)
            total_tokens += _estimate_tokens(str(msg.content))
        
        # Stop if over total budget
        if total_tokens > total_budget:
            stats['truncated'] = True
            logger.warning(f"Message history truncated at {total_tokens} tokens")
            break
    
    stats['total_tokens'] = total_tokens
    return compressed, stats


def build_interpretation_summary(
    user_query: str,
    tool_results: List[Dict[str, Any]],
    context_insights: List[str],
    max_tokens: int = 2000
) -> str:
    """
    Build concise interpretation prompt using summary-only approach.
    
    Does NOT include full tool results, only key metrics and insights.
    """
    parts = []
    
    # User query (always include)
    parts.append(f"USER QUERY: {user_query[:200]}")
    parts.append("")
    
    # Context insights (bullet points)
    if context_insights:
        parts.append("KEY INSIGHTS:")
        for insight in context_insights[:5]:  # Max 5 insights
            parts.append(f"• {insight[:100]}")
        parts.append("")
    
    # Tool results summary (NO raw data)
    parts.append("TOOL EXECUTION SUMMARY:")
    for i, result in enumerate(tool_results[:3]):  # Max 3 tool results
        if isinstance(result, dict):
            summary = []
            if result.get('success'):
                summary.append("✓ Success")
            if result.get('file_name'):
                summary.append(f"File: {result['file_name']}")
            if result.get('domain_name'):
                summary.append(f"Domain: {result['domain_name']}")
            if result.get('row_count'):
                summary.append(f"{result['row_count']} rows")
            
            parts.append(f"{i+1}. {' | '.join(summary)}")
    parts.append("")
    
    # Instructions (concise)
    parts.append("TASK: Provide a clear, actionable interpretation of these results.")
    parts.append("Format: 2-3 sentences highlighting key findings and recommendations.")
    
    summary = "\n".join(parts)
    
    # Enforce token budget
    if _estimate_tokens(summary) > max_tokens:
        summary = summary[:max_tokens * 4] + "\n\n[Truncated to fit budget]"
    
    return summary
