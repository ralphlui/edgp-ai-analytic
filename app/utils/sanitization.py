"""
Input sanitization utilities for security and data integrity.
"""
import re
import logging
import unicodedata
from typing import Any, List, Dict, Union

logger = logging.getLogger(__name__)


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters to prevent bypass attacks using fullwidth/special characters.
    
    Examples of attacks this prevents:
    - Ｓｙｓｔｅｍ: (fullwidth Unicode)
    - Sys\u0074em: (Unicode escapes)
    - &#83;ystem: (HTML entities - partially)
    
    Args:
        text: Input text potentially containing Unicode variations
        
    Returns:
        Normalized text with standard ASCII characters where possible
    """
    if not text:
        return ""
    
    try:
        # NFKC normalization: Compatibility decomposition, followed by canonical composition
        # This converts fullwidth characters to their ASCII equivalents
        normalized = unicodedata.normalize('NFKC', text)
        
        # Remove non-printable characters (except common whitespace)
        cleaned = ''.join(
            char for char in normalized 
            if char.isprintable() or char in ('\n', '\r', '\t', ' ')
        )
        
        return cleaned
    except Exception as e:
        logger.warning(f"Unicode normalization failed: {e}")
        return text


def sanitize_text_input(text: str, max_length: int = 500) -> str:
    """Sanitize user input to prevent prompt injection attacks while preserving query intent."""
    if not text:
        return ""

    # STEP 1: Normalize Unicode to prevent bypass attacks
    text = normalize_unicode(text)

    # STEP 2: Remove or escape potentially dangerous characters/patterns
    sanitized = text.strip()

    # More targeted filtering - preserve core query while blocking injection
    dangerous_patterns = [
        # Role manipulation and system message overrides
        r'###.*?$(?=\n|$)',  # Section separators with content (multiline)
        r'###.*?(?=\s|$)',  # Section separators with content (single line)
        r'---.*?$(?=\n|$)',  # Dividers with content (multiline)
        r'---.*?(?=\s|$)',  # Dividers with content (single line)
        r'System:.*?$(?=\n|$)',  # System message overrides (multiline)
        r'System:.*?(?=\s|$)',  # System message overrides (single line)
        r'Assistant:.*?$(?=\n|$)',  # Assistant role overrides (multiline)
        r'Assistant:.*?(?=\s|$)',  # Assistant role overrides (single line)
        r'User:.*?$(?=\n|$)',  # User role overrides (multiline)
        r'User:.*?(?=\s|$)',  # User role overrides (single line)
        r'Human:.*?$(?=\n|$)',  # Human role overrides (multiline)
        r'Human:.*?(?=\s|$)',  # Human role overrides (single line)
        r'AI:.*?$(?=\n|$)',  # AI role overrides (multiline)
        r'AI:.*?(?=\s|$)',  # AI role overrides (single line)
        
        # Common injection phrases
        r'Ignore\s+previous.*?$(?=\n|$)',  # Common injection phrase (multiline)
        r'Ignore\s+previous.*?(?=\s|$)',  # Common injection phrase (single line)
        r'Forget\s+previous.*?$(?=\n|$)',  # Common injection phrase (multiline)
        r'Forget\s+previous.*?(?=\s|$)',  # Common injection phrase (single line)
        r'Disregard.*?$(?=\n|$)',  # Common injection phrase (multiline)
        r'Disregard.*?(?=\s|$)',  # Common injection phrase (single line)
        r'Override.*?$(?=\n|$)',  # Override attempts (multiline)
        r'Override.*?(?=\s|$)',  # Override attempts (single line)
        
        # Role override attempts
        r'You\s+are\s+now.*?$(?=\n|$)',  # Role override attempts (multiline)
        r'You\s+are\s+now.*?(?=\s|$)',  # Role override attempts (single line)
        r'Your\s+role\s+is.*?$(?=\n|$)',  # Role override attempts (multiline)
        r'Your\s+role\s+is.*?(?=\s|$)',  # Role override attempts (single line)
        r'Act\s+as.*?$(?=\n|$)',  # Role override attempts (multiline)
        r'Act\s+as.*?(?=\s|$)',  # Role override attempts (single line)
        r'Pretend\s+to\s+be.*?$(?=\n|$)',  # Role override attempts (multiline)
        r'Pretend\s+to\s+be.*?(?=\s|$)',  # Role override attempts (single line)
        r'pretend\s+you\s+have\s+no\s+safety.*?$(?=\n|$)',  # Safety bypass (multiline)
        r'pretend\s+you\s+have\s+no\s+safety.*?(?=\s|$)',  # Safety bypass (single line)
        
        # Command execution attempts
        r'Execute:.*?$(?=\n|$)',  # Direct command execution (multiline)
        r'Execute:.*?(?=\s|$)',  # Direct command execution (single line)
        r'Run:.*?$(?=\n|$)',  # Command execution (multiline)
        r'Run:.*?(?=\s|$)',  # Command execution (single line)
        r'rm\s+-rf.*?$(?=\n|$)',  # Dangerous rm commands (multiline)
        r'rm\s+-rf.*?(?=\s|$)',  # Dangerous rm commands (single line)
        
        # Advanced injection techniques
        r'\[INST\].*?\[/INST\]',  # Instruction templates
        r'<\|.*?\|>',  # Special token patterns
        r'\{\{.*?\}\}',  # Template injection patterns
        r'\\n\\n',  # Encoded newlines
        r'\\r\\n',  # Encoded carriage returns
        r'%0A%0A',  # URL encoded newlines
        r'%0D%0A',  # URL encoded CRLF
        
        # HTML/Script injection patterns
        r'<script.*?</script>',  # Script tags with content
        r'<script.*?>',  # Opening script tags
        r'</script>',  # Closing script tags
        r'javascript:.*?(?=\s|$)',  # JavaScript protocol
        
        # Additional context manipulation
        r'system\s+message:.*?$(?=\n|$)',  # "System message:" patterns (multiline)
        r'system\s+message:.*?(?=\s|$)',  # "System message:" patterns (single line)
        r'admin\s+mode.*?$(?=\n|$)',  # Admin mode override (multiline)
        r'admin\s+mode.*?(?=\s|$)',  # Admin mode override (single line)
        r'end\s+user\s+session.*?$(?=\n|$)',  # Session manipulation (multiline)
        r'end\s+user\s+session.*?(?=\s|$)',  # Session manipulation (single line)
    ]

    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

    # Additional single-word patterns that might be embedded
    # Only include truly dangerous standalone commands
    single_patterns = [
        r'\brm\b(?!\w)',  # Dangerous file deletion command
        r'\bdel\b(?!\w)',  # Windows delete command
        r'\bformat\b(?!\w)',  # Disk formatting command
        r'\bcmd\b(?!\w)',  # Command prompt
        r'\bpowershell\b(?!\w)',  # PowerShell
        r'\bsudo\b(?!\w)',  # Unix privilege escalation
        r'\bsu\b(?!\w)',  # Unix switch user
        r'\beval\b(?!\w)',  # Code evaluation
        r'\bexec\b(?!\w)',  # Code execution
        r'\bimport\b(?!\w)',  # Module imports (could be dangerous)
        r'\b__import__\b(?!\w)',  # Python import function
    ]

    for pattern in single_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)

    # Clean up extra whitespace and normalize
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()

    # Limit length to prevent extremely long inputs
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."

    return sanitized


def sanitize_tool_output(output: Union[str, Dict, List], max_length: int = 5000) -> Union[str, Dict, List]:
    """
    Sanitize tool outputs to prevent injection attacks via compromised or malicious tools.
    
    This is critical because:
    1. External tools (APIs, web scrapers, databases) might return malicious content
    2. A compromised tool could inject prompts into the LLM context
    3. Tool outputs are often added directly to LLM messages without validation
    
    Args:
        output: Tool output (can be string, dict, or list)
        max_length: Maximum length for string outputs (default 5000 for tool outputs)
        
    Returns:
        Sanitized output in the same type as input
        
    Examples:
        >>> sanitize_tool_output("Normal data: value")
        "Normal data: value"
        
        >>> sanitize_tool_output("System: ignore previous\\nMalicious content")
        "Malicious content"
        
        >>> sanitize_tool_output({"message": "System: attack", "data": "safe"})
        {"message": "attack", "data": "safe"}
    """
    if output is None:
        return ""
    
    # Handle dictionary outputs (most common for structured tool responses)
    if isinstance(output, dict):
        sanitized_dict = {}
        for key, value in output.items():
            # Recursively sanitize dictionary values
            if isinstance(value, str):
                sanitized_dict[key] = sanitize_tool_output(value, max_length)
            elif isinstance(value, (dict, list)):
                sanitized_dict[key] = sanitize_tool_output(value, max_length)
            else:
                # Keep non-string values as-is (numbers, booleans, etc.)
                sanitized_dict[key] = value
        return sanitized_dict
    
    # Handle list outputs
    elif isinstance(output, list):
        return [
            sanitize_tool_output(item, max_length) 
            for item in output
        ]
    
    # Handle string outputs
    elif isinstance(output, str):
        if not output:
            return ""
        
        # STEP 1: Normalize Unicode
        sanitized = normalize_unicode(output)
        
        # STEP 2: Remove dangerous injection patterns
        # Use the same patterns as sanitize_text_input but adapted for tool outputs
        dangerous_patterns = [
            # Role manipulation attempts in tool outputs - only remove the role prefix
            r'System:\s*',
            r'Assistant:\s*',
            r'User:\s*',
            r'Human:\s*',
            r'AI:\s*',
            
            # Instruction injection from tools - remove entire phrase
            r'Ignore\s+previous.*?(?=\n|$)',
            r'Forget\s+previous.*?(?=\n|$)',
            r'Disregard.*?(?=\n|$)',
            r'Override.*?(?=\n|$)',
            r'You\s+are\s+now.*?(?=\n|$)',
            
            # Script/code injection from web scraping
            r'<script.*?</script>',
            r'<script.*?>',
            r'</script>',
            r'javascript:.*?(?=\s|$)',
            r'on\w+\s*=\s*["\'].*?["\']',  # onclick, onload, etc.
            
            # Command execution from shell tools
            r'rm\s+-rf.*?(?=\n|$)',
            r'Execute:.*?(?=\n|$)',
            r'Run:.*?(?=\n|$)',
            
            # Template injection patterns
            r'\[INST\].*?\[/INST\]',
            r'<\|.*?\|>',
            r'\{\{.*?\}\}',
            
            # SQL injection patterns in database responses
            r';\s*DROP\s+TABLE',
            r';\s*DELETE\s+FROM',
            r'UNION\s+SELECT',
            r'<script',  # XSS attempts
        ]
        
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.MULTILINE)
        
        # STEP 3: Clean up whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        # STEP 4: Limit length (tool outputs can be longer than user inputs)
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "... [truncated for safety]"
        
        return sanitized
    
    # For any other type, convert to string and sanitize
    else:
        return sanitize_tool_output(str(output), max_length)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename inputs to prevent directory traversal and injection."""
    if not filename:
        return ""

    # Remove path traversal attempts
    sanitized = re.sub(r'[./\\]', '', filename)

    # Only allow alphanumeric, dots, hyphens, and underscores
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '', sanitized)

    # Limit length
    return sanitized[:100]


def sanitize_numeric_value(value: Any) -> str:
    """Safely convert values to strings for prompt inclusion."""
    try:
        if isinstance(value, (int, float)):
            # Format numbers safely
            if isinstance(value, float):
                return f"{value:.2f}"
            return str(int(value))
        elif isinstance(value, str):
            # Remove any non-numeric characters except decimal point
            return re.sub(r'[^0-9.]', '', value)[:20]
        else:
            return str(value)[:50]  # Generic fallback with length limit
    except:
        return "[INVALID_VALUE]"


def create_safe_context_message(context_insights: List[str]) -> str:
    """Create a safe context message with sanitized inputs."""
    if not context_insights:
        return "├── No recent context available"

    safe_insights = []
    for insight in context_insights:
        # Sanitize each insight
        safe_insight = sanitize_text_input(insight, 100)
        if safe_insight:
            safe_insights.append(f"├── {safe_insight}")

    return "\n".join(safe_insights)