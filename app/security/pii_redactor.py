import re
import logging
from typing import Pattern, List, Tuple


class PIIRedactionFilter(logging.Filter):
    """
    Logging filter that automatically redacts PII from log messages.
    
    This filter processes log records before they're written and replaces
    sensitive information with redacted placeholders like [EMAIL_REDACTED].
    """
    
    # Compilation of regex patterns for different PII types
    PII_PATTERNS: List[Tuple[Pattern, str]] = [
        # JWT tokens (3 base64 segments separated by dots) - Must be before email
        # Relaxed to allow shorter segments (minimum 3 chars per segment)
        (
            re.compile(
                r'\beyJ[A-Za-z0-9_-]{3,}\.[A-Za-z0-9_-]{3,}\.[A-Za-z0-9_-]{3,}\b'
            ),
            '[JWT_REDACTED]'
        ),
        
        # Usernames in URLs (http://user:pass@host format) - Must be before email
        (
            re.compile(
                r'(https?://)[^:@\s]+:[^@\s]+@',
                re.IGNORECASE
            ),
            r'\1[CREDENTIALS_REDACTED]@'
        ),
        
        # Email addresses - comprehensive pattern with Unicode support
        (
            re.compile(
                r'\b[\w.%+-]+@[\w.-]+\.[A-Z|a-z]{2,}\b',
                re.IGNORECASE | re.UNICODE
            ),
            '[EMAIL_REDACTED]'
        ),
        
        # Credit card numbers (with or without separators) - Must be before phone
        (
            re.compile(
                r'\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b'
            ),
            '[CARD_REDACTED]'
        ),
        
        # Phone numbers - international formats
        # Matches: +1-234-567-8900, (123) 456-7890, 123-456-7890, +44 20 7123 4567
        (
            re.compile(
                r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
            ),
            '[PHONE_REDACTED]'
        ),
        
        # API keys (common patterns: 32-64 alphanumeric chars)
        (
            re.compile(
                r'\b(?:api[_-]?key|apikey|access[_-]?token)["\s:=]+([A-Za-z0-9_-]{32,64})\b',
                re.IGNORECASE
            ),
            r'\g<0>=[API_KEY_REDACTED]'
        ),
        
        # AWS Access Keys (AKIA... format, 20 chars)
        (
            re.compile(
                r'\b(AKIA[0-9A-Z]{16})\b'
            ),
            '[AWS_KEY_REDACTED]'
        ),
        
        # Generic token pattern (32+ hex chars) - Must be before AWS Secret to avoid conflict
        (
            re.compile(
                r'\b[a-fA-F0-9]{32,64}\b'
            ),
            '[TOKEN_REDACTED]'
        ),
        
        # AWS Secret Keys (40 alphanumeric chars) - More specific pattern
        (
            re.compile(
                r'\b([A-Za-z0-9/+=]{40})\b'
            ),
            '[AWS_SECRET_REDACTED]'
        ),
        
        # IP addresses (IPv4)
        (
            re.compile(
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            ),
            '[IP_REDACTED]'
        ),
        
        # Social Security Numbers (XXX-XX-XXXX format)
        (
            re.compile(
                r'\b\d{3}-\d{2}-\d{4}\b'
            ),
            '[SSN_REDACTED]'
        ),
        
        # Bearer tokens in Authorization headers
        (
            re.compile(
                r'\bBearer\s+[A-Za-z0-9_-]{20,}\b',
                re.IGNORECASE
            ),
            'Bearer [TOKEN_REDACTED]'
        ),
        
        # Password-like patterns in key-value pairs
        (
            re.compile(
                r'\b(password|passwd|pwd)["\s:=]+[^\s,}]{6,}',
                re.IGNORECASE
            ),
            r'\g<1>=[PASSWORD_REDACTED]'
        ),
    ]
    
    def __init__(self, name: str = ""):
        """
        Initialize the PII redaction filter.
        
        Args:
            name: Optional name for the filter (for logging.Filter compatibility)
        """
        super().__init__(name)
        self._redaction_count = 0
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter a log record by redacting PII from the message.
        
        This method is called automatically by the logging framework for each
        log record. It modifies the record's message in-place.
        
        Args:
            record: The log record to filter
            
        Returns:
            True to allow the record to be logged (always True for this filter)
        """
        # Redact PII from the main message
        original_msg = record.getMessage()
        redacted_msg = self._redact_pii(original_msg)
        
        # Update the record's message and args to prevent re-formatting issues
        if redacted_msg != original_msg:
            self._redaction_count += 1
            record.msg = redacted_msg
            record.args = ()
        
        # Always return True to allow the record to be logged
        return True
    
    def _redact_pii(self, text: str) -> str:
        """
        Apply all PII redaction patterns to the given text.
        
        Args:
            text: The text to redact
            
        Returns:
            Text with PII replaced by redaction placeholders
        """
        redacted = text
        
        for pattern, replacement in self.PII_PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        
        return redacted
    
    def get_redaction_count(self) -> int:
        """
        Get the total number of log records that had PII redacted.
        
        Returns:
            Count of redacted log records
        """
        return self._redaction_count
    
    def reset_count(self) -> None:
        """Reset the redaction counter to zero."""
        self._redaction_count = 0


def redact_pii(text: str) -> str:
    """
    Standalone function to redact PII from any text.
    
    This can be used independently of the logging filter for manual
    redaction of strings before logging or storing.
    
    Args:
        text: The text to redact
        
    Returns:
        Text with PII replaced by redaction placeholders
        
    Example:
        >>> redact_pii("Contact user@example.com or call 123-456-7890")
        'Contact [EMAIL_REDACTED] or call [PHONE_REDACTED]'
    """
    redacted = text
    
    for pattern, replacement in PIIRedactionFilter.PII_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    
    return redacted


# Convenience function for testing/debugging
def test_redaction_patterns():
    """
    Test all redaction patterns with sample PII data.
    
    This function is useful for validating that patterns work correctly
    and for debugging redaction behavior.
    """
    test_cases = [
        "Email: john.doe@example.com",
        "Phone: +1-234-567-8900 or (555) 123-4567",
        "JWT: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        "API Key: api_key=abcd1234efgh5678ijkl9012mnop3456qrst7890",
        "AWS: AKIAIOSFODNN7EXAMPLE",
        "IP: 192.168.1.100",
        "Card: 4532-1234-5678-9010",
        "SSN: 123-45-6789",
        "Bearer: Bearer abc123def456ghi789jkl012mno345pqr678",
        "Password: password=MySecretPass123!",
        "URL: https://user:pass@example.com/api",
        "Token: f47ac10b58cc4372a5670e02b2c3d479",
    ]
    
    print("PII Redaction Pattern Testing")
    print("=" * 60)
    
    for test in test_cases:
        redacted = redact_pii(test)
        print(f"Original:  {test}")
        print(f"Redacted:  {redacted}")
        print("-" * 60)


if __name__ == "__main__":
    # Run tests when module is executed directly
    test_redaction_patterns()
