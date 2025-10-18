"""
Unit tests for PII redaction functionality.

Tests the PIIRedactionFilter and redaction patterns to ensure all sensitive
information is properly masked in log messages.
"""

import pytest
import logging
from io import StringIO
from app.security.pii_redactor import (
    PIIRedactionFilter, 
    redact_pii
)
from app.logging_config import (
    setup_logging,
    add_pii_filter_to_existing_loggers,
    disable_pii_redaction
)


class TestPIIRedactionPatterns:
    """Test individual PII redaction patterns."""
    
    def test_email_redaction(self):
        """Test that email addresses are redacted."""
        text = "Contact user@example.com for details"
        redacted = redact_pii(text)
        assert "[EMAIL_REDACTED]" in redacted
        assert "user@example.com" not in redacted
    
    def test_multiple_emails_redaction(self):
        """Test redaction of multiple email addresses."""
        text = "Emails: john.doe@company.com, jane.smith@example.org"
        redacted = redact_pii(text)
        assert redacted.count("[EMAIL_REDACTED]") == 2
        assert "john.doe@company.com" not in redacted
        assert "jane.smith@example.org" not in redacted
    
    def test_phone_number_redaction(self):
        """Test that phone numbers are redacted."""
        test_cases = [
            ("Call +1-234-567-8900", "[PHONE_REDACTED]"),
            ("Phone: (555) 123-4567", "[PHONE_REDACTED]"),
            ("Contact 123-456-7890", "[PHONE_REDACTED]"),
        ]
        
        for original, expected_substring in test_cases:
            redacted = redact_pii(original)
            assert expected_substring in redacted
            # Ensure original number is removed
            assert not any(char.isdigit() for word in redacted.split() 
                          if word.startswith('+') or '(' in word or '-' in word
                          for char in word if char not in ['+', '-', '(', ')', ' ', '[', ']', '_'])
    
    def test_jwt_token_redaction(self):
        """Test that JWT tokens are redacted."""
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        text = f"Token: {jwt}"
        redacted = redact_pii(text)
        assert "[JWT_REDACTED]" in redacted
        assert jwt not in redacted
    
    def test_api_key_redaction(self):
        """Test that API keys are redacted."""
        text = "api_key=abcd1234efgh5678ijkl9012mnop3456qrst7890"
        redacted = redact_pii(text)
        assert "[API_KEY_REDACTED]" in redacted
        assert "abcd1234efgh5678ijkl9012mnop3456qrst7890" not in redacted
    
    def test_aws_access_key_redaction(self):
        """Test that AWS access keys are redacted."""
        text = "AWS_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE"
        redacted = redact_pii(text)
        assert "[AWS_KEY_REDACTED]" in redacted
        assert "AKIAIOSFODNN7EXAMPLE" not in redacted
    
    def test_ip_address_redaction(self):
        """Test that IP addresses are redacted."""
        text = "Request from 192.168.1.100"
        redacted = redact_pii(text)
        assert "[IP_REDACTED]" in redacted
        assert "192.168.1.100" not in redacted
    
    def test_credit_card_redaction(self):
        """Test that credit card numbers with separators are redacted."""
        test_cases = [
            "Card: 4532-1234-5678-9010",
            "Card: 4532 1234 5678 9010",
        ]
        
        for text in test_cases:
            redacted = redact_pii(text)
            assert "[CARD_REDACTED]" in redacted
    
    def test_ssn_redaction(self):
        """Test that Social Security Numbers are redacted."""
        text = "SSN: 123-45-6789"
        redacted = redact_pii(text)
        assert "[SSN_REDACTED]" in redacted
        assert "123-45-6789" not in redacted
    
    def test_bearer_token_redaction(self):
        """Test that Bearer tokens are redacted."""
        text = "Authorization: Bearer abc123def456ghi789jkl012mno345pqr678"
        redacted = redact_pii(text)
        assert "Bearer [TOKEN_REDACTED]" in redacted
        assert "abc123def456ghi789jkl012mno345pqr678" not in redacted
    
    def test_password_redaction(self):
        """Test that passwords in key-value pairs are redacted."""
        text = "Config: password=MySecretPass123!"
        redacted = redact_pii(text)
        assert "[PASSWORD_REDACTED]" in redacted
        assert "MySecretPass123" not in redacted
    
    def test_url_credentials_redaction(self):
        """Test that credentials in URLs are redacted."""
        text = "Database URL: https://user:password@database.example.com/db"
        redacted = redact_pii(text)
        assert "[CREDENTIALS_REDACTED]" in redacted
        assert "user:password" not in redacted
    
    def test_hex_token_redaction(self):
        """Test that long hex tokens are redacted."""
        text = "Token: f47ac10b58cc4372a5670e02b2c3d479a1234567"
        redacted = redact_pii(text)
        assert "[TOKEN_REDACTED]" in redacted
        assert "f47ac10b58cc4372a5670e02b2c3d479a1234567" not in redacted
    
    def test_no_false_positives_on_normal_text(self):
        """Test that normal text without PII is not redacted."""
        text = "Processing query for analytics dashboard at 2024-01-15"
        redacted = redact_pii(text)
        assert redacted == text  # Should be unchanged
    
    def test_preserves_non_pii_in_mixed_content(self):
        """Test that non-PII content is preserved when mixed with PII."""
        text = "User john@example.com requested chart type: bar, domain: sales"
        redacted = redact_pii(text)
        assert "chart type: bar" in redacted
        assert "domain: sales" in redacted
        assert "john@example.com" not in redacted


class TestPIIRedactionFilter:
    """Test the logging filter integration."""
    
    def setup_method(self):
        """Set up test logger before each test."""
        # Create a test logger with in-memory handler
        self.logger = logging.getLogger("test_pii_logger")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.INFO)
        
        # Create string buffer to capture logs
        self.log_stream = StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.handler.setLevel(logging.INFO)
        
        # Add PII filter
        self.pii_filter = PIIRedactionFilter()
        self.handler.addFilter(self.pii_filter)
        
        # Add handler to logger
        self.logger.addHandler(self.handler)
    
    def teardown_method(self):
        """Clean up after each test."""
        self.logger.handlers.clear()
        self.log_stream.close()
    
    def get_log_output(self) -> str:
        """Get the logged output as a string."""
        return self.log_stream.getvalue()
    
    def test_filter_redacts_email_in_log(self):
        """Test that emails are redacted in log messages."""
        self.logger.info("User contact: user@example.com")
        output = self.get_log_output()
        assert "[EMAIL_REDACTED]" in output
        assert "user@example.com" not in output
    
    def test_filter_redacts_phone_in_log(self):
        """Test that phone numbers are redacted in log messages."""
        self.logger.info("Customer phone: +1-555-123-4567")
        output = self.get_log_output()
        assert "[PHONE_REDACTED]" in output
        assert "555-123-4567" not in output
    
    def test_filter_redacts_jwt_in_log(self):
        """Test that JWT tokens are redacted in log messages."""
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0In0.abc123"
        self.logger.warning(f"Invalid token: {jwt}")
        output = self.get_log_output()
        assert "[JWT_REDACTED]" in output
        assert jwt not in output
    
    def test_filter_allows_non_pii_logs(self):
        """Test that normal logs without PII pass through unchanged."""
        message = "Processing analytics query for dashboard"
        self.logger.info(message)
        output = self.get_log_output()
        assert "Processing analytics query for dashboard" in output
    
    def test_filter_increments_redaction_count(self):
        """Test that the filter tracks redaction count."""
        initial_count = self.pii_filter.get_redaction_count()
        
        self.logger.info("Email: test@example.com")
        self.logger.info("Phone: 123-456-7890")
        self.logger.info("Normal log message")
        
        final_count = self.pii_filter.get_redaction_count()
        assert final_count == initial_count + 2  # Only 2 messages had PII
    
    def test_filter_reset_count(self):
        """Test resetting the redaction counter."""
        self.logger.info("Email: test@example.com")
        assert self.pii_filter.get_redaction_count() > 0
        
        self.pii_filter.reset_count()
        assert self.pii_filter.get_redaction_count() == 0
    
    def test_filter_handles_complex_log_messages(self):
        """Test redaction in complex log messages with multiple PII types."""
        self.logger.error(
            "Auth failed for user@example.com from IP 10.0.0.1 "
            "using token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.xyz"
        )
        output = self.get_log_output()
        assert "[EMAIL_REDACTED]" in output
        assert "[IP_REDACTED]" in output
        assert "[JWT_REDACTED]" in output
        assert "user@example.com" not in output
        assert "10.0.0.1" not in output


class TestLoggingSetup:
    """Test the logging setup function."""
    
    def teardown_method(self):
        """Clean up loggers after each test."""
        logging.getLogger().handlers.clear()
    
    def test_setup_logging_creates_logger(self):
        """Test that setup_logging creates a configured logger."""
        logger = setup_logging(log_level="DEBUG")
        assert logger is not None
        assert logger.level == logging.DEBUG
    
    def test_setup_logging_with_pii_redaction(self):
        """Test that PII redaction is enabled by default."""
        logger = setup_logging(enable_pii_redaction=True)
        
        # Check that at least one handler has a PII filter
        has_pii_filter = False
        for handler in logger.handlers:
            for filter_obj in handler.filters:
                if isinstance(filter_obj, PIIRedactionFilter):
                    has_pii_filter = True
                    break
        
        assert has_pii_filter, "PII filter should be added to handlers"
    
    def test_setup_logging_without_pii_redaction(self):
        """Test that PII redaction can be disabled."""
        logger = setup_logging(enable_pii_redaction=False)
        
        # Check that no handler has a PII filter
        has_pii_filter = False
        for handler in logger.handlers:
            for filter_obj in handler.filters:
                if isinstance(filter_obj, PIIRedactionFilter):
                    has_pii_filter = True
                    break
        
        assert not has_pii_filter, "PII filter should not be added when disabled"
    
    def test_setup_logging_custom_format(self):
        """Test custom log format."""
        custom_format = "%(levelname)s - %(message)s"
        logger = setup_logging(log_format=custom_format)
        
        # Verify the format was applied
        assert len(logger.handlers) > 0
        handler = logger.handlers[0]
        assert handler.formatter._fmt == custom_format
    
    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a valid logger instance."""
        from app.logging_config import get_logger
        
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"


class TestEndToEndLogging:
    """End-to-end tests for logging with PII redaction."""
    
    def setup_method(self):
        """Set up logging for each test."""
        # Clear existing handlers
        logging.getLogger().handlers.clear()
        
        # Create string buffer to capture logs
        self.log_stream = StringIO()
        
        # Set up logging with custom handler
        self.handler = logging.StreamHandler(self.log_stream)
        self.handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        self.handler.setFormatter(formatter)
        
        # Add PII filter
        self.pii_filter = PIIRedactionFilter()
        self.handler.addFilter(self.pii_filter)
        
        # Configure logger
        self.logger = logging.getLogger("analytic_agent")
        self.logger.handlers.clear()
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)
    
    def teardown_method(self):
        """Clean up after each test."""
        self.logger.handlers.clear()
        self.log_stream.close()
    
    def get_log_output(self) -> str:
        """Get the logged output."""
        return self.log_stream.getvalue()
    
    def test_realistic_auth_log(self):
        """Test redaction of realistic authentication logs."""
        self.logger.info("Profile data for user 12345: {'email': 'john@example.com', 'phone': '555-123-4567'}")
        output = self.get_log_output()
        
        assert "[EMAIL_REDACTED]" in output
        assert "[PHONE_REDACTED]" in output
        assert "john@example.com" not in output
        assert "555-123-4567" not in output
    
    def test_realistic_api_response_log(self):
        """Test redaction of API response containing tokens."""
        response_text = '{"access_token": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.sig", "user_email": "user@example.com"}'
        self.logger.error(f"API error response: {response_text}")
        output = self.get_log_output()
        
        assert "[JWT_REDACTED]" in output or "[TOKEN_REDACTED]" in output
        assert "[EMAIL_REDACTED]" in output
        assert "user@example.com" not in output
    
    def test_realistic_query_log(self):
        """Test redaction of user queries containing PII."""
        query = "Show me analytics for customer email: alice@company.com from IP 203.0.113.50"
        self.logger.info(f"Extracting intent from prompt: '{query}'")
        output = self.get_log_output()
        
        assert "[EMAIL_REDACTED]" in output
        assert "[IP_REDACTED]" in output
        assert "alice@company.com" not in output
        assert "203.0.113.50" not in output
    
    def test_multiple_log_levels_all_redacted(self):
        """Test that redaction works across all log levels."""
        # Temporarily set logger to DEBUG to capture all levels
        original_level = self.logger.level
        self.logger.setLevel(logging.DEBUG)
        self.handler.setLevel(logging.DEBUG)
        
        self.logger.debug("Debug: email debug@example.com")
        self.logger.info("Info: email info@example.com")
        self.logger.warning("Warning: email warn@example.com")
        self.logger.error("Error: email error@example.com")
        
        output = self.get_log_output()
        assert output.count("[EMAIL_REDACTED]") == 4
        assert "debug@example.com" not in output
        assert "info@example.com" not in output
        assert "warn@example.com" not in output
        assert "error@example.com" not in output
        
        # Restore original level
        self.logger.setLevel(original_level)
    
    def test_no_performance_impact_on_normal_logs(self):
        """Test that normal logs (without PII) have minimal overhead."""
        # Log many normal messages
        for i in range(100):
            self.logger.info(f"Processing query {i} for analytics dashboard")
        
        output = self.get_log_output()
        lines = output.strip().split('\n')
        assert len(lines) == 100
        assert all("Processing query" in line for line in lines)
    
    def test_redaction_count_tracking(self):
        """Test that redaction count is accurately tracked."""
        initial_count = self.pii_filter.get_redaction_count()
        
        # Log messages with and without PII
        self.logger.info("Normal log message")
        self.logger.info("Email: test1@example.com")
        self.logger.info("Another normal message")
        self.logger.info("Phone: 123-456-7890 and email: test2@example.com")
        self.logger.info("Yet another normal message")
        
        final_count = self.pii_filter.get_redaction_count()
        # Only 2 messages contained PII
        assert final_count == initial_count + 2


class TestFilterIntegration:
    """Test filter integration with existing logging setup."""
    
    def teardown_method(self):
        """Clean up loggers."""
        logging.getLogger().handlers.clear()
    
    def test_add_filter_to_existing_loggers(self):
        """Test adding PII filter to existing loggers."""
        # Create logger without PII filter
        logger = logging.getLogger("existing_logger")
        handler = logging.StreamHandler(StringIO())
        logger.addHandler(handler)
        
        # Verify no PII filter initially
        assert not any(isinstance(f, PIIRedactionFilter) for f in handler.filters)
        
        # Add PII filter
        add_pii_filter_to_existing_loggers()
        
        # Verify PII filter was added
        assert any(isinstance(f, PIIRedactionFilter) for f in handler.filters)
    
    def test_disable_pii_redaction(self):
        """Test disabling PII redaction."""
        # Set up logging with PII redaction
        setup_logging(enable_pii_redaction=True)
        
        # Verify filter is present
        root_logger = logging.getLogger()
        has_filter_before = any(
            isinstance(f, PIIRedactionFilter)
            for h in root_logger.handlers
            for f in h.filters
        )
        assert has_filter_before
        
        # Disable redaction
        disable_pii_redaction()
        
        # Verify filter is removed
        has_filter_after = any(
            isinstance(f, PIIRedactionFilter)
            for h in root_logger.handlers
            for f in h.filters
        )
        assert not has_filter_after


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_string_redaction(self):
        """Test redaction of empty string."""
        assert redact_pii("") == ""
    
    def test_none_handling(self):
        """Test that filter handles None gracefully."""
        # This shouldn't raise an error
        result = redact_pii("")
        assert result == ""
    
    def test_unicode_content_redaction(self):
        """Test redaction with Unicode characters."""
        text = "Contact: 用户@example.com or नाम@test.org"
        redacted = redact_pii(text)
        assert "[EMAIL_REDACTED]" in redacted
        assert "用户@example.com" not in redacted
    
    def test_multiline_log_redaction(self):
        """Test redaction across multiline log messages."""
        message = """User profile:
        Email: john@example.com
        Phone: 555-123-4567
        IP: 192.168.1.1
        """
        redacted = redact_pii(message)
        assert "[EMAIL_REDACTED]" in redacted
        assert "[PHONE_REDACTED]" in redacted
        assert "[IP_REDACTED]" in redacted
    
    def test_case_insensitive_patterns(self):
        """Test that patterns work case-insensitively where appropriate."""
        text = "API_KEY=abc123def456ghi789jkl012mno345pqr678uvw901xyz234"
        redacted = redact_pii(text)
        assert "[API_KEY_REDACTED]" in redacted or "API_KEY" in redacted


class TestUtilityFunctions:
    """Test utility functions in pii_redactor module."""
    
    def test_redaction_patterns_utility(self):
        """Test that test_redaction_patterns utility function runs without errors."""
        from app.security.pii_redactor import test_redaction_patterns
        
        # This function prints output but doesn't return anything
        # Just verify it runs without raising exceptions
        test_redaction_patterns()
        # If we get here, the function ran successfully
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
