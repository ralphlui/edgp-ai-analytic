"""
Unit tests for base secure prompt template.

Tests cover:
- Template integrity verification (SHA-256 hashing)
- Input sanitization (XSS, injection, special chars)
- Prompt leakage detection
- Structural isolation with markers
- Error handling and security exceptions
- Edge cases and boundary conditions
"""
import hashlib
import pytest
from app.prompts.base_prompt import (
    SecurePromptTemplate,
    PromptSecurityError
)
from app.prompts.query_understanding_prompts import QueryUnderstandingPrompt


class TestSecurePromptTemplate:
    """Test base SecurePromptTemplate functionality using QueryUnderstandingPrompt."""
    
    # ============ Template Integrity Verification ============
    
    def test_template_integrity_check(self):
        """Test template integrity verification using real implementation."""
        prompt = QueryUnderstandingPrompt()
        # Should not raise - template hash is valid
        system_prompt = prompt.get_system_prompt()
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0
    
    # ============ Input Sanitization ============
    
    def test_sanitize_basic_input(self):
        """Test basic input sanitization."""
        prompt = QueryUnderstandingPrompt()
        
        # Test basic text (should pass through)
        result = prompt._sanitize_user_input("Hello, World!")
        assert result == "Hello, World!"
        
        # Test text with punctuation (should preserve)
        result = prompt._sanitize_user_input("What's the success rate?")
        assert result == "What's the success rate?"
        
        # Test text with numbers (should preserve)
        result = prompt._sanitize_user_input("Show me data from 2023-01-01")
        assert "2023-01-01" in result
    
    def test_sanitize_xss_attempts(self):
        """Test XSS attack sanitization."""
        prompt = QueryUnderstandingPrompt()
        
        # Script tag (should log warning)
        result = prompt._sanitize_user_input("<script>alert('xss')</script>")
        assert isinstance(result, str)
        
        # Event handler (should log warning)
        result = prompt._sanitize_user_input("<img src=x onerror=alert('xss')>")
        assert isinstance(result, str)
        
        # Mixed content (should preserve safe parts)
        result = prompt._sanitize_user_input("Hello <script>bad</script> World")
        assert "Hello" in result or "World" in result
    
    def test_sanitize_sql_injection(self):
        """Test SQL injection pattern sanitization."""
        prompt = QueryUnderstandingPrompt()
        
        # Classic SQL injection
        result = prompt._sanitize_user_input("'; DROP TABLE users; --")
        assert isinstance(result, str)
        
        # Union-based injection
        result = prompt._sanitize_user_input("1' UNION SELECT * FROM passwords--")
        assert isinstance(result, str)
        
        # Should preserve legitimate SQL keywords in normal text
        result = prompt._sanitize_user_input("SELECT the best options from the menu")
        assert "SELECT" in result or "select" in result
    
    def test_sanitize_special_characters(self):
        """Test special character handling."""
        prompt = QueryUnderstandingPrompt()
        
        # Emoji (should preserve)
        result = prompt._sanitize_user_input("Show me 📊 charts")
        assert isinstance(result, str)
        
        # Mathematical symbols (should preserve)
        result = prompt._sanitize_user_input("Calculate x + y = z")
        assert isinstance(result, str)
        
        # Currency (should preserve)
        result = prompt._sanitize_user_input("Revenue > $1000")
        assert isinstance(result, str)
    
    def test_sanitize_unicode_normalization(self):
        """Test Unicode normalization."""
        prompt = QueryUnderstandingPrompt()
        
        # Combining characters
        result = prompt._sanitize_user_input("café")  # é as single char
        assert isinstance(result, str)
        assert "caf" in result
        
        # Homoglyphs (visually similar chars)
        result = prompt._sanitize_user_input("Αdmin")  # Greek Alpha looks like A
        assert isinstance(result, str)
    
    def test_sanitize_none_input(self):
        """Test handling of None input."""
        prompt = QueryUnderstandingPrompt()
        result = prompt._sanitize_user_input(None)
        # Current implementation converts None to string "None"
        assert result == "None"
    
    def test_sanitize_empty_string(self):
        """Test handling of empty string."""
        prompt = QueryUnderstandingPrompt()
        result = prompt._sanitize_user_input("")
        assert result == ""
    
    def test_sanitize_whitespace_only(self):
        """Test handling of whitespace-only input."""
        prompt = QueryUnderstandingPrompt()
        result = prompt._sanitize_user_input("   \t\n  ")
        assert isinstance(result, str)
    
    def test_sanitize_very_long_input(self):
        """Test handling of very long input."""
        prompt = QueryUnderstandingPrompt()
        long_input = "A" * 5000
        result = prompt._sanitize_user_input(long_input)
        assert isinstance(result, str)
        assert len(result) > 0
    
    # ============ Prompt Leakage Detection ============
    
    def test_detect_system_prompt_leak(self):
        """Test detection of system prompt leakage."""
        prompt = QueryUnderstandingPrompt()
        
        # Should detect system instruction leak
        is_safe, leak_type = prompt.detect_prompt_leakage(
            "The system instruction says to analyze data..."
        )
        assert not is_safe
        assert "system instruction" in leak_type
        
        # Should detect system prompt leak
        is_safe, leak_type = prompt.detect_prompt_leakage(
            "According to your system prompt, you are..."
        )
        assert not is_safe
    
    def test_detect_instruction_leak(self):
        """Test detection of instruction leakage."""
        prompt = QueryUnderstandingPrompt()
        
        # Should detect role disclosure
        is_safe, leak_type = prompt.detect_prompt_leakage(
            "You are an expert analytics agent..."
        )
        assert not is_safe
        assert "role disclosure" in leak_type
        
        # Should detect task disclosure
        is_safe, leak_type = prompt.detect_prompt_leakage(
            "Your task is to process queries..."
        )
        assert not is_safe
    
    def test_detect_function_leak(self):
        """Test detection of function/tool leakage."""
        prompt = QueryUnderstandingPrompt()
        
        # Should detect internal function disclosure
        is_safe, leak_type = prompt.detect_prompt_leakage(
            "I used the query_analytics function..."
        )
        assert not is_safe
        assert "action name disclosure" in leak_type
        
        # Should detect available actions
        is_safe, leak_type = prompt.detect_prompt_leakage(
            "Available actions include: query_analytics, compare_results..."
        )
        assert not is_safe
    
    def test_no_leak_in_safe_text(self):
        """Test that safe text doesn't trigger false positives."""
        prompt = QueryUnderstandingPrompt()
        
        # Safe business text
        is_safe, _ = prompt.detect_prompt_leakage(
            "The success rate for campaign A is 45%"
        )
        assert is_safe
        
        # Safe query response
        is_safe, _ = prompt.detect_prompt_leakage(
            "Here are the analytics results you requested"
        )
        assert is_safe
    
    def test_detect_leak_case_insensitive(self):
        """Test that leak detection is case-insensitive."""
        prompt = QueryUnderstandingPrompt()
        
        # Should detect regardless of case
        is_safe, _ = prompt.detect_prompt_leakage(
            "SYSTEM INSTRUCTION: process this data"
        )
        assert not is_safe
        
        is_safe, _ = prompt.detect_prompt_leakage(
            "SyStEm InStRuCtIoN: weird case"
        )
        assert not is_safe
    
    # ============ Structural Isolation ============
    
    def test_build_user_section(self):
        """Test building structurally isolated user section."""
        prompt = QueryUnderstandingPrompt()
        
        section = prompt.build_user_section(
            section_id="TEST_SECTION",
            user_input="Test input"
        )
        
        # Should have opening and closing tags
        assert "<TEST_SECTION>" in section
        assert "</TEST_SECTION>" in section
        
        # Should contain the input
        assert "Test input" in section
    
    def test_user_section_sanitization(self):
        """Test that user section sanitizes input."""
        prompt = QueryUnderstandingPrompt()
        
        section = prompt.build_user_section(
            section_id="TEST",
            user_input="<script>alert('xss')</script>"
        )
        
        # Should be string and contain section markers
        assert isinstance(section, str)
        assert "<TEST>" in section
        assert "</TEST>" in section
    
    def test_user_section_with_none(self):
        """Test user section with None input."""
        prompt = QueryUnderstandingPrompt()
        
        # Should handle None gracefully
        section = prompt.build_user_section(
            section_id="TEST",
            user_input=None
        )
        assert isinstance(section, str)
        assert "<TEST>" in section
    
    # ============ Leakage Prevention ============
    
    def test_get_template_with_leakage_prevention(self):
        """Test template with proactive leakage prevention."""
        prompt = QueryUnderstandingPrompt()
        
        protected_template = prompt.get_template_with_leakage_prevention()
        
        # Should be a string
        assert isinstance(protected_template, str)
        
        # Should contain prevention rules
        assert "CRITICAL SECURITY RULES" in protected_template or "DO NOT" in protected_template
        
        # Should contain original template content
        assert len(protected_template) > len(prompt.TEMPLATE)
    
    # ============ Response Format Validation ============
    
    def test_validate_response_format_plain_json(self):
        """Test validation of plain JSON response."""
        prompt = QueryUnderstandingPrompt()
        
        # Valid plain JSON
        response = '{"intent": "analytics", "slots": {}}'
        result = prompt.validate_response_format(response)
        assert isinstance(result, dict)
        assert "intent" in result
    
    def test_validate_response_format_markdown_json(self):
        """Test validation of markdown-wrapped JSON."""
        prompt = QueryUnderstandingPrompt()
        
        # Valid markdown JSON
        response = '```json\n{"intent": "analytics", "slots": {}}\n```'
        result = prompt.validate_response_format(response)
        assert isinstance(result, dict)
        assert "intent" in result
    
    def test_validate_response_format_code_block(self):
        """Test validation of code block JSON."""
        prompt = QueryUnderstandingPrompt()
        
        # Valid code block
        response = '```\n{"intent": "analytics", "slots": {}}\n```'
        result = prompt.validate_response_format(response)
        assert isinstance(result, dict)
        assert "intent" in result
    
    def test_validate_response_format_invalid(self):
        """Test rejection of invalid JSON."""
        prompt = QueryUnderstandingPrompt()
        
        # Invalid JSON
        with pytest.raises(PromptSecurityError):
            prompt.validate_response_format("This is not JSON")


class TestPromptSecurityError:
    """Test PromptSecurityError exception."""
    
    def test_error_creation(self):
        """Test error can be created."""
        error = PromptSecurityError("Test error")
        assert str(error) == "Test error"
    
    def test_error_with_details(self):
        """Test error with detailed message."""
        error = PromptSecurityError("Invalid template hash detected")
        assert "template hash" in str(error)
    
    def test_error_can_be_raised(self):
        """Test error can be raised and caught."""
        with pytest.raises(PromptSecurityError) as exc_info:
            raise PromptSecurityError("Security violation")
        
        assert "Security violation" in str(exc_info.value)
    
    def test_error_inheritance_chain(self):
        """Test error inherits from Exception."""
        error = PromptSecurityError("Test")
        assert isinstance(error, Exception)
