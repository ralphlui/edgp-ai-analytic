"""
Unit tests for secure prompt templates.

Tests the base functionality and security features of all prompt templates.
Focus on template integrity, basic formatting, and schema validation.
"""
import hashlib
import pytest
from app.prompts.base_prompt import PromptSecurityError
from app.prompts.query_understanding_prompts import QueryUnderstandingPrompt
from app.prompts.planner_prompts import PlannerPrompt
from app.prompts.simple_executor_prompts import (
    SimpleExecutorToolSelectionPrompt,
    SimpleExecutorResponseFormattingPrompt
)
from app.prompts.complex_executor_prompts import (
    ComplexExecutorToolSelectionPrompt,
    ComplexExecutorResponseFormattingPrompt
)


class TestQueryUnderstandingPrompt:
    """Test QueryUnderstandingPrompt."""
    
    def test_template_integrity(self):
        """Test template hash is valid."""
        prompt = QueryUnderstandingPrompt()
        expected_hash = hashlib.sha256(prompt.TEMPLATE.encode('utf-8')).hexdigest()
        assert prompt.TEMPLATE_HASH == expected_hash
    
    def test_get_system_prompt(self):
        """Test system prompt generation."""
        prompt = QueryUnderstandingPrompt()
        system_prompt = prompt.get_system_prompt()
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0
    
    def test_format_user_message(self):
        """Test user message formatting."""
        prompt = QueryUnderstandingPrompt()
        message = prompt.format_user_message(query="What's the success rate?")
        assert isinstance(message, str)
        assert "success rate" in message
        assert "<USER_QUERY>" in message  # Uses XML-style tags
    
    def test_format_user_message_sanitization(self):
        """Test that user input is sanitized (warns but doesn't escape in current implementation)."""
        prompt = QueryUnderstandingPrompt()
        message = prompt.format_user_message(query="<script>xss</script>test")
        # Current implementation logs warnings but includes content
        assert isinstance(message, str)
        assert "test" in message
    
    def test_format_user_message_empty(self):
        """Test empty query handling."""
        prompt = QueryUnderstandingPrompt()
        with pytest.raises(PromptSecurityError):
            prompt.format_user_message(query="")
    
    def test_validate_response_schema_valid(self):
        """Test valid response validation."""
        prompt = QueryUnderstandingPrompt()
        valid_response = {
            "intent": "success_rate",
            "query_type": "simple",
            "slots": {},
            "confidence": 0.9,
            "missing_required": [],
            "is_complete": True
        }
        assert prompt.validate_response_schema(valid_response) is True
    
    def test_validate_response_schema_missing_key(self):
        """Test validation fails for missing required keys."""
        prompt = QueryUnderstandingPrompt()
        with pytest.raises(PromptSecurityError):
            prompt.validate_response_schema({"intent": "success_rate"})


class TestPlannerPrompt:
    """Test PlannerPrompt."""
    
    def test_template_integrity(self):
        """Test template hash is valid."""
        prompt = PlannerPrompt()
        expected_hash = hashlib.sha256(prompt.TEMPLATE.encode('utf-8')).hexdigest()
        assert prompt.TEMPLATE_HASH == expected_hash
    
    def test_get_system_prompt(self):
        """Test system prompt generation."""
        prompt = PlannerPrompt()
        system_prompt = prompt.get_system_prompt()
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0


class TestSimpleExecutorToolSelectionPrompt:
    """Test SimpleExecutorToolSelectionPrompt."""
    
    def test_template_integrity(self):
        """Test template hash is valid."""
        prompt = SimpleExecutorToolSelectionPrompt()
        expected_hash = hashlib.sha256(prompt.TEMPLATE.encode('utf-8')).hexdigest()
        assert prompt.TEMPLATE_HASH == expected_hash
    
    def test_get_system_prompt(self):
        """Test system prompt generation."""
        prompt = SimpleExecutorToolSelectionPrompt()
        system_prompt = prompt.get_system_prompt()
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0


class TestSimpleExecutorResponseFormattingPrompt:
    """Test SimpleExecutorResponseFormattingPrompt."""
    
    def test_template_integrity(self):
        """Test template hash is valid."""
        prompt = SimpleExecutorResponseFormattingPrompt()
        expected_hash = hashlib.sha256(prompt.TEMPLATE.encode('utf-8')).hexdigest()
        assert prompt.TEMPLATE_HASH == expected_hash
    
    def test_get_system_prompt(self):
        """Test system prompt generation."""
        prompt = SimpleExecutorResponseFormattingPrompt()
        system_prompt = prompt.get_system_prompt()
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0


class TestComplexExecutorToolSelectionPrompt:
    """Test ComplexExecutorToolSelectionPrompt."""
    
    def test_template_integrity(self):
        """Test template hash is valid."""
        prompt = ComplexExecutorToolSelectionPrompt()
        expected_hash = hashlib.sha256(prompt.TEMPLATE.encode('utf-8')).hexdigest()
        assert prompt.TEMPLATE_HASH == expected_hash
    
    def test_get_system_prompt(self):
        """Test system prompt generation."""
        prompt = ComplexExecutorToolSelectionPrompt()
        system_prompt = prompt.get_system_prompt()
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0


class TestComplexExecutorResponseFormattingPrompt:
    """Test ComplexExecutorResponseFormattingPrompt."""
    
    def test_template_integrity(self):
        """Test template hash is valid."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        expected_hash = hashlib.sha256(prompt.TEMPLATE.encode('utf-8')).hexdigest()
        assert prompt.TEMPLATE_HASH == expected_hash
    
    def test_get_system_prompt(self):
        """Test system prompt generation."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        system_prompt = prompt.get_system_prompt()
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
