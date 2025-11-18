"""
Unit tests for complex executor prompt templates.

Tests cover:
- Template integrity verification
- Tool selection prompt formatting and validation
- Response formatting prompt
- Input sanitization and security validation
- Schema validation for comparison results
- Error handling and edge cases
"""
import hashlib
import pytest
from app.prompts.complex_executor_prompts import (
    ComplexExecutorToolSelectionPrompt,
    ComplexExecutorResponseFormattingPrompt
)
from app.prompts.base_prompt import PromptSecurityError


class TestComplexExecutorToolSelectionPrompt:
    """Test ComplexExecutorToolSelectionPrompt functionality."""
    
    # ============ Template Integrity ============
    
    def test_template_integrity(self):
        """Test template hash integrity verification."""
        prompt = ComplexExecutorToolSelectionPrompt()
        prompt.verify_integrity()
        assert prompt.TEMPLATE_HASH == hashlib.sha256(prompt.TEMPLATE.encode('utf-8')).hexdigest()
    
    def test_get_system_prompt(self):
        """Test system prompt generation with leakage prevention."""
        prompt = ComplexExecutorToolSelectionPrompt()
        system_prompt = prompt.get_system_prompt()
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0
        assert "CRITICAL SECURITY RULES" in system_prompt
    
    # ============ Message Formatting ============
    
    def test_format_message_success_rate_domain(self):
        """Test formatting for success_rate with domain target."""
        prompt = ComplexExecutorToolSelectionPrompt()
        message = prompt._format_message(
            metric_type="success_rate",
            target_type="domain_name",
            target="customer"
        )
        assert isinstance(message, str)
        assert "success_rate" in message
        assert "domain_name" in message
        assert "customer" in message
        assert "<TOOL_PARAMS>" in message
    
    def test_format_message_failure_rate_file(self):
        """Test formatting for failure_rate with file target."""
        prompt = ComplexExecutorToolSelectionPrompt()
        message = prompt._format_message(
            metric_type="failure_rate",
            target_type="file_name",
            target="data.csv"
        )
        assert isinstance(message, str)
        assert "failure_rate" in message
        assert "file_name" in message
        assert "data.csv" in message
    
    def test_format_message_sanitizes_target(self):
        """Test that target value is sanitized."""
        prompt = ComplexExecutorToolSelectionPrompt()
        message = prompt._format_message(
            metric_type="success_rate",
            target_type="domain_name",
            target="test<script>alert('xss')</script>"
        )
        assert isinstance(message, str)
        assert "test" in message
        # Sanitization logs warnings
    
    def test_format_message_invalid_metric_type_raises_error(self):
        """Test that invalid metric_type raises error."""
        prompt = ComplexExecutorToolSelectionPrompt()
        with pytest.raises(PromptSecurityError, match="metric_type must be one of"):
            prompt._format_message(
                metric_type="invalid_metric",
                target_type="domain_name",
                target="test"
            )
    
    def test_format_message_invalid_target_type_raises_error(self):
        """Test that invalid target_type raises error."""
        prompt = ComplexExecutorToolSelectionPrompt()
        with pytest.raises(PromptSecurityError, match="target_type must be one of"):
            prompt._format_message(
                metric_type="success_rate",
                target_type="invalid_type",
                target="test"
            )
    
    def test_format_message_empty_target_raises_error(self):
        """Test that empty target raises error."""
        prompt = ComplexExecutorToolSelectionPrompt()
        with pytest.raises(PromptSecurityError, match="target must be a non-empty string"):
            prompt._format_message(
                metric_type="success_rate",
                target_type="domain_name",
                target=""
            )
    
    def test_format_message_none_target_raises_error(self):
        """Test that None target raises error."""
        prompt = ComplexExecutorToolSelectionPrompt()
        with pytest.raises(PromptSecurityError, match="target must be a non-empty string"):
            prompt._format_message(
                metric_type="success_rate",
                target_type="domain_name",
                target=None
            )
    
    def test_format_message_target_not_string_raises_error(self):
        """Test that non-string target raises error."""
        prompt = ComplexExecutorToolSelectionPrompt()
        with pytest.raises(PromptSecurityError, match="target must be a non-empty string"):
            prompt._format_message(
                metric_type="success_rate",
                target_type="domain_name",
                target=123
            )
    
    def test_format_message_target_too_long_raises_error(self):
        """Test that overly long target raises error."""
        prompt = ComplexExecutorToolSelectionPrompt()
        with pytest.raises(PromptSecurityError, match="target exceeds maximum length"):
            prompt._format_message(
                metric_type="success_rate",
                target_type="domain_name",
                target="x" * 600
            )
    
    # ============ Schema Validation ============
    
    def test_validate_response_schema_success_rate_domain(self):
        """Test validation of success_rate tool with domain."""
        prompt = ComplexExecutorToolSelectionPrompt()
        data = {
            "tool": "generate_success_rate_report",
            "args": {
                "domain_name": "customer",
                "file_name": None
            }
        }
        assert prompt.validate_response_schema(data) is True
    
    def test_validate_response_schema_failure_rate_file(self):
        """Test validation of failure_rate tool with file."""
        prompt = ComplexExecutorToolSelectionPrompt()
        data = {
            "tool": "generate_failure_rate_report",
            "args": {
                "domain_name": None,
                "file_name": "data.csv"
            }
        }
        assert prompt.validate_response_schema(data) is True
    
    def test_validate_response_schema_missing_tool_key(self):
        """Test that missing 'tool' key raises error."""
        prompt = ComplexExecutorToolSelectionPrompt()
        data = {
            "args": {"domain_name": "test", "file_name": None}
        }
        with pytest.raises(PromptSecurityError, match="Response missing required keys"):
            prompt.validate_response_schema(data)
    
    def test_validate_response_schema_missing_args_key(self):
        """Test that missing 'args' key raises error."""
        prompt = ComplexExecutorToolSelectionPrompt()
        data = {
            "tool": "generate_success_rate_report"
        }
        with pytest.raises(PromptSecurityError, match="Response missing required keys"):
            prompt.validate_response_schema(data)
    
    def test_validate_response_schema_invalid_tool_name(self):
        """Test that invalid tool name raises error."""
        prompt = ComplexExecutorToolSelectionPrompt()
        data = {
            "tool": "malicious_tool",
            "args": {"domain_name": "test", "file_name": None}
        }
        with pytest.raises(PromptSecurityError, match="tool must be one of"):
            prompt.validate_response_schema(data)
    
    def test_validate_response_schema_args_not_dict(self):
        """Test that non-dict args raises error."""
        prompt = ComplexExecutorToolSelectionPrompt()
        data = {
            "tool": "generate_success_rate_report",
            "args": "invalid"
        }
        with pytest.raises(PromptSecurityError, match="'args' must be a dict"):
            prompt.validate_response_schema(data)
    
    def test_validate_response_schema_missing_domain_name_key(self):
        """Test that missing domain_name in args raises error."""
        prompt = ComplexExecutorToolSelectionPrompt()
        data = {
            "tool": "generate_success_rate_report",
            "args": {"file_name": "test.csv"}
        }
        with pytest.raises(PromptSecurityError, match="args missing required key: domain_name"):
            prompt.validate_response_schema(data)
    
    def test_validate_response_schema_missing_file_name_key(self):
        """Test that missing file_name in args raises error."""
        prompt = ComplexExecutorToolSelectionPrompt()
        data = {
            "tool": "generate_success_rate_report",
            "args": {"domain_name": "test"}
        }
        with pytest.raises(PromptSecurityError, match="args missing required key: file_name"):
            prompt.validate_response_schema(data)
    
    def test_validate_response_schema_both_none_raises_error(self):
        """Test that both domain_name and file_name being None raises error."""
        prompt = ComplexExecutorToolSelectionPrompt()
        data = {
            "tool": "generate_success_rate_report",
            "args": {
                "domain_name": None,
                "file_name": None
            }
        }
        with pytest.raises(PromptSecurityError, match="Either domain_name or file_name must be set"):
            prompt.validate_response_schema(data)
    
    def test_validate_response_schema_both_set_raises_error(self):
        """Test that both domain_name and file_name being set raises error."""
        prompt = ComplexExecutorToolSelectionPrompt()
        data = {
            "tool": "generate_success_rate_report",
            "args": {
                "domain_name": "customer",
                "file_name": "data.csv"
            }
        }
        with pytest.raises(PromptSecurityError, match="Cannot set both domain_name and file_name"):
            prompt.validate_response_schema(data)
    
    def test_validate_response_schema_domain_not_string_raises_error(self):
        """Test that non-string domain_name raises error."""
        prompt = ComplexExecutorToolSelectionPrompt()
        data = {
            "tool": "generate_success_rate_report",
            "args": {
                "domain_name": 123,
                "file_name": None
            }
        }
        with pytest.raises(PromptSecurityError, match="domain_name must be string"):
            prompt.validate_response_schema(data)
    
    def test_validate_response_schema_file_not_string_raises_error(self):
        """Test that non-string file_name raises error."""
        prompt = ComplexExecutorToolSelectionPrompt()
        data = {
            "tool": "generate_success_rate_report",
            "args": {
                "domain_name": None,
                "file_name": 123
            }
        }
        with pytest.raises(PromptSecurityError, match="file_name must be string"):
            prompt.validate_response_schema(data)
    
    def test_validate_response_schema_domain_too_long_raises_error(self):
        """Test that overly long domain_name raises error."""
        prompt = ComplexExecutorToolSelectionPrompt()
        data = {
            "tool": "generate_success_rate_report",
            "args": {
                "domain_name": "x" * 600,
                "file_name": None
            }
        }
        with pytest.raises(PromptSecurityError, match="domain_name exceeds maximum length"):
            prompt.validate_response_schema(data)
    
    def test_validate_response_schema_file_too_long_raises_error(self):
        """Test that overly long file_name raises error."""
        prompt = ComplexExecutorToolSelectionPrompt()
        data = {
            "tool": "generate_success_rate_report",
            "args": {
                "domain_name": None,
                "file_name": "x" * 600
            }
        }
        with pytest.raises(PromptSecurityError, match="file_name exceeds maximum length"):
            prompt.validate_response_schema(data)
    
    def test_validate_response_schema_sanitizes_domain(self):
        """Test that domain_name is sanitized during validation."""
        prompt = ComplexExecutorToolSelectionPrompt()
        data = {
            "tool": "generate_success_rate_report",
            "args": {
                "domain_name": "test<script>xss</script>",
                "file_name": None
            }
        }
        assert prompt.validate_response_schema(data) is True
        # Value should be sanitized
        assert isinstance(data['args']['domain_name'], str)
    
    def test_validate_response_schema_sanitizes_file(self):
        """Test that file_name is sanitized during validation."""
        prompt = ComplexExecutorToolSelectionPrompt()
        data = {
            "tool": "generate_success_rate_report",
            "args": {
                "domain_name": None,
                "file_name": "test<script>xss</script>.csv"
            }
        }
        assert prompt.validate_response_schema(data) is True
        # Value should be sanitized
        assert isinstance(data['args']['file_name'], str)


class TestComplexExecutorResponseFormattingPrompt:
    """Test ComplexExecutorResponseFormattingPrompt functionality."""
    
    # ============ Template Integrity ============
    
    def test_template_integrity(self):
        """Test template hash integrity verification."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        prompt.verify_integrity()
        assert prompt.TEMPLATE_HASH == hashlib.sha256(prompt.TEMPLATE.encode('utf-8')).hexdigest()
    
    def test_get_system_prompt(self):
        """Test system prompt generation with leakage prevention."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        system_prompt = prompt.get_system_prompt()
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0
        assert "CRITICAL SECURITY RULES" in system_prompt
    
    # ============ Message Formatting ============
    
    def test_format_message_with_chart(self):
        """Test formatting comparison results with chart."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        targets = ["customer", "payment"]
        details = [
            {
                "target": "customer",
                "metric_value": "85.5",
                "total_requests": 1000,
                "successful_requests": 855,
                "failed_requests": 145
            },
            {
                "target": "payment",
                "metric_value": "92.3",
                "total_requests": 800,
                "successful_requests": 738,
                "failed_requests": 62
            }
        ]
        message = prompt._format_message(
            user_query="Compare success rates",
            targets=targets,
            winner="payment",
            metric="success_rate",
            details=details,
            has_chart=True
        )
        assert isinstance(message, str)
        assert "customer" in message
        assert "payment" in message
        assert "85.5" in message
        assert "92.3" in message
        assert "<USER_QUERY>" in message
        assert "<COMPARISON_RESULTS>" in message
        assert "Available" in message
    
    def test_format_message_without_chart(self):
        """Test formatting comparison results without chart."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        targets = ["domain1"]
        details = [
            {
                "target": "domain1",
                "metric_value": "75.0",
                "total_requests": 500,
                "successful_requests": 375,
                "failed_requests": 125
            }
        ]
        message = prompt._format_message(
            user_query="Show failure rate",
            targets=targets,
            winner="domain1",
            metric="failure_rate",
            details=details,
            has_chart=False
        )
        assert isinstance(message, str)
        assert "domain1" in message
        assert "Not available" in message
        assert "failure_rate" in message
    
    def test_format_message_sanitizes_query(self):
        """Test that user query is sanitized."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        targets = ["test"]
        details = [
            {
                "target": "test",
                "metric_value": "80.0",
                "total_requests": 100,
                "successful_requests": 80,
                "failed_requests": 20
            }
        ]
        message = prompt._format_message(
            user_query="<script>xss</script>query",
            targets=targets,
            winner="test",
            metric="success_rate",
            details=details,
            has_chart=False
        )
        assert isinstance(message, str)
        assert "query" in message
    
    def test_format_message_sanitizes_target_names(self):
        """Test that target names in details are sanitized."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        targets = ["test<script>"]
        details = [
            {
                "target": "test<script>",
                "metric_value": "80.0",
                "total_requests": 100,
                "successful_requests": 80,
                "failed_requests": 20
            }
        ]
        message = prompt._format_message(
            user_query="query",
            targets=targets,
            winner="test<script>",
            metric="success_rate",
            details=details,
            has_chart=False
        )
        assert isinstance(message, str)
        assert "test" in message
    
    def test_format_message_empty_query_raises_error(self):
        """Test that empty query raises error."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        with pytest.raises(PromptSecurityError, match="user_query must be a non-empty string"):
            prompt._format_message(
                user_query="",
                targets=["test"],
                winner="test",
                metric="success_rate",
                details=[],
                has_chart=False
            )
    
    def test_format_message_none_query_raises_error(self):
        """Test that None query raises error."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        with pytest.raises(PromptSecurityError, match="user_query must be a non-empty string"):
            prompt._format_message(
                user_query=None,
                targets=["test"],
                winner="test",
                metric="success_rate",
                details=[],
                has_chart=False
            )
    
    def test_format_message_query_too_long_raises_error(self):
        """Test that overly long query raises error."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        with pytest.raises(PromptSecurityError, match="user_query exceeds maximum length"):
            prompt._format_message(
                user_query="x" * 6000,
                targets=["test"],
                winner="test",
                metric="success_rate",
                details=[],
                has_chart=False
            )
    
    def test_format_message_targets_not_list_raises_error(self):
        """Test that non-list targets raises error."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        with pytest.raises(PromptSecurityError, match="targets must be a list"):
            prompt._format_message(
                user_query="query",
                targets="not_a_list",
                winner="test",
                metric="success_rate",
                details=[],
                has_chart=False
            )
    
    def test_format_message_too_many_targets_raises_error(self):
        """Test that too many targets raises error."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        with pytest.raises(PromptSecurityError, match="targets exceeds maximum count"):
            prompt._format_message(
                user_query="query",
                targets=["target"] * 60,  # Exceeds 50 limit
                winner="test",
                metric="success_rate",
                details=[],
                has_chart=False
            )
    
    def test_format_message_invalid_metric_raises_error(self):
        """Test that invalid metric raises error."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        with pytest.raises(PromptSecurityError, match="metric must be one of"):
            prompt._format_message(
                user_query="query",
                targets=["test"],
                winner="test",
                metric="invalid_metric",
                details=[],
                has_chart=False
            )
    
    def test_format_message_has_chart_not_bool_raises_error(self):
        """Test that non-bool has_chart raises error."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        with pytest.raises(PromptSecurityError, match="has_chart must be a boolean"):
            prompt._format_message(
                user_query="query",
                targets=["test"],
                winner="test",
                metric="success_rate",
                details=[],
                has_chart="yes"
            )
    
    def test_format_message_details_not_list_raises_error(self):
        """Test that non-list details raises error."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        with pytest.raises(PromptSecurityError, match="details must be a list"):
            prompt._format_message(
                user_query="query",
                targets=["test"],
                winner="test",
                metric="success_rate",
                details="not_a_list",
                has_chart=False
            )
    
    def test_format_message_too_many_details_raises_error(self):
        """Test that too many details raises error."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        detail = {
            "target": "test",
            "metric_value": "80.0",
            "total_requests": 100,
            "successful_requests": 80,
            "failed_requests": 20
        }
        with pytest.raises(PromptSecurityError, match="details exceeds maximum count"):
            prompt._format_message(
                user_query="query",
                targets=["test"],
                winner="test",
                metric="success_rate",
                details=[detail] * 60,
                has_chart=False
            )
    
    def test_format_message_detail_not_dict_raises_error(self):
        """Test that non-dict detail raises error."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        with pytest.raises(PromptSecurityError, match="detail 0 must be a dict"):
            prompt._format_message(
                user_query="query",
                targets=["test"],
                winner="test",
                metric="success_rate",
                details=["not_a_dict"],
                has_chart=False
            )
    
    def test_format_message_detail_missing_key_raises_error(self):
        """Test that detail missing required key raises error."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        detail = {
            "target": "test",
            "metric_value": "80.0"
            # Missing other required keys
        }
        with pytest.raises(PromptSecurityError, match="detail 0 missing required key"):
            prompt._format_message(
                user_query="query",
                targets=["test"],
                winner="test",
                metric="success_rate",
                details=[detail],
                has_chart=False
            )
    
    # ============ Schema Validation ============
    
    def test_validate_response_schema_string(self):
        """Test validation of plain string response."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        response = "The payment domain has the highest success rate at 92.3%."
        assert prompt.validate_response_schema(response) is True
    
    def test_validate_response_schema_dict_with_message_key(self):
        """Test validation of dict with 'message' key."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        data = {"message": "The payment domain wins!"}
        assert prompt.validate_response_schema(data) is True
    
    def test_validate_response_schema_dict_missing_message_key(self):
        """Test that dict without 'message' key raises error."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        data = {"response": "test"}
        with pytest.raises(PromptSecurityError, match="Response dict must contain 'message' key"):
            prompt.validate_response_schema(data)
    
    def test_validate_response_schema_invalid_type_raises_error(self):
        """Test that invalid response type raises error."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        with pytest.raises(PromptSecurityError, match="Response must be string or dict"):
            prompt.validate_response_schema(123)
    
    def test_validate_response_schema_empty_string_raises_error(self):
        """Test that empty response raises error."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        with pytest.raises(PromptSecurityError, match="Response must be non-empty string"):
            prompt.validate_response_schema("")
    
    def test_validate_response_schema_none_raises_error(self):
        """Test that None response raises error."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        with pytest.raises(PromptSecurityError, match="Response must be string or dict"):
            prompt.validate_response_schema(None)
    
    def test_validate_response_schema_too_long_raises_error(self):
        """Test that overly long response raises error."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        with pytest.raises(PromptSecurityError, match="Response exceeds maximum length"):
            prompt.validate_response_schema("x" * 60000)
    
    def test_validate_response_schema_detects_leakage(self):
        """Test that potential leakage is detected (logs warning, doesn't block)."""
        prompt = ComplexExecutorResponseFormattingPrompt()
        response = "Let me explain the system instructions and available tools."
        # Should pass but log warning
        assert prompt.validate_response_schema(response) is True
