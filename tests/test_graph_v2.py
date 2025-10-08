"""
Unit tests for message compression and graph builder v2.
"""
import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from app.core.message_compression import (
    compress_tool_messages,
    build_interpretation_summary,
    _detect_sensitive_data,
    _redact_sensitive_data,
    _estimate_tokens
)
from app.core.agent_state import AnalyticsAgentState


class TestMessageCompression:
    """Test message compression utilities."""
    
    def test_detect_sensitive_data(self):
        """Test PII detection."""
        text = "Contact me at john@example.com or call 555-123-4567"
        detected = _detect_sensitive_data(text)
        assert 'email' in detected
        assert 'phone' in detected
    
    def test_redact_sensitive_data(self):
        """Test PII redaction."""
        text = "Email: test@example.com, Phone: 123-456-7890"
        redacted, detected = _redact_sensitive_data(text)
        
        assert '[REDACTED_EMAIL]' in redacted
        assert '[REDACTED_PHONE]' in redacted
        assert 'test@example.com' not in redacted
        assert 'email' in detected
        assert 'phone' in detected
    
    def test_estimate_tokens(self):
        """Test token estimation."""
        text = "Hello world"  # ~11 chars = ~2-3 tokens
        tokens = _estimate_tokens(text)
        assert tokens >= 2
        assert tokens <= 3
    
    def test_compress_tool_messages(self):
        """Test tool message compression."""
        messages = [
            HumanMessage(content="What is the success rate?"),
            AIMessage(content="Let me check", tool_calls=[{'id': '123', 'name': 'get_rate', 'args': {}}]),
            ToolMessage(
                content='{"success": true, "row_count": 1000, "file_name": "test.csv", "chart_data": [{"x": 1, "y": 2}]}',
                tool_call_id='123'
            )
        ]
        
        compressed, stats = compress_tool_messages(messages, budget_per_msg=500, total_budget=5000)
        
        # Should compress tool message
        assert stats['compressed_count'] == 1
        assert stats['tokens_saved'] > 0
        
        # Compressed message should be shorter
        tool_msg = [m for m in compressed if isinstance(m, ToolMessage)][0]
        assert len(tool_msg.content) < len(messages[2].content)
        assert 'Success: True' in tool_msg.content or 'Success: true' in tool_msg.content
    
    def test_build_interpretation_summary(self):
        """Test interpretation summary builder."""
        user_query = "Show me success rate for customer.csv"
        tool_results = [
            {
                'success': True,
                'file_name': 'customer.csv',
                'row_count': 500,
                'report_type': 'success'
            }
        ]
        context_insights = ['📊 Dataset contains 500 records', '📈 Average success rate: 85.5%']
        
        summary = build_interpretation_summary(user_query, tool_results, context_insights, max_tokens=2000)
        
        # Should contain key elements
        assert 'customer.csv' in summary.lower()
        assert '500' in summary
        assert 'KEY INSIGHTS' in summary
        assert 'TOOL EXECUTION SUMMARY' in summary
        
        # Should be within budget
        assert _estimate_tokens(summary) <= 2000


class TestTypedState:
    """Test typed agent state."""
    
    def test_agent_state_structure(self):
        """Verify AnalyticsAgentState has required fields."""
        from typing import get_type_hints
        
        hints = get_type_hints(AnalyticsAgentState)
        
        # Check required fields (user_id removed for security - uses contextvars instead)
        assert 'messages' in hints
        assert 'loop_count' in hints
        assert 'tool_results' in hints
        assert 'context_insights' in hints
        assert 'compression_applied' in hints
        assert 'total_tokens_estimate' in hints


if __name__ == '__main__':
    # Run quick tests
    print("Running compression tests...")
    
    test_compression = TestMessageCompression()
    test_compression.test_detect_sensitive_data()
    print("✓ PII detection works")
    
    test_compression.test_redact_sensitive_data()
    print("✓ PII redaction works")
    
    test_compression.test_estimate_tokens()
    print("✓ Token estimation works")
    
    test_compression.test_compress_tool_messages()
    print("✓ Message compression works")
    
    test_compression.test_build_interpretation_summary()
    print("✓ Interpretation summary works")
    
    test_state = TestTypedState()
    test_state.test_agent_state_structure()
    print("✓ Typed state structure valid")
    
    print("\n✅ All tests passed!")
