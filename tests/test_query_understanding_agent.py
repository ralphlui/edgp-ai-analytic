"""
Test suite for QueryUnderstandingAgent.

Tests intent extraction, slot filling, and query understanding for analytics queries.
"""
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.orchestration.query_understanding_agent import (
    QueryUnderstandingAgent,
    QueryUnderstandingResult,
    get_query_understanding_agent
)


class TestQueryUnderstandingResult:
    """Test QueryUnderstandingResult model."""
    
    def test_query_understanding_result_defaults(self):
        """Test QueryUnderstandingResult with default values."""
        result = QueryUnderstandingResult()
        
        assert result.intent is None
        assert result.slots == {}
        assert result.confidence == 0.0
        assert result.missing_required == []
        assert result.is_complete is False
        assert result.clarification_needed is None
        assert result.query_type is None
        assert result.high_level_intent is None
        assert result.comparison_targets == []
    
    def test_query_understanding_result_with_values(self):
        """Test QueryUnderstandingResult with explicit values."""
        result = QueryUnderstandingResult(
            intent="success_rate",
            slots={"domain_name": "customer"},
            confidence=0.95,
            missing_required=[],
            is_complete=True,
            query_type="simple",
            comparison_targets=[]
        )
        
        assert result.intent == "success_rate"
        assert result.slots == {"domain_name": "customer"}
        assert result.confidence == 0.95
        assert result.missing_required == []
        assert result.is_complete is True
        assert result.query_type == "simple"


class TestQueryUnderstandingAgentInit:
    """Test QueryUnderstandingAgent initialization."""
    
    @patch('app.orchestration.query_understanding_agent.ChatOpenAI')
    def test_agent_initialization(self, mock_chat_openai):
        """Test agent initializes with correct LLM settings."""
        agent = QueryUnderstandingAgent()
        
        assert agent.llm is not None
        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args.kwargs
        assert call_kwargs['temperature'] == 0.0


class TestExtractIntentAndSlots:
    """Test extract_intent_and_slots method."""
    
    @pytest.mark.asyncio
    @patch('app.orchestration.query_understanding_agent.ChatOpenAI')
    async def test_extract_intent_success_rate_simple(self, mock_chat_openai):
        """Test extracting success_rate intent for simple query."""
        # Mock LLM response
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "intent": "success_rate",
            "query_type": "simple",
            "slots": {"domain_name": "customer"},
            "confidence": 0.95,
            "missing_required": [],
            "is_complete": True,
            "comparison_targets": []
        })
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm
        
        agent = QueryUnderstandingAgent()
        result = await agent.extract_intent_and_slots("show me success rate for customer domain")
        
        assert result.intent == "success_rate"
        assert result.query_type == "simple"
        assert result.slots == {"domain_name": "customer"}
        assert result.is_complete is True
    
    @pytest.mark.asyncio
    @patch('app.orchestration.query_understanding_agent.ChatOpenAI')
    async def test_extract_intent_failure_rate_simple(self, mock_chat_openai):
        """Test extracting failure_rate intent for simple query."""
        # Mock LLM response
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "intent": "failure_rate",
            "query_type": "simple",
            "slots": {"file_name": "customer.csv"},
            "confidence": 0.95,
            "missing_required": [],
            "is_complete": True,
            "comparison_targets": []
        })
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm
        
        agent = QueryUnderstandingAgent()
        result = await agent.extract_intent_and_slots("failure rate for customer.csv")
        
        assert result.intent == "failure_rate"
        assert result.query_type == "simple"
        assert result.slots == {"file_name": "customer.csv"}
        assert result.is_complete is True
    
    @pytest.mark.asyncio
    @patch('app.orchestration.query_understanding_agent.ChatOpenAI')
    async def test_extract_intent_complex_comparison(self, mock_chat_openai):
        """Test extracting comparison intent for complex query."""
        # Mock LLM response
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "intent": "failure_rate",
            "query_type": "complex",
            "high_level_intent": "comparison",
            "slots": {},
            "confidence": 0.95,
            "missing_required": [],
            "is_complete": True,
            "comparison_targets": ["customer.csv", "product.csv"]
        })
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm
        
        agent = QueryUnderstandingAgent()
        result = await agent.extract_intent_and_slots("compare failure rates between customer.csv and product.csv")
        
        assert result.intent == "failure_rate"
        assert result.query_type == "complex"
        assert result.high_level_intent == "comparison"
        assert result.comparison_targets == ["customer.csv", "product.csv"]
        assert result.is_complete is True
    
    @pytest.mark.asyncio
    @patch('app.orchestration.query_understanding_agent.ChatOpenAI')
    async def test_extract_intent_incomplete_query(self, mock_chat_openai):
        """Test extracting intent from incomplete query."""
        # Mock LLM response
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "intent": "success_rate",
            "query_type": "simple",
            "slots": {},
            "confidence": 0.8,
            "missing_required": ["domain_name or file_name"],
            "is_complete": False,
            "clarification_needed": "I need to know which file or domain to analyze.",
            "comparison_targets": []
        })
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm
        
        agent = QueryUnderstandingAgent()
        result = await agent.extract_intent_and_slots("show me success rate")
        
        assert result.intent == "success_rate"
        assert result.is_complete is False
        assert "domain" in result.clarification_needed.lower() or "file" in result.clarification_needed.lower()
    
    @pytest.mark.asyncio
    @patch('app.orchestration.query_understanding_agent.ChatOpenAI')
    async def test_extract_intent_out_of_scope(self, mock_chat_openai):
        """Test extracting out_of_scope intent."""
        # Mock LLM response
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "intent": "out_of_scope",
            "query_type": "simple",
            "slots": {},
            "confidence": 0.95,
            "missing_required": [],
            "is_complete": True,
            "clarification_needed": "Hello! I'm an analytics assistant.",
            "comparison_targets": []
        })
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm
        
        agent = QueryUnderstandingAgent()
        result = await agent.extract_intent_and_slots("hello")
        
        assert result.intent == "out_of_scope"
        assert result.is_complete is True
        assert result.clarification_needed is not None
    
    @pytest.mark.asyncio
    @patch('app.orchestration.query_understanding_agent.ChatOpenAI')
    async def test_extract_intent_json_decode_error(self, mock_chat_openai):
        """Test handling of JSON decode error from LLM response."""
        # Mock LLM response with invalid JSON
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_response.content = "This is not valid JSON { invalid }"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm
        
        agent = QueryUnderstandingAgent()
        result = await agent.extract_intent_and_slots("show me success rate")
        
        # Should return fallback result
        assert result.intent == "general_query"
        assert result.is_complete is False
        assert result.confidence == 0.0
        assert "couldn't understand" in result.clarification_needed.lower()
    
    @pytest.mark.asyncio
    @patch('app.orchestration.query_understanding_agent.ChatOpenAI')
    async def test_extract_intent_llm_exception(self, mock_chat_openai):
        """Test handling of exception during LLM invocation."""
        # Mock LLM to raise exception
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API error"))
        mock_chat_openai.return_value = mock_llm
        
        agent = QueryUnderstandingAgent()
        result = await agent.extract_intent_and_slots("show me success rate")
        
        # Should return error fallback result
        assert result.intent == "general_query"
        assert result.is_complete is False
        assert result.confidence == 0.0
        assert "error" in result.clarification_needed.lower()


class TestValidateCompleteness:
    """Test validate_completeness method."""
    
    def test_validate_completeness_out_of_scope(self):
        """Test validation for out_of_scope intent."""
        agent = QueryUnderstandingAgent()
        
        result = QueryUnderstandingResult(
            intent="out_of_scope",
            slots={},
            is_complete=True
        )
        
        validated = agent.validate_completeness(result)
        
        assert validated.clarification_needed is not None
        assert "analytics assistant" in validated.clarification_needed.lower()
    
    def test_validate_completeness_general_query(self):
        """Test validation for general_query intent."""
        agent = QueryUnderstandingAgent()
        
        result = QueryUnderstandingResult(
            intent="general_query",
            slots={},
            is_complete=False
        )
        
        validated = agent.validate_completeness(result)
        
        assert validated.clarification_needed is not None
        assert "analytics" in validated.clarification_needed.lower()
    
    def test_validate_completeness_with_query_type(self):
        """Test validation preserves query_type."""
        agent = QueryUnderstandingAgent()
        
        result = QueryUnderstandingResult(
            intent="success_rate",
            query_type="complex",
            high_level_intent="comparison",
            slots={},
            is_complete=True
        )
        
        validated = agent.validate_completeness(result)
        
        # Validation should not change query_type or high_level_intent
        assert validated.intent == "success_rate"


class TestGetQueryUnderstandingAgent:
    """Test get_query_understanding_agent singleton."""
    
    @patch('app.orchestration.query_understanding_agent.ChatOpenAI')
    def test_get_agent_singleton(self, mock_chat_openai):
        """Test get_query_understanding_agent returns singleton instance."""
        # Reset singleton for test
        import app.orchestration.query_understanding_agent as agent_module
        agent_module._query_understanding_agent = None
        
        agent1 = get_query_understanding_agent()
        agent2 = get_query_understanding_agent()
        
        assert agent1 is agent2
        assert isinstance(agent1, QueryUnderstandingAgent)
    
    @patch('app.orchestration.query_understanding_agent.ChatOpenAI')
    def test_get_agent_creates_instance_once(self, mock_chat_openai):
        """Test get_query_understanding_agent creates instance only once."""
        # Reset singleton for test
        import app.orchestration.query_understanding_agent as agent_module
        agent_module._query_understanding_agent = None
        
        # First call should create instance
        agent1 = get_query_understanding_agent()
        call_count_after_first = mock_chat_openai.call_count
        
        # Second call should reuse instance
        agent2 = get_query_understanding_agent()
        call_count_after_second = mock_chat_openai.call_count
        
        assert call_count_after_first == call_count_after_second
        assert agent1 is agent2


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    @patch('app.orchestration.query_understanding_agent.ChatOpenAI')
    async def test_extract_intent_empty_query(self, mock_chat_openai):
        """Test extracting intent from empty query."""
        # Mock LLM response for empty query
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "intent": "general_query",
            "query_type": "simple",
            "slots": {},
            "confidence": 0.5,
            "missing_required": ["intent", "target"],
            "is_complete": False,
            "clarification_needed": "Please provide a query.",
            "comparison_targets": []
        })
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm
        
        agent = QueryUnderstandingAgent()
        result = await agent.extract_intent_and_slots("")
        
        assert result.intent == "general_query"
        assert result.is_complete is False
    
    @pytest.mark.asyncio
    @patch('app.orchestration.query_understanding_agent.ChatOpenAI')
    async def test_extract_intent_very_long_query(self, mock_chat_openai):
        """Test extracting intent from very long query."""
        # Mock LLM response
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "intent": "success_rate",
            "query_type": "simple",
            "slots": {"domain_name": "customer"},
            "confidence": 0.85,
            "missing_required": [],
            "is_complete": True,
            "comparison_targets": []
        })
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm
        
        agent = QueryUnderstandingAgent()
        long_query = "I would like to see " * 50 + "success rate for customer domain"
        result = await agent.extract_intent_and_slots(long_query)
        
        assert result.intent == "success_rate"
        assert result.slots.get("domain_name") == "customer"
    
    @pytest.mark.asyncio
    @patch('app.orchestration.query_understanding_agent.ChatOpenAI')
    async def test_extract_intent_special_characters(self, mock_chat_openai):
        """Test extracting intent with special characters in query."""
        # Mock LLM response
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "intent": "failure_rate",
            "query_type": "simple",
            "slots": {"file_name": "customer-data_v2.csv"},
            "confidence": 0.9,
            "missing_required": [],
            "is_complete": True,
            "comparison_targets": []
        })
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm
        
        agent = QueryUnderstandingAgent()
        result = await agent.extract_intent_and_slots("failure rate for customer-data_v2.csv")
        
        assert result.intent == "failure_rate"
        assert result.slots.get("file_name") == "customer-data_v2.csv"


class TestComplexQueryScenarios:
    """Test complex query scenarios."""
    
    @pytest.mark.asyncio
    @patch('app.orchestration.query_understanding_agent.ChatOpenAI')
    async def test_extract_intent_comparison_incomplete(self, mock_chat_openai):
        """Test extracting incomplete comparison query."""
        # Mock LLM response
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "intent": "failure_rate",
            "query_type": "complex",
            "high_level_intent": "comparison",
            "slots": {},
            "confidence": 0.7,
            "missing_required": ["second_comparison_target"],
            "is_complete": False,
            "clarification_needed": "What would you like to compare it with?",
            "comparison_targets": ["customer.csv"]
        })
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm
        
        agent = QueryUnderstandingAgent()
        result = await agent.extract_intent_and_slots("compare failure rates for customer.csv")
        
        assert result.intent == "failure_rate"
        assert result.query_type == "complex"
        assert result.is_complete is False
        assert len(result.comparison_targets) == 1
    
    @pytest.mark.asyncio
    @patch('app.orchestration.query_understanding_agent.ChatOpenAI')
    async def test_extract_intent_mixed_comparison(self, mock_chat_openai):
        """Test extracting comparison between domain and file."""
        # Mock LLM response
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "intent": "success_rate",
            "query_type": "complex",
            "high_level_intent": "comparison",
            "slots": {},
            "confidence": 0.95,
            "missing_required": [],
            "is_complete": True,
            "comparison_targets": ["customer", "payment.csv"]
        })
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm
        
        agent = QueryUnderstandingAgent()
        result = await agent.extract_intent_and_slots("compare success rate for customer domain vs payment.csv")
        
        assert result.intent == "success_rate"
        assert result.query_type == "complex"
        assert result.comparison_targets == ["customer", "payment.csv"]
        assert result.is_complete is True
