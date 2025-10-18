"""
Unit tests for QueryProcessor.

Tests cover:
- Prompt validation
- Query handler flow
- Authentication and authorization
- Intent extraction and validation
- Simple vs complex query routing
- Context management
- Error handling
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import ValidationError


class TestPromptRequestValidation:
    """Test PromptRequest model validation."""
    
    def test_valid_prompt(self):
        """Test valid prompt passes validation."""
        from app.services.query_processor import PromptRequest
        
        request = PromptRequest(prompt="What is the success rate for customer domain?")
        assert request.prompt == "What is the success rate for customer domain?"
    
    def test_empty_prompt_validation(self):
        """Test empty prompt raises validation error."""
        from app.services.query_processor import PromptRequest
        
        with pytest.raises(ValidationError) as exc_info:
            PromptRequest(prompt="")
        
        assert "Prompt cannot be empty" in str(exc_info.value)
    
    def test_whitespace_only_prompt_validation(self):
        """Test whitespace-only prompt raises validation error."""
        from app.services.query_processor import PromptRequest
        
        with pytest.raises(ValidationError) as exc_info:
            PromptRequest(prompt="   ")
        
        assert "Prompt cannot be empty" in str(exc_info.value)
    
    def test_prompt_too_long(self):
        """Test prompt exceeding max length raises validation error."""
        from app.services.query_processor import PromptRequest
        
        long_prompt = "a" * 5001
        with pytest.raises(ValidationError) as exc_info:
            PromptRequest(prompt=long_prompt)
        
        assert "Prompt too long" in str(exc_info.value)
    
    @patch('app.services.query_processor.validate_user_prompt')
    def test_prompt_security_validation_fails(self, mock_validate):
        """Test prompt security validation failure."""
        from app.services.query_processor import PromptRequest
        
        mock_validate.return_value = (False, "Suspicious pattern detected")
        
        with pytest.raises(ValidationError) as exc_info:
            PromptRequest(prompt="SELECT * FROM users")
        
        assert "Suspicious pattern" in str(exc_info.value)
    
    @patch('app.services.query_processor.validate_user_prompt')
    def test_prompt_security_validation_passes(self, mock_validate):
        """Test prompt security validation success."""
        from app.services.query_processor import PromptRequest
        
        mock_validate.return_value = (True, None)
        
        request = PromptRequest(prompt="What is the success rate?")
        assert request.prompt == "What is the success rate?"


class TestQueryHandlerAuthentication:
    """Test query handler authentication flow."""
    
    @pytest.fixture
    def mock_credentials(self):
        """Create mock HTTP authorization credentials."""
        return HTTPAuthorizationCredentials(scheme="Bearer", credentials="test_token")
    
    @pytest.fixture
    def mock_http_request(self):
        """Create mock HTTP request."""
        return Mock()
    
    @pytest.fixture
    def processor(self):
        """Create QueryProcessor instance."""
        from app.services.query_processor import QueryProcessor
        return QueryProcessor()
    
    @pytest.mark.asyncio
    @patch('app.services.query_processor.validate_user_profile_with_response')
    async def test_authentication_failure(self, mock_validate, processor, mock_http_request, mock_credentials):
        """Test authentication failure returns error response."""
        from app.services.query_processor import PromptRequest
        
        mock_validate.return_value = {
            "success": False,
            "message": "Invalid token"
        }
        
        request = PromptRequest(prompt="Test query", session_id="test-session")
        result = await processor.query_handler(request, mock_http_request, mock_credentials)
        
        assert result["success"] is False
        assert "Invalid token" in result["message"]
    
    @pytest.mark.asyncio
    @patch('app.services.query_processor.get_query_understanding_agent')
    @patch('app.services.query_processor.validate_user_profile_with_response')
    async def test_missing_user_id_in_jwt(self, mock_validate, mock_agent, processor, mock_http_request, mock_credentials):
        """Test missing user_id in JWT returns error response."""
        from app.services.query_processor import PromptRequest
        
        # Mock successful auth but missing user_id
        mock_validate.return_value = {
            "success": True,
            "payload": {}  # No 'sub' field
        }
        
        request = PromptRequest(prompt="Test query")
        
        # Should return error response (catches ValueError internally)
        result = await processor.query_handler(request, mock_http_request, mock_credentials)
        
        assert result["success"] is False
        assert "missing required claims" in result["message"].lower() or "invalid request" in result["message"].lower()


class TestQueryHandlerIntentExtraction:
    """Test intent extraction and validation."""
    
    @pytest.fixture
    def processor(self):
        """Create QueryProcessor instance."""
        from app.services.query_processor import QueryProcessor
        return QueryProcessor()
    
    @pytest.fixture
    def mock_auth_success(self):
        """Mock successful authentication."""
        return {
            "success": True,
            "payload": {
                "sub": "user-123",
                "orgId": "org-456"
            }
        }
    
    @pytest.mark.asyncio
    @patch('app.services.query_processor.get_query_context_service')
    @patch('app.services.query_processor.get_query_understanding_agent')
    @patch('app.services.query_processor.validate_user_profile_with_response')
    async def test_out_of_scope_query(self, mock_validate, mock_agent_func, mock_context_service, processor, mock_auth_success):
        """Test out-of-scope query returns clarification message."""
        from app.services.query_processor import PromptRequest
        
        mock_validate.return_value = mock_auth_success
        
        # Mock agent to return out-of-scope result
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.intent = "unknown"
        mock_result.slots = {}
        mock_result.is_complete = False
        mock_result.clarification_needed = "I'm specialized in analytics. Please ask about data analysis."
        mock_result.query_type = "unknown"
        mock_result.high_level_intent = None
        
        mock_agent.extract_intent_and_slots = AsyncMock(return_value=mock_result)
        mock_agent.validate_completeness = Mock(return_value=mock_result)
        mock_agent_func.return_value = mock_agent
        
        request = PromptRequest(prompt="What's the weather?")
        result = await processor.query_handler(request, Mock(), Mock())
        
        assert result["success"] is False
        assert "specialized in analytics" in result["message"]
        assert result["chart_image"] is None


class TestQueryHandlerSimpleQuery:
    """Test simple query processing."""
    
    @pytest.fixture
    def processor(self):
        """Create QueryProcessor instance."""
        from app.services.query_processor import QueryProcessor
        return QueryProcessor()
    
    @pytest.fixture
    def mock_auth_success(self):
        """Mock successful authentication."""
        return {
            "success": True,
            "payload": {
                "sub": "user-123",
                "orgId": "org-456"
            }
        }
    
    @pytest.mark.asyncio
    @patch('app.services.query_processor.get_query_context_service')
    @patch('app.services.query_processor.get_query_understanding_agent')
    @patch('app.services.query_processor.validate_user_profile_with_response')
    async def test_simple_query_success(self, mock_validate, mock_agent_func, mock_context_service, processor, mock_auth_success):
        """Test simple query processes successfully."""
        from app.services.query_processor import PromptRequest
        
        mock_validate.return_value = mock_auth_success
        
        # Mock agent to return complete simple query
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.intent = "success_rate"
        mock_result.slots = {"domain_name": "customer", "file_name": None}
        mock_result.is_complete = True
        mock_result.clarification_needed = None
        mock_result.query_type = "simple"
        mock_result.high_level_intent = "success_rate"
        mock_result.comparison_targets = []
        
        mock_agent.extract_intent_and_slots = AsyncMock(return_value=mock_result)
        mock_agent.validate_completeness = Mock(return_value=mock_result)
        mock_agent_func.return_value = mock_agent
        
        # Mock context service
        mock_context = Mock()
        mock_context.save_query_context = Mock(return_value={
            "intent": "success_rate",
            "slots": {"domain_name": "customer", "file_name": None}
        })
        mock_context.get_query_context = Mock(return_value=None)
        mock_context.should_save_context = Mock(return_value=True)
        mock_context_service.return_value = mock_context
        
        # Mock simple query executor (imported as: from app.orchestration.simple_query_executor import run_analytics_query)
        with patch('app.orchestration.simple_query_executor.run_analytics_query') as mock_executor:
            mock_executor.return_value = {
                "success": True,
                "message": "Analysis complete",
                "chart_image": "base64_image_data"
            }
            
            # Mock validate_llm_output
            with patch('app.services.query_processor.validate_llm_output') as mock_validate_output:
                mock_validate_output.return_value = (True, None)
                
                request = PromptRequest(prompt="What is the success rate for customer domain?")
                result = await processor.query_handler(request, Mock(), Mock())
                
                assert result["success"] is True
                assert "Analysis complete" in result["message"]


class TestQueryHandlerComplexQuery:
    """Test complex query processing."""
    
    @pytest.fixture
    def processor(self):
        """Create QueryProcessor instance."""
        from app.services.query_processor import QueryProcessor
        return QueryProcessor()
    
    @pytest.fixture
    def mock_auth_success(self):
        """Mock successful authentication."""
        return {
            "success": True,
            "payload": {
                "sub": "user-123",
                "orgId": "org-456"
            }
        }
    
    @pytest.mark.asyncio
    @patch('app.services.query_processor.get_query_context_service')
    @patch('app.services.query_processor.get_query_understanding_agent')
    @patch('app.services.query_processor.validate_user_profile_with_response')
    async def test_complex_query_missing_targets_and_intent(self, mock_validate, mock_agent_func, mock_context_service, processor, mock_auth_success):
        """Test complex query with missing targets and intent."""
        from app.services.query_processor import PromptRequest
        
        mock_validate.return_value = mock_auth_success
        
        # Mock agent to return complex query without complete info
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.intent = "compare"
        mock_result.slots = {}
        mock_result.is_complete = False
        mock_result.clarification_needed = None
        mock_result.query_type = "complex"
        mock_result.comparison_targets = []
        
        mock_agent.extract_intent_and_slots = AsyncMock(return_value=mock_result)
        mock_agent.validate_completeness = Mock(return_value=mock_result)
        mock_agent_func.return_value = mock_agent
        
        # Mock context service
        mock_context = Mock()
        mock_context.save_query_context = Mock(return_value={
            "intent": "",
            "slots": {},
            "comparison_targets": []
        })
        mock_context.get_query_context = Mock(return_value=None)
        mock_context_service.return_value = mock_context
        
        request = PromptRequest(prompt="Compare them")
        result = await processor.query_handler(request, Mock(), Mock())
        
        assert result["success"] is False
        assert "Incomplete comparison query" in result["message"]
    
    @pytest.mark.asyncio
    @patch('app.services.query_processor.get_query_context_service')
    @patch('app.services.query_processor.get_query_understanding_agent')
    @patch('app.services.query_processor.validate_user_profile_with_response')
    async def test_complex_query_missing_targets(self, mock_validate, mock_agent_func, mock_context_service, processor, mock_auth_success):
        """Test complex query with missing comparison targets."""
        from app.services.query_processor import PromptRequest
        
        mock_validate.return_value = mock_auth_success
        
        # Mock agent
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.intent = "success_rate"
        mock_result.slots = {}
        mock_result.is_complete = False
        mock_result.clarification_needed = None
        mock_result.query_type = "complex"
        mock_result.comparison_targets = []
        
        mock_agent.extract_intent_and_slots = AsyncMock(return_value=mock_result)
        mock_agent.validate_completeness = Mock(return_value=mock_result)
        mock_agent_func.return_value = mock_agent
        
        # Mock context service
        mock_context = Mock()
        mock_context.save_query_context = Mock(return_value={
            "intent": "success_rate",
            "slots": {},
            "comparison_targets": []
        })
        mock_context_service.return_value = mock_context
        
        request = PromptRequest(prompt="Compare success rates")
        result = await processor.query_handler(request, Mock(), Mock())
        
        assert result["success"] is False
        assert "Missing comparison targets" in result["message"]
    
    @pytest.mark.asyncio
    @patch('app.services.query_processor.get_query_context_service')
    @patch('app.services.query_processor.get_query_understanding_agent')
    @patch('app.services.query_processor.validate_user_profile_with_response')
    async def test_complex_query_missing_intent(self, mock_validate, mock_agent_func, mock_context_service, processor, mock_auth_success):
        """Test complex query with missing intent."""
        from app.services.query_processor import PromptRequest
        
        mock_validate.return_value = mock_auth_success
        
        # Mock agent
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.intent = "compare"
        mock_result.slots = {}
        mock_result.is_complete = False
        mock_result.clarification_needed = None
        mock_result.query_type = "complex"
        mock_result.comparison_targets = ["customer", "product"]
        
        mock_agent.extract_intent_and_slots = AsyncMock(return_value=mock_result)
        mock_agent.validate_completeness = Mock(return_value=mock_result)
        mock_agent_func.return_value = mock_agent
        
        # Mock context service
        mock_context = Mock()
        mock_context.save_query_context = Mock(return_value={
            "intent": "",
            "slots": {},
            "comparison_targets": ["customer", "product"]
        })
        mock_context.get_query_context = Mock(return_value=None)
        mock_context_service.return_value = mock_context
        
        request = PromptRequest(prompt="Compare customer and product")
        result = await processor.query_handler(request, Mock(), Mock())
        
        assert result["success"] is False
        assert "Missing analysis type" in result["message"]


class TestQueryHandlerErrorHandling:
    """Test error handling in query processor."""
    
    @pytest.fixture
    def processor(self):
        """Create QueryProcessor instance."""
        from app.services.query_processor import QueryProcessor
        return QueryProcessor()
    
    @pytest.fixture
    def mock_auth_success(self):
        """Mock successful authentication."""
        return {
            "success": True,
            "payload": {
                "sub": "user-123",
                "orgId": "org-456"
            }
        }
    
    @pytest.mark.asyncio
    @patch('app.services.query_processor.get_query_understanding_agent')
    @patch('app.services.query_processor.validate_user_profile_with_response')
    async def test_agent_exception_handling(self, mock_validate, mock_agent_func, processor, mock_auth_success):
        """Test exception during intent extraction returns error response."""
        from app.services.query_processor import PromptRequest
        
        mock_validate.return_value = mock_auth_success
        
        # Mock agent to raise exception
        mock_agent = Mock()
        mock_agent.extract_intent_and_slots = AsyncMock(side_effect=Exception("Agent error"))
        mock_agent_func.return_value = mock_agent
        
        request = PromptRequest(prompt="Test query")
        
        # Should handle exception gracefully and return error response
        result = await processor.query_handler(request, Mock(), Mock())
        
        assert result["success"] is False
        assert "error" in result["message"].lower() or "processing failed" in result["message"].lower()


class TestCreateErrorResponse:
    """Test _create_error_response method."""
    
    def test_create_error_response(self):
        """Test error response creation."""
        from app.services.query_processor import QueryProcessor
        
        processor = QueryProcessor()
        result = processor._create_error_response("validation", "Invalid input")
        
        assert result["success"] is False
        assert result["error"] == "validation"
        assert "Invalid input" in result["message"]
        assert result["chart_image"] is None


class TestQueryHandlerContextInheritance:
    """Test context inheritance from previous queries."""
    
    @pytest.fixture
    def processor(self):
        """Create QueryProcessor instance."""
        from app.services.query_processor import QueryProcessor
        return QueryProcessor()
    
    @pytest.fixture
    def mock_auth_success(self):
        """Mock successful authentication."""
        return {
            "success": True,
            "payload": {
                "sub": "user-123",
                "orgId": "org-456"
            }
        }
    
    @pytest.mark.asyncio
    @patch('app.services.query_processor.get_query_context_service')
    @patch('app.services.query_processor.get_query_understanding_agent')
    @patch('app.services.query_processor.validate_user_profile_with_response')
    async def test_inherit_intent_from_previous_context(self, mock_validate, mock_agent_func, mock_context_service, processor, mock_auth_success):
        """Test inheriting intent (report_type) from previous context."""
        from app.services.query_processor import PromptRequest
        
        mock_validate.return_value = mock_auth_success
        
        # Mock agent to return query without intent
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.intent = "unknown"
        mock_result.slots = {"domain_name": "customer", "file_name": None}
        mock_result.is_complete = False
        mock_result.clarification_needed = None
        mock_result.query_type = "simple"
        mock_result.comparison_targets = []
        
        mock_agent.extract_intent_and_slots = AsyncMock(return_value=mock_result)
        mock_agent.validate_completeness = Mock(return_value=mock_result)
        mock_agent_func.return_value = mock_agent
        
        # Mock context service with previous context
        mock_context = Mock()
        mock_context.get_query_context = Mock(return_value={
            "report_type": "success_rate",
            "slots": {},
            "updated_at": "2024-01-01T00:00:00"
        })
        mock_context.should_save_context = Mock(return_value=True)
        mock_context.save_query_context = Mock(return_value={
            "intent": "success_rate",
            "slots": {"domain_name": "customer"}
        })
        mock_context_service.return_value = mock_context
        
        # Mock executor
        with patch('app.orchestration.simple_query_executor.run_analytics_query') as mock_executor:
            mock_executor.return_value = {
                "success": True,
                "message": "Analysis complete",
                "chart_image": "base64_image_data"
            }
            
            with patch('app.services.query_processor.validate_llm_output') as mock_validate_output:
                mock_validate_output.return_value = (True, None)
                
                request = PromptRequest(prompt="for customer domain")
                result = await processor.query_handler(request, Mock(), Mock())
                
                # Should succeed by inheriting intent
                assert result["success"] is True
    
    @pytest.mark.asyncio
    @patch('app.services.query_processor.get_query_context_service')
    @patch('app.services.query_processor.get_query_understanding_agent')
    @patch('app.services.query_processor.validate_user_profile_with_response')
    async def test_inherit_target_from_previous_context(self, mock_validate, mock_agent_func, mock_context_service, processor, mock_auth_success):
        """Test inheriting target (domain/file) from previous context."""
        from app.services.query_processor import PromptRequest
        
        mock_validate.return_value = mock_auth_success
        
        # Mock agent to return query without target
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.intent = "success_rate"
        mock_result.slots = {"domain_name": None, "file_name": None}
        mock_result.is_complete = False
        mock_result.clarification_needed = None
        mock_result.query_type = "simple"
        mock_result.comparison_targets = []
        
        mock_agent.extract_intent_and_slots = AsyncMock(return_value=mock_result)
        mock_agent.validate_completeness = Mock(return_value=mock_result)
        mock_agent_func.return_value = mock_agent
        
        # Mock context service with previous context
        mock_context = Mock()
        mock_context.get_query_context = Mock(return_value={
            "report_type": "failure_rate",
            "slots": {"domain_name": "customer"},
            "updated_at": "2024-01-01T00:00:00"
        })
        mock_context.should_save_context = Mock(return_value=True)
        mock_context.save_query_context = Mock(return_value={
            "intent": "success_rate",
            "slots": {"domain_name": "customer"}
        })
        mock_context_service.return_value = mock_context
        
        # Mock executor
        with patch('app.orchestration.simple_query_executor.run_analytics_query') as mock_executor:
            mock_executor.return_value = {
                "success": True,
                "message": "Analysis complete",
                "chart_image": "base64_image_data"
            }
            
            with patch('app.services.query_processor.validate_llm_output') as mock_validate_output:
                mock_validate_output.return_value = (True, None)
                
                request = PromptRequest(prompt="show success rate")
                result = await processor.query_handler(request, Mock(), Mock())
                
                # Should succeed by inheriting target
                assert result["success"] is True


class TestQueryHandlerConflictDetection:
    """Test target conflict detection."""
    
    @pytest.fixture
    def processor(self):
        """Create QueryProcessor instance."""
        from app.services.query_processor import QueryProcessor
        return QueryProcessor()
    
    @pytest.fixture
    def mock_auth_success(self):
        """Mock successful authentication."""
        return {
            "success": True,
            "payload": {
                "sub": "user-123",
                "orgId": "org-456"
            }
        }
    
    @pytest.mark.asyncio
    @patch('app.services.query_processor.get_query_context_service')
    @patch('app.services.query_processor.get_query_understanding_agent')
    @patch('app.services.query_processor.validate_user_profile_with_response')
    async def test_conflict_domain_to_file(self, mock_validate, mock_agent_func, mock_context_service, processor, mock_auth_success):
        """Test conflict detection when switching from domain to file."""
        from app.services.query_processor import PromptRequest
        
        mock_validate.return_value = mock_auth_success
        
        # Mock agent to return file target
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.intent = "success_rate"
        mock_result.slots = {"domain_name": None, "file_name": "product.csv"}
        mock_result.is_complete = True
        mock_result.clarification_needed = None
        mock_result.query_type = "simple"
        mock_result.comparison_targets = []
        
        mock_agent.extract_intent_and_slots = AsyncMock(return_value=mock_result)
        mock_agent.validate_completeness = Mock(return_value=mock_result)
        mock_agent_func.return_value = mock_agent
        
        # Mock context service with previous domain context
        mock_context = Mock()
        mock_context.get_query_context = Mock(return_value={
            "report_type": "success_rate",
            "slots": {"domain_name": "customer", "file_name": None},
            "updated_at": "2024-01-01T00:00:00"
        })
        mock_context.should_save_context = Mock(return_value=True)
        mock_context.save_query_context = Mock(return_value={
            "intent": "success_rate",
            "slots": {"file_name": "product.csv", "_conflict_pending": True}
        })
        mock_context_service.return_value = mock_context
        
        request = PromptRequest(prompt="show success rate for product.csv")
        result = await processor.query_handler(request, Mock(), Mock())
        
        # Should detect conflict
        assert result["success"] is False
        assert "Target Conflict" in result["message"] or "conflict" in result["message"].lower()


class TestQueryHandlerIncompleteQueries:
    """Test incomplete query handling."""
    
    @pytest.fixture
    def processor(self):
        """Create QueryProcessor instance."""
        from app.services.query_processor import QueryProcessor
        return QueryProcessor()
    
    @pytest.fixture
    def mock_auth_success(self):
        """Mock successful authentication."""
        return {
            "success": True,
            "payload": {
                "sub": "user-123",
                "orgId": "org-456"
            }
        }
    
    @pytest.mark.asyncio
    @patch('app.services.query_processor.get_query_context_service')
    @patch('app.services.query_processor.get_query_understanding_agent')
    @patch('app.services.query_processor.validate_user_profile_with_response')
    async def test_missing_both_intent_and_target(self, mock_validate, mock_agent_func, mock_context_service, processor, mock_auth_success):
        """Test query missing both intent and target."""
        from app.services.query_processor import PromptRequest
        
        mock_validate.return_value = mock_auth_success
        
        # Mock agent to return incomplete query
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.intent = "unknown"
        mock_result.slots = {"domain_name": None, "file_name": None}
        mock_result.is_complete = False
        mock_result.clarification_needed = None
        mock_result.query_type = "simple"
        mock_result.comparison_targets = []
        mock_result.missing_required = ["report_type", "target"]
        
        mock_agent.extract_intent_and_slots = AsyncMock(return_value=mock_result)
        mock_agent.validate_completeness = Mock(return_value=mock_result)
        mock_agent_func.return_value = mock_agent
        
        # Mock context service (no previous context)
        mock_context = Mock()
        mock_context.get_query_context = Mock(return_value=None)
        mock_context.should_save_context = Mock(return_value=False)
        mock_context_service.return_value = mock_context
        
        request = PromptRequest(prompt="show me something")
        result = await processor.query_handler(request, Mock(), Mock())
        
        # Should return error about missing info
        assert result["success"] is False
        assert "Missing" in result["message"] or "specify" in result["message"].lower()
    
    @pytest.mark.asyncio
    @patch('app.services.query_processor.get_query_context_service')
    @patch('app.services.query_processor.get_query_understanding_agent')
    @patch('app.services.query_processor.validate_user_profile_with_response')
    async def test_missing_only_target(self, mock_validate, mock_agent_func, mock_context_service, processor, mock_auth_success):
        """Test query missing only target."""
        from app.services.query_processor import PromptRequest
        
        mock_validate.return_value = mock_auth_success
        
        # Mock agent to return query with intent but no target
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.intent = "success_rate"
        mock_result.slots = {"domain_name": None, "file_name": None}
        mock_result.is_complete = False
        mock_result.clarification_needed = None
        mock_result.query_type = "simple"
        mock_result.comparison_targets = []
        mock_result.missing_required = ["target"]
        
        mock_agent.extract_intent_and_slots = AsyncMock(return_value=mock_result)
        mock_agent.validate_completeness = Mock(return_value=mock_result)
        mock_agent_func.return_value = mock_agent
        
        # Mock context service (no previous context)
        mock_context = Mock()
        mock_context.get_query_context = Mock(return_value=None)
        mock_context.should_save_context = Mock(return_value=True)
        mock_context.save_query_context = Mock(return_value={
            "intent": "success_rate",
            "slots": {}
        })
        mock_context_service.return_value = mock_context
        
        request = PromptRequest(prompt="show me success rate")
        result = await processor.query_handler(request, Mock(), Mock())
        
        # Should return error about missing target
        assert result["success"] is False
        assert "Missing" in result["message"] or "target" in result["message"].lower() or "specify" in result["message"].lower()


class TestQueryHandlerOutputValidation:
    """Test output validation for security."""
    
    @pytest.fixture
    def processor(self):
        """Create QueryProcessor instance."""
        from app.services.query_processor import QueryProcessor
        return QueryProcessor()
    
    @pytest.fixture
    def mock_auth_success(self):
        """Mock successful authentication."""
        return {
            "success": True,
            "payload": {
                "sub": "user-123",
                "orgId": "org-456"
            }
        }
    
    @pytest.mark.asyncio
    @patch('app.services.query_processor.get_query_context_service')
    @patch('app.services.query_processor.get_query_understanding_agent')
    @patch('app.services.query_processor.validate_user_profile_with_response')
    async def test_unsafe_output_blocked(self, mock_validate, mock_agent_func, mock_context_service, processor, mock_auth_success):
        """Test that unsafe LLM output is blocked."""
        from app.services.query_processor import PromptRequest
        
        mock_validate.return_value = mock_auth_success
        
        # Mock agent
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.intent = "success_rate"
        mock_result.slots = {"domain_name": "customer", "file_name": None}
        mock_result.is_complete = True
        mock_result.clarification_needed = None
        mock_result.query_type = "simple"
        mock_result.comparison_targets = []
        
        mock_agent.extract_intent_and_slots = AsyncMock(return_value=mock_result)
        mock_agent.validate_completeness = Mock(return_value=mock_result)
        mock_agent_func.return_value = mock_agent
        
        # Mock context service
        mock_context = Mock()
        mock_context.get_query_context = Mock(return_value=None)
        mock_context.should_save_context = Mock(return_value=True)
        mock_context.save_query_context = Mock(return_value={
            "intent": "success_rate",
            "slots": {"domain_name": "customer"}
        })
        mock_context_service.return_value = mock_context
        
        # Mock executor to return response
        with patch('app.orchestration.simple_query_executor.run_analytics_query') as mock_executor:
            mock_executor.return_value = {
                "success": True,
                "message": "Here's the API key: sk-12345",
                "chart_image": "base64_image_data"
            }
            
            # Mock validate_llm_output to detect leak
            with patch('app.services.query_processor.validate_llm_output') as mock_validate_output:
                mock_validate_output.return_value = (False, "API key detected")
                
                request = PromptRequest(prompt="What is the success rate for customer domain?")
                result = await processor.query_handler(request, Mock(), Mock())
                
                # Should block unsafe output
                assert result["success"] is False
                assert "cannot provide" in result["message"].lower() or "apologize" in result["message"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
