"""
Unit tests for analytic_api.py
Tests FastAPI endpoints with mocked dependencies.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import ValidationError

from app.analytic_api import app


class TestAnalyticReportEndpoint:
    """Test the /api/analytics/report endpoint."""
    
    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)
        self.valid_token = "Bearer valid.jwt.token"
        self.headers = {"Authorization": self.valid_token}
    
    @patch('app.analytic_api.validate_jwt_token')
    @patch('app.analytic_api.query_processor.query_handler')
    @patch('app.analytic_api.get_audit_sqs_service')
    def test_successful_analytics_query(self, mock_audit, mock_query_handler, mock_validate_jwt):
        """Test successful analytics query with valid token."""
        # Setup mocks
        mock_validate_jwt.return_value = {
            "sub": "user-123",
            "userName": "john.doe",
            "email": "john@example.com"
        }
        
        mock_query_handler.return_value = {
            "success": True,
            "message": "Success rate for customer domain: 85%",
            "chart_image": "base64_chart_data"
        }
        
        mock_audit_service = Mock()
        mock_audit.return_value = mock_audit_service
        
        # Make request
        response = self.client.post(
            "/api/analytics/report",
            json={"prompt": "show me success rate for customer domain"},
            headers=self.headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Success rate" in data["message"]
        assert data["chart_image"] is not None
        
        # Verify audit log was called
        mock_audit_service.send_analytics_query_audit.assert_called_once()
        call_args = mock_audit_service.send_analytics_query_audit.call_args[1]
        assert call_args["statusCode"] == 200
        assert call_args["user_id"] == "user-123"
        assert call_args["success"] is True
    
    @patch('app.analytic_api.validate_jwt_token')
    @patch('app.analytic_api.get_audit_sqs_service')
    def test_validation_error_empty_prompt(self, mock_audit, mock_validate_jwt):
        """Test validation error when prompt is empty."""
        mock_validate_jwt.return_value = {"sub": "user-123", "userName": "john"}
        mock_audit_service = Mock()
        mock_audit.return_value = mock_audit_service
        
        response = self.client.post(
            "/api/analytics/report",
            json={"prompt": ""},
            headers=self.headers
        )
        
        assert response.status_code == 200  # App returns 200 with error in body
        data = response.json()
        assert data["success"] is False
        assert "empty" in data["message"].lower() or "invalid" in data["message"].lower()
        
        # Verify audit log for failure
        mock_audit_service.send_analytics_query_audit.assert_called_once()
        call_args = mock_audit_service.send_analytics_query_audit.call_args[1]
        assert call_args["statusCode"] == 400
        assert call_args["success"] is False
    
    @patch('app.analytic_api.validate_jwt_token')
    @patch('app.analytic_api.query_processor.query_handler')
    @patch('app.analytic_api.get_audit_sqs_service')
    def test_query_handler_exception(self, mock_audit, mock_query_handler, mock_validate_jwt):
        """Test handling of unexpected exceptions in query handler."""
        mock_validate_jwt.return_value = {"sub": "user-123", "userName": "john"}
        mock_query_handler.side_effect = Exception("Database connection failed")
        mock_audit_service = Mock()
        mock_audit.return_value = mock_audit_service
        
        response = self.client.post(
            "/api/analytics/report",
            json={"prompt": "show me data"},
            headers=self.headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "unexpected error" in data["message"].lower()
        
        # Verify audit log for error
        mock_audit_service.send_analytics_query_audit.assert_called_once()
        call_args = mock_audit_service.send_analytics_query_audit.call_args[1]
        assert call_args["statusCode"] == 500
        assert call_args["success"] is False
    
    @patch('app.analytic_api.validate_jwt_token')
    @patch('app.analytic_api.query_processor.query_handler')
    @patch('app.analytic_api.get_audit_sqs_service')
    def test_query_failure_response(self, mock_audit, mock_query_handler, mock_validate_jwt):
        """Test when query returns failure."""
        mock_validate_jwt.return_value = {"sub": "user-123", "userName": "john"}
        mock_query_handler.return_value = {
            "success": False,
            "message": "No data found for domain",
            "chart_image": None
        }
        mock_audit_service = Mock()
        mock_audit.return_value = mock_audit_service
        
        response = self.client.post(
            "/api/analytics/report",
            json={"prompt": "show me data for unknown domain"},
            headers=self.headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "No data found" in data["message"]
        
        # Verify audit includes error message
        call_args = mock_audit_service.send_analytics_query_audit.call_args[1]
        assert call_args["message"] == "No data found for domain"


class TestClearConversationEndpoint:
    """Test the /api/analytics/conversation/clear endpoint."""
    
    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)
        self.valid_token = "Bearer valid.jwt.token"
        self.headers = {"Authorization": self.valid_token}
    
    @patch('app.analytic_api.validate_jwt_token')
    @patch('app.services.query_context_service.QueryContextService')
    @patch('app.analytic_api.get_audit_sqs_service')
    def test_clear_conversation_success(self, mock_audit, mock_context_service_class, mock_validate_jwt):
        """Test successful conversation clearing."""
        mock_validate_jwt.return_value = {
            "sub": "user-123",
            "userName": "john.doe"
        }
        
        mock_context_service = Mock()
        mock_context_service.clear_query_context.return_value = True
        mock_context_service_class.return_value = mock_context_service
        
        mock_audit_service = Mock()
        mock_audit.return_value = mock_audit_service
        
        response = self.client.delete(
            "/api/analytics/conversation/clear",
            headers=self.headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "cleared successfully" in data["message"].lower()
        
        # Verify clear was called
        mock_context_service.clear_query_context.assert_called_once_with("user-123")
        
        # Verify audit log
        mock_audit_service.send_analytics_query_audit.assert_called_once()
        call_args = mock_audit_service.send_analytics_query_audit.call_args[1]
        assert call_args["prompt"] == "[CLEAR_CONVERSATION_HISTORY]"
        assert call_args["success"] is True
    
    @patch('app.analytic_api.validate_jwt_token')
    @patch('app.services.query_context_service.QueryContextService')
    def test_clear_conversation_failure(self, mock_context_service_class, mock_validate_jwt):
        """Test when clearing conversation fails."""
        mock_validate_jwt.return_value = {
            "sub": "user-123",
            "userName": "john.doe"
        }
        
        mock_context_service = Mock()
        mock_context_service.clear_query_context.return_value = False
        mock_context_service_class.return_value = mock_context_service
        
        response = self.client.delete(
            "/api/analytics/conversation/clear",
            headers=self.headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "failed" in data["message"].lower()
    
    @patch('app.analytic_api.validate_jwt_token')
    def test_clear_conversation_no_user_id(self, mock_validate_jwt):
        """Test when JWT doesn't contain user ID."""
        mock_validate_jwt.return_value = {
            "userName": "john.doe"
            # Missing "sub"
        }
        
        response = self.client.delete(
            "/api/analytics/conversation/clear",
            headers=self.headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "invalid authentication" in data["message"].lower()
    
    @patch('app.analytic_api.validate_jwt_token')
    @patch('app.services.query_context_service.QueryContextService')
    @patch('app.analytic_api.get_audit_sqs_service')
    def test_clear_conversation_exception(self, mock_audit, mock_context_service_class, mock_validate_jwt):
        """Test handling of exceptions during clear."""
        mock_validate_jwt.return_value = {
            "sub": "user-123",
            "userName": "john.doe"
        }
        
        mock_context_service_class.side_effect = Exception("DynamoDB connection error")
        mock_audit_service = Mock()
        mock_audit.return_value = mock_audit_service
        
        response = self.client.delete(
            "/api/analytics/conversation/clear",
            headers=self.headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "unexpected error" in data["message"].lower()
        
        # Verify error audit log
        call_args = mock_audit_service.send_analytics_query_audit.call_args[1]
        assert call_args["statusCode"] == 500
        assert call_args["success"] is False
    
    def test_clear_conversation_no_auth(self):
        """Test clearing without authentication."""
        response = self.client.delete("/api/analytics/conversation/clear")
        
        # Should return 403 (no auth header)
        assert response.status_code == 403


class TestAPILifecycle:
    """Test API lifecycle events."""
    
    def test_api_initialization(self):
        """Test that API initializes correctly."""
        assert app.title == "Analytic Agent API"
        assert app.version == "2.0.0"
        assert "stateless" in app.description.lower()
    
    def test_health_check_routes_exist(self):
        """Test that routes are properly registered."""
        client = TestClient(app)
        routes = [route.path for route in app.routes]
        
        assert "/api/analytics/report" in routes
        assert "/api/analytics/conversation/clear" in routes


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=app.analytic_api", "--cov-report=term-missing"])
