"""
Unit tests for authentication and authorization.

Tests cover:
- JWT token validation
- User profile validation
- Error handling for invalid/expired tokens
- Integration with admin API
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt
import httpx


class TestJWTValidation:
    """Test cases for JWT token validation."""
    
    @pytest.fixture
    def mock_credentials(self):
        """Create mock HTTP authorization credentials."""
        return HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="test.jwt.token"
        )
    
    @pytest.fixture
    def sample_jwt_payload(self):
        """Sample JWT payload for testing."""
        return {
            "sub": "user-123-456",
            "email": "test@example.com",
            "userName": "testuser",
            "exp": 9999999999  # Far future
        }
    
    @patch('app.security.auth.jwt.decode')
    def test_valid_jwt_token(self, mock_decode, mock_credentials, sample_jwt_payload):
        """Test validation of a valid JWT token."""
        from app.security.auth import validate_jwt_token
        
        mock_decode.return_value = sample_jwt_payload
        
        result = validate_jwt_token(mock_credentials)
        
        assert result == sample_jwt_payload
        assert result["sub"] == "user-123-456"
        assert result["email"] == "test@example.com"
    
    @patch('app.security.auth.jwt.decode')
    def test_expired_jwt_token(self, mock_decode, mock_credentials):
        """Test handling of expired JWT token."""
        from app.security.auth import validate_jwt_token
        from jose import JWTError
        
        mock_decode.side_effect = JWTError("Token expired")
        
        with pytest.raises(HTTPException) as exc_info:
            validate_jwt_token(mock_credentials)
        
        assert exc_info.value.status_code == 401
        assert "Invalid or expired token" in str(exc_info.value.detail)
    
    @patch('app.security.auth.jwt.decode')
    def test_invalid_jwt_token(self, mock_decode, mock_credentials):
        """Test handling of invalid JWT token."""
        from app.security.auth import validate_jwt_token
        from jose import JWTError
        
        mock_decode.side_effect = JWTError("Invalid token")
        
        with pytest.raises(HTTPException) as exc_info:
            validate_jwt_token(mock_credentials)
        
        assert exc_info.value.status_code == 401
    
    @patch('app.security.auth.jwt.decode')
    def test_jwt_missing_required_claims(self, mock_decode, mock_credentials):
        """Test JWT token missing required claims."""
        from app.security.auth import validate_jwt_token
        
        # Payload without 'sub' claim
        incomplete_payload = {
            "email": "test@example.com"
        }
        mock_decode.return_value = incomplete_payload
        
        # Token is valid, but missing claims will be caught downstream
        result = validate_jwt_token(mock_credentials)
        assert "sub" not in result


class TestUserProfileValidation:
    """Test cases for user profile validation via admin API."""
    
    @pytest.fixture
    def mock_credentials(self):
        """Create mock HTTP authorization credentials."""
        return HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="valid.jwt.token"
        )
    
    @pytest.fixture
    def sample_jwt_payload(self):
        """Sample JWT payload."""
        return {
            "sub": "user-123",
            "email": "active@example.com",
            "userName": "activeuser"
        }
    
    @pytest.mark.asyncio
    @patch('app.security.auth.validate_jwt_token')
    @patch('httpx.AsyncClient')
    async def test_active_user_validation(
        self,
        mock_httpx_client,
        mock_validate_jwt,
        mock_credentials,
        sample_jwt_payload
    ):
        """Test validation of active user profile."""
        from app.security.auth import validate_user_profile_with_response
        
        # Mock JWT validation
        mock_validate_jwt.return_value = sample_jwt_payload
        
        # Mock admin API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "data": {
                "userID": "user-123",
                "email": "active@example.com",
                "active": True
            }
        }
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_httpx_client.return_value.__aenter__.return_value = mock_client
        
        # Test validation
        result = await validate_user_profile_with_response(mock_credentials)
        
        assert result["success"] is True
        assert "authenticated and active" in result["message"].lower()
        assert result["payload"] == sample_jwt_payload
    
    @pytest.mark.asyncio
    @patch('app.security.auth.validate_jwt_token')
    @patch('httpx.AsyncClient')
    async def test_inactive_user_validation(
        self,
        mock_httpx_client,
        mock_validate_jwt,
        mock_credentials,
        sample_jwt_payload
    ):
        """Test validation of inactive user profile."""
        from app.security.auth import validate_user_profile_with_response
        
        mock_validate_jwt.return_value = sample_jwt_payload
        
        # Mock admin API response for inactive user
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": False,
            "data": {
                "userID": "user-123",
                "active": False
            }
        }
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_httpx_client.return_value.__aenter__.return_value = mock_client
        
        result = await validate_user_profile_with_response(mock_credentials)
        
        assert result["success"] is False
        assert "not active" in result["message"].lower()
    
    @pytest.mark.asyncio
    @patch('app.security.auth.validate_jwt_token')
    @patch('httpx.AsyncClient')
    async def test_user_not_found(
        self,
        mock_httpx_client,
        mock_validate_jwt,
        mock_credentials,
        sample_jwt_payload
    ):
        """Test validation when user not found in admin system."""
        from app.security.auth import validate_user_profile_with_response
        
        mock_validate_jwt.return_value = sample_jwt_payload
        
        # Mock 404 response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "User not found"
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_httpx_client.return_value.__aenter__.return_value = mock_client
        
        result = await validate_user_profile_with_response(mock_credentials)
        
        assert result["success"] is False
        assert "not found" in result["message"].lower()
    
    @pytest.mark.asyncio
    @patch('app.security.auth.validate_jwt_token')
    async def test_invalid_token_in_profile_validation(
        self,
        mock_validate_jwt,
        mock_credentials
    ):
        """Test profile validation with invalid JWT token."""
        from app.security.auth import validate_user_profile_with_response
        
        # Mock JWT validation failure
        mock_validate_jwt.side_effect = HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )
        
        result = await validate_user_profile_with_response(mock_credentials)
        
        assert result["success"] is False
        assert "authentication failed" in result["message"].lower()
    
    @pytest.mark.asyncio
    @patch('app.security.auth.validate_jwt_token')
    @patch('httpx.AsyncClient')
    async def test_admin_api_timeout(
        self,
        mock_httpx_client,
        mock_validate_jwt,
        mock_credentials,
        sample_jwt_payload
    ):
        """Test handling of admin API timeout."""
        from app.security.auth import validate_user_profile_with_response
        
        mock_validate_jwt.return_value = sample_jwt_payload
        
        # Mock timeout exception
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        mock_httpx_client.return_value.__aenter__.return_value = mock_client
        
        result = await validate_user_profile_with_response(mock_credentials)
        
        assert result["success"] is False
        assert "timeout" in result["message"].lower()
    
    @pytest.mark.asyncio
    @patch('app.security.auth.validate_jwt_token')
    @patch('httpx.AsyncClient')
    async def test_admin_api_connection_error(
        self,
        mock_httpx_client,
        mock_validate_jwt,
        mock_credentials,
        sample_jwt_payload
    ):
        """Test handling of admin API connection error."""
        from app.security.auth import validate_user_profile_with_response
        
        mock_validate_jwt.return_value = sample_jwt_payload
        
        # Mock connection error
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.RequestError("Connection failed"))
        mock_httpx_client.return_value.__aenter__.return_value = mock_client
        
        result = await validate_user_profile_with_response(mock_credentials)
        
        assert result["success"] is False
        assert "connect" in result["message"].lower()
    
    @pytest.mark.asyncio
    @patch('app.security.auth.validate_jwt_token')
    async def test_missing_user_id_in_jwt(
        self,
        mock_validate_jwt,
        mock_credentials
    ):
        """Test handling of JWT without user ID (sub claim)."""
        from app.security.auth import validate_user_profile_with_response
        
        # Mock JWT payload without 'sub' claim
        mock_validate_jwt.return_value = {
            "email": "test@example.com"
        }
        
        result = await validate_user_profile_with_response(mock_credentials)
        
        assert result["success"] is False
        assert "missing user id" in result["message"].lower()
    
    @pytest.mark.asyncio
    @patch('app.security.auth.validate_jwt_token')
    @patch('httpx.AsyncClient')
    async def test_admin_api_500_error(
        self,
        mock_httpx_client,
        mock_validate_jwt,
        mock_credentials,
        sample_jwt_payload
    ):
        """Test handling of admin API internal server error."""
        from app.security.auth import validate_user_profile_with_response
        
        mock_validate_jwt.return_value = sample_jwt_payload
        
        # Mock 500 response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_httpx_client.return_value.__aenter__.return_value = mock_client
        
        result = await validate_user_profile_with_response(mock_credentials)
        
        assert result["success"] is False
        assert "unable to verify" in result["message"].lower()


class TestAuthorizationEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def mock_credentials(self):
        """Create mock credentials."""
        return HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="token"
        )
    
    @pytest.mark.asyncio
    @patch('app.security.auth.validate_jwt_token')
    @patch('httpx.AsyncClient')
    async def test_unexpected_exception(
        self,
        mock_httpx_client,
        mock_validate_jwt,
        mock_credentials
    ):
        """Test handling of unexpected exceptions."""
        from app.security.auth import validate_user_profile_with_response
        
        mock_validate_jwt.return_value = {"sub": "user-123"}
        
        # Mock unexpected exception
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Unexpected error"))
        mock_httpx_client.return_value.__aenter__.return_value = mock_client
        
        result = await validate_user_profile_with_response(mock_credentials)
        
        assert result["success"] is False
        assert "unexpected error" in result["message"].lower()
    
    def test_empty_token(self):
        """Test handling of empty token."""
        from app.security.auth import validate_jwt_token
        from jose import JWTError
        
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=""
        )
        
        with patch('app.security.auth.jwt.decode', side_effect=JWTError("Invalid token")):
            with pytest.raises(HTTPException):
                validate_jwt_token(credentials)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
