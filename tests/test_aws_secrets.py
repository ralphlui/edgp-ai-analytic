"""
Unit tests for AWS Secrets Manager integration.

Tests cover:
- Secrets Manager initialization with various credential scenarios
- Secret retrieval with caching
- JSON secret parsing
- JWT public key extraction from environment-specific secrets
- OpenAI API key extraction from environment-specific secrets
- Error handling and fallback mechanisms
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError


class TestSecretsManagerInitialization:
    """Test SecretsManager initialization under various conditions."""
    
    @patch('boto3.client')
    def test_successful_initialization(self, mock_boto_client):
        """Test successful initialization with valid AWS credentials."""
        from app.services.aws_secrets import SecretsManager
        
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        manager = SecretsManager(region_name="us-east-1")
        
        assert manager.available is True
        assert manager.region_name == "us-east-1"
        assert manager._client is not None
        mock_boto_client.assert_called_once_with('secretsmanager', region_name="us-east-1")
    
    @patch.dict('os.environ', {'AWS_REGION': 'eu-west-1'})
    @patch('boto3.client')
    def test_region_from_env_aws_region(self, mock_boto_client):
        """Test region defaults to AWS_REGION environment variable."""
        from app.services.aws_secrets import SecretsManager
        
        manager = SecretsManager()
        
        assert manager.region_name == "eu-west-1"
    
    @patch.dict('os.environ', {'AWS_DEFAULT_REGION': 'ap-south-1'}, clear=True)
    @patch('boto3.client')
    def test_region_from_env_aws_default_region(self, mock_boto_client):
        """Test region defaults to AWS_DEFAULT_REGION if AWS_REGION not set."""
        from app.services.aws_secrets import SecretsManager
        
        manager = SecretsManager()
        
        assert manager.region_name == "ap-south-1"
    
    @patch('boto3.client')
    def test_no_credentials_error(self, mock_boto_client):
        """Test initialization with NoCredentialsError."""
        from app.services.aws_secrets import SecretsManager
        
        mock_boto_client.side_effect = NoCredentialsError()
        
        manager = SecretsManager()
        
        assert manager.available is False
        assert manager._client is None
    
    @patch('boto3.client')
    def test_partial_credentials_error(self, mock_boto_client):
        """Test initialization with PartialCredentialsError."""
        from app.services.aws_secrets import SecretsManager
        
        mock_boto_client.side_effect = PartialCredentialsError(provider='test', cred_var='test_var')
        
        manager = SecretsManager()
        
        assert manager.available is False
        assert manager._client is None
    
    @patch('boto3.client')
    def test_generic_init_error(self, mock_boto_client):
        """Test initialization with generic exception."""
        from app.services.aws_secrets import SecretsManager
        
        mock_boto_client.side_effect = Exception("Connection error")
        
        manager = SecretsManager()
        
        assert manager.available is False
        assert manager._client is None


class TestGetSecret:
    """Test get_secret method."""
    
    @patch('boto3.client')
    def test_get_secret_success_plain_string(self, mock_boto_client):
        """Test successful retrieval of plain string secret."""
        from app.services.aws_secrets import SecretsManager
        
        mock_client = Mock()
        mock_client.get_secret_value.return_value = {
            'SecretString': 'my-secret-value'
        }
        mock_boto_client.return_value = mock_client
        
        manager = SecretsManager()
        # Clear LRU cache
        manager.get_secret.cache_clear()
        
        result = manager.get_secret('test-secret')
        
        assert result == 'my-secret-value'
        assert 'test-secret' in manager._cache
    
    @patch('boto3.client')
    def test_get_secret_json_single_key(self, mock_boto_client):
        """Test retrieval of JSON secret with single key."""
        from app.services.aws_secrets import SecretsManager
        
        mock_client = Mock()
        mock_client.get_secret_value.return_value = {
            'SecretString': json.dumps({'api_key': 'secret-key-123'})
        }
        mock_boto_client.return_value = mock_client
        
        manager = SecretsManager()
        manager.get_secret.cache_clear()
        
        result = manager.get_secret('test-secret')
        
        assert result == 'secret-key-123'
    
    @patch('boto3.client')
    def test_get_secret_json_with_key_field(self, mock_boto_client):
        """Test retrieval of JSON secret with 'key' field."""
        from app.services.aws_secrets import SecretsManager
        
        mock_client = Mock()
        mock_client.get_secret_value.return_value = {
            'SecretString': json.dumps({'key': 'the-actual-key', 'other': 'data'})
        }
        mock_boto_client.return_value = mock_client
        
        manager = SecretsManager()
        manager.get_secret.cache_clear()
        
        result = manager.get_secret('test-secret')
        
        assert result == 'the-actual-key'
    
    @patch('boto3.client')
    def test_get_secret_complex_json(self, mock_boto_client):
        """Test retrieval of complex JSON secret."""
        from app.services.aws_secrets import SecretsManager
        
        secret_data = {'username': 'user', 'password': 'pass', 'host': 'localhost'}
        mock_client = Mock()
        mock_client.get_secret_value.return_value = {
            'SecretString': json.dumps(secret_data)
        }
        mock_boto_client.return_value = mock_client
        
        manager = SecretsManager()
        manager.get_secret.cache_clear()
        
        result = manager.get_secret('test-secret')
        
        # Complex JSON returned as JSON string
        assert json.loads(result) == secret_data
    
    @patch('boto3.client')
    def test_get_secret_resource_not_found(self, mock_boto_client):
        """Test handling of ResourceNotFoundException."""
        from app.services.aws_secrets import SecretsManager
        
        mock_client = Mock()
        error_response = {'Error': {'Code': 'ResourceNotFoundException'}}
        mock_client.get_secret_value.side_effect = ClientError(error_response, 'GetSecretValue')
        mock_boto_client.return_value = mock_client
        
        manager = SecretsManager()
        manager.get_secret.cache_clear()
        
        result = manager.get_secret('missing-secret', fallback_value='fallback')
        
        assert result == 'fallback'
        assert manager.available is True  # Still available
    
    @patch('boto3.client')
    def test_get_secret_access_denied(self, mock_boto_client):
        """Test handling of AccessDeniedException."""
        from app.services.aws_secrets import SecretsManager
        
        mock_client = Mock()
        error_response = {'Error': {'Code': 'AccessDeniedException'}}
        mock_client.get_secret_value.side_effect = ClientError(error_response, 'GetSecretValue')
        mock_boto_client.return_value = mock_client
        
        manager = SecretsManager()
        manager.get_secret.cache_clear()
        
        result = manager.get_secret('forbidden-secret', fallback_value='fallback')
        
        assert result == 'fallback'
        assert manager.available is False  # Mark unavailable
    
    @patch('boto3.client')
    def test_get_secret_when_unavailable(self, mock_boto_client):
        """Test get_secret when manager is unavailable."""
        from app.services.aws_secrets import SecretsManager
        
        mock_boto_client.side_effect = NoCredentialsError()
        
        manager = SecretsManager()
        manager.get_secret.cache_clear()
        
        result = manager.get_secret('any-secret', fallback_value='fallback')
        
        assert result == 'fallback'
    
    @patch('boto3.client')
    def test_get_secret_from_cache(self, mock_boto_client):
        """Test secret retrieval from cache."""
        from app.services.aws_secrets import SecretsManager
        
        mock_client = Mock()
        mock_client.get_secret_value.return_value = {
            'SecretString': 'cached-value'
        }
        mock_boto_client.return_value = mock_client
        
        manager = SecretsManager()
        manager.get_secret.cache_clear()
        
        # First call
        result1 = manager.get_secret('test-secret')
        # Second call should use cache
        result2 = manager.get_secret('test-secret')
        
        assert result1 == result2 == 'cached-value'
        # Should only call AWS once
        mock_client.get_secret_value.assert_called_once()


class TestGetSecretJson:
    """Test get_secret_json method."""
    
    @patch('boto3.client')
    def test_get_secret_json_success(self, mock_boto_client):
        """Test successful JSON secret parsing."""
        from app.services.aws_secrets import SecretsManager
        
        secret_data = {'db_host': 'localhost', 'db_port': 5432}
        mock_client = Mock()
        mock_client.get_secret_value.return_value = {
            'SecretString': json.dumps(secret_data)
        }
        mock_boto_client.return_value = mock_client
        
        manager = SecretsManager()
        manager.get_secret.cache_clear()
        
        result = manager.get_secret_json('test-secret')
        
        assert result == secret_data
    
    @patch('boto3.client')
    def test_get_secret_json_invalid_json(self, mock_boto_client):
        """Test handling of non-JSON secret."""
        from app.services.aws_secrets import SecretsManager
        
        mock_client = Mock()
        mock_client.get_secret_value.return_value = {
            'SecretString': 'not-valid-json'
        }
        mock_boto_client.return_value = mock_client
        
        manager = SecretsManager()
        manager.get_secret.cache_clear()
        
        result = manager.get_secret_json('test-secret', fallback_dict={'default': 'value'})
        
        assert result == {'default': 'value'}
    
    @patch('boto3.client')
    def test_get_secret_json_not_found(self, mock_boto_client):
        """Test JSON secret not found."""
        from app.services.aws_secrets import SecretsManager
        
        mock_client = Mock()
        error_response = {'Error': {'Code': 'ResourceNotFoundException'}}
        mock_client.get_secret_value.side_effect = ClientError(error_response, 'GetSecretValue')
        mock_boto_client.return_value = mock_client
        
        manager = SecretsManager()
        manager.get_secret.cache_clear()
        
        result = manager.get_secret_json('missing-secret', fallback_dict={'default': 'value'})
        
        assert result == {'default': 'value'}


class TestClearCache:
    """Test cache clearing functionality."""
    
    @patch('boto3.client')
    def test_clear_cache(self, mock_boto_client):
        """Test clearing secret cache."""
        from app.services.aws_secrets import SecretsManager
        
        mock_client = Mock()
        mock_client.get_secret_value.return_value = {
            'SecretString': 'secret-value'
        }
        mock_boto_client.return_value = mock_client
        
        manager = SecretsManager()
        manager.get_secret.cache_clear()
        
        # Add to cache
        manager.get_secret('test-secret')
        assert 'test-secret' in manager._cache
        
        # Clear cache
        manager.clear_cache()
        
        assert len(manager._cache) == 0


class TestGetJwtPublicKey:
    """Test JWT public key retrieval."""
    
    @patch.dict('os.environ', {'APP_ENV': 'sit'})
    @patch('app.services.aws_secrets.get_secrets_manager')
    def test_get_jwt_key_sit_environment(self, mock_get_manager):
        """Test JWT key retrieval in SIT environment."""
        from app.services.aws_secrets import get_jwt_public_key
        
        mock_manager = Mock()
        mock_manager.get_secret.return_value = json.dumps({
            'jwt_public_key': 'MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A...'
        })
        mock_get_manager.return_value = mock_manager
        
        result = get_jwt_public_key()
        
        assert result.startswith('-----BEGIN PUBLIC KEY-----')
        assert result.endswith('-----END PUBLIC KEY-----')
        mock_manager.get_secret.assert_called_with('sit/edgp/secret', None)
    
    @patch.dict('os.environ', {'APP_ENV': 'prd'})
    @patch('app.services.aws_secrets.get_secrets_manager')
    def test_get_jwt_key_prod_environment(self, mock_get_manager):
        """Test JWT key retrieval in production environment."""
        from app.services.aws_secrets import get_jwt_public_key
        
        mock_manager = Mock()
        mock_manager.get_secret.return_value = json.dumps({
            'jwt_public_key': '-----BEGIN PUBLIC KEY-----\nMII...\n-----END PUBLIC KEY-----'
        })
        mock_get_manager.return_value = mock_manager
        
        result = get_jwt_public_key()
        
        assert '-----BEGIN PUBLIC KEY-----' in result
        mock_manager.get_secret.assert_called_with('prod/edgp/secret', None)
    
    @patch.dict('os.environ', {'APP_ENV': 'development'})
    @patch('app.services.aws_secrets.get_secrets_manager')
    def test_get_jwt_key_development_environment(self, mock_get_manager):
        """Test JWT key retrieval in development environment."""
        from app.services.aws_secrets import get_jwt_public_key
        
        mock_manager = Mock()
        mock_manager.get_secret.return_value = json.dumps({
            'jwt_public_key': 'test-key'
        })
        mock_get_manager.return_value = mock_manager
        
        result = get_jwt_public_key()
        
        assert result is not None
        mock_manager.get_secret.assert_called_with('/config/edgpv2', None)
    
    @patch.dict('os.environ', {'APP_ENV': 'unknown_env'})
    @patch('app.services.aws_secrets.get_secrets_manager')
    def test_get_jwt_key_unknown_environment(self, mock_get_manager):
        """Test JWT key retrieval with unknown environment defaults to SIT."""
        from app.services.aws_secrets import get_jwt_public_key
        
        mock_manager = Mock()
        mock_manager.get_secret.return_value = None
        mock_get_manager.return_value = mock_manager
        
        get_jwt_public_key(fallback_key='fallback')
        
        # Should default to SIT secret
        assert mock_manager.get_secret.call_args_list[0][0][0] == 'sit/edgp/secret'
    
    @patch('app.services.aws_secrets.get_secrets_manager')
    def test_get_jwt_key_fallback(self, mock_get_manager):
        """Test JWT key fallback when secret not found."""
        from app.services.aws_secrets import get_jwt_public_key
        
        mock_manager = Mock()
        # Return None for both calls (environment-specific and generic)
        mock_manager.get_secret.side_effect = [None, 'fallback-key']
        mock_get_manager.return_value = mock_manager
        
        result = get_jwt_public_key(fallback_key='fallback-key')
        
        assert result == 'fallback-key'


class TestGetOpenAiApiKey:
    """Test OpenAI API key retrieval."""
    
    @patch.dict('os.environ', {'APP_ENV': 'sit'})
    @patch('app.services.aws_secrets.get_secrets_manager')
    def test_get_openai_key_sit_environment(self, mock_get_manager):
        """Test OpenAI key retrieval in SIT environment."""
        from app.services.aws_secrets import get_openai_api_key
        
        mock_manager = Mock()
        mock_manager.get_secret.return_value = json.dumps({
            'ai_agent_api_key': 'sk-test-key-123'
        })
        mock_get_manager.return_value = mock_manager
        
        result = get_openai_api_key()
        
        assert result == 'sk-test-key-123'
        mock_manager.get_secret.assert_called_with('sit/edgp/secret', None)
    
    @patch.dict('os.environ', {'APP_ENV': 'production'})
    @patch('app.services.aws_secrets.get_secrets_manager')
    def test_get_openai_key_prod_environment(self, mock_get_manager):
        """Test OpenAI key retrieval in production environment."""
        from app.services.aws_secrets import get_openai_api_key
        
        mock_manager = Mock()
        mock_manager.get_secret.return_value = json.dumps({
            'ai_agent_api_key': 'sk-prod-key-456'
        })
        mock_get_manager.return_value = mock_manager
        
        result = get_openai_api_key()
        
        assert result == 'sk-prod-key-456'
        mock_manager.get_secret.assert_called_with('prod/edgp/secret', None)
    
    @patch('app.services.aws_secrets.get_secrets_manager')
    def test_get_openai_key_fallback(self, mock_get_manager):
        """Test OpenAI key fallback when secret not found."""
        from app.services.aws_secrets import get_openai_api_key
        
        mock_manager = Mock()
        # Return None for both calls (environment-specific and generic)
        mock_manager.get_secret.side_effect = [None, 'fallback-openai-key']
        mock_get_manager.return_value = mock_manager
        
        result = get_openai_api_key(fallback_key='fallback-openai-key')
        
        assert result == 'fallback-openai-key'
    
    @patch('app.services.aws_secrets.get_secrets_manager')
    def test_get_openai_key_plain_string(self, mock_get_manager):
        """Test OpenAI key retrieval as plain string (not JSON)."""
        from app.services.aws_secrets import get_openai_api_key
        
        mock_manager = Mock()
        mock_manager.get_secret.return_value = 'sk-plain-key'
        mock_get_manager.return_value = mock_manager
        
        result = get_openai_api_key()
        
        assert result == 'sk-plain-key'


class TestGetSecretsManager:
    """Test global secrets manager instance."""
    
    @patch('boto3.client')
    def test_get_secrets_manager_singleton(self, mock_boto_client):
        """Test that get_secrets_manager returns singleton instance."""
        from app.services.aws_secrets import get_secrets_manager
        
        # Reset global instance
        import app.services.aws_secrets as aws_secrets_module
        aws_secrets_module._secrets_manager = None
        
        manager1 = get_secrets_manager()
        manager2 = get_secrets_manager()
        
        assert manager1 is manager2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
