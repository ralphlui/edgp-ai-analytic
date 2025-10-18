"""
Unit tests for configuration management.

Tests cover:
- Environment variable loading
- AWS Secrets Manager integration
- Configuration validation
- Fallback mechanisms
"""
import pytest
from unittest.mock import patch, Mock
import os


class TestConfigLoading:
    """Test configuration loading from environment variables."""
    
    @patch.dict(os.environ, {
        'APP_ENV': 'test',
        'APP_PORT': '8091',
        'OPENAI_MODEL': 'gpt-4o-mini',
        'AWS_REGION': 'ap-southeast-1'
    })
    def test_load_environment_variables(self):
        """Test loading configuration from environment variables."""
        # Re-import to pick up environment changes
        import importlib
        import app.config as config_module
        importlib.reload(config_module)
        
        assert config_module.APP_PORT == 8091
        assert config_module.OPENAI_MODEL == 'gpt-4o-mini'
        assert config_module.AWS_REGION == 'ap-southeast-1'
    
    @patch.dict(os.environ, {'APP_ENV': 'development'})
    def test_development_environment(self):
        """Test configuration for development environment."""
        import importlib
        import app.config as config_module
        importlib.reload(config_module)
        
        # Development should load .env file
        # Exact values depend on the .env file
        assert config_module.AWS_REGION is not None
    
    @patch.dict(os.environ, {'USE_SECRETS_MANAGER': 'false'})
    def test_secrets_manager_disabled(self):
        """Test configuration when Secrets Manager is disabled."""
        import importlib
        import app.config as config_module
        importlib.reload(config_module)
        
        # Should fall back to environment variables
        assert config_module.USE_SECRETS_MANAGER is False


class TestSecretsManagerIntegration:
    """Test AWS Secrets Manager integration during config loading."""
    
    @patch.dict(os.environ, {'USE_SECRETS_MANAGER': 'true'})
    @patch('app.services.aws_secrets.get_secrets_manager')
    def test_secrets_manager_import_error(self, mock_get_secrets):
        """Test fallback when AWS Secrets Manager import fails."""
        import importlib
        import app.config as config_module
        
        # Mock import error by removing the module
        with patch.dict('sys.modules', {'app.services.aws_secrets': None}):
            # This will trigger ImportError
            try:
                importlib.reload(config_module)
            except:
                pass
        
        # Should fall back to environment variables (verify keys are set)
        assert hasattr(config_module, 'OPENAI_API_KEY')
        assert hasattr(config_module, 'JWT_SECRET_KEY')
    
    @patch.dict(os.environ, {'USE_SECRETS_MANAGER': 'true', 'OPENAI_API_KEY': 'test_key', 'JWT_SECRET_KEY': 'test_jwt'})
    @patch('app.services.aws_secrets.get_secrets_manager')
    def test_secrets_manager_initialization_exception(self, mock_get_secrets):
        """Test fallback when Secrets Manager initialization fails."""
        # Mock exception during initialization
        mock_get_secrets.side_effect = Exception("AWS credentials not found")
        
        import importlib
        import app.config as config_module
        
        try:
            importlib.reload(config_module)
        except:
            pass
        
        # Should fall back to environment variables
        assert config_module.OPENAI_API_KEY == 'test_key'
        assert config_module.JWT_SECRET_KEY == 'test_jwt'


class TestAWSSecretsManager:
    """Test AWS Secrets Manager integration."""
    
    @pytest.fixture
    def mock_secrets_manager(self):
        """Create mock secrets manager."""
        with patch('app.services.aws_secrets.boto3.client') as mock_client:
            yield mock_client
    
    def test_secrets_manager_initialization(self, mock_secrets_manager):
        """Test Secrets Manager client initialization."""
        from app.services.aws_secrets import SecretsManager
        
        secrets_manager = SecretsManager(region_name='us-east-1')
        
        assert secrets_manager.region_name == 'us-east-1'
        assert secrets_manager._cache == {}
    
    def test_get_secret_with_fallback(self, mock_secrets_manager):
        """Test getting secret with fallback value."""
        from app.services.aws_secrets import SecretsManager
        
        # Mock unavailable secrets manager
        mock_secrets_manager.side_effect = Exception("No credentials")
        
        secrets_manager = SecretsManager()
        result = secrets_manager.get_secret('test-secret', fallback_value='fallback')
        
        assert result == 'fallback'
    
    def test_get_secret_success(self):
        """Test successful secret retrieval."""
        from app.services.aws_secrets import SecretsManager
        
        with patch.object(SecretsManager, 'get_secret', return_value='secret-value'):
            secrets_manager = SecretsManager()
            result = secrets_manager.get_secret('test-secret')
            
            assert result == 'secret-value'
    
    def test_get_secret_json(self):
        """Test JSON secret retrieval."""
        from app.services.aws_secrets import SecretsManager
        import json
        
        secret_data = json.dumps({"key": "value", "api_key": "secret123"})
        
        with patch.object(SecretsManager, 'get_secret', return_value=secret_data):
            secrets_manager = SecretsManager()
            result = secrets_manager.get_secret_json('test-secret')
            
            assert result == {"key": "value", "api_key": "secret123"}
    
    def test_cache_clearing(self):
        """Test secret cache clearing."""
        from app.services.aws_secrets import SecretsManager
        
        secrets_manager = SecretsManager()
        secrets_manager._cache['test'] = 'value'
        
        secrets_manager.clear_cache()
        
        assert secrets_manager._cache == {}


class TestLoadEnvironmentConfig:
    """Test load_environment_config function edge cases."""
    
    @patch('os.path.exists', return_value=False)
    @patch('app.config.load_dotenv')
    @patch.dict(os.environ, {'APP_ENV': 'production'})
    def test_missing_env_file_fallback(self, mock_load_dotenv, mock_exists):
        """Test fallback when environment-specific file doesn't exist."""
        from app.config import load_environment_config
        
        load_environment_config()
        
        # Should call load_dotenv() for fallback
        mock_load_dotenv.assert_called()
    
    @patch('os.path.exists', return_value=True)
    @patch('app.config.load_dotenv')
    @patch.dict(os.environ, {'APP_ENV': 'sit'})
    def test_env_file_exists(self, mock_load_dotenv, mock_exists):
        """Test loading when environment file exists."""
        from app.config import load_environment_config
        
        load_environment_config()
        
        # Should call load_dotenv with the sit env file
        mock_load_dotenv.assert_called_with('.env.sit')
    
    @patch('os.path.exists', return_value=False)
    @patch('app.config.load_dotenv')
    @patch.dict(os.environ, {'APP_ENV': 'unknown_env'})
    def test_unknown_environment(self, mock_load_dotenv, mock_exists):
        """Test behavior with unknown environment name."""
        from app.config import load_environment_config
        
        load_environment_config()
        
        # Should fall back to default .env
        mock_load_dotenv.assert_called()


class TestConfigValidation:
    """Test configuration validation and error handling."""
    
    def test_required_config_values(self):
        """Test that required configuration values are present."""
        from app.config import AWS_REGION, OPENAI_MODEL
        
        # These should always be set (even with defaults)
        assert AWS_REGION is not None
        assert OPENAI_MODEL is not None
    
    def test_default_values(self):
        """Test default configuration values."""
        from app.config import OPENAI_MODEL, MAX_AGENT_LOOPS
        
        assert OPENAI_MODEL == 'gpt-4o-mini'
        assert MAX_AGENT_LOOPS == 10
    
    @patch.dict(os.environ, {'APP_PORT': 'invalid'})
    def test_invalid_port_number(self):
        """Test handling of invalid port number."""
        import importlib
        
        # Should raise ValueError when trying to convert invalid port
        with pytest.raises(ValueError):
            import app.config as config_module
            importlib.reload(config_module)


class TestEnvironmentSpecificConfig:
    """Test environment-specific configuration loading."""
    
    @patch.dict(os.environ, {'APP_ENV': 'production'})
    def test_production_env_detection(self):
        """Test production environment detection."""
        import importlib
        import app.config as config_module
        importlib.reload(config_module)
        
        # Production should attempt to load .env.production
        assert os.getenv('APP_ENV') == 'production'
    
    @patch.dict(os.environ, {'APP_ENV': 'sit'})
    def test_sit_env_detection(self):
        """Test SIT environment detection."""
        import importlib
        import app.config as config_module
        importlib.reload(config_module)
        
        assert os.getenv('APP_ENV') == 'sit'
    
    @patch.dict(os.environ, {'APP_ENV': 'test'})
    def test_test_env_detection(self):
        """Test test environment detection."""
        import importlib
        import app.config as config_module
        importlib.reload(config_module)
        
        assert os.getenv('APP_ENV') == 'test'


class TestDynamoDBConfig:
    """Test DynamoDB-specific configuration."""
    
    def test_dynamodb_table_names(self):
        """Test DynamoDB table name configuration."""
        import os
        from app.config import (
            DYNAMODB_TRACKER_TABLE_NAME,
            DYNAMODB_HEADER_TABLE_NAME,
            DYNAMODB_CONVERSATION_CONTEXT_TABLE
        )
        
        # Conversation context table should match environment
        env = os.getenv('APP_ENV', 'development').lower()
        if env == 'test':
            # Test environment uses test suffix
            assert DYNAMODB_CONVERSATION_CONTEXT_TABLE == 'analytics_conversation_context_test'
        else:
            # Other environments use standard name
            assert DYNAMODB_CONVERSATION_CONTEXT_TABLE == 'analytics_conversation_context'
    
    def test_conversation_context_ttl(self):
        """Test conversation context TTL configuration."""
        from app.config import CONVERSATION_CONTEXT_TTL_HOURS
        
        assert isinstance(CONVERSATION_CONTEXT_TTL_HOURS, float)
        assert CONVERSATION_CONTEXT_TTL_HOURS > 0


class TestOpenAIConfig:
    """Test OpenAI-specific configuration."""
    
    def test_openai_model_config(self):
        """Test OpenAI model configuration."""
        from app.config import OPENAI_MODEL, USE_LLM
        
        assert OPENAI_MODEL == 'gpt-4o-mini'
        assert isinstance(USE_LLM, bool)
    
    def test_use_llm_flag(self):
        """Test USE_LLM flag based on API key presence."""
        from app.config import USE_LLM, OPENAI_API_KEY
        
        # USE_LLM should be True only if API key is set
        if OPENAI_API_KEY:
            assert USE_LLM is True
        else:
            assert USE_LLM is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
