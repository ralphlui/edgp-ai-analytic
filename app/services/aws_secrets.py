"""
AWS Secrets Manager integration for secure key management.
Provides secure retrieval of JWT public keys and API keys from AWS Secrets Manager.
"""
import json
import logging
import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from typing import Optional, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)

class SecretsManager:
    """
    AWS Secrets Manager client for retrieving sensitive configuration.
    Provides caching and fallback mechanisms for production resilience.
    """

    def __init__(self, region_name: str = None):
        self.region_name = region_name or os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1"))
        self._client = None
        self._cache = {}
        self._available = False

        try:
            self._client = boto3.client('secretsmanager', region_name=self.region_name)
            # Mark as available - we'll test connection on first actual secret retrieval
            self._available = True
            logger.info(f"AWS Secrets Manager client initialized for region {self.region_name}")
        except NoCredentialsError:
            logger.warning("AWS credentials not found - falling back to environment variables")
        except PartialCredentialsError:
            logger.warning("Incomplete AWS credentials - falling back to environment variables")
        except Exception as e:
            logger.warning(f"Error initializing AWS Secrets Manager client: {e} - falling back to environment variables")

    @property
    def available(self) -> bool:
        """Check if AWS Secrets Manager is available and properly configured."""
        return self._available and self._client is not None

    @lru_cache(maxsize=10)
    def get_secret(self, secret_name: str, fallback_value: Optional[str] = None) -> Optional[str]:
        """
        Retrieve a secret from AWS Secrets Manager with caching.

        Args:
            secret_name: Name or ARN of the secret
            fallback_value: Value to return if secret retrieval fails

        Returns:
            Secret value as string, or fallback_value if unavailable
        """
        if not self.available:
            logger.debug(f"AWS Secrets Manager unavailable, using fallback for {secret_name}")
            return fallback_value

        # Check cache first
        if secret_name in self._cache:
            return self._cache[secret_name]

        try:
            response = self._client.get_secret_value(SecretId=secret_name)

            if 'SecretString' in response:
                secret_value = response['SecretString']
                # Try to parse as JSON first (for complex secrets)
                try:
                    secret_data = json.loads(secret_value)
                    # If it's a JSON object with a single key, return the value
                    if isinstance(secret_data, dict) and len(secret_data) == 1:
                        secret_value = list(secret_data.values())[0]
                    elif isinstance(secret_data, dict) and 'key' in secret_data:
                        secret_value = secret_data['key']
                    else:
                        # For complex JSON, return as JSON string
                        secret_value = json.dumps(secret_data)
                except json.JSONDecodeError:
                    # Not JSON, use as plain string
                    pass

                self._cache[secret_name] = secret_value
                logger.info(f"Retrieved secret {secret_name} from AWS Secrets Manager")
                return secret_value

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'ResourceNotFoundException':
                logger.warning(f"Secret {secret_name} not found in AWS Secrets Manager")
            elif error_code == 'AccessDeniedException':
                logger.warning(f"Access denied for secret {secret_name} - insufficient permissions")
                # Mark as unavailable for future calls to avoid repeated attempts
                self._available = False
            elif error_code in ['UnauthorizedOperation', 'InvalidUserID.NotFound', 'TokenRefreshRequired']:
                logger.warning(f"AWS credentials issue for {secret_name}: {error_code}")
                self._available = False
            else:
                logger.warning(f"AWS error retrieving secret {secret_name}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error retrieving secret {secret_name}: {e}")
            # For network or other issues, don't mark as unavailable immediately

        logger.debug(f"Using fallback value for {secret_name}")
        return fallback_value

    def get_secret_json(self, secret_name: str, fallback_dict: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve a secret and parse it as JSON.

        Args:
            secret_name: Name or ARN of the secret
            fallback_dict: Dictionary to return if retrieval/parsing fails

        Returns:
            Parsed JSON dictionary, or fallback_dict if unavailable
        """
        secret_value = self.get_secret(secret_name)
        if not secret_value:
            return fallback_dict or {}

        try:
            return json.loads(secret_value)
        except json.JSONDecodeError:
            logger.warning(f"Secret {secret_name} is not valid JSON")
            return fallback_dict or {}

    def clear_cache(self):
        """Clear the secret cache (useful for testing or forced refresh)."""
        self._cache.clear()
        self.get_secret.cache_clear()
        logger.debug("Cleared AWS Secrets Manager cache")

# Global instance for application-wide use
_secrets_manager = None

def get_secrets_manager() -> SecretsManager:
    """Get or create the global AWS Secrets Manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager

def get_jwt_public_key(fallback_key: Optional[str] = None) -> Optional[str]:
    """
    Retrieve JWT public key from AWS Secrets Manager.
    
    Uses environment-based secret naming similar to Spring Boot approach.
    Handles both JSON secrets and plain string secrets.

    Args:
        fallback_key: JWT public key from environment variables

    Returns:
        JWT public key, or fallback if unavailable
    """
    secrets_manager = get_secrets_manager()
    
    # Get environment from environment variable (avoiding circular import)
    app_env = os.getenv('APP_ENV', 'development').lower()

    if app_env in ['sit']:
        secret_name = 'sit/edgp/secret'
    elif app_env in ['prd', 'production']:
        secret_name = 'prod/edgp/secret'
    elif app_env in ['development']:
        secret_name = '/config/edgpv2'
    else:
        logger.warning(f"WARNING: Unknown environment '{app_env}', defaulting to SIT secret")
        secret_name = 'sit/edgp/secret'
    
    logger.info(f"Using secret name '{secret_name}' for JWT public key retrieval")
    
    # Initialize return variable
    jwt_key = None
    
    # Try environment-specific first
    secret_data = secrets_manager.get_secret(secret_name, None)

    # If we got a secret, try to extract JWT key from JSON
    if secret_data:
        try:
            # Parse as JSON and extract jwt_public_key
            secret_json = json.loads(secret_data)
            if isinstance(secret_json, dict) and 'jwt_public_key' in secret_json:
                raw_key = secret_json['jwt_public_key']
                # Ensure proper PEM formatting
                if not raw_key.startswith('-----BEGIN'):
                    # Add PEM headers if missing
                    jwt_key = f"-----BEGIN PUBLIC KEY-----\n{raw_key}\n-----END PUBLIC KEY-----"
                else:
                    jwt_key = raw_key
                logger.info("Extracted and formatted JWT public key from JSON secret")
        except json.JSONDecodeError:
            # Not JSON, assume it's a plain PEM string
            jwt_key = secret_data
    
    # Fallback to simple name if environment-specific failed
    if jwt_key is None:
        jwt_key = secrets_manager.get_secret("jwt-public-key", fallback_key)
    
    return jwt_key

def get_openai_api_key(fallback_key: Optional[str] = None) -> Optional[str]:
    """
    Retrieve OpenAI API key from AWS Secrets Manager.
    
    Uses environment-based secret naming similar to Spring Boot approach.
    Handles both JSON secrets and plain string secrets.

    Args:
        fallback_key: OpenAI API key from environment variables

    Returns:
        OpenAI API key, or fallback if unavailable
    """
    secrets_manager = get_secrets_manager()
    
    # Get environment from environment variable (avoiding circular import)
    app_env = os.getenv('APP_ENV', 'development').lower()

    if app_env in ['sit']:
        secret_name = 'sit/edgp/secret'
    elif app_env in ['prd', 'production']:
        secret_name = 'prod/edgp/secret'
    elif app_env in ['development']:
        secret_name = '/config/edgpv2'
    else:
        logger.warning(f" WARNING: Unknown environment '{app_env}', defaulting to SIT secret")
        secret_name = 'sit/edgp/secret'
    
    logger.info(f"Using secret name '{secret_name}' for OpenAI API key retrieval")
    
    # Initialize return variable
    openai_key = None
    
    # Try environment-specific first
    secret_data = secrets_manager.get_secret(secret_name, None)
    
    # If we got a secret, try to extract OpenAI key from JSON
    if secret_data:
        try:
            # Parse as JSON and extract openai_api_key (try multiple key names)
            secret_json = json.loads(secret_data)
            if isinstance(secret_json, dict):
                # Try different possible key names
                for key_name in ['openai_api_key', 'OPENAI_API_KEY', 'openai_key', 'api_key']:
                    if key_name in secret_json:
                        openai_key = secret_json[key_name]
                        logger.info(f"Extracted OpenAI API key from JSON secret (key: {key_name})")
                        break
        except json.JSONDecodeError:
            # Not JSON, assume it's a plain API key string
            openai_key = secret_data
    
    # Fallback to simple name if environment-specific failed
    if openai_key is None:
        openai_key = secrets_manager.get_secret("openai-api-key", fallback_key)

    return openai_key
