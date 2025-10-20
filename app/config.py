import os
from dotenv import load_dotenv
from datetime import datetime

def load_environment_config():
    """
    Load environment-specific configuration based on ENV variable.
    
    Environments:
    - production: Uses .env.production
    - sit: Uses .env.sit
    - test: Uses .env.test
    - development: Uses .env (default)
    """
    env = os.getenv('APP_ENV', 'development').lower()
    
    # Environment file mapping
    env_files = {
        'production': '.env.production',
        'sit': '.env.sit',
        'test': '.env.test',
        'development': '.env',
    }
    
    # Load the appropriate .env file
    env_file = env_files.get(env, '.env')
    
    if os.path.exists(env_file):
        load_dotenv(env_file)
        print(f"🔧 Loaded configuration from: {env_file}")
    else:
        # Fallback to default .env
        load_dotenv()
        print(f"⚠️  Environment file {env_file} not found, using default .env")
        if env != 'development':
            print(f"💡 Create {env_file} for {env} environment configuration")

# Load environment-specific configuration
load_environment_config()



# AWS Secrets Manager integration (conditional)
USE_SECRETS_MANAGER: bool = os.getenv('USE_SECRETS_MANAGER', 'true').lower() == 'true'

if USE_SECRETS_MANAGER:
    try:
        from app.services.aws_secrets import get_jwt_public_key, get_openai_api_key, get_secrets_manager
        
        secrets_manager = get_secrets_manager()
        
        if secrets_manager.available:
            # Retrieve secrets from AWS Secrets Manager with environment variable fallbacks
            OPENAI_API_KEY = get_openai_api_key(os.getenv("OPENAI_API_KEY"))
            JWT_SECRET_KEY = get_jwt_public_key(os.getenv("JWT_SECRET_KEY"))
            
            print(f"🔐 AWS Secrets Manager: Connected - Using secure secrets from region {secrets_manager.region_name}")
        else:
            # Fall back to environment variables
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
            JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
            print("⚠️  AWS Secrets Manager: Unavailable - Using environment variables")
            
    except ImportError as e:
        print(f"⚠️  AWS Secrets Manager import failed: {e} - Using environment variables")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    except Exception as e:
        print(f"⚠️  AWS Secrets Manager initialization failed: {e} - Using environment variables")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
else:
    # Use environment variables only
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    print("🔧 AWS Secrets Manager: Disabled - Using environment variables only")

# Other configuration variables
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
USE_LLM = bool(OPENAI_API_KEY)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
# default to ap-southeast-1 if not provided
AWS_DEFAULT_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

# DynamoDB table names
DYNAMODB_TRACKER_TABLE_NAME = os.getenv("DYNAMODB_TRACKER_TABLE_NAME")
DYNAMODB_HEADER_TABLE_NAME = os.getenv("DYNAMODB_HEADER_TABLE_NAME")
DYNAMODB_CONVERSATION_CONTEXT_TABLE = os.getenv("DYNAMODB_CONVERSATION_CONTEXT_TABLE", "analytics_conversation_context")

CONVERSATION_CONTEXT_TTL_HOURS = float(os.getenv("CONVERSATION_CONTEXT_TTL_HOURS", "24"))

# AWS Region for services
AWS_REGION = AWS_DEFAULT_REGION

# Admin API configuration
ADMIN_API_BASE_URL = os.getenv("ADMIN_URL")

# AWS SQS Audit Logging Configuration
AUDIT_SQS_QUEUE_URL = os.getenv("AUDIT_SQS_QUEUE_URL")

# JWT configuration
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "RS256")


# Maximum number of assistant->tool cycles before we force-stop the agent
MAX_AGENT_LOOPS = 10

# Application port configuration
APP_PORT = int(os.getenv("APP_PORT", "8091"))


# Parse CORS origins from comma-separated string
cors_origins_str = os.getenv("CORS_ORIGINS", "")
CORS_ORIGINS = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]

CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"

# Parse CORS methods
cors_methods_str = os.getenv("CORS_ALLOW_METHODS", "*")
CORS_ALLOW_METHODS = [method.strip() for method in cors_methods_str.split(",") if method.strip()] if cors_methods_str != "*" else ["*"]

# Parse CORS headers
cors_headers_str = os.getenv("CORS_ALLOW_HEADERS", "*")
CORS_ALLOW_HEADERS = [header.strip() for header in cors_headers_str.split(",") if header.strip()] if cors_headers_str != "*" else ["*"]

CORS_MAX_AGE = int(os.getenv("CORS_MAX_AGE", "600"))
