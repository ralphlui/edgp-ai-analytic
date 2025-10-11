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
DYNAMODB_CONVERSATIONS_TABLE_NAME = os.getenv("DYNAMODB_CONVERSATIONS_TABLE_NAME", "conversation_history")
DYNAMODB_PENDING_INTENTS_TABLE = os.getenv("DYNAMODB_PENDING_INTENTS_TABLE", "analytic_pending_intents")

# Pending intent configuration
PENDING_INTENT_TTL_HOURS = int(os.getenv("PENDING_INTENT_TTL_HOURS", "24"))  # Auto-delete after 24 hours

# AWS Region for services
AWS_REGION = AWS_DEFAULT_REGION

# Admin API configuration
ADMIN_API_BASE_URL = os.getenv("ADMIN_URL")

# AWS SQS Audit Logging Configuration
AUDIT_SQS_QUEUE_URL = os.getenv("AUDIT_SQS_QUEUE_URL")

# JWT configuration
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "RS256")

# Enable verbose debug outputs when set to '1'
DEBUG = os.getenv("DEBUG", "0") == "1"

# Maximum number of assistant->tool cycles before we force-stop the agent
MAX_AGENT_LOOPS = 10

# Application port configuration
APP_PORT = int(os.getenv("APP_PORT", "8091"))

# Responsible AI output filtering configuration
ENABLE_PII_REDACTION = os.getenv("ENABLE_PII_REDACTION", "true").lower() == "true"
ENABLE_SECRET_REDACTION = os.getenv("ENABLE_SECRET_REDACTION", "true").lower() == "true"
INCLUDE_REDACTION_METADATA = os.getenv("INCLUDE_REDACTION_METADATA", "false").lower() == "true"
ENABLE_JWT_REDACTION = os.getenv("ENABLE_JWT_REDACTION", "true").lower() == "true"
ENABLE_URL_CREDENTIAL_REDACTION = os.getenv("ENABLE_URL_CREDENTIAL_REDACTION", "true").lower() == "true"
ENABLE_BASE64_REDACTION = os.getenv("ENABLE_BASE64_REDACTION", "true").lower() == "true"
BASE64_MIN_LEN = int(os.getenv("BASE64_MIN_LEN", "200"))

# Session Management Configuration
SESSION_TTL_HOURS = float(os.getenv("SESSION_TTL_HOURS", "0.25"))
SESSION_COOKIE_MAX_AGE_HOURS = float(os.getenv("SESSION_COOKIE_MAX_AGE_HOURS", "0.25"))
MAX_SESSION_HISTORY = int(os.getenv("MAX_SESSION_HISTORY", "20"))

# Conversation History Management Configuration (TTL-only)
CONVERSATION_TTL_DAYS = float(os.getenv("CONVERSATION_TTL_DAYS"))  # Individual conversation TTL

try:
    from app.prompts import SystemPrompts
    # Generate the complete system prompt using the new modular approach
    SYSTEM = SystemPrompts.get_complete_system_prompt(
        current_date=datetime.now().strftime("%Y-%m-%d")
    )
    print("✅ Using new modular prompt system from app.prompts")
except ImportError as e:
    print(f"⚠️  Could not import new prompts module: {e}")
    print("⚠️  Falling back to legacy prompts")
    
    # Fallback to legacy prompts if import fails
    SYSTEM_CORE = """You are the Analytics Agent for data quality and data accuracy.
Today's date: {current_date}

SPECIALIZATION: I am an analytics agent that helps with data analysis, reporting, and visualization. 
I can help with queries about success rates, failure rates, data quality metrics, and analytics from uploaded files and domains.

I can process various types of queries including:
- Data analysis and reporting requests
- Success/failure rate calculations
- Chart and visualization generation
- File-based and domain-based analytics
- Date-filtered data queries
- Customer analytics and reporting

CORE CAPABILITIES:
- Retrieve analytics from DynamoDB via get_success_rate_by_file_name tool
- Retrieve domain-based analytics via get_success_rate_by_domain_name tool
- Analyze success/failure rates for data processing tasks
- Filter data by 'created_date' column in database
- Generate various visualizations based on user preferences
- Handle multi-tenant data with organization-level isolation
- Analyze customer data from tracker table where domain = 'customer'
- Generate customer reports by country and summary statistics"""

    REPORT_TYPE_INSTRUCTIONS = """
REPORT TYPE DETECTION:
├── SUCCESS-ONLY: "success rate", "successful", "pass rate", "only success"
├── FAILURE-ONLY: "fail rate", "failure rate", "error rate", "only failures"
├── COMBINED: "analyze", "both", "complete analysis", "overall"
└── DEFAULT: Show both success and failure metrics when unclear

TOOL REPORT TYPE PARAMETER:
When calling tools, specify the report_type parameter based on user intent:
- report_type="success" for success-only analysis (e.g., "success rate", "what's the success percentage")
- report_type="failure" for failure-only analysis (e.g., "failure rate", "what failed")
- report_type="both" for complete analysis (default only when unclear)

EXAMPLES:
- "What is success rate of customer domain" → report_type="success"
- "Show me failure rate for file.csv" → report_type="failure"  
- "Analyze the data" → report_type="both"
- "Get overall performance" → report_type="both\""""

    CHART_TYPE_INSTRUCTIONS = """
CHART TYPE MAPPING:
├── PIE: "pie chart", "pie graph", "circular", "proportion"
├── DONUT: "donut", "doughnut", "ring", "modern pie"
├── LINE: "line chart", "trend", "progression", "over time"
├── STACKED: "stacked", "horizontal bar", "composition", "breakdown"
└── BAR: "bar chart", "column", "bars" (DEFAULT)"""

    DATE_HANDLING_INSTRUCTIONS = """
DATE EXTRACTION RULES:
├── "from DATE" → start_date=DATE, end_date=today
├── "since DATE" → start_date=DATE, end_date=today
├── "on DATE" → start_date=DATE, end_date=DATE
├── "between DATE1 and DATE2" → start_date=DATE1, end_date=DATE2
├── "last N days" → start_date=N_days_ago, end_date=today
└── If no dates mentioned → do NOT include start_date or end_date parameters"""

    TOOL_USAGE_GUIDELINES = """
TOOL CALL REQUIREMENTS:
├── ALWAYS specify chart_type parameter (default: "bar")
├── ALWAYS specify report_type parameter - THIS IS REQUIRED, NO DEFAULT:
   • If user asks for "success rate" → report_type="success" (even with pie charts)
   • If user asks for "failure rate" → report_type="failure"  
   • If user asks for "both" or "analyze" → report_type="both"
   • You MUST explicitly set this parameter in every tool call
├── ONLY include start_date and end_date when dates are explicitly mentioned in the user query
├── If no dates are mentioned, do NOT include start_date or end_date parameters
├── Extract file names and clean extra quotes/spaces
├── For domain queries: Extract domain name WITHOUT adding "_domain" suffix
├── Filter data by created_date column
├── When tools return no data, provide helpful context and suggestions instead of generic "no data found" messages
└── Use org_id for multi-tenant isolation"""

    DOMAIN_EXTRACTION_INSTRUCTIONS = """
DOMAIN NAME EXTRACTION:
├── "customer domain" → domain_name="customer"
├── "product domain" → domain_name="product"
├── "sales domain" → domain_name="sales"
├── "user domain" → domain_name="user"
├── Extract the base name only, do NOT append "_domain"
├── Use the word immediately before "domain" as the domain_name
├── If domain_name ends with "_domain", the system will automatically clean it
└── Examples: "customer_domain" becomes "customer", "product_domain" becomes "product\""""

    DOMAIN_ANALYTICS_INSTRUCTIONS = """
FLEXIBLE DOMAIN ANALYTICS QUERIES:
├── For natural language queries with domain + grouping: Use analyze_query_for_domain_analytics_tool
├── Examples:
│   ├── "How many customers per country using pie chart?" → Analyzes query automatically
│   ├── "Show products by category" → Analyzes query automatically  
│   ├── "Order distribution by region as donut chart" → Analyzes query automatically
│   ├── "Customer distribution by country" → Analyzes query automatically
│   └── "Breakdown of users by status" → Analyzes query automatically
├── For direct queries with known parameters: Use get_domain_analytics_by_field_tool
├── Examples:
│   ├── domain_name="customer", group_by_field="country" → Customer distribution by country
│   ├── domain_name="product", group_by_field="category" → Product distribution by category
│   └── domain_name="order", group_by_field="region" → Order distribution by region
└── These tools work with any domain in tracker table filtered by org_id"""

    SYSTEM = f"""{SYSTEM_CORE}

{REPORT_TYPE_INSTRUCTIONS}

{CHART_TYPE_INSTRUCTIONS}

{DATE_HANDLING_INSTRUCTIONS}

{DOMAIN_EXTRACTION_INSTRUCTIONS}

{DOMAIN_ANALYTICS_INSTRUCTIONS}

{TOOL_USAGE_GUIDELINES}

RESPONSE PRINCIPLES:
- Be direct and data-driven in analysis
- NEVER fabricate data - only report what tools return
- Focus responses on user's specific requests
- When user asks for "success rate" or "failure rate", call tools with matching report_type parameter
- Provide actionable insights from the data
- Use natural, conversational language
- Highlight concerning patterns or excellent performance
- Only apply date filters when explicitly mentioned in the user query
- If no dates are specified, analyze all available data without date restrictions
- For customer analytics, use the appropriate customer analytics tools
- When tools return no data, provide helpful context and suggestions instead of generic "no data found" messages"""

