# 🤖 EDGP AI Analytics Agent

**Intelligent Data Analytics API powered by LangGraph and OpenAI GPT models**

A scalable, multi-tenant analytics platform that provides AI-driven insights from your data with natural language processing capabilities, advanced visualization, comprehensive session management, and enterprise-grade security.

> **📢 Latest Update (October 2025)**: Added 15-layer secure prompt template system, achieved 83% test coverage (463 tests), fixed intent classification bugs, and implemented multi-turn conversation support with database context persistence.

[![Tests](https://img.shields.io/badge/tests-463%20passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-83%25-green)]()
[![Python](https://img.shields.io/badge/python-3.13%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

---

## 🚀 Features

### 🧠 **AI-Powered Analytics**
- **Natural Language Queries**: Ask questions in plain English about your data
- **LangGraph Workflows**: Sophisticated AI agent workflows with loop protection
- **GPT-4 Integration**: Powered by OpenAI's latest models for intelligent data analysis
- **Smart Report Classification**: Automatically detects if you want success, failure, or combined analytics
- **Multi-Turn Conversations**: Context-aware sessions with database persistence for incomplete queries
- **Secure Prompt Templates**: 15-layer defense-in-depth security architecture protecting against prompt injection

### 📊 **Advanced Visualization**
- **Chart Types**: Bar charts
- **Dynamic Chart Generation**: Matplotlib and Seaborn integration
- **Base64 Image Output**: Ready for web integration

### 🏢 **Enterprise-Ready Architecture**
- **JWT Authentication**: Secure token-based authentication
- **Session Management**: Stateful conversation tracking
- **AWS DynamoDB**: Scalable cloud database integration
- **FastAPI Framework**: High-performance async API


---

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.13+
- **AI/ML**: LangChain, LangGraph, OpenAI GPT-4o-mini
- **Database**: AWS DynamoDB
- **Visualization**: Matplotlib, Seaborn
- **Authentication**: JWT, Python-JOSE
- **Testing**: Pytest (463 tests, 83% coverage)
- **Security**: Secure Prompt Templates with SHA-256 integrity verification
- **Deployment**: Uvicorn ASGI server

---

## 📋 Prerequisites

- **Python 3.13+**
- **OpenAI API Key** 
- **AWS Account** (DynamoDB access)
- **Virtual Environment** (recommended)

---

## ⚡ Quick Start

### 1. **Clone & Setup**
```bash
git clone https://github.com/ralphlui/edgp-ai-analytic.git
cd edgp-ai-analytic
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. **Environment Configuration**
Create `.env` file in project root:
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=ap-southeast-1

# JWT Configuration  
JWT_SECRET_KEY=your_jwt_secret_key
JWT_ALGORITHM=RS256

# Admin API (Optional)
ADMIN_API_BASE_URL=https://your-admin-api.com

# Debug Mode
DEBUG=1
```

### 3. **Run the Server**
```bash
# Development server
uvicorn app.analytic_api:app --reload --host 0.0.0.0 --port 8091

# Production server  
uvicorn app.analytic_api:app --host 0.0.0.0 --port 8091
```

### 4. **Test the API**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Show me customer success rate by country as pie chart"
  }'
```

---

## 🔧 API Documentation

### **Main Endpoint**
- **POST** `/query` - Process natural language analytics queries

### **Request Format**
```json
{
  "prompt": "Show customer analytics by country as pie chart"
}
```

### **Response Format**
```json
{
  "success": true,
  "message": "Analysis complete. Found 150 customers across 5 countries.",
  "chart_image": "base64_encoded_chart_image"
}
```

---

## 📝 Example Queries

### **Simple Queries**
```bash
# Success/Failure Analysis
"What's the failure rate for file.csv?"
"compare success rate of product.csv and customer.csv"

# Date-Filtered Queries
"Customer data from 2024-01-01 to 2024-12-31" 
"Show data from last 7 days"
"Analytics since January 2024"
```

### **Multi-Turn Conversations** 🆕
The system supports natural multi-turn conversations with context persistence:

```bash
# Turn 1: User provides partial information
User: "success rate report"
Bot: "I understand you want a success rate report. Which file or domain would you like to analyze?"

# Turn 2: User provides missing information
User: "customer domain"
Bot: "The customer domain has an 80% success rate (1,200 successful requests out of 1,500 total)."

# Turn 3: Continue the conversation
User: "show me failure rate"
Bot: "The customer domain has a 20% failure rate (300 failures out of 1,500 total requests)."
```

**How it works:**
- System saves partial context (domain/file) to DynamoDB
- Retrieves context in subsequent turns
- Combines old + new information automatically
- No need to repeat yourself!

---

## 🧪 Testing

### **Run Test Suite**
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=app --cov-report=term-missing

# Generate HTML coverage report
python -m pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html
```

### **Current Test Coverage**
- **463 Total Tests** ✅
- **83% Overall Coverage** 🎯
- **Well-tested areas**: 
  - Query Understanding Agent (100%)
  - Analytics Repository (96%)
  - Query Context Service (95%)
  - Query Processor (91%)
  - Domain Analytics Tools (86%+)
  - Complex Query Executor (78%)
  - Simple Query Executor (74%)
- **Recent Improvements**: +7pp coverage increase through comprehensive test suite expansion

### **Test Categories**
- **Unit Tests**: Core business logic and utilities
- **Integration Tests**: End-to-end workflow testing  
- **Agent Tests**: Query understanding, planning, execution
- **Auth Tests**: JWT and session management
- **Repository Tests**: Analytics data access patterns
- **Service Tests**: Query processing and context management
- **Security Tests**: Prompt template validation and sanitization

---

## 📁 Project Structure

```
edgp-ai-analytic/
├── app/
│   ├── prompts/              # 🔐 Secure prompt templates (NEW!)
│   │   ├── base_prompt.py            # Security foundation (15 layers)
│   │   ├── query_understanding_prompts.py  # Intent extraction
│   │   ├── planner_prompts.py        # Execution planning
│   │   ├── simple_executor_prompts.py    # Simple query tools
│   │   └── complex_executor_prompts.py   # Complex comparison tools
│   ├── orchestration/        # AI agent orchestration
│   │   ├── query_understanding_agent.py   # Intent detection (100% coverage)
│   │   ├── planner_agent.py          # Plan generation
│   │   ├── simple_query_executor.py      # Simple workflows (74% coverage)
│   │   └── complex_query_executor.py     # Complex workflows (78% coverage)
│   ├── services/             # External service integrations
│   │   ├── query_processor.py        # Main orchestrator (91% coverage)
│   │   ├── query_context_service.py      # Session persistence (95% coverage)
│   │   ├── aws_secrets.py            # AWS Secrets Manager integration
│   │   ├── audit_sqs_service.py      # SQS audit logging
│   │   └── chart_service.py          # Visualization generation
│   ├── repositories/         # Data access layer
│   │   └── analytics_repository.py       # Analytics queries (96% coverage)
│   ├── tools/                # LangGraph tools
│   │   └── analytics_tools.py         # Domain analytics & rate analysis tools
│   ├── security/             # Security utilities
│   │   ├── auth.py                # JWT authentication
│   │   ├── prompt_validator.py        # Input validation
│   │   └── pii_redactor.py           # PII protection
│   ├── analytic_api.py       # FastAPI application
│   ├── config.py            # Configuration management
│   └── logging_config.py    # Logging setup
├── tests/                   # Test suite (463 tests, 83% coverage)
├── htmlcov/                # Coverage reports
├── requirements.txt        # Dependencies
├── .env                   # Environment variables (create this)
└── README.md             # This file
```

---

## 🔍 Key Components

### **🔐 Secure Prompt Templates** (`app/prompts/`)
**15-Layer Defense-in-Depth Security Architecture**

Comprehensive security system protecting against prompt injection, jailbreaking, and information leakage:

#### **Security Layers**
1. **Template Integrity**: SHA-256 hash verification prevents template tampering
2. **Input Sanitization**: 7 sublayers of validation
   - Length validation (prevents DoS attacks)
   - Unicode normalization (NFKC)
   - Control character filtering
   - Newline normalization
   - Injection pattern detection
   - Suspicious pattern blocking
   - Content validation
3. **Structural Isolation**: XML boundaries separate user input from system prompts
4. **Proactive Leakage Prevention**: Embedded rules prevent prompt disclosure
5. **Response Format Validation**: Ensures valid output structure
6. **Reactive Leakage Detection**: Pattern-based scanning of responses
7. **Response Schema Validation**: Strict allowlists and type checking
8. **Security Logging**: Comprehensive audit trails

#### **Secure Templates**
- **`base_prompt.py`**: Abstract foundation with core security methods
- **`query_understanding_prompts.py`**: Intent/slot extraction with validation
- **`planner_prompts.py`**: Execution plan generation for complex queries
- **`simple_executor_prompts.py`**: Tool selection + response formatting
- **`complex_executor_prompts.py`**: Multi-target comparison handling

#### **Recent Fixes** (Oct 2025)
- ✅ Fixed "success rate report" classification bug (was returning `general_query` instead of `success_rate`)
- ✅ Corrected typo in success rate example clarification message
- ✅ Strengthened intent detection rules with explicit MUST-classify guidelines
- ✅ Fixed LangChain template compatibility issue in planner agent
- ✅ Updated all test mocks for direct message construction pattern

### **Analytics Engine** (`query_processor.py`)
- Natural language processing for analytics queries
- Smart report type classification (success/failure/both)
- Conversation context extraction and enhancement
- Multi-step workflow coordination
- 91% test coverage

### **Query Understanding Agent** (`query_understanding_agent.py`)
- Intent detection (success_rate, failure_rate, comparison, etc.)
- Slot extraction (domain_name, file_name, date ranges)
- Completeness validation
- Smart clarification for incomplete queries
- 100% test coverage ✅

### **Planner Agent** (`planner_agent.py`)  
- Execution plan generation for complex queries
- Multi-target comparison orchestration
- Dependency tracking between steps
- JSON response validation
- Direct LLM message construction (no template layer)

### **Simple Query Executor** (`simple_query_executor.py`)
- Single-target analytics execution
- Tool selection based on intent
- Response formatting with chart integration
- 74% test coverage

### **Complex Query Executor** (`complex_query_executor.py`)
- Multi-target comparison workflows
- Parallel data retrieval
- Result aggregation and comparison
- 78% test coverage

### **Chart Generator** (`chart_service.py`)
- Dynamic visualization creation
- Support for bar charts
- Matplotlib and Seaborn integration
- Base64 encoding for web delivery

### **Analytics Tools** (`analytics_tools.py`)
- Domain analytics (customer, product, sales)
- Success/failure rate analysis
- Country/category/region grouping
- Flexible query patterns for any data domain

---

## 🎯 Secure Prompt Template System

The project includes a sophisticated secure prompt template system with 15-layer defense-in-depth security:

### Features

- **🔐 15-Layer Security**: Comprehensive protection against prompt injection and jailbreaking
- **📋 Structured Templates**: Organized by purpose (understanding, planning, execution)
- **🛡️ Input Sanitization**: 7 sublayers of validation and normalization
- **✅ Template Integrity**: SHA-256 hash verification prevents tampering
- **🔍 Leakage Detection**: Proactive and reactive prompt leakage prevention
- **📊 Schema Validation**: Strict allowlists and type checking
- **🔖 Versioning**: Track and manage prompt versions over time
- **🛠️ Tool Guidance**: Intelligent tool selection and parameter extraction

### Security Architecture

```python
# All prompts inherit from SecurePromptTemplate
class QueryUnderstandingPrompt(SecurePromptTemplate):
    # 1. Template Integrity
    TEMPLATE_HASH = hashlib.sha256(TEMPLATE.encode('utf-8')).hexdigest()
    
    # 2. Input Sanitization (7 layers)
    def _sanitize_user_input(self, value):
        # Length, Unicode, control chars, injection patterns, etc.
        
    # 3. Structural Isolation
    def build_user_section(self, section_id, user_input):
        # XML boundaries separate user input from system
        
    # 4-8. Format validation, leakage detection, schema validation, logging
```

### Usage

```python
# Initialize secure prompt template
from app.prompts.query_understanding_prompts import QueryUnderstandingPrompt

prompt = QueryUnderstandingPrompt()

# Get system prompt with leakage prevention
system_prompt = prompt.get_system_prompt()

# Format user message with sanitization
user_message = prompt.format_user_message(
    user_query="show me success rate for customer",
    conversation_history=[]
)

# Validate response
result = prompt.validate_response_schema(llm_response)
```

### Testing

All prompt templates are thoroughly tested:
- **Template integrity verification**
- **Input sanitization edge cases**
- **Schema validation with invalid inputs**
- **Leakage detection with malicious inputs**
- **Integration with LLM agents**

```bash
# Run prompt security tests
python -m pytest tests/test_*_prompts.py -v
```

---

## 🐛 Troubleshooting

### **Recent Bug Fixes** (October 2025)

**1. Success Rate Intent Classification** ✅
- **Issue**: "success rate report" was incorrectly classified as `general_query` instead of `success_rate`
- **Root Cause**: Missing direct example in prompt template, LLM confusion between metric keywords
- **Fix**: Added explicit "success rate report" example, strengthened intent detection rules
- **Impact**: All success/failure rate queries now correctly classified

**2. LangChain Template Compatibility** ✅
- **Issue**: `ChatPromptTemplate` expecting template variables that don't exist in formatted strings
- **Root Cause**: Mixing two systems - secure templates (plain strings) + LangChain templates (variable placeholders)
- **Fix**: Replaced `ChatPromptTemplate.from_messages()` with direct `SystemMessage`/`HumanMessage` construction
- **Impact**: All 463 tests passing, cleaner integration pattern

**3. Out-of-Scope Query Detection** ✅
- **Issue**: Valid queries like "analyze product domain" were rejected as out-of-scope
- **Root Cause**: `if result.clarification_needed is not None:` treated ALL clarifications as errors
- **Fix**: Changed to `if result.intent == "out_of_scope":` for precise checking
- **Impact**: Multi-turn conversations work correctly, partial queries saved to database

### **Common Issues**

**1. OpenAI API Key Issues**
```bash
# Check your API key is set
echo $OPENAI_API_KEY
# Verify API key is valid at https://platform.openai.com
```

**2. AWS DynamoDB Connection**
```bash
# Verify AWS credentials
aws configure list
# Test DynamoDB access
aws dynamodb list-tables --region ap-southeast-1
```

**3. JWT Authentication**
```bash
# Verify JWT secret is set
echo $JWT_SECRET_KEY
# Check token format and expiration
```

---

## 🚀 Deployment

### **Development**
```bash
uvicorn app.analytic_api:app --reload --host 0.0.0.0 --port 8091
```

### **Production**
```bash
# With Gunicorn + Uvicorn workers
gunicorn app.analytic_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8091
```

### **Docker** (Optional)
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.analytic_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🌐 CORS Configuration

### **Overview**
The application supports environment-aware CORS (Cross-Origin Resource Sharing) configuration for secure cross-origin requests across multiple deployment environments.

### **Setup Instructions**

#### **Step 1: Environment Variables**

Add CORS configuration to your `.env` file:

**Development Environment:**
```env
# Environment
DEBUG=1

# CORS Configuration - Development
CORS_ORIGINS=http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000,http://localhost:5173
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=*
CORS_ALLOW_HEADERS=*
CORS_MAX_AGE=600
```

**Production Environment:**
```env
# Environment
DEBUG=0

# CORS Configuration - Production
CORS_ORIGINS=https://your-prod-frontend.com,https://www.your-prod-frontend.com
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=GET,POST,PUT,DELETE,OPTIONS
CORS_ALLOW_HEADERS=Authorization,Content-Type,Accept
CORS_MAX_AGE=3600
```

#### **Step 2: Update `app/config.py`**

Add CORS configuration variables:

```python
import os
from typing import List

# ... existing config ...

# CORS Configuration
CORS_ORIGINS: List[str] = [
    origin.strip() 
    for origin in os.getenv("CORS_ORIGINS", "").split(",") 
    if origin.strip()
]
CORS_ALLOW_CREDENTIALS: bool = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"

CORS_ALLOW_METHODS_STR: str = os.getenv("CORS_ALLOW_METHODS", "*")
CORS_ALLOW_METHODS: List[str] = (
    [method.strip() for method in CORS_ALLOW_METHODS_STR.split(",") if method.strip()] 
    if CORS_ALLOW_METHODS_STR != "*" 
    else ["*"]
)

CORS_ALLOW_HEADERS_STR: str = os.getenv("CORS_ALLOW_HEADERS", "*")
CORS_ALLOW_HEADERS: List[str] = (
    [header.strip() for header in CORS_ALLOW_HEADERS_STR.split(",") if header.strip()] 
    if CORS_ALLOW_HEADERS_STR != "*" 
    else ["*"]
)

CORS_MAX_AGE: int = int(os.getenv("CORS_MAX_AGE", "600"))
```

#### **Step 3: Update `app/analytic_api.py`**

Add CORS middleware after app creation:

```python
from fastapi.middleware.cors import CORSMiddleware
from app.config import (
    DEBUG,
    CORS_ORIGINS,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_HEADERS,
    CORS_MAX_AGE
)

# ... existing code ...

app = FastAPI(
    title="Analytic Agent API",
    description="Scalable analytic agent (stateless, DynamoDB conversation history)",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware - Environment-aware configuration
if DEBUG:
    # Development environment - more permissive
    logger.info(f"🔓 CORS enabled for DEVELOPMENT with origins: {CORS_ORIGINS}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS if CORS_ORIGINS else ["*"],
        allow_credentials=CORS_ALLOW_CREDENTIALS,
        allow_methods=CORS_ALLOW_METHODS,
        allow_headers=CORS_ALLOW_HEADERS,
        max_age=CORS_MAX_AGE,
    )
else:
    # Production environment - strict whitelist
    if not CORS_ORIGINS:
        logger.warning("⚠️  CORS_ORIGINS not configured for production! CORS will be disabled.")
    else:
        logger.info(f"🔒 CORS enabled for PRODUCTION with origins: {CORS_ORIGINS}")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=CORS_ORIGINS,
            allow_credentials=CORS_ALLOW_CREDENTIALS,
            allow_methods=CORS_ALLOW_METHODS,
            allow_headers=CORS_ALLOW_HEADERS,
            max_age=CORS_MAX_AGE,
            expose_headers=["X-Request-ID"],
        )
```

### **Environment-Specific Examples**

#### **Development (.env)**
```env
DEBUG=1
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=*
CORS_ALLOW_HEADERS=*
CORS_MAX_AGE=600
```

#### **Staging (.env.staging)**
```env
DEBUG=0
CORS_ORIGINS=https://staging-frontend.company.com,https://staging-admin.company.com
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=GET,POST,PUT,DELETE,OPTIONS
CORS_ALLOW_HEADERS=Authorization,Content-Type,Accept,X-Request-ID
CORS_MAX_AGE=3600
```

#### **Production (.env.production)**
```env
DEBUG=0
CORS_ORIGINS=https://frontend.company.com,https://www.company.com
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=GET,POST,PUT,DELETE,OPTIONS
CORS_ALLOW_HEADERS=Authorization,Content-Type,Accept
CORS_MAX_AGE=3600
```

### **Testing CORS**

#### **Preflight Request (OPTIONS)**
```bash
curl -X OPTIONS http://localhost:8000/api/analytics/report \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Authorization,Content-Type" \
  -v
```

#### **Actual Request with CORS**
```bash
curl -X POST http://localhost:8000/api/analytics/report \
  -H "Origin: http://localhost:3000" \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test query"}' \
  -v
```

#### **Expected Response Headers**
```
Access-Control-Allow-Origin: http://localhost:3000
Access-Control-Allow-Credentials: true
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: Authorization, Content-Type, Accept
Access-Control-Max-Age: 600
```

### **Environment Matrix**

| Environment | DEBUG | CORS Origins | Methods | Headers | Max Age |
|-------------|-------|--------------|---------|---------|---------|
| **Local Dev** | `1` | `localhost:*` | `*` (All) | `*` (All) | 600s |
| **Staging** | `0` | Staging domains | Specific | Specific | 3600s |
| **Production** | `0` | Prod domains only | Specific | Specific | 3600s |

### **Security Best Practices**

✅ **DO:**
- Specify exact origin domains in production (e.g., `https://frontend.company.com`)
- Use HTTPS for all production origins
- Limit HTTP methods to only what you need
- Set reasonable `max_age` to cache preflight requests
- Log CORS configuration on startup for debugging

❌ **DON'T:**
- Never use `allow_origins=["*"]` with `allow_credentials=True` (security risk)
- Don't use wildcards in production origins
- Don't allow unnecessary HTTP methods
- Don't hardcode domains in code (use environment variables)

### **Troubleshooting**

**Issue: CORS error "Origin not allowed"**
```bash
# Check CORS_ORIGINS includes your frontend domain
echo $CORS_ORIGINS
# Example: http://localhost:3000,http://localhost:8080
```

**Issue: Credentials not working**
```bash
# Ensure CORS_ALLOW_CREDENTIALS is set to true
echo $CORS_ALLOW_CREDENTIALS
# Should output: true
```

**Issue: Custom headers blocked**
```bash
# Add your custom headers to CORS_ALLOW_HEADERS
CORS_ALLOW_HEADERS=Authorization,Content-Type,X-Custom-Header
```

---

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`  
3. **Add tests** for your changes
4. **Run test suite**: `python -m pytest tests/ -v`
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open Pull Request**

### **Development Guidelines**
- Maintain test coverage above 80%
- Follow PEP 8 style guidelines
- Add docstrings to all public methods
- Update README.md for significant changes
- Run tests before submitting PR

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Support

- **Documentation**: Check this README and inline code comments
- **Issues**: Open GitHub issues for bugs and feature requests  
- **Testing**: Run the comprehensive test suite before changes
- **Coverage**: Maintain test coverage above 80%

---

## 🗺️ Roadmap

### ✅ **Completed**
- [x] **Modular prompt system with versioning** 
- [x] **ReAct (Reasoning-Acting-Observing) pattern**
- [x] **Plan-and-Execute framework for complex queries**
- [x] **15-Layer Secure Prompt Template System** (Oct 2025)
- [x] **83% Test Coverage** with 463 comprehensive tests
- [x] **Multi-Turn Conversation Support** with database context persistence
- [x] **Intent Detection Improvements** (success/failure rate classification)
- [x] **LangChain Direct Message Integration** (removed template layer)

### 🚧 **In Progress**
- [ ] **Increase test coverage** to 90%+
- [ ] **Complete Phase 2**: Integrate secure templates into remaining agents

### 📋 **Planned**
- [ ] **Add authentication service tests**
- [ ] **Implement caching layer** (Redis integration)
- [ ] **Add real-time analytics** (WebSocket support)
- [ ] **Support additional chart types** (scatter, heatmap, area)
- [ ] **Add data export functionality** (CSV, Excel, PDF)
- [ ] **Implement rate limiting** (per user/organization)
- [ ] **Add monitoring and logging** (Prometheus, Grafana)
- [ ] **Performance optimization** for large datasets
- [ ] **API versioning** (v1, v2 support)
- [ ] **GraphQL endpoint** (alternative to REST)

---

## 🎖️ Achievements

- 🏆 **463 Tests Passing** - Comprehensive test coverage
- 🔒 **15-Layer Security** - Enterprise-grade prompt protection
- 📊 **83% Coverage** - Well-tested codebase
- 🤖 **Multi-Turn AI** - Context-aware conversations
- ⚡ **High Performance** - Async FastAPI architecture
- 🌐 **Multi-Tenant** - Organization-level isolation
- 📈 **Production Ready** - Battle-tested components

---

**Built with ❤️ using LangGraph, OpenAI, and FastAPI**

**Maintained by**: [Ralph Lui](https://github.com/ralphlui)  
**Repository**: [edgp-ai-analytic](https://github.com/ralphlui/edgp-ai-analytic)  
**Branch**: `task/handle-system-prompt-security`
