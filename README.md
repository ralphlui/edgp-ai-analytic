# 🤖 EDGP AI Analytics Agent

**Intelligent Data Analytics API powered by LangGraph and OpenAI GPT models**

A scalable, multi-tenant analytics platform that provides AI-driven insights from your data with natural language processing capabilities, advanced visualization, and comprehensive session management.

## 🚀 Features

### 🧠 **AI-Powered Analytics**
- **Natural Language Queries**: Ask questions in plain English about your data
- **LangGraph Workflows**: Sophisticated AI agent workflows with loop protection
- **GPT-4 Integration**: Powered by OpenAI's latest models for intelligent data analysis
- **Smart Report Classification**: Automatically detects if you want success, failure, or combined analytics

### 📊 **Advanced Visualization**
- **Multiple Chart Types**: Bar, pie, donut, line, and stacked charts
- **Dynamic Chart Generation**: Matplotlib and Seaborn integration
- **Customizable Visualizations**: User-specified chart types and styling
- **Base64 Image Output**: Ready for web integration

### 🏢 **Enterprise-Ready Architecture**
- **Multi-Tenant Support**: Organization-level data isolation
- **JWT Authentication**: Secure token-based authentication
- **Session Management**: Stateful conversation tracking
- **AWS DynamoDB**: Scalable cloud database integration
- **FastAPI Framework**: High-performance async API

### 📈 **Domain Analytics**
- **Customer Analytics**: Analyze customer data by country, status, demographics
- **Product Analytics**: Product performance and categorization
- **Sales Analytics**: Revenue analysis and sales tracking  
- **Flexible Domain Support**: Extensible for any data domain
- **Success/Failure Rate Analysis**: Data quality and accuracy metrics

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.13+
- **AI/ML**: LangChain, LangGraph, OpenAI GPT-4
- **Database**: AWS DynamoDB
- **Visualization**: Matplotlib, Seaborn
- **Authentication**: JWT, Python-JOSE
- **Testing**: Pytest (75+ tests, 38% coverage)
- **Deployment**: Uvicorn ASGI server

## 📋 Prerequisites

- **Python 3.13+**
- **OpenAI API Key** 
- **AWS Account** (DynamoDB access)
- **Virtual Environment** (recommended)

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
uvicorn app.analytic_api:app --reload --host 0.0.0.0 --port 8000

# Production server  
uvicorn app.analytic_api:app --host 0.0.0.0 --port 8000
```

### 4. **Test the API**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me customer success rate by country as pie chart"
  }'
```

## 🔧 API Documentation

### **Main Endpoint**
- **POST** `/query` - Process natural language analytics queries

### **Request Format**
```json
{
  "query": "Show customer analytics by country as pie chart"
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

## 📝 Example Queries

```bash
# Customer Analytics
"Show customer distribution by country as pie chart"
"What's the success rate for customer domain?"
"How many customers from last 30 days?"

# Product Analytics  
"Display product breakdown by category"
"Show product performance as bar chart"

# Success/Failure Analysis
"What's the failure rate for file.csv?"
"Analyze success rate since 2024-01-01"
"Show both success and failure metrics"

# Date-Filtered Queries
"Customer data from 2024-01-01 to 2024-12-31" 
"Show data from last 7 days"
"Analytics since January 2024"
```

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
- **75 Total Tests**
- **38% Overall Coverage**
- **Well-tested areas**: Core analytics (80%+), domain tools (86%+)
- **Areas needing tests**: Database service (0%), Auth system (18%)

### **Test Categories**
- **Unit Tests**: Core business logic and utilities
- **Integration Tests**: End-to-end workflow testing  
- **Auth Tests**: JWT and session management
- **Domain Analytics Tests**: Customer/product data analysis
- **Graph Builder Tests**: LangGraph workflow validation

## 📁 Project Structure

```
edgp-ai-analytic/
├── app/
│   ├── core/                 # Core business logic
│   │   ├── analytic_service.py    # Main analytics engine
│   │   └── graph_builder.py       # LangGraph workflow builder
│   ├── services/             # External service integrations
│   │   ├── database_service.py    # DynamoDB operations
│   │   └── query_coordinator.py   # Request orchestration
│   ├── tools/                # LangGraph tools
│   │   ├── domain_analytics_tools.py  # Domain-specific analytics
│   │   ├── rate_analysis_tools.py     # Success/failure rate tools
│   │   └── session_manager.py         # Session management tools
│   ├── generators/           # Visualization generation
│   │   └── chart_generator.py         # Chart creation engine
│   ├── utils/                # Utility functions
│   │   ├── report_type.py         # Report classification logic
│   │   ├── formatting.py          # Data formatting utilities
│   │   └── sanitization.py        # Input sanitization
│   ├── analytic_api.py       # FastAPI application
│   ├── auth.py              # JWT authentication
│   └── config.py            # Configuration management
├── tests/                   # Test suite
├── htmlcov/                # Coverage reports
├── requirements.txt        # Dependencies
├── .env                   # Environment variables (create this)
└── README.md             # This file
```

## 🔍 Key Components

### **Analytics Engine** (`analytic_service.py`)
- Natural language processing for analytics queries
- Smart report type classification (success/failure/both)
- Conversation context extraction and enhancement
- Multi-step workflow coordination

### **Graph Builder** (`graph_builder.py`)  
- LangGraph workflow construction
- AI agent loop protection (max 10 cycles)
- State management and message tracking
- Tool integration and error handling

### **Chart Generator** (`chart_generator.py`)
- Dynamic visualization creation
- Support for bar, pie, donut, line, stacked charts
- Matplotlib and Seaborn integration
- Base64 encoding for web delivery

### **Domain Analytics Tools**
- Flexible domain data analysis
- Customer, product, sales analytics
- Country/category/region grouping
- Success rate and performance metrics

## 🚀 Deployment

### **Development**
```bash
uvicorn app.analytic_api:app --reload --host 0.0.0.0 --port 8000
```


## 🐛 Troubleshooting

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

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`  
3. **Add tests** for your changes
4. **Run test suite**: `python -m pytest tests/ -v`
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open Pull Request**

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Support

- **Documentation**: Check this README and inline code comments
- **Issues**: Open GitHub issues for bugs and feature requests  
- **Testing**: Run the comprehensive test suite before changes
- **Coverage**: Maintain test coverage above 40%

## 🗺️ Roadmap

- [ ] **Increase test coverage** to 80%+
- [ ] **Add authentication service tests**
- [ ] **Implement caching layer**
- [ ] **Add real-time analytics**
- [ ] **Support additional chart types**
- [ ] **Add data export functionality**
- [ ] **Implement rate limiting**
- [ ] **Add monitoring and logging**
- [x] **Modular prompt system with versioning** ✅
- [x] **ReAct (Reasoning-Acting-Observing) pattern** ✅
- [x] **Plan-and-Execute framework for complex queries** ✅

## 🎯 New: Modular Prompt System

The project now includes a sophisticated modular prompt system with support for advanced AI patterns:

### Features

- **📋 Structured Prompts**: Separated from config, organized by purpose
- **🔄 ReAct Pattern**: Systematic Reasoning → Acting → Observing cycles
- **📝 Plan-and-Execute**: Multi-step planning for complex analytics queries
- **🔖 Versioning**: Track and manage prompt versions over time
- **🛠️ Tool Guidance**: Intelligent tool selection and parameter extraction

### Usage

```python
# Use system prompts
from app.prompts import SystemPrompts
system_prompt = SystemPrompts.get_complete_system_prompt()

# Enable ReAct pattern (add to .env)
USE_REACT_PROMPTS=true

# Use Plan-and-Execute for complex queries
from app.prompts import PlanExecutePrompts
planner = PlanExecutePrompts.get_planner_system_prompt()
```

### Documentation

- **Full Guide**: See [`app/prompts/README.md`](app/prompts/README.md)
- **Examples**: Run `python examples/prompts_demo.py`
- **Configuration**: Check `.env.prompts.example`

---

**Built with ❤️ using LangGraph, OpenAI, and FastAPI**
