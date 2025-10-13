# Test Suite for EDGP AI Analytics

## Overview

Comprehensive test suite covering security, authentication, chart generation, and configuration management for the EDGP AI Analytics application.

## Test Structure

```
tests/
├── __init__.py                    # Test package initialization
├── test_prompt_validator.py      # Security validation tests (100+ tests)
├── test_chart_service.py          # Chart generation tests (40+ tests)
├── test_auth.py                   # Authentication tests (20+ tests)
├── test_config.py                 # Configuration tests (15+ tests)
└── README.md                      # This file
```

## Test Categories

### 1. Security Tests (`test_prompt_validator.py`)

**Coverage:**
- ✅ Prompt injection detection (role manipulation, instruction override, identity hijacking)
- ✅ System probing attempts
- ✅ Command injection
- ✅ Context breaking and template injection
- ✅ Security bypass attempts
- ✅ Unicode/homoglyph attacks
- ✅ Output validation for information leakage
- ✅ Credential leak detection
- ✅ Database detail exposure

**Test Classes:**
- `TestPromptInputValidation` - 50+ tests for input validation
- `TestOutputValidation` - 30+ tests for output validation
- `TestValidatorIntegration` - Integration tests
- `TestPatternCoverage` - Pattern coverage verification

### 2. Chart Generation Tests (`test_chart_service.py`)

**Coverage:**
- ✅ Success/failure rate bar charts
- ✅ Chart type filtering (success-only, failure-only)
- ✅ Data validation and edge cases
- ✅ Base64 encoding verification
- ✅ Large and small number handling
- ✅ Special characters in names

**Test Classes:**
- `TestBarChartGeneration` - 25+ tests for chart creation
- `TestChartDataValidation` - Data validation tests

### 3. Authentication Tests (`test_auth.py`)

**Coverage:**
- ✅ JWT token validation
- ✅ Expired/invalid token handling
- ✅ User profile validation via admin API
- ✅ Active/inactive user handling
- ✅ API timeout and error handling
- ✅ Missing claims detection

**Test Classes:**
- `TestJWTValidation` - JWT token tests
- `TestUserProfileValidation` - User profile tests
- `TestAuthorizationEdgeCases` - Edge case tests

### 4. Configuration Tests (`test_config.py`)

**Coverage:**
- ✅ Environment variable loading
- ✅ AWS Secrets Manager integration
- ✅ Environment-specific configs (dev, sit, prod)
- ✅ DynamoDB configuration
- ✅ OpenAI configuration
- ✅ Fallback mechanisms

**Test Classes:**
- `TestConfigLoading` - Basic config loading
- `TestAWSSecretsManager` - Secrets Manager tests
- `TestConfigValidation` - Validation tests
- `TestEnvironmentSpecificConfig` - Environment tests
- `TestDynamoDBConfig` - DynamoDB tests
- `TestOpenAIConfig` - OpenAI tests

## Running Tests

### Run All Tests

```bash
# Run all tests with coverage
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=app --cov-report=html
```

### Run Specific Test Files

```bash
# Security tests only
pytest tests/test_prompt_validator.py

# Chart tests only
pytest tests/test_chart_service.py

# Auth tests only
pytest tests/test_auth.py

# Config tests only
pytest tests/test_config.py
```

### Run Specific Test Classes

```bash
# Run specific test class
pytest tests/test_prompt_validator.py::TestPromptInputValidation

# Run specific test method
pytest tests/test_prompt_validator.py::TestPromptInputValidation::test_safe_analytics_query
```

### Run Tests by Markers

```bash
# Run only unit tests
pytest -m unit

# Run only security tests
pytest -m security

# Run only slow tests
pytest -m slow
```

## Test Coverage

Current coverage targets:
- **Overall:** 70%+ code coverage
- **Security Module:** 95%+ coverage
- **Chart Service:** 85%+ coverage
- **Auth Module:** 80%+ coverage

View detailed coverage report:
```bash
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

## Writing New Tests

### Test Structure Template

```python
"""
Module description.

Tests cover:
- Feature 1
- Feature 2
"""
import pytest


class TestFeatureName:
    """Test cases for specific feature."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing test data."""
        return {"key": "value"}
    
    def test_basic_functionality(self, sample_data):
        """Test basic feature functionality."""
        result = function_under_test(sample_data)
        assert result is not None
    
    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            function_under_test(invalid_input)
```

### Best Practices

1. **Descriptive Names:** Use clear, descriptive test names
2. **Single Assert:** Test one thing per test method
3. **Fixtures:** Use fixtures for reusable test data
4. **Mocking:** Mock external dependencies (APIs, databases)
5. **Async Tests:** Use `@pytest.mark.asyncio` for async functions
6. **Markers:** Add markers for test categorization

### Example Test

```python
@pytest.mark.security
def test_sql_injection_detection():
    """Test detection of SQL injection attempts."""
    malicious_input = "'; DROP TABLE users; --"
    is_safe, error = validate_user_prompt(malicious_input)
    assert is_safe is False
    assert "injection" in error.lower()
```

## Continuous Integration

Tests are run automatically on:
- Pull requests
- Commits to main branch
- Pre-deployment checks

CI Configuration:
```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    pytest --cov=app --cov-report=xml
    pytest --cov=app --cov-fail-under=70
```

## Debugging Tests

### Run with Debug Output

```bash
# Show print statements
pytest -s

# Show local variables on failure
pytest -l

# Drop into debugger on failure
pytest --pdb
```

### Common Issues

**Issue:** Import errors
```bash
# Ensure app is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest
```

**Issue:** Async tests not running
```bash
# Install async plugin
pip install pytest-asyncio
```

**Issue:** Mock not working
```bash
# Check patch path matches actual import
# Use full module path in patch decorator
```

## Performance

Test execution times (approximate):
- Security tests: ~5 seconds
- Chart tests: ~3 seconds
- Auth tests: ~2 seconds
- Config tests: ~1 second
- **Total:** ~11 seconds

## Future Enhancements

Planned test additions:
- [ ] API endpoint integration tests
- [ ] Database interaction tests
- [ ] Load/performance tests
- [ ] End-to-end workflow tests
- [ ] Chaos engineering tests

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure tests pass locally
3. Maintain 70%+ coverage
4. Update this README if adding new test categories

## Support

For test-related questions:
- Check existing tests for examples
- Review pytest documentation
- Contact the development team

---

**Last Updated:** October 2025
**Test Count:** 175+ tests
**Coverage:** 70%+
