"""
Pytest configuration file for the analytics system tests.
"""
import pytest
import sys
import os

# Set test environment BEFORE any imports from app
os.environ['ENVIRONMENT'] = 'test'

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

# Configure pytest markers
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment to use .env.test file."""
    # Environment already set at module level to ensure proper loading
    # This fixture is kept for any additional test-specific setup
    yield
    # Cleanup after tests if needed

@pytest.fixture
def mock_org_context():
    """Provide mock organization context for tests."""
    return {
        'org_id': 'test-org-123',
        'user_id': 'test-user-456',
        'session_id': 'test-session-789'
    }

@pytest.fixture
def sample_chart_data():
    """Provide sample chart data for tests."""
    return [
        {"country": "USA", "customer_count": 100, "status": "success"},
        {"country": "Canada", "customer_count": 50, "status": "success"},
        {"country": "Mexico", "customer_count": 25, "status": "failure"}
    ]

@pytest.fixture
def sample_conversation_history():
    """Provide sample conversation history for tests."""
    return [
        {
            "user_prompt": "Show customer analytics",
            "response_summary": {
                "file_name": "customers.csv",
                "domain_name": "customer", 
                "report_type": "both",
                "row_count": 175
            }
        },
        {
            "user_prompt": "What about products?",
            "response_summary": {
                "domain_name": "product",
                "report_type": "success", 
                "row_count": 85
            }
        }
    ]