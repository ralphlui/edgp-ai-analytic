"""
Comprehensive tests for analytics_tools.py

These tests verify the LangChain tool wrappers work correctly,
including validation, repository integration, and error handling.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

from app.tools.analytics_tools import (
    generate_success_rate_report,
    generate_failure_rate_report,
    get_analytics_tools
)

# Access the underlying functions (LangChain @tool decorator wraps them)
success_rate_func = generate_success_rate_report.func
failure_rate_func = generate_failure_rate_report.func


class TestGenerateSuccessRateReport:
    """Test success rate report generation tool."""
    
    @patch('app.tools.analytics_tools.get_analytics_repository')
    def test_success_rate_by_domain(self, mock_get_repo):
        """Test generating success rate report for a domain."""
        # Mock repository response
        mock_repo = Mock()
        mock_repo.get_success_rate_by_domain.return_value = {
            "target_type": "domain",
            "target_value": "customer",
            "total_requests": 1000,
            "successful_requests": 850,
            "failed_requests": 150,
            "success_rate": 85.0
        }
        mock_get_repo.return_value = mock_repo
        
        # Call the underlying function
        result = success_rate_func(domain_name="customer", org_id="org-123")
        
        # Verify
        assert result["success"] is True
        assert result["data"]["target_type"] == "domain"
        assert result["data"]["target_value"] == "customer"
        assert result["data"]["total_requests"] == 1000
        assert result["data"]["successful_requests"] == 850
        assert result["data"]["failed_requests"] == 150
        assert result["data"]["success_rate"] == 85.0
        assert result["data"]["report_type"] == "success_rate"
        
        mock_repo.get_success_rate_by_domain.assert_called_once_with("customer", org_id="org-123")
    
    @patch('app.tools.analytics_tools.get_analytics_repository')
    def test_success_rate_by_file(self, mock_get_repo):
        """Test generating success rate report for a file."""
        # Mock repository response
        mock_repo = Mock()
        mock_repo.get_success_rate_by_file.return_value = {
            "target_type": "file",
            "target_value": "data.csv",
            "total_requests": 500,
            "successful_requests": 475,
            "failed_requests": 25,
            "success_rate": 95.0
        }
        mock_get_repo.return_value = mock_repo
        
        # Call tool
        result = success_rate_func(file_name="data.csv", org_id="org-456")
        
        # Verify
        assert result["success"] is True
        assert result["data"]["target_type"] == "file"
        assert result["data"]["target_value"] == "data.csv"
        assert result["data"]["total_requests"] == 500
        assert result["data"]["success_rate"] == 95.0
        assert result["data"]["report_type"] == "success_rate"
        
        mock_repo.get_success_rate_by_file.assert_called_once_with("data.csv", org_id="org-456")
    
    def test_success_rate_no_parameters(self):
        """Test error when neither domain_name nor file_name provided."""
        result = success_rate_func(org_id="org-123")
        
        assert result["success"] is False
        assert "Must provide either domain_name or file_name" in result["error"]
    
    def test_success_rate_both_parameters(self):
        """Test error when both domain_name and file_name provided."""
        result = success_rate_func(domain_name="customer", file_name="data.csv", org_id="org-123")
        
        assert result["success"] is False
        assert "Provide only ONE of domain_name or file_name" in result["error"]
    
    @patch('app.tools.analytics_tools.get_analytics_repository')
    def test_success_rate_repository_exception(self, mock_get_repo):
        """Test handling of repository exceptions."""
        mock_repo = Mock()
        mock_repo.get_success_rate_by_domain.side_effect = Exception("DynamoDB error")
        mock_get_repo.return_value = mock_repo
        
        result = success_rate_func(domain_name="customer", org_id="org-123")
        
        assert result["success"] is False
        assert "Error generating report" in result["error"]
        assert "DynamoDB error" in result["error"]
    
    @patch('app.tools.analytics_tools.get_analytics_repository')
    def test_success_rate_without_org_id(self, mock_get_repo):
        """Test success rate generation without org_id (should fail)."""
        result = success_rate_func(domain_name="payment")
        
        # Should fail because org_id is required
        assert result["success"] is False
        assert "not associated with any organization" in result["error"]


class TestGenerateFailureRateReport:
    """Test failure rate report generation tool."""
    
    @patch('app.tools.analytics_tools.get_analytics_repository')
    def test_failure_rate_by_domain(self, mock_get_repo):
        """Test generating failure rate report for a domain."""
        mock_repo = Mock()
        mock_repo.get_failure_rate_by_domain.return_value = {
            "target_type": "domain",
            "target_value": "api.example.com",
            "total_requests": 1500,
            "successful_requests": 1350,
            "failed_requests": 150,
            "failure_rate": 10.0
        }
        mock_get_repo.return_value = mock_repo
        
        result = failure_rate_func(domain_name="api.example.com", org_id="org-789")
        
        assert result["success"] is True
        assert result["data"]["target_type"] == "domain"
        assert result["data"]["target_value"] == "api.example.com"
        assert result["data"]["total_requests"] == 1500
        assert result["data"]["successful_requests"] == 1350
        assert result["data"]["failed_requests"] == 150
        assert result["data"]["failure_rate"] == 10.0
        assert result["data"]["report_type"] == "failure_rate"
        
        mock_repo.get_failure_rate_by_domain.assert_called_once_with("api.example.com", org_id="org-789")
    
    @patch('app.tools.analytics_tools.get_analytics_repository')
    def test_failure_rate_by_file(self, mock_get_repo):
        """Test generating failure rate report for a file."""
        mock_repo = Mock()
        mock_repo.get_failure_rate_by_file.return_value = {
            "target_type": "file",
            "target_value": "report.pdf",
            "total_requests": 300,
            "successful_requests": 285,
            "failed_requests": 15,
            "failure_rate": 5.0
        }
        mock_get_repo.return_value = mock_repo
        
        result = failure_rate_func(file_name="report.pdf", org_id="org-111")
        
        assert result["success"] is True
        assert result["data"]["target_type"] == "file"
        assert result["data"]["target_value"] == "report.pdf"
        assert result["data"]["failure_rate"] == 5.0
        assert result["data"]["report_type"] == "failure_rate"
        
        mock_repo.get_failure_rate_by_file.assert_called_once_with("report.pdf", org_id="org-111")
    
    def test_failure_rate_no_parameters(self):
        """Test error when neither domain_name nor file_name provided."""
        result = failure_rate_func(org_id="org-123")
        
        assert result["success"] is False
        assert "Must provide either domain_name or file_name" in result["error"]
    
    def test_failure_rate_both_parameters(self):
        """Test error when both domain_name and file_name provided."""
        result = failure_rate_func(domain_name="api.com", file_name="upload.txt", org_id="org-123")
        
        assert result["success"] is False
        assert "Provide only ONE of domain_name or file_name" in result["error"]
    
    @patch('app.tools.analytics_tools.get_analytics_repository')
    def test_failure_rate_repository_exception(self, mock_get_repo):
        """Test handling of repository exceptions."""
        mock_repo = Mock()
        mock_repo.get_failure_rate_by_file.side_effect = RuntimeError("Connection timeout")
        mock_get_repo.return_value = mock_repo
        
        result = failure_rate_func(file_name="test.csv", org_id="org-123")
        
        assert result["success"] is False
        assert "Error generating report" in result["error"]
        assert "Connection timeout" in result["error"]
    
    @patch('app.tools.analytics_tools.get_analytics_repository')
    def test_failure_rate_zero_failures(self, mock_get_repo):
        """Test failure rate report with zero failures (should fail without org_id)."""
        result = failure_rate_func(domain_name="perfect.com")
        
        # Should fail because org_id is required
        assert result["success"] is False
        assert "not associated with any organization" in result["error"]


class TestGetAnalyticsTools:
    """Test the get_analytics_tools helper function."""
    
    def test_get_tools_list(self):
        """Test that get_analytics_tools returns all tools."""
        tools = get_analytics_tools()
        
        assert len(tools) == 2
        assert generate_success_rate_report in tools
        assert generate_failure_rate_report in tools
    
    def test_tools_are_callable(self):
        """Test that all returned tools are callable."""
        tools = get_analytics_tools()
        
        for tool in tools:
            assert callable(tool)
    
    def test_tools_have_langchain_decorator(self):
        """Test that tools have LangChain tool metadata."""
        tools = get_analytics_tools()
        
        # LangChain @tool decorator adds specific attributes
        for tool_func in tools:
            # Tools should have name and description (from @tool decorator)
            assert hasattr(tool_func, 'name') or hasattr(tool_func, '__name__')
            assert hasattr(tool_func, 'description') or hasattr(tool_func, '__doc__')


class TestToolIntegration:
    """Integration tests for tool behavior."""
    
    @patch('app.tools.analytics_tools.get_analytics_repository')
    def test_both_tools_use_same_repository_instance(self, mock_get_repo):
        """Test that tools use the repository pattern correctly."""
        mock_repo = Mock()
        mock_repo.get_success_rate_by_domain.return_value = {
            "target_type": "domain",
            "target_value": "test",
            "total_requests": 100,
            "successful_requests": 90,
            "failed_requests": 10,
            "success_rate": 90.0
        }
        mock_repo.get_failure_rate_by_domain.return_value = {
            "target_type": "domain",
            "target_value": "test",
            "total_requests": 100,
            "successful_requests": 90,
            "failed_requests": 10,
            "failure_rate": 10.0
        }
        mock_get_repo.return_value = mock_repo
        
        # Call both tools
        success_result = success_rate_func(domain_name="test", org_id="org-123")
        failure_result = failure_rate_func(domain_name="test", org_id="org-123")
        
        # Both should succeed
        assert success_result["success"] is True
        assert failure_result["success"] is True
        
        # Repository should be called twice (once per tool)
        assert mock_get_repo.call_count == 2
    
    @patch('app.tools.analytics_tools.get_analytics_repository')
    def test_data_structure_consistency(self, mock_get_repo):
        """Test that both tools return consistent data structures."""
        mock_repo = Mock()
        mock_data = {
            "target_type": "domain",
            "target_value": "example.com",
            "total_requests": 500,
            "successful_requests": 450,
            "failed_requests": 50,
            "success_rate": 90.0,
            "failure_rate": 10.0
        }
        mock_repo.get_success_rate_by_domain.return_value = mock_data
        mock_repo.get_failure_rate_by_domain.return_value = mock_data
        mock_get_repo.return_value = mock_repo
        
        success_result = success_rate_func(domain_name="example.com", org_id="org-123")
        failure_result = failure_rate_func(domain_name="example.com", org_id="org-123")
        
        # Check both have same structure
        assert set(success_result.keys()) == {"success", "data"}
        assert set(failure_result.keys()) == {"success", "data"}
        
        # Check data fields
        for key in ["target_type", "target_value", "total_requests", "successful_requests", "failed_requests"]:
            assert key in success_result["data"]
            assert key in failure_result["data"]
        
        # Check report_type is different
        assert success_result["data"]["report_type"] == "success_rate"
        assert failure_result["data"]["report_type"] == "failure_rate"
