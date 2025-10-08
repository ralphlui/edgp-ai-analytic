"""
Unit tests for query parsing and domain analytics tools.
"""
import pytest
from unittest.mock import Mock, patch
from app.tools.domain_analytics_tools import (
    _parse_analytics_query,
    get_domain_analytics_by_field_tool,
    analyze_query_for_domain_analytics_tool
)


class TestDomainAnalyticsParsing:
    """Test cases for the _parse_analytics_query function."""

    def test_customer_country_pie_chart(self):
        """Test parsing 'How many customers per country using pie chart?'"""
        domain, field, chart = _parse_analytics_query("How many customers per country using pie chart?")
        assert domain == "customer"
        assert field == "country"
        assert chart == "pie"

    def test_products_by_category(self):
        """Test parsing 'Show products by category'"""
        domain, field, chart = _parse_analytics_query("Show products by category")
        assert domain == "product"
        assert field == "category"
        assert chart is None  # Default to bar

    def test_orders_by_region_donut(self):
        """Test parsing 'Orders by region as donut chart'"""
        domain, field, chart = _parse_analytics_query("Orders by region as donut chart")
        assert domain == "order"
        assert field == "region"
        assert chart == "donut"

    def test_users_breakdown_by_status(self):
        """Test parsing 'Users breakdown by status'"""
        domain, field, chart = _parse_analytics_query("Users breakdown by status")
        assert domain == "user"
        assert field == "status"
        assert chart is None

    def test_customer_distribution_by_country(self):
        """Test parsing 'Customer distribution by country'"""
        domain, field, chart = _parse_analytics_query("Customer distribution by country")
        assert domain == "customer"
        assert field == "country"
        assert chart is None

    def test_how_many_pattern(self):
        """Test parsing 'How many orders per region'"""
        domain, field, chart = _parse_analytics_query("How many orders per region")
        assert domain == "order"
        assert field == "region"
        assert chart is None

    def test_show_pattern(self):
        """Test parsing 'Show users by status'"""
        domain, field, chart = _parse_analytics_query("Show users by status")
        assert domain == "user"
        assert field == "status"
        assert chart is None

    def test_unknown_query(self):
        """Test parsing an unrecognized query"""
        domain, field, chart = _parse_analytics_query("What is the weather today?")
        assert domain is None
        assert field is None
        assert chart is None

    def test_case_insensitive(self):
        """Test that parsing is case insensitive"""
        domain, field, chart = _parse_analytics_query("HOW MANY CUSTOMERS PER COUNTRY USING PIE CHART?")
        assert domain == "customer"
        assert field == "country"
        assert chart == "pie"

    def test_various_chart_types(self):
        """Test different chart type extractions"""
        test_cases = [
            ("Customers per country bar chart", "bar"),
            ("Products by category line chart", "line"),
            ("Orders by region pie chart", "pie"),
            ("Users by status donut chart", "donut"),
            ("Orders by region stacked chart", None),  # Not supported, should be None
        ]
        for query, expected_chart in test_cases:
            _, _, chart = _parse_analytics_query(query)
            assert chart == expected_chart, f"Failed for query: {query}"

    def test_edge_cases(self):
        """Test edge cases and malformed queries"""
        edge_cases = [
            ("", (None, None, None)),
            ("   ", (None, None, None)),
            ("customers", (None, None, None)),  # Incomplete
            ("per country", (None, None, None)),  # Missing domain
            ("breakdown", (None, None, None)),  # Too generic
        ]
        for query, expected in edge_cases:
            result = _parse_analytics_query(query)
            assert result == expected, f"Failed for edge case: {query}"

    def test_plural_handling(self):
        """Test that plural forms are correctly handled"""
        test_cases = [
            ("customers per country", "customer"),
            ("products by category", "product"),
            ("orders by region", "order"),
            ("users by status", "user"),
        ]
        for query, expected_domain in test_cases:
            domain, _, _ = _parse_analytics_query(query)
            assert domain == expected_domain, f"Failed plural handling for: {query}"


class TestDomainAnalyticsTools:
    """Test domain analytics tool functions."""

    @patch('app.tools.domain_analytics_tools.get_org_id_for_tool')
    @patch('app.tools.domain_analytics_tools.execute_async_tool_call')
    def test_get_domain_analytics_by_field_success(self, mock_execute, mock_get_org_id):
        """Test successful domain analytics retrieval."""
        mock_get_org_id.return_value = "org123"
        mock_execute.return_value = {
            "success": True,
            "chart_data": [{"country": "USA", "count": 100}],
            "row_count": 1
        }
        
        # Import the actual function, not the tool wrapper
        from app.tools.domain_analytics_tools import get_domain_analytics_by_field_tool
        
        result = get_domain_analytics_by_field_tool.func(
            domain_name="customer",
            group_by_field="country",
            chart_type="pie"
        )
        
        # Parse JSON result
        import json
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "chart_data" in result_data
        mock_execute.assert_called_once()

    @patch('app.tools.domain_analytics_tools.get_org_id_for_tool')
    def test_get_domain_analytics_by_field_missing_params(self, mock_get_org_id):
        """Test domain analytics with missing parameters."""
        mock_get_org_id.return_value = "org123"
        
        from app.tools.domain_analytics_tools import get_domain_analytics_by_field_tool
        
        result = get_domain_analytics_by_field_tool.func(
            domain_name="",  # Missing domain
            group_by_field="country"
        )
        
        import json
        result_data = json.loads(result)
        
        assert result_data["success"] is False
        assert "Both domain_name and group_by_field are required" in result_data["error"]

    @patch('app.tools.domain_analytics_tools.get_org_id_for_tool')
    def test_get_domain_analytics_no_auth(self, mock_get_org_id):
        """Test domain analytics without authentication."""
        mock_get_org_id.return_value = None  # No org_id
        
        from app.tools.domain_analytics_tools import get_domain_analytics_by_field_tool
        
        result = get_domain_analytics_by_field_tool.func(
            domain_name="customer",
            group_by_field="country"
        )
        
        import json
        result_data = json.loads(result)
        
        assert result_data["success"] is False
        # The error response contains an "error" field with the exception message
        assert "error" in result_data

    @patch('app.tools.domain_analytics_tools.get_org_id_for_tool')
    @patch('app.tools.domain_analytics_tools.execute_async_tool_call')
    def test_analyze_query_tool_success(self, mock_execute, mock_get_org_id):
        """Test query analysis tool success."""
        mock_get_org_id.return_value = "org123"
        mock_execute.return_value = {
            "success": True,
            "chart_data": [{"country": "USA", "customer_count": 100}],
            "row_count": 1
        }
        
        from app.tools.domain_analytics_tools import analyze_query_for_domain_analytics_tool
        
        result = analyze_query_for_domain_analytics_tool.func(
            user_query="How many customers per country using pie chart?",
            chart_type="bar"
        )
        
        import json
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "parsed_query" in result_data
        assert result_data["parsed_query"]["extracted_domain"] == "customer"
        assert result_data["parsed_query"]["extracted_field"] == "country"
        assert result_data["parsed_query"]["extracted_chart_type"] == "pie"

    @patch('app.tools.domain_analytics_tools.get_org_id_for_tool')
    def test_analyze_query_tool_unparseable(self, mock_get_org_id):
        """Test query analysis with unparseable query."""
        mock_get_org_id.return_value = "org123"
        
        from app.tools.domain_analytics_tools import analyze_query_for_domain_analytics_tool
        
        result = analyze_query_for_domain_analytics_tool.func(
            user_query="What is the weather today?",
            chart_type="bar"
        )
        
        import json
        result_data = json.loads(result)
        
        assert result_data["success"] is False
        assert "Could not identify domain and grouping field" in result_data["message"]


class TestToolUtils:
    """Test tool utility functions."""

    def test_validate_chart_type_valid(self):
        """Test chart type validation with valid types."""
        from app.tools.tool_utils import validate_chart_type
        
        valid_types = ['bar', 'pie', 'donut', 'line', 'stacked']
        for chart_type in valid_types:
            result = validate_chart_type(chart_type)
            assert result == chart_type

    def test_validate_chart_type_invalid(self):
        """Test chart type validation with invalid types."""
        from app.tools.tool_utils import validate_chart_type
        
        invalid_types = ['scatter', 'bubble', 'invalid', '', None]
        for chart_type in invalid_types:
            result = validate_chart_type(chart_type)
            assert result == 'bar'  # Default

    def test_validate_report_type_valid(self):
        """Test report type validation with valid types."""
        from app.tools.tool_utils import validate_report_type
        
        valid_types = ['success', 'failure', 'both']
        for report_type in valid_types:
            result = validate_report_type(report_type)
            assert result == report_type

    def test_validate_report_type_invalid(self):
        """Test report type validation with invalid types."""
        from app.tools.tool_utils import validate_report_type
        
        invalid_types = ['invalid', 'error', '', None]
        for report_type in invalid_types:
            result = validate_report_type(report_type)
            assert result == ''  # Default empty string