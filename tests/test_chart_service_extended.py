"""
Extended tests for chart_service.py to increase coverage to 87-90%

This module tests the chart generation methods that were previously untested:
- Line charts
- Area charts  
- Donut charts
- Horizontal bar charts
- Grouped bar charts
"""

import pytest
import base64
from unittest.mock import Mock, patch
from io import BytesIO

from app.services.chart_service import (
    AnalyticsChartGenerator,
    generate_analytics_chart,
    generate_comparison_chart
)


class TestLineChartGeneration:
    """Test line chart generation"""
    
    def test_generate_line_chart_success_rate(self):
        """Test line chart generation for success rate"""
        generator = AnalyticsChartGenerator()
        data = {
            "target_type": "domain",
            "target_value": "customer",
            "total_requests": 1000,
            "successful_requests": 950,
            "failed_requests": 50,
            "success_rate": 95.0
        }
        
        result = generator.generate_line_chart(data, "success_rate")
        
        assert result is not None
        assert isinstance(result, str)
        # Verify it's valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0
    
    def test_generate_line_chart_failure_rate(self):
        """Test line chart generation for failure rate"""
        generator = AnalyticsChartGenerator()
        data = {
            "target_type": "file",
            "target_value": "uploads.csv",
            "total_requests": 500,
            "successful_requests": 400,
            "failed_requests": 100,
            "failure_rate": 20.0
        }
        
        result = generator.generate_line_chart(data, "failure_rate")
        
        assert result is not None
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert len(decoded) > 0
    
    def test_generate_line_chart_zero_total(self):
        """Test line chart with zero total requests"""
        generator = AnalyticsChartGenerator()
        data = {
            "target_type": "domain",
            "target_value": "test",
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "success_rate": 0
        }
        
        result = generator.generate_line_chart(data, "success_rate")
        
        assert result is None
    
    def test_generate_line_chart_exception(self):
        """Test line chart with exception handling"""
        generator = AnalyticsChartGenerator()
        invalid_data = {"invalid": "data"}
        
        result = generator.generate_line_chart(invalid_data, "success_rate")
        
        assert result is None
    
    def test_generate_line_chart_with_100_percent(self):
        """Test line chart with 100% success rate"""
        generator = AnalyticsChartGenerator()
        data = {
            "target_type": "domain",
            "target_value": "perfect",
            "total_requests": 1000,
            "successful_requests": 1000,
            "failed_requests": 0,
            "success_rate": 100.0
        }
        
        result = generator.generate_line_chart(data, "success_rate")
        
        assert result is not None
    
    def test_generate_line_chart_with_0_percent(self):
        """Test line chart with 0% success rate"""
        generator = AnalyticsChartGenerator()
        data = {
            "target_type": "file",
            "target_value": "broken.csv",
            "total_requests": 100,
            "successful_requests": 0,
            "failed_requests": 100,
            "success_rate": 0.0
        }
        
        result = generator.generate_line_chart(data, "success_rate")
        
        assert result is not None


class TestAreaChartGeneration:
    """Test area chart generation"""
    
    def test_generate_area_chart_success_rate(self):
        """Test area chart generation for success rate"""
        generator = AnalyticsChartGenerator()
        data = {
            "target_type": "domain",
            "target_value": "customer",
            "total_requests": 1000,
            "successful_requests": 950,
            "failed_requests": 50,
            "success_rate": 95.0
        }
        
        result = generator.generate_area_chart(data, "success_rate")
        
        assert result is not None
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert len(decoded) > 0
    
    def test_generate_area_chart_failure_rate(self):
        """Test area chart generation for failure rate"""
        generator = AnalyticsChartGenerator()
        data = {
            "target_type": "file",
            "target_value": "errors.csv",
            "total_requests": 200,
            "successful_requests": 150,
            "failed_requests": 50,
            "failure_rate": 25.0
        }
        
        result = generator.generate_area_chart(data, "failure_rate")
        
        assert result is not None
        assert isinstance(result, str)
    
    def test_generate_area_chart_zero_total(self):
        """Test area chart with zero total requests"""
        generator = AnalyticsChartGenerator()
        data = {
            "target_type": "domain",
            "target_value": "test",
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "success_rate": 0
        }
        
        result = generator.generate_area_chart(data, "success_rate")
        
        assert result is None
    
    def test_generate_area_chart_exception(self):
        """Test area chart with exception handling"""
        generator = AnalyticsChartGenerator()
        invalid_data = None
        
        result = generator.generate_area_chart(invalid_data, "success_rate")
        
        assert result is None
    
    def test_generate_area_chart_large_numbers(self):
        """Test area chart with large numbers"""
        generator = AnalyticsChartGenerator()
        data = {
            "target_type": "domain",
            "target_value": "high-traffic",
            "total_requests": 1000000,
            "successful_requests": 999500,
            "failed_requests": 500,
            "success_rate": 99.95
        }
        
        result = generator.generate_area_chart(data, "success_rate")
        
        assert result is not None


class TestDonutChartGeneration:
    """Test donut chart generation"""
    
    def test_generate_donut_chart_success(self):
        """Test donut chart generation with normal data"""
        generator = AnalyticsChartGenerator()
        data = {
            "target_type": "domain",
            "target_value": "customer",
            "total_requests": 1000,
            "successful_requests": 750,
            "failed_requests": 250,
            "success_rate": 75.0
        }
        
        result = generator.generate_donut_chart(data)
        
        assert result is not None
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert len(decoded) > 0
    
    def test_generate_donut_chart_with_file_target(self):
        """Test donut chart with file target"""
        generator = AnalyticsChartGenerator()
        data = {
            "target_type": "file",
            "target_value": "data.csv",
            "total_requests": 500,
            "successful_requests": 400,
            "failed_requests": 100,
            "success_rate": 80.0
        }
        
        result = generator.generate_donut_chart(data)
        
        assert result is not None
    
    def test_generate_donut_chart_zero_total(self):
        """Test donut chart with zero total requests"""
        generator = AnalyticsChartGenerator()
        data = {
            "target_type": "domain",
            "target_value": "test",
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0
        }
        
        result = generator.generate_donut_chart(data)
        
        assert result is None
    
    def test_generate_donut_chart_100_percent_success(self):
        """Test donut chart with 100% success rate"""
        generator = AnalyticsChartGenerator()
        data = {
            "target_type": "domain",
            "target_value": "perfect",
            "total_requests": 1000,
            "successful_requests": 1000,
            "failed_requests": 0,
            "success_rate": 100.0
        }
        
        result = generator.generate_donut_chart(data)
        
        assert result is not None
    
    def test_generate_donut_chart_100_percent_failure(self):
        """Test donut chart with 100% failure rate"""
        generator = AnalyticsChartGenerator()
        data = {
            "target_type": "file",
            "target_value": "broken.csv",
            "total_requests": 100,
            "successful_requests": 0,
            "failed_requests": 100,
            "success_rate": 0.0
        }
        
        result = generator.generate_donut_chart(data)
        
        assert result is not None
    
    def test_generate_donut_chart_exception(self):
        """Test donut chart with exception handling"""
        generator = AnalyticsChartGenerator()
        invalid_data = {"missing": "required_fields"}
        
        result = generator.generate_donut_chart(invalid_data)
        
        assert result is None


class TestGenerateAnalyticsChartWithAllStyles:
    """Test generate_analytics_chart function with different styles"""
    
    def test_generate_line_chart_style(self):
        """Test generating chart with line style"""
        data = {
            "target_type": "domain",
            "target_value": "customer",
            "total_requests": 1000,
            "successful_requests": 950,
            "failed_requests": 50,
            "success_rate": 95.0
        }
        
        result = generate_analytics_chart(data, chart_type="success_rate", style="line")
        
        assert result is not None
        assert isinstance(result, str)
    
    def test_generate_area_chart_style(self):
        """Test generating chart with area style"""
        data = {
            "target_type": "domain",
            "target_value": "customer",
            "total_requests": 1000,
            "successful_requests": 950,
            "failed_requests": 50,
            "success_rate": 95.0
        }
        
        result = generate_analytics_chart(data, chart_type="success_rate", style="area")
        
        assert result is not None
        assert isinstance(result, str)
    
    def test_generate_donut_chart_style(self):
        """Test generating chart with donut style"""
        data = {
            "target_type": "domain",
            "target_value": "customer",
            "total_requests": 1000,
            "successful_requests": 950,
            "failed_requests": 50,
            "success_rate": 95.0
        }
        
        result = generate_analytics_chart(data, chart_type="success_rate", style="donut")
        
        assert result is not None
        assert isinstance(result, str)
    
    def test_generate_horizontal_bar_chart_style(self):
        """Test generating chart with horizontal_bar style"""
        data = {
            "target_type": "domain",
            "target_value": "customer",
            "total_requests": 1000,
            "successful_requests": 950,
            "failed_requests": 50,
            "success_rate": 95.0
        }
        
        result = generate_analytics_chart(data, chart_type="success_rate", style="horizontal_bar")
        
        assert result is not None
        assert isinstance(result, str)
    
    def test_generate_with_failure_rate_and_line(self):
        """Test generating line chart with failure rate"""
        data = {
            "target_type": "file",
            "target_value": "errors.csv",
            "total_requests": 500,
            "successful_requests": 400,
            "failed_requests": 100,
            "failure_rate": 20.0
        }
        
        result = generate_analytics_chart(data, chart_type="failure_rate", style="line")
        
        assert result is not None
    
    def test_generate_with_failure_rate_and_area(self):
        """Test generating area chart with failure rate"""
        data = {
            "target_type": "file",
            "target_value": "errors.csv",
            "total_requests": 500,
            "successful_requests": 400,
            "failed_requests": 100,
            "failure_rate": 20.0
        }
        
        result = generate_analytics_chart(data, chart_type="failure_rate", style="area")
        
        assert result is not None


class TestGenerateComparisonChartExtended:
    """Test generate_comparison_chart with different chart types"""
    
    def test_comparison_chart_horizontal_bar(self):
        """Test comparison chart with horizontal_bar type"""
        comparison_data = {
            "targets": ["customer", "location"],
            "metric": "success_rate",
            "winner": "customer",
            "comparison_details": [
                {"target": "customer", "metric_value": 95.0, "total_requests": 1000, "successful_requests": 950, "failed_requests": 50},
                {"target": "location", "metric_value": 85.0, "total_requests": 800, "successful_requests": 680, "failed_requests": 120}
            ]
        }
        chart_type = "horizontal_bar"
        
        result = generate_comparison_chart(comparison_data, chart_type)
        
        assert result is not None
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert len(decoded) > 0
    
    def test_comparison_chart_grouped_bar(self):
        """Test comparison chart with grouped_bar type"""
        comparison_data = {
            "targets": ["customer", "location"],
            "metric": "success_rate",
            "winner": "customer",
            "comparison_details": [
                {"target": "customer", "metric_value": 95.0, "total_requests": 1000, "successful_requests": 950, "failed_requests": 50},
                {"target": "location", "metric_value": 85.0, "total_requests": 800, "successful_requests": 680, "failed_requests": 120}
            ]
        }
        
        result = generate_comparison_chart(comparison_data, "grouped_bar")
        assert result is not None
    
    def test_comparison_chart_line(self):
        """Test comparison chart with line type"""
        comparison_data = {
            "targets": ["customer", "location"],
            "metric": "success_rate",
            "winner": "customer",
            "comparison_details": [
                {"target": "customer", "metric_value": 95.0, "total_requests": 1000, "successful_requests": 950, "failed_requests": 50},
                {"target": "location", "metric_value": 85.0, "total_requests": 800, "successful_requests": 680, "failed_requests": 120}
            ]
        }
        
        result = generate_comparison_chart(comparison_data, "line")
        assert result is not None
    
    def test_comparison_chart_pie(self):
        """Test comparison chart with pie type"""
        comparison_data = {
            "targets": ["customer", "location"],
            "metric": "success_rate",
            "winner": "customer",
            "comparison_details": [
                {"target": "customer", "metric_value": 95.0, "total_requests": 1000, "successful_requests": 950, "failed_requests": 50},
                {"target": "location", "metric_value": 85.0, "total_requests": 800, "successful_requests": 680, "failed_requests": 120}
            ]
        }
        
        result = generate_comparison_chart(comparison_data, "pie")
        assert result is not None
    
    def test_comparison_chart_donut(self):
        """Test comparison chart with donut type"""
        comparison_data = {
            "targets": ["customer", "location"],
            "metric": "success_rate",
            "winner": "customer",
            "comparison_details": [
                {"target": "customer", "metric_value": 95.0, "total_requests": 1000, "successful_requests": 950, "failed_requests": 50},
                {"target": "location", "metric_value": 85.0, "total_requests": 800, "successful_requests": 680, "failed_requests": 120}
            ]
        }
        
        result = generate_comparison_chart(comparison_data, "donut")
        assert result is not None
    
    def test_comparison_chart_area(self):
        """Test comparison chart with area type"""
        comparison_data = {
            "targets": ["customer", "location"],
            "metric": "success_rate",
            "winner": "customer",
            "comparison_details": [
                {"target": "customer", "metric_value": 95.0, "total_requests": 1000, "successful_requests": 950, "failed_requests": 50},
                {"target": "location", "metric_value": 85.0, "total_requests": 800, "successful_requests": 680, "failed_requests": 120}
            ]
        }
        
        result = generate_comparison_chart(comparison_data, "area")
        assert result is not None
    
    def test_comparison_chart_with_three_targets_grouped_bar(self):
        """Test grouped bar comparison chart with three targets"""
        comparison_data = {
            "targets": ["customer", "location", "product"],
            "metric": "success_rate",
            "winner": "customer",
            "comparison_details": [
                {"target": "customer", "metric_value": 95.0, "total_requests": 1000, "successful_requests": 950, "failed_requests": 50},
                {"target": "location", "metric_value": 85.0, "total_requests": 800, "successful_requests": 680, "failed_requests": 120},
                {"target": "product", "metric_value": 90.0, "total_requests": 900, "successful_requests": 810, "failed_requests": 90}
            ]
        }
        
        result = generate_comparison_chart(comparison_data, "grouped_bar")
        assert result is not None
    
    def test_comparison_chart_with_four_targets_line(self):
        """Test line comparison chart with four targets"""
        comparison_data = {
            "targets": ["customer", "location", "product", "order"],
            "metric": "failure_rate",
            "winner": "customer",
            "comparison_details": [
                {"target": "customer", "metric_value": 5.0, "total_requests": 1000, "successful_requests": 950, "failed_requests": 50},
                {"target": "location", "metric_value": 15.0, "total_requests": 800, "successful_requests": 680, "failed_requests": 120},
                {"target": "product", "metric_value": 10.0, "total_requests": 900, "successful_requests": 810, "failed_requests": 90},
                {"target": "order", "metric_value": 8.0, "total_requests": 950, "successful_requests": 874, "failed_requests": 76}
            ]
        }
        
        result = generate_comparison_chart(comparison_data, "line")
        assert result is not None
    
    def test_comparison_chart_with_failure_rate_horizontal_bar(self):
        """Test horizontal bar comparison chart with failure rate"""
        comparison_data = {
            "targets": ["customer", "location"],
            "metric": "failure_rate",
            "winner": "customer",
            "comparison_details": [
                {"target": "customer", "metric_value": 5.0, "total_requests": 1000, "successful_requests": 950, "failed_requests": 50},
                {"target": "location", "metric_value": 15.0, "total_requests": 800, "successful_requests": 680, "failed_requests": 120}
            ]
        }
        
        result = generate_comparison_chart(comparison_data, "horizontal_bar")
        assert result is not None
    
    def test_comparison_chart_default_to_bar_on_unknown_type(self):
        """Test comparison chart defaults to bar for unknown chart type"""
        comparison_data = {
            "targets": ["customer", "location"],
            "metric": "success_rate",
            "winner": "customer",
            "comparison_details": [
                {"target": "customer", "metric_value": 95.0, "total_requests": 1000, "successful_requests": 950, "failed_requests": 50},
                {"target": "location", "metric_value": 85.0, "total_requests": 800, "successful_requests": 680, "failed_requests": 120}
            ]
        }
        
        # Should default to bar chart
        result = generate_comparison_chart(comparison_data, "unknown_type")
        assert result is not None
    
    def test_comparison_chart_with_single_target(self):
        """Test comparison chart with only one target (edge case)"""
        comparison_data = {
            "targets": ["customer"],
            "metric": "success_rate",
            "winner": "customer",
            "comparison_details": [
                {"target": "customer", "metric_value": 95.0, "total_requests": 1000, "successful_requests": 950, "failed_requests": 50}
            ]
        }
        
        # Should still generate chart even with single target
        result = generate_comparison_chart(comparison_data, "grouped_bar")
        assert result is not None


class TestHorizontalBarChartGeneration:
    """Test horizontal bar chart generation directly"""
    
    def test_horizontal_bar_in_generate_analytics_chart(self):
        """Test horizontal bar generation through generate_analytics_chart"""
        data = {
            "target_type": "domain",
            "target_value": "customer",
            "total_requests": 1000,
            "successful_requests": 950,
            "failed_requests": 50,
            "success_rate": 95.0
        }
        
        result = generate_analytics_chart(data, chart_type="success_rate", style="horizontal_bar")
        
        assert result is not None
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert len(decoded) > 0
    
    def test_horizontal_bar_with_failure_rate(self):
        """Test horizontal bar with failure rate"""
        data = {
            "target_type": "file",
            "target_value": "errors.csv",
            "total_requests": 500,
            "successful_requests": 400,
            "failed_requests": 100,
            "failure_rate": 20.0
        }
        
        result = generate_analytics_chart(data, chart_type="failure_rate", style="horizontal_bar")
        
        assert result is not None
        assert isinstance(result, str)
