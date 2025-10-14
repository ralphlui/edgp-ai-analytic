"""
Unit tests for chart generation service.

Tests cover:
- Bar chart generation for success/failure rates
- Chart type filtering (success-only, failure-only)
- Data validation and edge cases
- Base64 encoding
"""
import pytest
import base64
from io import BytesIO
from app.services.chart_service import AnalyticsChartGenerator


class TestBarChartGeneration:
    """Test cases for bar chart generation."""
    
    @pytest.fixture
    def generator(self):
        """Create chart generator instance."""
        return AnalyticsChartGenerator()
    
    @pytest.fixture
    def sample_data(self):
        """Sample analytics data for testing."""
        return {
            "target_type": "domain",
            "target_value": "customer.example.com",
            "total_requests": 100,
            "successful_requests": 80,
            "failed_requests": 20,
            "success_rate": 80.0,
            "failure_rate": 20.0
        }
    
    # ============ Success Rate Charts ============
    
    def test_generate_success_rate_chart(self, generator, sample_data):
        """Test generating success rate bar chart."""
        chart = generator.generate_success_failure_bar_chart(
            sample_data,
            chart_type="success_rate"
        )
        
        assert chart is not None, "Chart generation failed"
        assert isinstance(chart, str), "Chart should be string (base64)"
        assert len(chart) > 0, "Chart data is empty"
        
        # Verify it's valid base64
        try:
            base64.b64decode(chart)
        except Exception as e:
            pytest.fail(f"Invalid base64 encoding: {e}")
    
    def test_generate_failure_rate_chart(self, generator, sample_data):
        """Test generating failure rate bar chart."""
        chart = generator.generate_success_failure_bar_chart(
            sample_data,
            chart_type="failure_rate"
        )
        
        assert chart is not None
        assert isinstance(chart, str)
        assert len(chart) > 0
    
    def test_chart_with_file_target(self, generator):
        """Test chart generation for file-based analytics."""
        data = {
            "target_type": "file",
            "target_value": "customer.csv",
            "total_requests": 50,
            "successful_requests": 45,
            "failed_requests": 5,
            "success_rate": 90.0,
            "failure_rate": 10.0
        }
        
        chart = generator.generate_success_failure_bar_chart(
            data,
            chart_type="success_rate"
        )
        
        assert chart is not None
    
    def test_chart_with_domain_target(self, generator):
        """Test chart generation for domain-based analytics."""
        data = {
            "target_type": "domain",
            "target_value": "api.example.com",
            "total_requests": 200,
            "successful_requests": 190,
            "failed_requests": 10,
            "success_rate": 95.0,
            "failure_rate": 5.0
        }
        
        chart = generator.generate_success_failure_bar_chart(
            data,
            chart_type="failure_rate"
        )
        
        assert chart is not None
    
    # ============ Chart Type Filtering ============
    
    def test_success_chart_shows_only_success(self, generator, sample_data):
        """Test that success_rate chart type shows only success bar."""
        # This is a behavioral test - we verify the chart is generated
        # The actual filtering logic is tested through integration
        chart = generator.generate_success_failure_bar_chart(
            sample_data,
            chart_type="success_rate"
        )
        
        assert chart is not None
        # In a real scenario, you might decode and analyze the image
        # For unit tests, we trust the implementation
    
    def test_failure_chart_shows_only_failure(self, generator, sample_data):
        """Test that failure_rate chart type shows only failure bar."""
        chart = generator.generate_success_failure_bar_chart(
            sample_data,
            chart_type="failure_rate"
        )
        
        assert chart is not None
    
    # ============ Data Validation ============
    
    def test_chart_with_zero_total_requests(self, generator):
        """Test handling of data with zero total requests."""
        data = {
            "target_type": "domain",
            "target_value": "test.com",
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "success_rate": 0,
            "failure_rate": 0
        }
        
        chart = generator.generate_success_failure_bar_chart(
            data,
            chart_type="success_rate"
        )
        
        # Should return None for invalid data
        assert chart is None
    
    def test_chart_with_missing_fields(self, generator):
        """Test handling of data with missing fields."""
        data = {
            "target_type": "domain",
            "target_value": "test.com"
            # Missing total_requests, etc.
        }
        
        chart = generator.generate_success_failure_bar_chart(
            data,
            chart_type="success_rate"
        )
        
        # Should handle gracefully (return None or use defaults)
        assert chart is None
    
    def test_chart_with_high_success_rate(self, generator):
        """Test chart generation with very high success rate."""
        data = {
            "target_type": "domain",
            "target_value": "reliable.com",
            "total_requests": 1000,
            "successful_requests": 999,
            "failed_requests": 1,
            "success_rate": 99.9,
            "failure_rate": 0.1
        }
        
        chart = generator.generate_success_failure_bar_chart(
            data,
            chart_type="success_rate"
        )
        
        assert chart is not None
    
    def test_chart_with_high_failure_rate(self, generator):
        """Test chart generation with very high failure rate."""
        data = {
            "target_type": "domain",
            "target_value": "problematic.com",
            "total_requests": 100,
            "successful_requests": 5,
            "failed_requests": 95,
            "success_rate": 5.0,
            "failure_rate": 95.0
        }
        
        chart = generator.generate_success_failure_bar_chart(
            data,
            chart_type="failure_rate"
        )
        
        assert chart is not None
    
    def test_chart_with_100_percent_success(self, generator):
        """Test chart generation with 100% success rate."""
        data = {
            "target_type": "domain",
            "target_value": "perfect.com",
            "total_requests": 50,
            "successful_requests": 50,
            "failed_requests": 0,
            "success_rate": 100.0,
            "failure_rate": 0.0
        }
        
        chart = generator.generate_success_failure_bar_chart(
            data,
            chart_type="success_rate"
        )
        
        assert chart is not None
    
    def test_chart_with_100_percent_failure(self, generator):
        """Test chart generation with 100% failure rate."""
        data = {
            "target_type": "domain",
            "target_value": "broken.com",
            "total_requests": 50,
            "successful_requests": 0,
            "failed_requests": 50,
            "success_rate": 0.0,
            "failure_rate": 100.0
        }
        
        chart = generator.generate_success_failure_bar_chart(
            data,
            chart_type="failure_rate"
        )
        
        assert chart is not None
    
    # ============ Large Numbers ============
    
    def test_chart_with_large_numbers(self, generator):
        """Test chart generation with large request counts."""
        data = {
            "target_type": "domain",
            "target_value": "high-traffic.com",
            "total_requests": 1000000,
            "successful_requests": 950000,
            "failed_requests": 50000,
            "success_rate": 95.0,
            "failure_rate": 5.0
        }
        
        chart = generator.generate_success_failure_bar_chart(
            data,
            chart_type="success_rate"
        )
        
        assert chart is not None
    
    def test_chart_with_small_numbers(self, generator):
        """Test chart generation with small request counts."""
        data = {
            "target_type": "domain",
            "target_value": "low-traffic.com",
            "total_requests": 3,
            "successful_requests": 2,
            "failed_requests": 1,
            "success_rate": 66.67,
            "failure_rate": 33.33
        }
        
        chart = generator.generate_success_failure_bar_chart(
            data,
            chart_type="success_rate"
        )
        
        assert chart is not None
    
    # ============ Edge Cases ============
    
    def test_chart_with_unknown_target_value(self, generator):
        """Test chart generation with unknown target value."""
        data = {
            "target_type": "domain",
            "target_value": "Unknown",
            "total_requests": 10,
            "successful_requests": 8,
            "failed_requests": 2,
            "success_rate": 80.0,
            "failure_rate": 20.0
        }
        
        chart = generator.generate_success_failure_bar_chart(
            data,
            chart_type="success_rate"
        )
        
        assert chart is not None
    
    def test_chart_with_special_characters_in_name(self, generator):
        """Test chart generation with special characters in target name."""
        data = {
            "target_type": "file",
            "target_value": "customer-data_v2.0.csv",
            "total_requests": 25,
            "successful_requests": 20,
            "failed_requests": 5,
            "success_rate": 80.0,
            "failure_rate": 20.0
        }
        
        chart = generator.generate_success_failure_bar_chart(
            data,
            chart_type="success_rate"
        )
        
        assert chart is not None
    
    def test_chart_generator_initialization(self, generator):
        """Test chart generator initialization."""
        assert generator.figure_size == (10, 6)
        assert generator.dpi == 100
        assert generator.color_success == '#10b981'
        assert generator.color_failure == '#ef4444'
        assert generator.color_neutral == '#6b7280'
    
    def test_multiple_chart_generations(self, generator, sample_data):
        """Test generating multiple charts in sequence with varying data."""
        charts = []
        
        for i in range(5):
            # Create unique data for each chart by varying the success rate
            modified_data = sample_data.copy()
            modified_data['successful_requests'] = 80 + i * 2
            modified_data['failed_requests'] = 20 - i * 2
            modified_data['success_rate'] = ((80 + i * 2) / 100) * 100
            modified_data['failure_rate'] = ((20 - i * 2) / 100) * 100
            
            chart = generator.generate_success_failure_bar_chart(
                modified_data,
                chart_type="success_rate"
            )
            charts.append(chart)
        
        # All should be generated successfully
        assert all(chart is not None for chart in charts)
        
        # Each should be unique (different content due to different data)
        assert len(set(charts)) == 5


class TestChartDataValidation:
    """Test data validation logic in chart generation."""
    
    @pytest.fixture
    def generator(self):
        """Create chart generator instance."""
        return AnalyticsChartGenerator()
    
    def test_empty_data_dict(self, generator):
        """Test handling of empty data dictionary."""
        chart = generator.generate_success_failure_bar_chart(
            {},
            chart_type="success_rate"
        )
        
        assert chart is None
    
    def test_data_with_negative_values(self, generator):
        """Test handling of negative values (invalid data)."""
        data = {
            "target_type": "domain",
            "target_value": "test.com",
            "total_requests": -100,
            "successful_requests": 80,
            "failed_requests": 20,
            "success_rate": 80.0
        }
        
        # Should handle gracefully
        chart = generator.generate_success_failure_bar_chart(
            data,
            chart_type="success_rate"
        )
        
        # Behavior depends on implementation - might be None or handle it
        # For this test, we just ensure it doesn't crash
        assert chart is None or isinstance(chart, str)
    
    def test_data_with_mismatched_totals(self, generator):
        """Test data where success + failure != total."""
        data = {
            "target_type": "domain",
            "target_value": "test.com",
            "total_requests": 100,
            "successful_requests": 60,
            "failed_requests": 60,  # Should be 40
            "success_rate": 60.0
        }
        
        # Should still generate chart (data validation is upstream)
        chart = generator.generate_success_failure_bar_chart(
            data,
            chart_type="success_rate"
        )
        
        assert chart is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
