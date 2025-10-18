"""
Unit tests for AnalyticsRepository.

Tests cover:
- Repository initialization
- Success rate queries by domain
- Success rate queries by file
- Failure rate queries by domain
- Failure rate queries by file
- DynamoDB query operations
- Metrics calculations
- Time series building
- Organization filtering
- Error handling
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from decimal import Decimal


class TestAnalyticsRepositoryInitialization:
    """Test repository initialization."""
    
    @patch('app.repositories.analytics_repository.boto3.resource')
    def test_init_with_default_table_name(self, mock_boto_resource):
        """Test initialization with default table name."""
        from app.repositories.analytics_repository import AnalyticsRepository
        
        mock_dynamodb = MagicMock()
        mock_boto_resource.return_value = mock_dynamodb
        
        repo = AnalyticsRepository()
        
        assert repo.table_name == "analytics_events"
        mock_boto_resource.assert_called_once_with('dynamodb')
        mock_dynamodb.Table.assert_any_call("analytics_events")
    
    @patch('app.repositories.analytics_repository.boto3.resource')
    def test_init_with_custom_table_name(self, mock_boto_resource):
        """Test initialization with custom table name."""
        from app.repositories.analytics_repository import AnalyticsRepository
        
        mock_dynamodb = MagicMock()
        mock_boto_resource.return_value = mock_dynamodb
        
        repo = AnalyticsRepository(table_name="custom_table")
        
        assert repo.table_name == "custom_table"
        mock_dynamodb.Table.assert_any_call("custom_table")


class TestGetSuccessRateByDomain:
    """Test get_success_rate_by_domain method."""
    
    @pytest.fixture
    def mock_repository(self):
        """Create mock repository."""
        with patch('app.repositories.analytics_repository.boto3.resource'):
            from app.repositories.analytics_repository import AnalyticsRepository
            return AnalyticsRepository()
    
    def test_success_rate_calculation_all_successful(self, mock_repository):
        """Test success rate when all requests succeed."""
        mock_items = [
            {"final_status": "success", "timestamp": "2024-01-01T10:00:00"},
            {"final_status": "success", "timestamp": "2024-01-01T11:00:00"},
            {"final_status": "success", "timestamp": "2024-01-01T12:00:00"}
        ]
        
        with patch.object(mock_repository, '_query_by_domain', return_value=mock_items):
            result = mock_repository.get_success_rate_by_domain("customer")
        
        assert result["total_requests"] == 3
        assert result["successful_requests"] == 3
        assert result["failed_requests"] == 0
        assert result["success_rate"] == 100.0
        assert result["target_type"] == "domain"
        assert result["target_value"] == "customer"
    
    def test_success_rate_calculation_mixed(self, mock_repository):
        """Test success rate with mixed success/failure."""
        mock_items = [
            {"final_status": "success", "timestamp": "2024-01-01T10:00:00"},
            {"final_status": "failure", "timestamp": "2024-01-01T11:00:00"},
            {"final_status": "success", "timestamp": "2024-01-01T12:00:00"}
        ]
        
        with patch.object(mock_repository, '_query_by_domain', return_value=mock_items):
            result = mock_repository.get_success_rate_by_domain("product")
        
        assert result["total_requests"] == 3
        assert result["successful_requests"] == 2
        assert result["failed_requests"] == 1
        assert abs(result["success_rate"] - 66.67) < 0.01
    
    def test_success_rate_no_data(self, mock_repository):
        """Test success rate when no data exists."""
        with patch.object(mock_repository, '_query_by_domain', return_value=[]):
            result = mock_repository.get_success_rate_by_domain("vendor")
        
        assert result["total_requests"] == 0
        assert result["successful_requests"] == 0
        assert result["failed_requests"] == 0
        assert result["success_rate"] == 0.0
    
    def test_success_rate_with_org_id(self, mock_repository):
        """Test success rate filtering by org_id."""
        mock_items = [
            {"status": "success", "org_id": "org-123", "timestamp": "2024-01-01T10:00:00"}
        ]
        
        with patch.object(mock_repository, '_query_by_domain', return_value=mock_items) as mock_query:
            result = mock_repository.get_success_rate_by_domain("location", org_id="org-123")
        
        mock_query.assert_called_once_with("location", org_id="org-123")
        assert result["total_requests"] == 1


class TestGetSuccessRateByFile:
    """Test get_success_rate_by_file method."""
    
    @pytest.fixture
    def mock_repository(self):
        """Create mock repository."""
        with patch('app.repositories.analytics_repository.boto3.resource'):
            from app.repositories.analytics_repository import AnalyticsRepository
            return AnalyticsRepository()
    
    def test_success_rate_by_file_calculation(self, mock_repository):
        """Test success rate calculation for file-based query."""
        mock_items = [
            {"final_status": "success", "file_name": "customers.csv", "timestamp": "2024-01-01T10:00:00"},
            {"final_status": "failure", "file_name": "customers.csv", "timestamp": "2024-01-01T11:00:00"}
        ]
        
        with patch.object(mock_repository, '_query_by_file', return_value=mock_items):
            result = mock_repository.get_success_rate_by_file("customers.csv")
        
        assert result["total_requests"] == 2
        assert result["successful_requests"] == 1
        assert result["failed_requests"] == 1
        assert result["success_rate"] == 50.0
        assert result["target_type"] == "file"
        assert result["target_value"] == "customers.csv"
    
    def test_success_rate_by_file_with_org_id(self, mock_repository):
        """Test file-based success rate with org_id filtering."""
        mock_items = [
            {"status": "success", "org_id": "org-456", "timestamp": "2024-01-01T10:00:00"}
        ]
        
        with patch.object(mock_repository, '_query_by_file', return_value=mock_items) as mock_query:
            result = mock_repository.get_success_rate_by_file("products.xlsx", org_id="org-456")
        
        mock_query.assert_called_once_with("products.xlsx", org_id="org-456")
        assert result["total_requests"] == 1


class TestGetFailureRateByDomain:
    """Test get_failure_rate_by_domain method."""
    
    @pytest.fixture
    def mock_repository(self):
        """Create mock repository."""
        with patch('app.repositories.analytics_repository.boto3.resource'):
            from app.repositories.analytics_repository import AnalyticsRepository
            return AnalyticsRepository()
    
    def test_failure_rate_all_failed(self, mock_repository):
        """Test failure rate when all requests fail."""
        mock_items = [
            {"status": "failure", "timestamp": "2024-01-01T10:00:00"},
            {"status": "failure", "timestamp": "2024-01-01T11:00:00"}
        ]
        
        with patch.object(mock_repository, '_query_by_domain', return_value=mock_items):
            result = mock_repository.get_failure_rate_by_domain("vendor")
        
        assert result["total_requests"] == 2
        assert result["failed_requests"] == 2
        assert result["successful_requests"] == 0
        assert result["failure_rate"] == 100.0
    
    def test_failure_rate_mixed(self, mock_repository):
        """Test failure rate with mixed results."""
        mock_items = [
            {"final_status": "success", "timestamp": "2024-01-01T10:00:00"},
            {"final_status": "failure", "timestamp": "2024-01-01T11:00:00"},
            {"final_status": "failure", "timestamp": "2024-01-01T12:00:00"},
            {"final_status": "failure", "timestamp": "2024-01-01T13:00:00"}
        ]
        
        with patch.object(mock_repository, '_query_by_domain', return_value=mock_items):
            result = mock_repository.get_failure_rate_by_domain("customer")
        
        assert result["total_requests"] == 4
        assert result["failed_requests"] == 3
        assert result["successful_requests"] == 1
        assert result["failure_rate"] == 75.0


class TestGetFailureRateByFile:
    """Test get_failure_rate_by_file method."""
    
    @pytest.fixture
    def mock_repository(self):
        """Create mock repository."""
        with patch('app.repositories.analytics_repository.boto3.resource'):
            from app.repositories.analytics_repository import AnalyticsRepository
            return AnalyticsRepository()
    
    def test_failure_rate_by_file(self, mock_repository):
        """Test failure rate calculation for file-based query."""
        mock_items = [
            {"final_status": "failure", "timestamp": "2024-01-01T10:00:00"},
            {"final_status": "failure", "timestamp": "2024-01-01T11:00:00"},
            {"final_status": "success", "timestamp": "2024-01-01T12:00:00"}
        ]
        
        with patch.object(mock_repository, '_query_by_file', return_value=mock_items):
            result = mock_repository.get_failure_rate_by_file("data.json")
        
        assert result["total_requests"] == 3
        assert result["failed_requests"] == 2
        assert result["failure_rate"] == pytest.approx(66.67, rel=0.01)
        assert result["target_type"] == "file"


class TestQueryByDomain:
    """Test _query_by_domain private method."""
    
    @pytest.fixture
    def mock_repository(self):
        """Create mock repository with mocked table."""
        with patch('app.repositories.analytics_repository.boto3.resource') as mock_boto:
            from app.repositories.analytics_repository import AnalyticsRepository
            
            mock_table = MagicMock()
            mock_boto.return_value.Table.return_value = mock_table
            
            repo = AnalyticsRepository()
            repo.table = mock_table
            return repo
    
    def test_query_by_domain_basic(self, mock_repository):
        """Test basic domain query without filters."""
        mock_response = {
            "Items": [
                {"domain_name": "customer", "final_status": "success"},
                {"domain_name": "customer", "final_status": "failure"}
            ]
        }
        mock_repository.table.scan.return_value = mock_response
        
        result = mock_repository._query_by_domain("customer")
        
        assert len(result) == 2
        mock_repository.table.scan.assert_called_once()
    
    def test_query_by_domain_with_org_id(self, mock_repository):
        """Test domain query with org_id filtering."""
        mock_response = {
            "Items": [
                {"domain_name": "product", "org_id": "org-123", "final_status": "success"}
            ]
        }
        mock_repository.table.scan.return_value = mock_response
        
        result = mock_repository._query_by_domain("product", org_id="org-123")
        
        assert len(result) == 1
        assert result[0]["org_id"] == "org-123"
    
    def test_query_by_domain_pagination(self, mock_repository):
        """Test domain query with pagination."""
        # First call returns data with LastEvaluatedKey
        mock_repository.table.scan.side_effect = [
            {
                "Items": [{"domain_name": "location", "final_status": "success"}],
                "LastEvaluatedKey": {"domain_name": "location", "timestamp": "2024-01-01"}
            },
            {
                "Items": [{"domain_name": "location", "final_status": "failure"}]
            }
        ]
        
        result = mock_repository._query_by_domain("location")
        
        # Should have called scan twice due to pagination
        assert mock_repository.table.scan.call_count == 2
        assert len(result) == 2


class TestQueryByFile:
    """Test _query_by_file private method."""
    
    @pytest.fixture
    def mock_repository(self):
        """Create mock repository with mocked table."""
        with patch('app.repositories.analytics_repository.boto3.resource') as mock_boto:
            from app.repositories.analytics_repository import AnalyticsRepository
            
            mock_table = MagicMock()
            mock_boto.return_value.Table.return_value = mock_table
            
            repo = AnalyticsRepository()
            repo.table = mock_table
            return repo
    
    def test_query_by_file_basic(self, mock_repository):
        """Test basic file query returns empty when no data found."""
        # Mock header table to return no items  
        mock_repository.header_table.scan.return_value = {"Items": []}
        
        result = mock_repository._query_by_file("data.csv")
        
        # Should return empty list when no header found
        assert result == []
        mock_repository.header_table.scan.assert_called_once()
    
    def test_query_by_file_with_org_id(self, mock_repository):
        """Test file query with org_id filtering returns empty when no data."""
        # Mock header table to return no items
        mock_repository.header_table.scan.return_value = {"Items": []}
        
        result = mock_repository._query_by_file("report.xlsx", org_id="org-789")
        
        # Should return empty list when no header found
        assert result == []
        # Verify org_id was part of the query intent
        mock_repository.header_table.scan.assert_called_once()


class TestCalculateMetrics:
    """Test _calculate_metrics private method."""
    
    @pytest.fixture
    def mock_repository(self):
        """Create mock repository."""
        with patch('app.repositories.analytics_repository.boto3.resource'):
            from app.repositories.analytics_repository import AnalyticsRepository
            return AnalyticsRepository()
    
    def test_calculate_metrics_empty_list(self, mock_repository):
        """Test metrics calculation with empty list."""
        result = mock_repository._calculate_metrics([])
        
        assert result["total"] == 0
        assert result["successful"] == 0
        assert result["failed"] == 0
        assert result["success_rate"] == 0.0
    
    def test_calculate_metrics_all_successful(self, mock_repository):
        """Test metrics with all successful requests."""
        items = [
            {"final_status": "success"},
            {"final_status": "success"},
            {"final_status": "success"}
        ]
        
        result = mock_repository._calculate_metrics(items)
        
        assert result["total"] == 3
        assert result["successful"] == 3
        assert result["failed"] == 0
        assert result["success_rate"] == 100.0
    
    def test_calculate_metrics_mixed(self, mock_repository):
        """Test metrics with mixed results."""
        items = [
            {"final_status": "success"},
            {"final_status": "failure"},
            {"final_status": "success"},
            {"final_status": "failure"},
            {"final_status": "failure"}
        ]
        
        result = mock_repository._calculate_metrics(items)
        
        assert result["total"] == 5
        assert result["successful"] == 2
        assert result["failed"] == 3
        assert result["success_rate"] == 40.0


class TestBuildTimeSeries:
    """Test _build_time_series private method."""
    
    @pytest.fixture
    def mock_repository(self):
        """Create mock repository."""
        with patch('app.repositories.analytics_repository.boto3.resource'):
            from app.repositories.analytics_repository import AnalyticsRepository
            return AnalyticsRepository()
    
    def test_build_time_series_hourly(self, mock_repository):
        """Test building time series data."""
        items = [
            {"timestamp": "2024-01-01T10:00:00", "final_status": "success"},
            {"timestamp": "2024-01-01T10:30:00", "final_status": "failure"},
            {"timestamp": "2024-01-01T11:00:00", "final_status": "success"}
        ]
        
        result = mock_repository._build_time_series(items, metric_type="success_rate")
        
        assert isinstance(result, list)
        # Should have at least one day of data
        assert len(result) >= 1
    
    def test_build_time_series_daily(self, mock_repository):
        """Test building daily time series data."""
        items = [
            {"timestamp": "2024-01-01T10:00:00", "final_status": "success"},
            {"timestamp": "2024-01-02T10:00:00", "final_status": "failure"}
        ]
        
        result = mock_repository._build_time_series(items, metric_type="failure_rate")
        
        assert isinstance(result, list)
        assert len(result) >= 2


class TestDebugScanSample:
    """Test debug_scan_sample method."""
    
    @pytest.fixture
    def mock_repository(self):
        """Create mock repository with mocked table."""
        with patch('app.repositories.analytics_repository.boto3.resource') as mock_boto:
            from app.repositories.analytics_repository import AnalyticsRepository
            
            mock_table = MagicMock()
            mock_boto.return_value.Table.return_value = mock_table
            
            repo = AnalyticsRepository()
            repo.table = mock_table
            return repo
    
    def test_debug_scan_sample_default_limit(self, mock_repository):
        """Test debug scan with default limit."""
        mock_response = {
            "Items": [
                {"id": "1", "status": "success"},
                {"id": "2", "status": "failure"}
            ]
        }
        mock_repository.table.scan.return_value = mock_response
        
        result = mock_repository.debug_scan_sample()
        
        assert len(result) == 2
        mock_repository.table.scan.assert_called_once_with(Limit=5)
    
    def test_debug_scan_sample_custom_limit(self, mock_repository):
        """Test debug scan with custom limit."""
        mock_response = {
            "Items": [{"id": str(i)} for i in range(10)]
        }
        mock_repository.table.scan.return_value = mock_response
        
        result = mock_repository.debug_scan_sample(limit=10)
        
        assert len(result) == 10
        mock_repository.table.scan.assert_called_once_with(Limit=10)


class TestGetAnalyticsRepository:
    """Test get_analytics_repository singleton function."""
    
    @patch('app.repositories.analytics_repository.boto3.resource')
    @patch('app.repositories.analytics_repository.DYNAMODB_TRACKER_TABLE_NAME', 'test_table')
    def test_get_analytics_repository_default_table(self, mock_boto_resource):
        """Test that get_analytics_repository uses default table from config."""
        from app.repositories.analytics_repository import get_analytics_repository
        
        repo = get_analytics_repository()
        
        # Should use the table name from config
        assert repo.table_name == 'test_table'
    
    @patch('app.repositories.analytics_repository.boto3.resource')
    def test_get_analytics_repository_custom_table(self, mock_boto_resource):
        """Test get_analytics_repository with custom table name."""
        from app.repositories.analytics_repository import get_analytics_repository
        
        repo = get_analytics_repository(table_name="custom_analytics")
        
        assert repo.table_name == "custom_analytics"


class TestQueryByDomainEdgeCases:
    """Test edge cases in _query_by_domain method."""
    
    @patch('app.repositories.analytics_repository.boto3.resource')
    def test_query_by_domain_no_items_found(self, mock_boto_resource):
        """Test _query_by_domain when no items match domain."""
        from app.repositories.analytics_repository import AnalyticsRepository
        
        mock_dynamodb = MagicMock()
        mock_boto_resource.return_value = mock_dynamodb
        
        # Mock scan to return empty items list
        mock_table = Mock()
        mock_table.scan.return_value = {'Items': []}
        mock_dynamodb.Table.return_value = mock_table
        
        repo = AnalyticsRepository()
        result = repo._query_by_domain('nonexistent_domain')
        
        # Should return empty list and log warning
        assert result == []
    
    @patch('app.repositories.analytics_repository.boto3.resource')
    def test_query_by_domain_exception_handling(self, mock_boto_resource):
        """Test _query_by_domain handles exceptions."""
        from app.repositories.analytics_repository import AnalyticsRepository
        
        mock_dynamodb = MagicMock()
        mock_boto_resource.return_value = mock_dynamodb
        
        # Mock scan to raise exception
        mock_table = Mock()
        mock_table.scan.side_effect = Exception("DynamoDB error")
        mock_dynamodb.Table.return_value = mock_table
        
        repo = AnalyticsRepository()
        result = repo._query_by_domain('customer')
        
        # Should return empty list on error
        assert result == []


class TestQueryByFileEdgeCases:
    """Test edge cases in _query_by_file method."""
    
    @patch('app.repositories.analytics_repository.boto3.resource')
    def test_query_by_file_no_header_found(self, mock_boto_resource):
        """Test _query_by_file when file not found in header table."""
        from app.repositories.analytics_repository import AnalyticsRepository
        
        mock_dynamodb = MagicMock()
        mock_boto_resource.return_value = mock_dynamodb
        
        # Mock header table scan to return empty
        mock_header_table = Mock()
        mock_header_table.scan.return_value = {'Items': []}
        
        mock_tracker_table = Mock()
        
        # Setup Table() to return different tables for header vs tracker
        def table_selector(name):
            if 'header' in name:
                return mock_header_table
            else:
                return mock_tracker_table
        
        mock_dynamodb.Table.side_effect = table_selector
        
        repo = AnalyticsRepository()
        result = repo._query_by_file('nonexistent.csv')
        
        # Should return empty list when file not found
        assert result == []
    
    @patch('app.repositories.analytics_repository.boto3.resource')
    def test_query_by_file_missing_id_field(self, mock_boto_resource):
        """Test _query_by_file when header record has no 'id' field."""
        from app.repositories.analytics_repository import AnalyticsRepository
        
        mock_dynamodb = MagicMock()
        mock_boto_resource.return_value = mock_dynamodb
        
        # Mock header table scan to return item without 'id' field
        mock_header_table = Mock()
        mock_header_table.scan.return_value = {
            'Items': [{'file_name': 'customer.csv'}]  # Missing 'id' field
        }
        
        mock_tracker_table = Mock()
        
        def table_selector(name):
            if 'header' in name:
                return mock_header_table
            else:
                return mock_tracker_table
        
        mock_dynamodb.Table.side_effect = table_selector
        
        repo = AnalyticsRepository()
        result = repo._query_by_file('customer.csv')
        
        # Should return empty list when id field missing
        assert result == []
    
    @patch('app.repositories.analytics_repository.boto3.resource')
    def test_query_by_file_with_pagination(self, mock_boto_resource):
        """Test _query_by_file handles pagination correctly."""
        from app.repositories.analytics_repository import AnalyticsRepository
        
        mock_dynamodb = MagicMock()
        mock_boto_resource.return_value = mock_dynamodb
        
        # Create persistent mock tables
        mock_header_table = Mock()
        mock_header_table.scan.return_value = {
            'Items': [{'file_name': 'customer.csv', 'id': 'file-123'}]
        }
        
        # Mock tracker table with pagination
        mock_tracker_table = Mock()
        mock_tracker_table.scan.side_effect = [
            {
                'Items': [
                    {'file_id': 'file-123', 'final_status': 'success'}
                ],
                'LastEvaluatedKey': {'id': 'last-key'}
            },
            {
                'Items': [
                    {'file_id': 'file-123', 'final_status': 'failure'}
                ]
            }
        ]
        
        # Use a dictionary to map table names to mock tables
        table_map = {
            'analytics_events': mock_tracker_table,
            'MasterDataHeaderTEST': mock_header_table
        }
        
        mock_dynamodb.Table.side_effect = lambda name: table_map.get(name, mock_tracker_table)
        
        repo = AnalyticsRepository()
        result = repo._query_by_file('customer.csv')
        
        # Should return all items from both pages
        assert len(result) == 2
        assert mock_tracker_table.scan.call_count == 2


class TestBuildTimeSeriesEdgeCases:
    """Test edge cases in _build_time_series method."""
    
    @patch('app.repositories.analytics_repository.boto3.resource')
    def test_build_time_series_with_no_timestamp(self, mock_boto_resource):
        """Test _build_time_series handles items without timestamp."""
        from app.repositories.analytics_repository import AnalyticsRepository
        
        mock_dynamodb = MagicMock()
        mock_boto_resource.return_value = mock_dynamodb
        
        repo = AnalyticsRepository()
        
        items = [
            {'final_status': 'success'},  # No timestamp
            {'timestamp': '', 'final_status': 'success'},  # Empty timestamp
        ]
        
        result = repo._build_time_series(items)
        
        # Should skip items without valid timestamps
        assert result == []
    
    @patch('app.repositories.analytics_repository.boto3.resource')
    def test_build_time_series_with_zero_total(self, mock_boto_resource):
        """Test _build_time_series handles zero total edge case."""
        from app.repositories.analytics_repository import AnalyticsRepository
        
        mock_dynamodb = MagicMock()
        mock_boto_resource.return_value = mock_dynamodb
        
        repo = AnalyticsRepository()
        
        # Items that will result in 0 success and 0 failure
        items = [
            {'timestamp': '2024-01-01T10:00:00Z', 'final_status': 'unknown'}
        ]
        
        result = repo._build_time_series(items)
        
        # Should handle zero total gracefully (though won't appear in output)
        assert isinstance(result, list)


class TestDebugScanSampleEdgeCases:
    """Test edge cases in debug_scan_sample method."""
    
    @patch('app.repositories.analytics_repository.boto3.resource')
    def test_debug_scan_sample_exception_handling(self, mock_boto_resource):
        """Test debug_scan_sample handles exceptions."""
        from app.repositories.analytics_repository import AnalyticsRepository
        
        mock_dynamodb = MagicMock()
        mock_boto_resource.return_value = mock_dynamodb
        
        # Mock scan to raise exception
        mock_table = Mock()
        mock_table.scan.side_effect = Exception("DynamoDB error")
        mock_dynamodb.Table.return_value = mock_table
        
        repo = AnalyticsRepository()
        result = repo.debug_scan_sample()
        
        # Should return empty list on error
        assert result == []
    
    @patch('app.repositories.analytics_repository.boto3.resource')
    def test_debug_scan_sample_with_items(self, mock_boto_resource):
        """Test debug_scan_sample returns items successfully."""
        from app.repositories.analytics_repository import AnalyticsRepository
        
        mock_dynamodb = MagicMock()
        mock_boto_resource.return_value = mock_dynamodb
        
        # Mock scan to return sample items
        mock_table = Mock()
        mock_table.scan.return_value = {
            'Items': [
                {'domain_name': 'customer', 'final_status': 'success'},
                {'domain_name': 'payment', 'final_status': 'failure'}
            ]
        }
        mock_dynamodb.Table.return_value = mock_table
        
        repo = AnalyticsRepository()
        result = repo.debug_scan_sample(limit=5)
        
        # Should return the items
        assert len(result) == 2
        assert result[0]['domain_name'] == 'customer'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
