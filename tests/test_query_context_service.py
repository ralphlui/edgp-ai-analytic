"""
Unit tests for query_context_service.py
Tests DynamoDB conversation context management.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from botocore.exceptions import ClientError

from app.services.query_context_service import QueryContextService


class TestQueryContextServiceInitialization:
    """Test QueryContextService initialization."""
    
    @patch('app.services.query_context_service.boto3')
    def test_service_initialization(self, mock_boto3):
        """Test successful service initialization."""
        mock_dynamodb = Mock()
        mock_boto3.resource.return_value = mock_dynamodb
        
        # Fix: Mock client to return proper dict
        mock_client = Mock()
        mock_client.list_tables.return_value = {'TableNames': ['analytics_conversation_context_test']}
        mock_boto3.client.return_value = mock_client
        
        service = QueryContextService()
        
        assert service.dynamodb == mock_dynamodb
        mock_boto3.resource.assert_called_once_with('dynamodb', region_name='us-east-1')
    
    @patch('app.services.query_context_service.boto3')
    def test_table_creation_if_not_exists(self, mock_boto3):
        """Test table creation when it doesn't exist."""
        mock_client = Mock()
        mock_client.list_tables.return_value = {'TableNames': []}
        mock_boto3.client.return_value = mock_client
        mock_boto3.resource.return_value = Mock()
        
        service = QueryContextService()
        
        # Should attempt to create table
        mock_client.create_table.assert_called_once()


class TestSaveQueryContext:
    """Test saving query context."""
    
    def setup_method(self):
        """Setup test service with mocked DynamoDB."""
        with patch('app.services.query_context_service.boto3'):
            self.service = QueryContextService()
            self.service.table = Mock()
    
    def test_save_context_success(self):
        """Test successful context save."""
        self.service.table.put_item.return_value = {}
        
        result = self.service.save_query_context(
            user_id="user-123",
            intent="success_rate",
            slots={"domain_name": "customer", "file_name": None},
            original_prompt="show me success rate for customer"
        )
        
        # Method returns a dict with saved values, not True
        assert result is not None
        assert result['intent'] == 'success_rate'
        assert result['slots'] == {"domain_name": "customer", "file_name": None}
        self.service.table.put_item.assert_called_once()
        
        # Verify item structure
        call_args = self.service.table.put_item.call_args[1]
        item = call_args['Item']
        assert item['user_id'] == "user-123"
        assert item['report_type'] == "success_rate"
        assert item['slots']['domain_name'] == "customer"
        assert 'timestamp' in item
        assert 'ttl' in item
    
    def test_save_context_with_file_name(self):
        """Test saving context with file_name."""
        self.service.table.put_item.return_value = {}
        
        result = self.service.save_query_context(
            user_id="user-456",
            intent="failure_rate",
            slots={"domain_name": None, "file_name": "data.csv"},
            original_prompt="show failures in data.csv"
        )
        
        # Method returns a dict with saved values, not True
        assert result is not None
        assert result['intent'] == 'failure_rate'
        assert result['slots'] == {"domain_name": None, "file_name": "data.csv"}
        call_args = self.service.table.put_item.call_args[1]
        item = call_args['Item']
        assert item['slots']['file_name'] == "data.csv"
        assert item['report_type'] == "failure_rate"
    
    def test_save_context_dynamodb_error(self):
        """Test handling of DynamoDB errors."""
        self.service.table.put_item.side_effect = ClientError(
            {'Error': {'Code': 'ProvisionedThroughputExceededException', 'Message': 'Throughput exceeded'}},
            'PutItem'
        )
        
        result = self.service.save_query_context(
            user_id="user-123",
            intent="success_rate",
            slots={"domain_name": "customer"},
            original_prompt="test"
        )
        
        assert result is None
    
    def test_save_context_with_empty_intent(self):
        """Test saving context with empty intent and slots."""
        self.service.table.put_item.return_value = {}
        
        result = self.service.save_query_context(
            user_id="user-123",
            intent=None,
            slots={},
            original_prompt="hello"
        )
        
        # Method saves even with empty intent (caller should check should_save_context)
        assert result is not None
        self.service.table.put_item.assert_called_once()


class TestGetFullContext:
    """Test retrieving full context."""
    
    def setup_method(self):
        """Setup test service."""
        with patch('app.services.query_context_service.boto3'):
            self.service = QueryContextService()
            self.service.table = Mock()
    
    def test_get_context_success(self):
        """Test successful context retrieval."""
        mock_item = {
            'user_id': 'user-123',
            'timestamp': 1234567890,
            'report_type': 'success_rate',
            'slots': {'domain_name': 'customer', 'file_name': None},
            'prompts': ['show me success rate']
        }
        
        self.service.table.query.return_value = {
            'Items': [mock_item]
        }
        
        context = self.service.get_full_context('user-123')
        
        # The method transforms report_type to intent
        assert context['intent'] == 'success_rate'
        assert context['slots'] == {'domain_name': 'customer', 'file_name': None}
        assert context['prompts'] == ['show me success rate']
        assert context['timestamp'] == 1234567890
        self.service.table.query.assert_called_once()
    
    def test_get_context_no_data(self):
        """Test when no context exists."""
        self.service.table.query.return_value = {'Items': []}
        
        context = self.service.get_full_context('user-999')
        
        assert context is None
    
    def test_get_context_error(self):
        """Test error handling during retrieval."""
        self.service.table.query.side_effect = ClientError(
            {'Error': {'Code': 'ResourceNotFoundException'}},
            'Query'
        )
        
        context = self.service.get_full_context('user-123')
        
        assert context is None


class TestClearQueryContext:
    """Test clearing query context."""
    
    def setup_method(self):
        """Setup test service."""
        with patch('app.services.query_context_service.boto3'):
            self.service = QueryContextService()
            self.service.table = Mock()
    
    def test_clear_context_success(self):
        """Test successful context clearing."""
        mock_context = {
            'user_id': 'user-123',
            'timestamp': 1234567890
        }
        
        with patch.object(self.service, 'get_full_context', return_value=mock_context):
            result = self.service.clear_query_context('user-123')
            
            assert result is True
            self.service.table.delete_item.assert_called_once_with(
                Key={'user_id': 'user-123', 'timestamp': 1234567890}
            )
    
    def test_clear_context_no_existing_context(self):
        """Test clearing when no context exists."""
        with patch.object(self.service, 'get_full_context', return_value=None):
            result = self.service.clear_query_context('user-999')
            
            assert result is True
            self.service.table.delete_item.assert_not_called()
    
    def test_clear_context_error(self):
        """Test error handling during clear."""
        mock_context = {'user_id': 'user-123', 'timestamp': 1234567890}
        
        with patch.object(self.service, 'get_full_context', return_value=mock_context):
            self.service.table.delete_item.side_effect = ClientError(
                {'Error': {'Code': 'ConditionalCheckFailedException'}},
                'DeleteItem'
            )
            
            result = self.service.clear_query_context('user-123')
            
            assert result is False


class TestShouldSaveContext:
    """Test should_save_context logic."""
    
    def setup_method(self):
        """Setup test service."""
        with patch('app.services.query_context_service.boto3'):
            self.service = QueryContextService()
    
    def test_should_save_with_success_rate_intent(self):
        """Test saving when intent is success_rate."""
        result = self.service.should_save_context(
            "success_rate",
            {"domain_name": "customer"}
        )
        assert result is True
    
    def test_should_save_with_failure_rate_intent(self):
        """Test saving when intent is failure_rate."""
        result = self.service.should_save_context(
            "failure_rate",
            {"file_name": "data.csv"}
        )
        assert result is True
    
    def test_should_save_with_domain_name_only(self):
        """Test saving when only domain_name is present."""
        result = self.service.should_save_context(
            None,
            {"domain_name": "payment"}
        )
        assert result is True
    
    def test_should_save_with_file_name_only(self):
        """Test saving when only file_name is present."""
        result = self.service.should_save_context(
            None,
            {"file_name": "transactions.csv"}
        )
        assert result is True
    
    def test_should_not_save_without_criteria(self):
        """Test not saving when no criteria met."""
        result = self.service.should_save_context(
            None,
            {}
        )
        assert result is False
    
    def test_should_not_save_with_wrong_intent(self):
        """Test not saving with unsupported intent."""
        result = self.service.should_save_context(
            "unknown_intent",
            {}
        )
        assert result is False


class TestUpdateContextSlots:
    """Test updating context slots."""
    
    def setup_method(self):
        """Setup test service."""
        with patch('app.services.query_context_service.boto3'):
            self.service = QueryContextService()
            self.service.table = Mock()
    
    def test_update_slots_success(self):
        """Test successful slot update."""
        mock_context = {
            'user_id': 'user-123',
            'timestamp': 1234567890,
            'report_type': 'success_rate',
            'domain_name': 'customer'
        }
        
        with patch.object(self.service, 'get_full_context', return_value=mock_context):
            self.service.table.update_item.return_value = {}
            
            result = self.service.update_context_slots(
                'user-123',
                timestamp=1234567890,
                new_slots={'domain_name': 'payment'}
            )
            
            assert result is True
            self.service.table.update_item.assert_called_once()
    
    def test_update_slots_no_existing_context(self):
        """Test updating when no context exists (DynamoDB will create it)."""
        self.service.table.update_item.return_value = {}
        
        result = self.service.update_context_slots(
            'user-123',
            timestamp=1234567890,
            new_slots={'domain_name': 'payment'}
        )
        
        # DynamoDB update_item will succeed even if item doesn't exist
        assert result is True


class TestTTLCalculation:
    """Test TTL calculation for context expiry."""
    
    def setup_method(self):
        """Setup test service."""
        with patch('app.services.query_context_service.boto3'):
            self.service = QueryContextService()
            self.service.ttl_hours = 1.0  # 1 hour TTL
    
    def test_ttl_calculation(self):
        """Test that TTL is calculated correctly."""
        import time
        
        current_time = int(time.time())
        expected_ttl = current_time + int(3600)  # 1 hour in seconds
        
        # Mock the save to capture TTL
        self.service.table = Mock()
        self.service.save_query_context(
            user_id="user-123",
            intent="success_rate",
            slots={"domain_name": "customer"},
            original_prompt="test"
        )
        
        call_args = self.service.table.put_item.call_args[1]
        item_ttl = call_args['Item']['ttl']
        
        # Allow 5 second tolerance for test execution time
        assert abs(item_ttl - expected_ttl) < 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=app.services.query_context_service", "--cov-report=term-missing"])
