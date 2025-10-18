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


class TestGetFullContext:
    """Test getting full context."""
    
    def setup_method(self):
        """Setup test service."""
        with patch('app.services.query_context_service.boto3'):
            self.service = QueryContextService()
            self.service.table = Mock()
    
    def test_get_full_context_success(self):
        """Test successful retrieval of full context."""
        mock_items = [{
            'user_id': 'user-123',
            'timestamp': 1234567890,
            'report_type': 'success_rate',
            'slots': {'domain_name': 'customer'},
            'prompts': [{'prompt': 'test', 'timestamp': '2024-01-01T00:00:00'}],
            'ttl': 1234999999
        }]
        
        self.service.table.query.return_value = {'Items': mock_items}
        
        result = self.service.get_full_context('user-123')
        
        assert result is not None
        # get_full_context returns 'intent' key (not 'user_id')
        assert result['intent'] == 'success_rate'
        assert result['slots'] == {'domain_name': 'customer'}
    
    def test_get_full_context_no_items(self):
        """Test when no context exists."""
        self.service.table.query.return_value = {'Items': []}
        
        result = self.service.get_full_context('user-123')
        
        assert result is None
    
    def test_get_full_context_dynamodb_error(self):
        """Test DynamoDB error handling."""
        error_response = {'Error': {'Code': 'ProvisionedThroughputExceededException'}}
        self.service.table.query.side_effect = ClientError(error_response, 'Query')
        
        result = self.service.get_full_context('user-123')
        
        assert result is None


class TestUpdateExistingRecord:
    """Test _update_existing_record method."""
    
    def setup_method(self):
        """Setup test service."""
        with patch('app.services.query_context_service.boto3'):
            self.service = QueryContextService()
            self.service.table = Mock()
    
    def test_update_existing_record_success(self):
        """Test successful record update."""
        self.service.table.update_item.return_value = {}
        
        result = self.service._update_existing_record(
            user_id='user-123',
            timestamp=1234567890,
            new_intent='success_rate',
            new_slots={'domain_name': 'customer'},
            new_prompt='show me success rate'
        )
        
        assert result is True
        self.service.table.update_item.assert_called_once()
    
    def test_update_existing_record_with_comparison_targets(self):
        """Test updating with comparison targets."""
        self.service.table.update_item.return_value = {}
        
        result = self.service._update_existing_record(
            user_id='user-123',
            timestamp=1234567890,
            new_intent='compare',
            new_slots={'domain_name': 'customer'},
            new_prompt='compare them',
            new_comparison_targets=['customer.csv', 'product.csv']
        )
        
        assert result is True
    
    def test_update_existing_record_no_prompt(self):
        """Test update fails when no prompt provided."""
        result = self.service._update_existing_record(
            user_id='user-123',
            timestamp=1234567890,
            new_intent='success_rate',
            new_slots={'domain_name': 'customer'},
            new_prompt=''  # Empty prompt
        )
        
        assert result is False
    
    def test_update_existing_record_dynamodb_error(self):
        """Test DynamoDB error handling during update."""
        error_response = {'Error': {'Code': 'ConditionalCheckFailedException'}}
        self.service.table.update_item.side_effect = ClientError(error_response, 'UpdateItem')
        
        result = self.service._update_existing_record(
            user_id='user-123',
            timestamp=1234567890,
            new_intent='success_rate',
            new_slots={'domain_name': 'customer'},
            new_prompt='test'
        )
        
        assert result is False


class TestSaveQueryContextUpdateScenarios:
    """Test save_query_context when updating existing records."""
    
    def setup_method(self):
        """Setup test service."""
        with patch('app.services.query_context_service.boto3'):
            self.service = QueryContextService()
            self.service.table = Mock()
    
    def test_save_updates_existing_record(self):
        """Test that save updates an existing record."""
        # Mock get_full_context to return existing record
        existing_record = {
            'user_id': 'user-123',
            'timestamp': 1234567890,
            'intent': 'success_rate',
            'slots': {'domain_name': 'customer'},
            'prompts': [{'prompt': 'first', 'timestamp': '2024-01-01T00:00:00'}]
        }
        
        with patch.object(self.service, 'get_full_context', return_value=existing_record):
            with patch.object(self.service, '_update_existing_record', return_value=True):
                result = self.service.save_query_context(
                    user_id='user-123',
                    intent='failure_rate',  # Different intent
                    slots={'domain_name': 'payment'},  # Different slots
                    original_prompt='show failures'
                )
                
                assert result is not None
    
    def test_save_creates_new_record_when_no_existing(self):
        """Test that save creates new record when none exists."""
        with patch.object(self.service, 'get_full_context', return_value=None):
            self.service.table.put_item.return_value = {}
            
            result = self.service.save_query_context(
                user_id='user-456',
                intent='success_rate',
                slots={'file_name': 'data.csv'},
                original_prompt='analyze data.csv'
            )
            
            assert result is not None
            assert result['intent'] == 'success_rate'
            self.service.table.put_item.assert_called_once()


class TestClearQueryContext:
    """Test clearing query context."""
    
    def setup_method(self):
        """Setup test service."""
        with patch('app.services.query_context_service.boto3'):
            self.service = QueryContextService()
            self.service.table = Mock()
    
    def test_clear_context_success(self):
        """Test successful context deletion."""
        mock_items = [{
            'user_id': 'user-123',
            'timestamp': 1234567890
        }]
        self.service.table.query.return_value = {'Items': mock_items}
        self.service.table.delete_item.return_value = {}
        
        result = self.service.clear_query_context('user-123')
        
        assert result is True
        self.service.table.delete_item.assert_called_once()
    
    def test_clear_context_no_items(self):
        """Test clearing when no context exists."""
        # Mock get_full_context to return None
        with patch.object(self.service, 'get_full_context', return_value=None):
            result = self.service.clear_query_context('user-123')
            
            # Code returns True when no context exists (nothing to clear)
            assert result is True
    
    def test_clear_context_dynamodb_error(self):
        """Test DynamoDB error during deletion."""
        mock_items = [{'user_id': 'user-123', 'timestamp': 1234567890}]
        self.service.table.query.return_value = {'Items': mock_items}
        
        error_response = {'Error': {'Code': 'ItemNotFoundException'}}
        self.service.table.delete_item.side_effect = ClientError(error_response, 'DeleteItem')
        
        result = self.service.clear_query_context('user-123')
        
        assert result is False


class TestGetQueryContextService:
    """Test singleton factory function."""
    
    def test_get_service_returns_singleton(self):
        """Test that get_query_context_service returns singleton."""
        from app.services.query_context_service import get_query_context_service
        
        # Reset singleton
        import app.services.query_context_service as module
        module._query_context_service = None
        
        with patch('app.services.query_context_service.boto3'):
            service1 = get_query_context_service()
            service2 = get_query_context_service()
            
            assert service1 is service2


class TestContextWithComparisonTargets:
    """Test context handling with comparison targets."""
    
    def setup_method(self):
        """Setup test service."""
        with patch('app.services.query_context_service.boto3'):
            self.service = QueryContextService()
            self.service.table = Mock()
    
    def test_save_with_comparison_targets(self):
        """Test saving context with comparison targets."""
        self.service.table.put_item.return_value = {}
        
        with patch.object(self.service, 'get_full_context', return_value=None):
            result = self.service.save_query_context(
                user_id='user-123',
                intent='compare',
                slots={'domain_name': 'customer'},
                original_prompt='compare customer and product',
                comparison_targets=['customer.csv', 'product.csv']
            )
            
            assert result is not None
            call_args = self.service.table.put_item.call_args[1]
            item = call_args['Item']
            assert 'comparison_targets' in item
            assert item['comparison_targets'] == ['customer.csv', 'product.csv']


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def setup_method(self):
        """Setup test service."""
        with patch('app.services.query_context_service.boto3'):
            self.service = QueryContextService()
            self.service.table = Mock()
    
    def test_save_handles_unexpected_exception(self):
        """Test handling of unexpected exceptions."""
        with patch.object(self.service, 'get_full_context', side_effect=Exception("Unexpected error")):
            result = self.service.save_query_context(
                user_id='user-123',
                intent='success_rate',
                slots={'domain_name': 'customer'},
                original_prompt='test'
            )
            
            assert result is None
    
    def test_update_slots_handles_exception(self):
        """Test update_context_slots handles exceptions."""
        self.service.table.update_item.side_effect = Exception("Unexpected error")
        
        result = self.service.update_context_slots(
            'user-123',
            timestamp=1234567890,
            new_slots={'domain_name': 'payment'}
        )
        
        assert result is False


class TestTableCreationEdgeCases:
    """Test edge cases in table creation."""
    
    @patch('app.services.query_context_service.boto3')
    def test_table_creation_resource_in_use_exception(self, mock_boto3):
        """Test handling of ResourceInUseException during table creation."""
        mock_client = Mock()
        mock_client.list_tables.return_value = {'TableNames': []}
        
        # Simulate ResourceInUseException (table being created)
        error_response = {'Error': {'Code': 'ResourceInUseException', 'Message': 'Table being created'}}
        mock_client.create_table.side_effect = ClientError(error_response, 'create_table')
        
        mock_boto3.client.return_value = mock_client
        mock_boto3.resource.return_value = Mock()
        
        # Should not raise exception, just log
        service = QueryContextService()
        assert service is not None
    
    @patch('app.services.query_context_service.boto3')
    def test_table_creation_other_client_error(self, mock_boto3):
        """Test handling of other ClientError during table creation."""
        mock_client = Mock()
        mock_client.list_tables.return_value = {'TableNames': []}
        
        # Simulate other ClientError
        error_response = {'Error': {'Code': 'AccessDeniedException', 'Message': 'Access denied'}}
        mock_client.create_table.side_effect = ClientError(error_response, 'create_table')
        
        mock_boto3.client.return_value = mock_client
        mock_boto3.resource.return_value = Mock()
        
        # Should raise exception
        with pytest.raises(ClientError):
            QueryContextService()
    
    @patch('app.services.query_context_service.boto3')
    def test_table_creation_unexpected_exception(self, mock_boto3):
        """Test handling of unexpected exception during table creation."""
        mock_client = Mock()
        mock_client.list_tables.return_value = {'TableNames': []}
        mock_client.create_table.side_effect = Exception("Unexpected error")
        
        mock_boto3.client.return_value = mock_client
        mock_boto3.resource.return_value = Mock()
        
        # Should raise exception
        with pytest.raises(Exception):
            QueryContextService()


class TestGetQueryContextWithTTL:
    """Test get_query_context with TTL validation."""
    
    def setup_method(self):
        """Setup test service with mocked DynamoDB."""
        with patch('app.services.query_context_service.boto3'):
            self.service = QueryContextService()
            self.service.table = Mock()
    
    def test_get_query_context_expired_ttl(self):
        """Test get_query_context returns None when TTL expired."""
        import time
        current_time = int(time.time())
        expired_ttl = current_time - 3600  # 1 hour ago
        
        self.service.table.query.return_value = {
            'Items': [{
                'user_id': 'user-123',
                'timestamp': current_time - 7200,
                'report_type': 'success_rate',
                'slots': {'domain_name': 'customer'},
                'ttl': expired_ttl,
                'updated_at': '2024-01-01T12:00:00'
            }]
        }
        
        result = self.service.get_query_context('user-123')
        
        # Should return None due to expired TTL
        assert result is None
    
    def test_get_query_context_no_items(self):
        """Test get_query_context returns None when no items found."""
        self.service.table.query.return_value = {'Items': []}
        
        result = self.service.get_query_context('user-123')
        
        assert result is None
    
    def test_get_query_context_client_error(self):
        """Test get_query_context handles ClientError."""
        error_response = {'Error': {'Code': 'ProvisionedThroughputExceededException', 'Message': 'Rate exceeded'}}
        self.service.table.query.side_effect = ClientError(error_response, 'query')
        
        result = self.service.get_query_context('user-123')
        
        assert result is None
    
    def test_get_query_context_unexpected_exception(self):
        """Test get_query_context handles unexpected exception."""
        self.service.table.query.side_effect = Exception("Unexpected error")
        
        result = self.service.get_query_context('user-123')
        
        assert result is None


class TestUpdateExistingRecordEdgeCases:
    """Test edge cases in _update_existing_record."""
    
    def setup_method(self):
        """Setup test service with mocked DynamoDB."""
        with patch('app.services.query_context_service.boto3'):
            self.service = QueryContextService()
            self.service.table = Mock()
    
    def test_update_existing_record_client_error(self):
        """Test _update_existing_record handles ClientError."""
        error_response = {'Error': {'Code': 'ConditionalCheckFailedException', 'Message': 'Condition failed'}}
        self.service.table.update_item.side_effect = ClientError(error_response, 'update_item')
        
        result = self.service._update_existing_record(
            user_id='user-123',
            timestamp=1234567890,
            new_intent='success_rate',
            new_slots={'domain_name': 'customer'},
            new_prompt='show me success rate',
            new_comparison_targets=None
        )
        
        assert result is False
    
    def test_update_existing_record_unexpected_exception(self):
        """Test _update_existing_record handles unexpected exception."""
        self.service.table.update_item.side_effect = Exception("Unexpected error")
        
        result = self.service._update_existing_record(
            user_id='user-123',
            timestamp=1234567890,
            new_intent='success_rate',
            new_slots={'domain_name': 'customer'},
            new_prompt='show me success rate',
            new_comparison_targets=None
        )
        
        assert result is False


class TestSaveContextUnexpectedException:
    """Test save_query_context unexpected exception handling."""
    
    def setup_method(self):
        """Setup test service with mocked DynamoDB."""
        with patch('app.services.query_context_service.boto3'):
            self.service = QueryContextService()
            self.service.table = Mock()
    
    def test_save_context_unexpected_exception_during_put(self):
        """Test save_query_context handles unexpected exception during put_item."""
        self.service.table.put_item.side_effect = Exception("Unexpected error")
        
        result = self.service.save_query_context(
            user_id='user-123',
            intent='success_rate',
            slots={'domain_name': 'customer'},
            original_prompt='show me success rate'
        )
        
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=app.services.query_context_service", "--cov-report=term-missing"])
