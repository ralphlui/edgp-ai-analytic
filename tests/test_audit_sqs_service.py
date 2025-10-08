"""
Tests for AWS SQS Audit Logging Service.

Tests audit log formatting, SQS integration, and error handling.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from botocore.exceptions import ClientError

from app.services.audit_sqs_service import AuditSQSService, get_audit_sqs_service


class TestAuditSQSService:
    """Test AWS SQS audit logging functionality."""
    
    @patch('app.services.audit_sqs_service.boto3.client')
    def test_sqs_client_initialization_with_credentials(self, mock_boto_client):
        """Test SQS client initialization with AWS credentials."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        with patch('app.services.audit_sqs_service.AWS_ACCESS_KEY_ID', 'test_key'):
            with patch('app.services.audit_sqs_service.AWS_SECRET_ACCESS_KEY', 'test_secret'):
                with patch('app.services.audit_sqs_service.AWS_DEFAULT_REGION', 'us-east-1'):
                    service = AuditSQSService(queue_url="https://sqs.us-east-1.amazonaws.com/123456789/test-queue")
        
        mock_boto_client.assert_called_once_with(
            'sqs',
            aws_access_key_id='test_key',
            aws_secret_access_key='test_secret',
            region_name='us-east-1'
        )
        assert service.sqs_client == mock_client
    
    @patch('app.services.audit_sqs_service.boto3.client')
    def test_sqs_client_initialization_without_credentials(self, mock_boto_client):
        """Test SQS client initialization using default AWS credentials."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        with patch('app.services.audit_sqs_service.AWS_ACCESS_KEY_ID', ''):
            with patch('app.services.audit_sqs_service.AWS_SECRET_ACCESS_KEY', ''):
                with patch('app.services.audit_sqs_service.AWS_DEFAULT_REGION', 'us-west-2'):
                    service = AuditSQSService(queue_url="https://sqs.us-west-2.amazonaws.com/123456789/test-queue")
        
        mock_boto_client.assert_called_once_with('sqs', region_name='us-west-2')
        assert service.sqs_client == mock_client
    
    def test_service_without_queue_url(self):
        """Test service behavior when no queue URL is configured."""
        with patch.object(AuditSQSService, '_get_queue_url_from_config', return_value=None):
            service = AuditSQSService()
            
            result = service.send_audit_log("user123", "testuser", "test-activity")
            assert result is False
    
    @patch('app.services.audit_sqs_service.boto3.client')
    def test_successful_audit_log_send(self, mock_boto_client):
        """Test successful audit log sending to SQS."""
        mock_client = Mock()
        mock_client.send_message.return_value = {'MessageId': 'msg-123'}
        mock_boto_client.return_value = mock_client
        
        service = AuditSQSService(queue_url="https://sqs.us-east-1.amazonaws.com/123456789/test-queue")
        
        result = service.send_audit_log(
            statusCode=200,
            user_id="user123",
            username="testuser",
            activity_type="Analytics-Query",
            activity_description="Test analytics query",
            request_endpoint="api/analytics/query",
            response_status="SUCCESS"
        )
        
        assert result is True
        mock_client.send_message.assert_called_once()
        
        # Verify message structure
        call_args = mock_client.send_message.call_args
        message_body = json.loads(call_args[1]['MessageBody'])
        
        assert message_body['statusCode'] == 200
        assert message_body['userId'] == "user123"
        assert message_body['username'] == "testuser"
        assert message_body['activityType'] == "Analytics-Query"
        assert message_body['requestActionEndpoint'] == "api/analytics/query"
        assert message_body['responseStatus'] == "SUCCESS"
        assert 'timestamp' in message_body
    
    @patch('app.services.audit_sqs_service.boto3.client')
    def test_sqs_client_error_handling(self, mock_boto_client):
        """Test error handling for SQS client errors."""
        mock_client = Mock()
        mock_client.send_message.side_effect = ClientError(
            error_response={'Error': {'Code': 'InvalidParameterValue'}},
            operation_name='SendMessage'
        )
        mock_boto_client.return_value = mock_client
        
        service = AuditSQSService(queue_url="https://sqs.us-east-1.amazonaws.com/123456789/test-queue")
        
        result = service.send_audit_log(statusCode=200, user_id="user123", username="testuser")
        assert result is False
    
    @patch('app.services.audit_sqs_service.boto3.client')
    def test_analytics_query_audit_success(self, mock_boto_client):
        """Test analytics-specific audit logging for successful queries."""
        mock_client = Mock()
        mock_client.send_message.return_value = {'MessageId': 'msg-456'}
        mock_boto_client.return_value = mock_client
        
        service = AuditSQSService(queue_url="https://sqs.us-east-1.amazonaws.com/123456789/test-queue")
        
        result = service.send_analytics_query_audit(
            statusCode=200,
            user_id="user456",
            username="analyst",
            prompt="show me success rate for customer.csv",
            success=True,
            processing_time_ms=1250
        )
        
        assert result is True
        
        # Verify message content
        call_args = mock_client.send_message.call_args
        message_body = json.loads(call_args[1]['MessageBody'])
        
        assert message_body['statusCode'] == 200
        assert message_body['userId'] == "user456"
        assert message_body['username'] == "analyst"
        assert message_body['activityType'] == "Analytics-Query"
        assert message_body['responseStatus'] == "SUCCESS"
        assert message_body['processingTimeMs'] == 1250
        assert "show me success rate for customer.csv" in message_body['activityDescription']
    
    @patch('app.services.audit_sqs_service.boto3.client')
    def test_analytics_query_audit_failure(self, mock_boto_client):
        """Test analytics-specific audit logging for failed queries."""
        mock_client = Mock()
        mock_client.send_message.return_value = {'MessageId': 'msg-789'}
        mock_boto_client.return_value = mock_client
        
        service = AuditSQSService(queue_url="https://sqs.us-east-1.amazonaws.com/123456789/test-queue")
        
        result = service.send_analytics_query_audit(
            statusCode=400,
            user_id="user789",
            username="analyst2",
            prompt="invalid query with malicious content",
            success=False,
            processing_time_ms=50,
            error_message="Request blocked for security reasons"
        )
        
        assert result is True
        
        # Verify error logging
        call_args = mock_client.send_message.call_args
        message_body = json.loads(call_args[1]['MessageBody'])
        
        assert message_body['statusCode'] == 400
        assert message_body['responseStatus'] == "FAILURE"
        assert "Request blocked for security reasons" in message_body['remarks']
        assert "Error: Request blocked for security reasons" in message_body['activityDescription']
    
    def test_long_prompt_truncation(self):
        """Test that long prompts are properly truncated for privacy."""
        with patch.object(AuditSQSService, 'send_audit_log', return_value=True) as mock_send:
            service = AuditSQSService(queue_url="https://test.queue.url")
            
            long_prompt = "a" * 200  # 200 character prompt
            service.send_analytics_query_audit(
                statusCode=200,
                user_id="user123",
                username="test",
                prompt=long_prompt,
                success=True
            )
            
            # Verify that activity_description was truncated
            call_args = mock_send.call_args[1]
            activity_desc = call_args['activity_description']
            assert len(activity_desc) < len(long_prompt) + 50  # Should be truncated
            assert "..." in activity_desc
    
    @patch('app.services.audit_sqs_service.boto3.client')
    def test_health_check_success(self, mock_boto_client):
        """Test health check for properly configured SQS service."""
        mock_client = Mock()
        mock_client.get_queue_attributes.return_value = {
            'Attributes': {
                'QueueArn': 'arn:aws:sqs:us-east-1:123456789:test-queue'
            }
        }
        mock_boto_client.return_value = mock_client
        
        service = AuditSQSService(queue_url="https://sqs.us-east-1.amazonaws.com/123456789/test-queue")
        health_info = service.health_check()
        
        assert health_info['healthy'] is True
        assert 'queue_arn' in health_info
        assert health_info['queue_url'] == "https://sqs.us-east-1.amazonaws.com/123456789/test-queue"
    
    @patch.dict('os.environ', {'AUDIT_SQS_QUEUE_URL': ''}, clear=False)
    @patch('app.services.audit_sqs_service.boto3.client')
    def test_health_check_no_queue_url(self, mock_boto_client):
        """Test health check when no queue URL is configured."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        service = AuditSQSService(queue_url=None)
        health_info = service.health_check()
        
        # When queue_url is None, health check should return False without calling AWS
        assert health_info['healthy'] is False
        assert "Queue URL not configured" in health_info['error']
        # Verify that get_queue_attributes was NOT called since queue_url is None
        mock_client.get_queue_attributes.assert_not_called()
    
    @patch('app.services.audit_sqs_service.boto3.client')
    def test_health_check_sqs_error(self, mock_boto_client):
        """Test health check when SQS access fails."""
        mock_client = Mock()
        mock_client.get_queue_attributes.side_effect = ClientError(
            error_response={'Error': {'Code': 'AccessDenied'}},
            operation_name='GetQueueAttributes'
        )
        mock_boto_client.return_value = mock_client
        
        service = AuditSQSService(queue_url="https://sqs.us-east-1.amazonaws.com/123456789/test-queue")
        health_info = service.health_check()
        
        assert health_info['healthy'] is False
        assert "AccessDenied" in health_info['error']


class TestAuditSQSServiceSingleton:
    """Test singleton behavior of audit service."""
    
    def test_singleton_instance(self):
        """Test that get_audit_sqs_service returns the same instance."""
        service1 = get_audit_sqs_service()
        service2 = get_audit_sqs_service()
        
        assert service1 is service2  # Same instance
    
    @patch('app.services.audit_sqs_service._audit_sqs_service', None)
    def test_singleton_initialization(self):
        """Test that singleton is properly initialized on first call."""
        with patch.object(AuditSQSService, '__init__', return_value=None) as mock_init:
            get_audit_sqs_service()
            mock_init.assert_called_once()


class TestAuditLogMessageFormat:
    """Test audit log message format compliance."""
    
    @patch('app.services.audit_sqs_service.boto3.client')
    def test_message_format_matches_requirements(self, mock_boto_client):
        """Test that audit log format matches the required sample format."""
        mock_client = Mock()
        mock_client.send_message.return_value = {'MessageId': 'test'}
        mock_boto_client.return_value = mock_client
        
        service = AuditSQSService(queue_url="https://test.queue")
        
        service.send_audit_log(
            statusCode=200,
            user_id="afcf8527-ce7e-453f-86bf-3aa232fca2c4",
            username="Khin",
            activity_type="Admin-Authentication-LoginUser",
            activity_description="analyst@gmail.com login successfully",
            request_endpoint="api/admin/users/login",
            response_status="SUCCESS",
            request_type="POST"
        )
        
        call_args = mock_client.send_message.call_args
        message_body = json.loads(call_args[1]['MessageBody'])
        
        # Verify all required fields are present and match sample format
        required_fields = [
            'statusCode', 'userId', 'username', 'activityType',
            'activityDescription', 'requestActionEndpoint', 
            'responseStatus', 'requestType', 'remarks'
        ]
        
        for field in required_fields:
            assert field in message_body
        
        # Verify specific values match sample
        assert message_body['userId'] == "afcf8527-ce7e-453f-86bf-3aa232fca2c4"
        assert message_body['username'] == "Khin"
        assert message_body['responseStatus'] == "SUCCESS"
        assert message_body['requestType'] == "POST"