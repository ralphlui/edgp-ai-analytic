"""
Comprehensive tests for audit_sqs_service.py

Tests AWS SQS audit logging service including message formatting,
error handling, and health checks.
"""
import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError, BotoCoreError

from app.services.audit_sqs_service import AuditSQSService, get_audit_sqs_service


class TestAuditSQSServiceInitialization:
    """Test service initialization and configuration."""
    
    @patch('app.services.audit_sqs_service.boto3')
    @patch('app.services.audit_sqs_service.AWS_ACCESS_KEY_ID', 'test-key')
    @patch('app.services.audit_sqs_service.AWS_SECRET_ACCESS_KEY', 'test-secret')
    @patch('app.services.audit_sqs_service.AWS_DEFAULT_REGION', 'us-east-1')
    def test_init_with_credentials(self, mock_boto3):
        """Test initialization with AWS credentials."""
        mock_sqs_client = Mock()
        mock_boto3.client.return_value = mock_sqs_client
        
        service = AuditSQSService(queue_url="https://sqs.us-east-1.amazonaws.com/123/test")
        
        assert service.queue_url == "https://sqs.us-east-1.amazonaws.com/123/test"
        assert service.sqs_client == mock_sqs_client
        mock_boto3.client.assert_called_once_with(
            'sqs',
            aws_access_key_id='test-key',
            aws_secret_access_key='test-secret',
            region_name='us-east-1'
        )
    
    @patch('app.services.audit_sqs_service.boto3')
    @patch('app.services.audit_sqs_service.AWS_ACCESS_KEY_ID', None)
    @patch('app.services.audit_sqs_service.AWS_SECRET_ACCESS_KEY', None)
    @patch('app.services.audit_sqs_service.AWS_DEFAULT_REGION', 'ap-southeast-1')
    def test_init_without_credentials(self, mock_boto3):
        """Test initialization using IAM role (no explicit credentials)."""
        mock_sqs_client = Mock()
        mock_boto3.client.return_value = mock_sqs_client
        
        service = AuditSQSService(queue_url="https://sqs.ap-southeast-1.amazonaws.com/456/prod")
        
        assert service.sqs_client == mock_sqs_client
        mock_boto3.client.assert_called_once_with('sqs', region_name='ap-southeast-1')
    
    @patch('app.services.audit_sqs_service.boto3')
    def test_init_client_failure(self, mock_boto3):
        """Test handling of client initialization failure."""
        mock_boto3.client.side_effect = Exception("AWS credentials not found")
        
        service = AuditSQSService(queue_url="https://sqs.us-east-1.amazonaws.com/123/test")
        
        assert service.sqs_client is None
        assert service.queue_url is not None
    
    @patch.dict('os.environ', {'AUDIT_SQS_QUEUE_URL': 'https://sqs.us-west-2.amazonaws.com/789/env-queue'})
    @patch('app.services.audit_sqs_service.boto3')
    def test_init_queue_url_from_env(self, mock_boto3):
        """Test reading queue URL from environment variable."""
        mock_boto3.client.return_value = Mock()
        
        service = AuditSQSService()
        
        assert service.queue_url == 'https://sqs.us-west-2.amazonaws.com/789/env-queue'


class TestSendAuditLog:
    """Test sending audit logs to SQS."""
    
    def setup_method(self):
        """Setup test service with mocked SQS client."""
        with patch('app.services.audit_sqs_service.boto3'):
            self.service = AuditSQSService(queue_url="https://sqs.us-east-1.amazonaws.com/123/test")
            self.service.sqs_client = Mock()
    
    def test_send_audit_log_success(self):
        """Test successfully sending audit log."""
        self.service.sqs_client.send_message.return_value = {
            'MessageId': 'msg-12345',
            'MD5OfMessageBody': 'abc123'
        }
        
        result = self.service.send_audit_log(
            statusCode=200,
            user_id="user-123",
            username="john.doe",
            activity_type="Analytics-Query",
            activity_description="Success rate query",
            request_endpoint="api/analytics/report",
            response_status="SUCCESS",
            request_type="POST",
            remarks="Query completed successfully"
        )
        
        assert result is True
        self.service.sqs_client.send_message.assert_called_once()
        
        # Verify message structure
        call_args = self.service.sqs_client.send_message.call_args[1]
        assert call_args['QueueUrl'] == "https://sqs.us-east-1.amazonaws.com/123/test"
        
        message_body = json.loads(call_args['MessageBody'])
        assert message_body['statusCode'] == 200
        assert message_body['userId'] == "user-123"
        assert message_body['username'] == "john.doe"
        assert message_body['activityType'] == "Analytics-Query"
        assert message_body['responseStatus'] == "SUCCESS"
        assert message_body['requestType'] == "POST"
        assert 'timestamp' in message_body
        assert 'region' in message_body
    
    def test_send_audit_log_with_additional_data(self):
        """Test sending audit log with additional custom data."""
        self.service.sqs_client.send_message.return_value = {'MessageId': 'msg-456'}
        
        result = self.service.send_audit_log(
            statusCode=200,
            user_id="user-456",
            username="jane.smith",
            activity_type="Report-Generation",
            activity_description="Chart created",
            request_endpoint="api/analytics/chart",
            response_status="SUCCESS",
            request_type="GET",
            remarks="",
            additional_data={
                "chart_type": "bar",
                "data_points": 50,
                "processing_time_ms": 1234
            }
        )
        
        assert result is True
        call_args = self.service.sqs_client.send_message.call_args[1]
        message_body = json.loads(call_args['MessageBody'])
        
        assert message_body['chart_type'] == "bar"
        assert message_body['data_points'] == 50
        assert message_body['processing_time_ms'] == 1234
    
    def test_send_audit_log_no_queue_url(self):
        """Test handling when queue URL is not configured."""
        self.service.queue_url = None
        
        result = self.service.send_audit_log(
            statusCode=200,
            user_id="user-789",
            username="test.user",
            activity_type="Test",
            activity_description="Test activity",
            request_endpoint="api/test",
            response_status="SUCCESS",
            request_type="POST"
        )
        
        assert result is False
        self.service.sqs_client.send_message.assert_not_called()
    
    def test_send_audit_log_no_sqs_client(self):
        """Test handling when SQS client is not initialized."""
        self.service.sqs_client = None
        
        result = self.service.send_audit_log(
            statusCode=200,
            user_id="user-999",
            username="test.user",
            activity_type="Test",
            activity_description="Test activity",
            request_endpoint="api/test",
            response_status="SUCCESS",
            request_type="POST"
        )
        
        assert result is False
    
    def test_send_audit_log_client_error(self):
        """Test handling of AWS ClientError."""
        self.service.sqs_client.send_message.side_effect = ClientError(
            {
                'Error': {
                    'Code': 'InvalidMessageContents',
                    'Message': 'Message contains invalid characters'
                }
            },
            'SendMessage'
        )
        
        result = self.service.send_audit_log(
            statusCode=400,
            user_id="user-error",
            username="error.user",
            activity_type="Test",
            activity_description="Test error",
            request_endpoint="api/test",
            response_status="FAILURE",
            request_type="POST"
        )
        
        assert result is False
    
    def test_send_audit_log_botocore_error(self):
        """Test handling of BotoCoreError."""
        self.service.sqs_client.send_message.side_effect = BotoCoreError()
        
        result = self.service.send_audit_log(
            statusCode=500,
            user_id="user-botocore",
            username="botocore.user",
            activity_type="Test",
            activity_description="BotoCore error",
            request_endpoint="api/test",
            response_status="FAILURE",
            request_type="POST"
        )
        
        assert result is False
    
    def test_send_audit_log_generic_exception(self):
        """Test handling of generic exceptions."""
        self.service.sqs_client.send_message.side_effect = Exception("Network error")
        
        result = self.service.send_audit_log(
            statusCode=500,
            user_id="user-exception",
            username="exception.user",
            activity_type="Test",
            activity_description="Generic error",
            request_endpoint="api/test",
            response_status="FAILURE",
            request_type="POST"
        )
        
        assert result is False
    
    def test_message_attributes_format(self):
        """Test that message attributes are correctly formatted."""
        self.service.sqs_client.send_message.return_value = {'MessageId': 'msg-attrs'}
        
        self.service.send_audit_log(
            statusCode=200,
            user_id="user-attrs",
            username="attrs.user",
            activity_type="Analytics-Query",
            activity_description="Testing attributes",
            request_endpoint="api/analytics/report",
            response_status="SUCCESS",
            request_type="POST"
        )
        
        call_args = self.service.sqs_client.send_message.call_args[1]
        message_attrs = call_args['MessageAttributes']
        
        assert 'ActivityType' in message_attrs
        assert message_attrs['ActivityType']['StringValue'] == "Analytics-Query"
        assert message_attrs['ActivityType']['DataType'] == 'String'
        
        assert 'UserId' in message_attrs
        assert message_attrs['UserId']['StringValue'] == "user-attrs"
        
        assert 'Timestamp' in message_attrs


class TestSendAnalyticsQueryAudit:
    """Test analytics-specific audit logging."""
    
    def setup_method(self):
        """Setup test service."""
        with patch('app.services.audit_sqs_service.boto3'):
            self.service = AuditSQSService(queue_url="https://sqs.us-east-1.amazonaws.com/123/test")
            self.service.sqs_client = Mock()
            self.service.sqs_client.send_message.return_value = {'MessageId': 'msg-analytics'}
    
    def test_send_analytics_query_success(self):
        """Test sending audit for successful analytics query."""
        result = self.service.send_analytics_query_audit(
            statusCode=200,
            user_id="user-query",
            username="query.user",
            prompt="Show me success rate for customer domain",
            success=True,
            message=None
        )
        
        assert result is True
        self.service.sqs_client.send_message.assert_called_once()
        
        call_args = self.service.sqs_client.send_message.call_args[1]
        message_body = json.loads(call_args['MessageBody'])
        
        assert message_body['activityType'] == "Analytics-Query"
        assert message_body['requestType'] == "POST"
        assert message_body['responseStatus'] == "SUCCESS"
        assert "Show me success rate for customer domain" in message_body['activityDescription']
    
    def test_send_analytics_query_failure(self):
        """Test sending audit for failed analytics query."""
        result = self.service.send_analytics_query_audit(
            statusCode=400,
            user_id="user-fail",
            username="fail.user",
            prompt="Invalid query",
            success=False,
            message="Validation error: prompt contains malicious content"
        )
        
        assert result is True
        call_args = self.service.sqs_client.send_message.call_args[1]
        message_body = json.loads(call_args['MessageBody'])
        
        assert message_body['responseStatus'] == "FAILURE"
        assert "Error:" in message_body['activityDescription']
        assert "Validation error" in message_body['remarks']
    
    def test_prompt_truncation(self):
        """Test that long prompts are truncated for privacy."""
        long_prompt = "A" * 150  # 150 characters
        
        self.service.send_analytics_query_audit(
            statusCode=200,
            user_id="user-long",
            username="long.user",
            prompt=long_prompt,
            success=True,
            message=None
        )
        
        call_args = self.service.sqs_client.send_message.call_args[1]
        message_body = json.loads(call_args['MessageBody'])
        
        # Should be truncated to 100 chars + "..."
        assert len(message_body['activityDescription']) < len(long_prompt) + 50
        assert "..." in message_body['activityDescription']
    
    def test_error_message_truncation(self):
        """Test that long error messages are truncated."""
        long_error = "E" * 300  # 300 characters
        
        self.service.send_analytics_query_audit(
            statusCode=500,
            user_id="user-error-long",
            username="error.user",
            prompt="test",
            success=False,
            message=long_error
        )
        
        call_args = self.service.sqs_client.send_message.call_args[1]
        message_body = json.loads(call_args['MessageBody'])
        
        # Error message in remarks should be truncated to 200 chars
        assert len(message_body['remarks']) <= 200


class TestHealthCheck:
    """Test health check functionality."""
    
    def setup_method(self):
        """Setup test service."""
        with patch('app.services.audit_sqs_service.boto3'):
            self.service = AuditSQSService(queue_url="https://sqs.us-east-1.amazonaws.com/123/test")
            self.service.sqs_client = Mock()
    
    def test_health_check_healthy(self):
        """Test health check when service is healthy."""
        self.service.sqs_client.get_queue_attributes.return_value = {
            'Attributes': {
                'QueueArn': 'arn:aws:sqs:us-east-1:123456789:test-queue'
            }
        }
        
        health = self.service.health_check()
        
        assert health['healthy'] is True
        assert health['queue_url'] == "https://sqs.us-east-1.amazonaws.com/123/test"
        assert health['queue_arn'] == 'arn:aws:sqs:us-east-1:123456789:test-queue'
        assert 'region' in health
    
    def test_health_check_no_queue_url(self):
        """Test health check when queue URL is not configured."""
        self.service.queue_url = None
        
        health = self.service.health_check()
        
        assert health['healthy'] is False
        assert health['error'] == "Queue URL not configured"
        assert health['queue_url'] is None
    
    def test_health_check_no_client(self):
        """Test health check when SQS client is not initialized."""
        self.service.sqs_client = None
        
        health = self.service.health_check()
        
        assert health['healthy'] is False
        assert health['error'] == "SQS client not initialized"
        assert health['queue_url'] is not None
    
    def test_health_check_queue_access_error(self):
        """Test health check when queue is inaccessible."""
        self.service.sqs_client.get_queue_attributes.side_effect = ClientError(
            {'Error': {'Code': 'QueueDoesNotExist', 'Message': 'Queue not found'}},
            'GetQueueAttributes'
        )
        
        health = self.service.health_check()
        
        assert health['healthy'] is False
        assert 'error' in health
        assert health['queue_url'] is not None


class TestGetAuditSQSService:
    """Test singleton pattern for service instance."""
    
    @patch('app.services.audit_sqs_service.boto3')
    def test_singleton_returns_same_instance(self, mock_boto3):
        """Test that get_audit_sqs_service returns singleton instance."""
        mock_boto3.client.return_value = Mock()
        
        # Reset singleton
        import app.services.audit_sqs_service
        app.services.audit_sqs_service._audit_sqs_service = None
        
        service1 = get_audit_sqs_service()
        service2 = get_audit_sqs_service()
        
        assert service1 is service2
    
    @patch('app.services.audit_sqs_service.boto3')
    def test_singleton_initialization(self, mock_boto3):
        """Test that singleton is initialized only once."""
        mock_boto3.client.return_value = Mock()
        
        # Reset singleton
        import app.services.audit_sqs_service
        app.services.audit_sqs_service._audit_sqs_service = None
        
        service = get_audit_sqs_service()
        
        assert service is not None
        assert isinstance(service, AuditSQSService)
