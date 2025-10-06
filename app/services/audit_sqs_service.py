"""
AWS SQS Audit Logging Service for Analytics API.

Sends structured audit logs to SQS queue for analytics activities.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError, BotoCoreError

from app.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION

logger = logging.getLogger(__name__)


class AuditSQSService:
    """Service for sending audit logs to AWS SQS queue."""
    
    def __init__(self, queue_url: Optional[str] = None):
        """
        Initialize SQS audit service.
        
        Args:
            queue_url: SQS queue URL. If not provided, will be read from config.
        """
        self.queue_url = queue_url or self._get_queue_url_from_config()
        self.sqs_client = None
        self._initialize_sqs_client()
    
    def _get_queue_url_from_config(self) -> Optional[str]:
        """Get SQS queue URL from environment configuration."""
        import os
        return os.getenv("AUDIT_SQS_QUEUE_URL")
    
    def _initialize_sqs_client(self):
        """Initialize AWS SQS client with proper error handling."""
        try:
            if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
                self.sqs_client = boto3.client(
                    'sqs',
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    region_name=AWS_DEFAULT_REGION
                )
            else:
                # Use default credentials (IAM role, profile, etc.)
                self.sqs_client = boto3.client('sqs', region_name=AWS_DEFAULT_REGION)
            
            logger.info(f"✅ SQS client initialized for region: {AWS_DEFAULT_REGION}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize SQS client: {e}")
            self.sqs_client = None
    
    def send_audit_log(
        self,
        statusCode: int,
        user_id: str,
        username: str,
        activity_type: str = "Analytics-Query",
        activity_description: str = "Analytics query processed",
        request_endpoint: str = "api/analytics/query",
        response_status: str = "SUCCESS",
        request_type: str = "POST",
        remarks: str = "",
        additional_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send audit log to SQS queue.
        
        Args:
            user_id: User ID from JWT token
            username: Username from JWT token  
            activity_type: Type of activity (e.g., "Analytics-Query")
            activity_description: Description of the activity
            request_endpoint: API endpoint accessed
            response_status: SUCCESS or FAILURE
            request_type: HTTP method (POST, GET, etc.)
            remarks: Additional remarks or notes
            additional_data: Extra data to include in the audit log
            
        Returns:
            bool: True if log was sent successfully, False otherwise
        """
        if not self.queue_url:
            logger.warning("⚠️ SQS queue URL not configured, skipping audit log")
            return False
        
        if not self.sqs_client:
            logger.warning("⚠️ SQS client not initialized, skipping audit log")
            return False
        
        try:
            # Build audit log message matching the required format
            audit_log = {
                "statusCode": statusCode,  # Default success, can be overridden
                "userId": user_id,
                "username": username,
                "activityType": activity_type,
                "activityDescription": activity_description,
                "requestActionEndpoint": request_endpoint,
                "responseStatus": response_status,
                "requestType": request_type,
                "remarks": remarks,
                "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                "region": AWS_DEFAULT_REGION
            }
            
            # Add any additional data
            if additional_data:
                audit_log.update(additional_data)
            
            # Send to SQS
            response = self.sqs_client.send_message(
                QueueUrl=self.queue_url,
                MessageBody=json.dumps(audit_log),
                MessageAttributes={
                    'ActivityType': {
                        'StringValue': activity_type,
                        'DataType': 'String'
                    },
                    'UserId': {
                        'StringValue': user_id,
                        'DataType': 'String'
                    },
                    'Timestamp': {
                        'StringValue': audit_log["timestamp"],
                        'DataType': 'String'
                    }
                }
            )
            
            message_id = response.get('MessageId')
            logger.info(f"✅ Audit log sent to SQS: MessageId={message_id}")
            return True
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            logger.error(f"❌ AWS SQS ClientError: {error_code} - {e}")
            return False
        except BotoCoreError as e:
            logger.error(f"❌ AWS SQS BotoCoreError: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to send audit log to SQS: {e}")
            return False
    
    def send_analytics_query_audit(
        self,
        user_id: str,
        username: str,
        prompt: str,
        success: bool,
        processing_time_ms: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Send audit log specifically for analytics query processing.
        
        Args:
            user_id: User ID from JWT
            username: Username from JWT
            prompt: User's query prompt (will be truncated for privacy)
            success: Whether the query was processed successfully
            processing_time_ms: Processing time in milliseconds
            error_message: Error message if query failed
            
        Returns:
            bool: True if audit log sent successfully
        """
        # Truncate prompt for privacy (don't log full user queries)
        safe_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
        
        activity_description = f"Analytics query: '{safe_prompt}'"
        if not success and error_message:
            activity_description += f" - Error: {error_message[:100]}"
        
        additional_data = {}
        if processing_time_ms is not None:
            additional_data["processingTimeMs"] = processing_time_ms
        
        return self.send_audit_log(
            user_id=user_id,
            username=username,
            activity_type="Analytics-Query",
            activity_description=activity_description,
            request_endpoint="api/analytics/query",
            response_status="SUCCESS" if success else "FAILURE",
            request_type="POST",
            remarks=error_message[:200] if error_message else "",
            additional_data=additional_data
        )
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the SQS audit service.
        
        Returns:
            dict: Health status information
        """
        if not self.queue_url:
            return {
                "healthy": False,
                "error": "Queue URL not configured",
                "queue_url": None
            }
        
        if not self.sqs_client:
            return {
                "healthy": False,
                "error": "SQS client not initialized",
                "queue_url": self.queue_url
            }
        
        try:
            # Test queue accessibility
            response = self.sqs_client.get_queue_attributes(
                QueueUrl=self.queue_url,
                AttributeNames=['QueueArn']
            )
            
            return {
                "healthy": True,
                "queue_url": self.queue_url,
                "queue_arn": response.get('Attributes', {}).get('QueueArn'),
                "region": AWS_DEFAULT_REGION
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "queue_url": self.queue_url
            }


# Global instance for reuse
_audit_sqs_service = None

def get_audit_sqs_service() -> AuditSQSService:
    """Get singleton instance of audit SQS service."""
    global _audit_sqs_service
    if _audit_sqs_service is None:
        _audit_sqs_service = AuditSQSService()
    return _audit_sqs_service