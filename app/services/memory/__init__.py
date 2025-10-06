"""
TTL-based conversation storage utilities.

Note: The app uses DynamoDB for conversation history keyed by user_id with
automatic TTL-based cleanup. Individual conversations expire automatically
while user_id records are preserved permanently.
"""

# Enhanced TTL-based conversation service (dual-layer automatic cleanup)
from .dynamo_conversation_ttl_enhanced_service import dynamo_conversation_ttl_enhanced, DynamoConversationTTLEnhancedService

# Export enhanced TTL service as the default
dynamo_conversation = dynamo_conversation_ttl_enhanced

__all__ = [
    'dynamo_conversation',                      # Enhanced TTL service (dual-layer cleanup)
    'dynamo_conversation_ttl_enhanced',         # Explicit enhanced TTL service
    'DynamoConversationTTLEnhancedService',     # Enhanced TTL service class
]