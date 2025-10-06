"""
Enhanced TTL service with DynamoDB native TTL support.
Provides both application-level and DynamoDB-level TTL cleanup.
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List

import boto3
from app import config as app_config
from app.utils.sanitization import sanitize_text_input

LOGGER = logging.getLogger("dynamo_conversation_ttl_enhanced")


class DynamoConversationTTLEnhancedService:
    """
    Enhanced conversation history storage with dual TTL approach:
    1. Application-level TTL filtering (immediate consistency)
    2. DynamoDB native TTL (guaranteed background cleanup)
    
    This ensures expired conversations are cleaned up even for inactive users.
    """

    def __init__(self):
        table_name = getattr(
            app_config,
            "DYNAMODB_CONVERSATIONS_TABLE_NAME",
            os.environ.get("DYNAMODB_CONVERSATIONS_TABLE_NAME", "conversation_history"),
        )

        self.max_history = getattr(app_config, "MAX_SESSION_HISTORY", 20)
        
        # TTL configuration (default: 30 days)
        self.conversation_ttl_days = getattr(
            app_config, 
            "CONVERSATION_TTL_DAYS", 
            float(os.environ.get("CONVERSATION_TTL_DAYS"))
        )

        self.dynamodb = boto3.resource(
            "dynamodb",
            aws_access_key_id=getattr(app_config, "AWS_ACCESS_KEY_ID", None),
            aws_secret_access_key=getattr(app_config, "AWS_SECRET_ACCESS_KEY", None),
            region_name=getattr(app_config, "AWS_DEFAULT_REGION", None),
        )
        self.table = self.dynamodb.Table(table_name)
        LOGGER.info(f"Enhanced TTL Service initialized for table: {table_name}")

    def _calculate_ttl(self, days: int = None) -> int:
        """Calculate Unix timestamp for TTL expiration."""
        ttl_days = days or self.conversation_ttl_days
        expiry_date = datetime.utcnow() + timedelta(days=ttl_days)
        return int(expiry_date.timestamp())

    def store_interaction(self, user_id: str, user_prompt: str, ttl_days: int = None) -> bool:
        """
        Store interaction with dual TTL approach:
        1. Individual conversation TTL (application-level filtering)
        2. Record-level TTL (DynamoDB native cleanup)
        """
        try:
            try:
                user_prompt_safe = sanitize_text_input(user_prompt, max_length=1000)
            except Exception:
                user_prompt_safe = (user_prompt or "").strip()[:1000]

            # Fetch existing record
            resp = self.table.get_item(Key={"user_id": user_id})
            item = resp.get("Item", {}) or {}

            # Get existing contexts and filter out expired ones (TTL check during write)
            contexts = item.get("context", [])
            if not isinstance(contexts, list):
                contexts = []
            
            # APPLICATION-LEVEL TTL CHECK - Remove expired conversations
            current_time = int(datetime.utcnow().timestamp())
            contexts = [
                ctx for ctx in contexts 
                if ctx.get("ttl", float('inf')) > current_time
            ]

            # Calculate TTL for new conversation
            conversation_ttl = self._calculate_ttl(ttl_days)
            
            # Build the new conversation entry with TTL
            context = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_prompt": user_prompt_safe,
                "ttl": conversation_ttl  # Individual conversation TTL
            }

            # Append and maintain bounded history
            contexts.append(context)
            if len(contexts) > self.max_history:
                contexts = contexts[-self.max_history:]

            # Calculate record-level TTL (for DynamoDB native cleanup)
            # Set record TTL to expire slightly after the last conversation
            max_conversation_ttl = max((ctx.get("ttl", 0) for ctx in contexts), default=conversation_ttl)
            record_ttl = max_conversation_ttl + 86400  # Add 1 day buffer

            # Update the item with both individual and record-level TTL
            item.update({
                "user_id": user_id,
                "context": contexts,
                "updated_at": datetime.utcnow().isoformat(),
                "record_ttl": record_ttl  # DynamoDB native TTL attribute
            })

            # Persist
            self.table.put_item(Item=item)
            LOGGER.debug(f"Stored interaction for {user_id[:8]}... with conversation TTL: {conversation_ttl}, record TTL: {record_ttl}")
            return True

        except Exception as e:
            LOGGER.warning(f"Failed to store interaction for {user_id[:8]}...: {e}")
            return False

    def get_conversation_history(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history with APPLICATION-LEVEL TTL filtering.
        Returns only non-expired conversations.
        """
        try:
            resp = self.table.get_item(Key={"user_id": user_id})
            item = resp.get("Item", {})
            contexts = item.get("context", [])
            
            # APPLICATION-LEVEL TTL CHECK - Filter out expired conversations
            current_time = int(datetime.utcnow().timestamp())
            active_contexts = [
                ctx for ctx in contexts 
                if ctx.get("ttl", float('inf')) > current_time
            ]
            
            # Clean up expired conversations from the record (and update record TTL)
            if len(active_contexts) != len(contexts):
                self._cleanup_expired_conversations(user_id, item, active_contexts)
            
            return active_contexts
            
        except Exception as e:
            LOGGER.warning(f"Failed to load conversation history for {user_id[:8]}...: {e}")
            return []

    def _cleanup_expired_conversations(self, user_id: str, item: Dict, active_contexts: List):
        """Clean up expired conversations and update record TTL."""
        try:
            # Update record TTL based on remaining conversations
            if active_contexts:
                max_conversation_ttl = max((ctx.get("ttl", 0) for ctx in active_contexts), default=0)
                record_ttl = max_conversation_ttl + 86400  # Add 1 day buffer
            else:
                # No active conversations - set record to expire soon (but keep user_id for a grace period)
                record_ttl = int(datetime.utcnow().timestamp()) + 7 * 86400  # 7 days grace period
            
            item.update({
                "context": active_contexts,
                "updated_at": datetime.utcnow().isoformat(),
                "record_ttl": record_ttl
            })
            
            self.table.put_item(Item=item)
            LOGGER.debug(f"Cleaned expired conversations for {user_id[:8]}..., updated record TTL: {record_ttl}")
        except Exception as e:
            LOGGER.warning(f"Failed to cleanup expired conversations: {e}")


    # def extend_conversation_ttl(self, user_id: str, additional_days: int = 30) -> bool:
    #     """Extend TTL for all active conversations of a user."""
    #     try:
    #         resp = self.table.get_item(Key={"user_id": user_id})
    #         item = resp.get("Item", {})
    #         contexts = item.get("context", [])
            
    #         current_time = int(datetime.utcnow().timestamp())
    #         updated = False
            
    #         for ctx in contexts:
    #             if ctx.get("ttl", 0) > current_time:  # Only extend active conversations
    #                 ctx["ttl"] = self._calculate_ttl(additional_days)
    #                 updated = True
            
    #         if updated:
    #             # Update record TTL as well
    #             max_conversation_ttl = max((ctx.get("ttl", 0) for ctx in contexts), default=0)
    #             record_ttl = max_conversation_ttl + 86400
                
    #             item.update({
    #                 "updated_at": datetime.utcnow().isoformat(),
    #                 "record_ttl": record_ttl
    #             })
                
    #             self.table.put_item(Item=item)
    #             LOGGER.info(f"Extended TTL by {additional_days} days for {user_id[:8]}...")
    #             return True
                
    #     except Exception as e:
    #         LOGGER.error(f"Failed to extend TTL for {user_id[:8]}...: {e}")
        
    #     return False


# Singleton instance
dynamo_conversation_ttl_enhanced = DynamoConversationTTLEnhancedService()