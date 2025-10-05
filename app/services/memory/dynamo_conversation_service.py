import logging
import os
from datetime import datetime
from typing import Any, Dict, List
from venv import logger

import boto3
from app import config as app_config
from app.utils.sanitization import sanitize_text_input

LOGGER = logging.getLogger("dynamo_conversation")


class DynamoConversationService:
    """
    Conversation history storage backed by DynamoDB, keyed by user_id.
    """

    def __init__(self):
        table_name = getattr(
            app_config,
            "DYNAMODB_CONVERSATIONS_TABLE_NAME",
            os.environ.get("DYNAMODB_CONVERSATIONS_TABLE_NAME", "conversation_history"),
        )

        self.max_history = getattr(app_config, "MAX_SESSION_HISTORY", 20)

        self.dynamodb = boto3.resource(
            "dynamodb",
            aws_access_key_id=getattr(app_config, "AWS_ACCESS_KEY_ID", None),
            aws_secret_access_key=getattr(app_config, "AWS_SECRET_ACCESS_KEY", None),
            region_name=getattr(app_config, "AWS_DEFAULT_REGION", None),
        )
        self.table = self.dynamodb.Table(table_name)
        LOGGER.info(f"DynamoConversationService initialized for table: {table_name}")

    # ✅ FIXED: now properly indented as a class method
    def store_interaction(self, user_id: str, user_prompt: str) -> bool:
        """Append an interaction for user_id and maintain bounded history/context."""
        try:
            try:
                user_prompt_safe = sanitize_text_input(user_prompt, max_length=1000)
            except Exception:
                user_prompt_safe = (user_prompt or "").strip()[:1000]

            # Fetch existing record (may not exist yet)
            resp = self.table.get_item(Key={"user_id": user_id})
            item = resp.get("Item", {}) or {}

            # Ensure contexts is a list before using it
            contexts = item.get("context", [])
            if not isinstance(contexts, list):
                contexts = []

            # Build the new entry
            context = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_prompt": user_prompt_safe,
            }

            # Append and trim to bounded history
            contexts.append(context)
            if len(contexts) > self.max_history:
                contexts = contexts[-self.max_history:]

            # Update the item while preserving other attributes
            item.update({
                "user_id": user_id,
                "context": contexts,
                "updated_at": datetime.utcnow().isoformat(),
            })

            # Persist
            self.table.put_item(Item=item)
            return True

        except Exception as e:
            LOGGER.warning(f"Failed to store interaction for {user_id[:8]}...: {e}")
            return False

    def get_conversation_history(self, user_id: str) -> List[Dict[str, Any]]:
        try:
            resp = self.table.get_item(Key={"user_id": user_id})
            item = resp.get("Item", {})
            return item.get("context", [])
        except Exception as e:
            LOGGER.warning(f"Failed to load conversation history for {user_id[:8]}...: {e}")
            return []

    # (Keep your other helper methods here, all indented inside the class)

# ✅ Singleton instance
dynamo_conversation = DynamoConversationService()
