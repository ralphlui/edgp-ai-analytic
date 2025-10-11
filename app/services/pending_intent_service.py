"""
Service for managing pending intents and slots in DynamoDB.

This service stores incomplete user queries to support multi-turn conversations.
"""
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import ClientError

from app.config import (
    AWS_REGION,
    DYNAMODB_PENDING_INTENTS_TABLE,
    PENDING_INTENT_TTL_HOURS
)

logger = logging.getLogger(__name__)


class PendingIntentService:
    """
    Service for managing pending intents in DynamoDB.
    
    Stores extracted intent and slots when user query is complete,
    allowing the system to track what analysis the user wants to perform.
    """
    
    def __init__(self):
        """Initialize DynamoDB client and table."""
        self.dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
        self.dynamodb_client = boto3.client('dynamodb', region_name=AWS_REGION)
        self.table_name = DYNAMODB_PENDING_INTENTS_TABLE
        self.ttl_hours = PENDING_INTENT_TTL_HOURS
        
        # Create table if it doesn't exist
        self._ensure_table_exists()
        
        self.table = self.dynamodb.Table(self.table_name)
        logger.info(f"PendingIntentService initialized with table: {self.table_name}")
    
    def _ensure_table_exists(self):
        """Create DynamoDB table if it doesn't exist."""
        try:
            # Check if table exists
            existing_tables = self.dynamodb_client.list_tables()['TableNames']
            
            if self.table_name in existing_tables:
                logger.info(f"Table {self.table_name} already exists")
                return
            
            logger.info(f"Creating DynamoDB table: {self.table_name}")
            
            # Create table
            self.dynamodb_client.create_table(
                TableName=self.table_name,
                KeySchema=[
                    {'AttributeName': 'user_id', 'KeyType': 'HASH'},  # Partition key
                    {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}  # Sort key
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'user_id', 'AttributeType': 'S'},
                    {'AttributeName': 'timestamp', 'AttributeType': 'N'}
                ],
                BillingMode='PAY_PER_REQUEST'  # On-demand pricing
            )
            
            # Wait for table to be created
            logger.info(f"Waiting for table {self.table_name} to be created...")
            waiter = self.dynamodb_client.get_waiter('table_exists')
            waiter.wait(TableName=self.table_name)
            
            # Enable TTL
            logger.info(f"Enabling TTL on table {self.table_name}")
            self.dynamodb_client.update_time_to_live(
                TableName=self.table_name,
                TimeToLiveSpecification={
                    'Enabled': True,
                    'AttributeName': 'ttl'
                }
            )
            
            logger.info(f"Table {self.table_name} created successfully with TTL enabled")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceInUseException':
                logger.info(f"Table {self.table_name} is already being created")
            else:
                logger.error(f"Failed to create table {self.table_name}: {e}")
                raise
        except Exception as e:
            logger.exception(f"Unexpected error ensuring table exists: {e}")
            raise
    
    def save_intent_and_slots(
        self,
        user_id: str,
        intent: str,
        slots: Dict[str, Any],
        original_prompt: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Save extracted intent and slots to DynamoDB.
        
        This is called when:
        - Intent is success_rate or failure_rate
        - OR domain_name OR file_name is present in slots
        
        If a record already exists for this user, the new prompt is appended to the prompts array.
        
        Args:
            user_id: The user's ID
            intent: Extracted intent (success_rate, failure_rate, etc.)
            slots: Extracted slot values (domain_name, file_name, chart_type, etc.)
            original_prompt: Optional original user prompt for context
            
        Returns:
            Dict with all saved values (user_id, intent, slots, prompts, timestamps) or None if save failed
        """
        try:
            logger.info(f"💾 Starting save for user {user_id}: intent={intent}, slots={slots}")
            
            # Check if user already has a pending intent
            existing = self.get_latest_intent(user_id)
            
            if existing:
                logger.info(f"📝 Existing record found, appending to prompts array")
                # Update existing record - append new prompt to prompts array
                updated = self._append_prompt_to_existing(
                    user_id=user_id,
                    timestamp=existing['timestamp'],
                    new_prompt=original_prompt,
                    new_slots=slots
                )
                
                if updated:
                    # Return the updated record
                    return self.get_latest_intent(user_id)
                else:
                    return None
            
            logger.info(f"📝 No existing record, creating new one")
            
            # Create new record
            ttl_timestamp = int((datetime.now() + timedelta(hours=self.ttl_hours)).timestamp())
            current_timestamp = int(time.time())
            
            # Build prompts array
            prompts = []
            if original_prompt:
                prompts.append({
                    'prompt': original_prompt,
                    'timestamp': datetime.now().isoformat()
                })
            
            item = {
                'user_id': user_id,  # Partition key
                'timestamp': current_timestamp,  # Sort key
                'report_type': intent,
                'slots': slots,
                'prompts': prompts,  # Array of prompts
                'ttl': ttl_timestamp,  # DynamoDB will auto-delete
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            logger.info(f"💾 Saving item to DynamoDB: {item}")
            
            # Save to DynamoDB
            self.table.put_item(Item=item)
            
            logger.info(
                f"✅ Successfully saved intent and slots for user {user_id}: "
                f"intent={intent}, slots={slots}, prompts_count={len(prompts)}"
            )
            
            # Return the saved record with all values
            return {
                'intent': intent,
                'slots': slots,
                'prompts': prompts,
            }
            
        except ClientError as e:
            logger.error(f"❌ DynamoDB ClientError for user {user_id}: {e}")
            logger.error(f"Error code: {e.response['Error']['Code']}")
            logger.error(f"Error message: {e.response['Error']['Message']}")
            return None
        except Exception as e:
            logger.exception(f"❌ Unexpected error saving intent and slots: {e}")
            return None
    
    def _append_prompt_to_existing(
        self,
        user_id: str,
        timestamp: int,
        new_prompt: str,
        new_slots: Dict[str, Any]
    ) -> bool:
        """
        Append a new prompt to an existing pending intent record.
        
        Args:
            user_id: The user's ID
            timestamp: Timestamp of the existing item
            new_prompt: New prompt to append
            new_slots: New slots to merge with existing
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            if not new_prompt:
                logger.warning("No prompt to append")
                return False
            
            # Create new prompt entry
            new_prompt_entry = {
                'prompt': new_prompt,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update: append to prompts list and merge slots
            response = self.table.update_item(
                Key={
                    'user_id': user_id,
                    'timestamp': timestamp
                },
                UpdateExpression='SET prompts = list_append(if_not_exists(prompts, :empty_list), :new_prompt), '
                                'slots = :slots, '
                                'updated_at = :updated_at',
                ExpressionAttributeValues={
                    ':empty_list': [],
                    ':new_prompt': [new_prompt_entry],
                    ':slots': new_slots,
                    ':updated_at': datetime.now().isoformat()
                },
                ReturnValues='UPDATED_NEW'
            )
            
            logger.info(f"Appended prompt to existing record for user {user_id}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to append prompt for user {user_id}: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error appending prompt: {e}")
            return False
    
    def get_latest_intent(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the most recent intent and slots for a user.
        
        Args:
            user_id: The user's ID
            
        Returns:
            Dict with intent, slots, prompts array, and metadata, or None if not found
        """
        try:
            response = self.table.query(
                KeyConditionExpression='user_id = :uid',
                ExpressionAttributeValues={':uid': user_id},
                ScanIndexForward=False,  # Sort descending by timestamp
                Limit=1
            )
            
            items = response.get('Items', [])
            
            if items:
                item = items[0]
                logger.info(
                    f"Retrieved intent for user {user_id}: {item.get('report_type')}, "
                    f"prompts_count={len(item.get('prompts', []))}"
                )
                return {
                    'intent': item.get('report_type'),  # Changed from 'intent' to 'report_type'
                    'slots': item.get('slots', {}),
                    'prompts': item.get('prompts', []),  # Array of prompts
                    'created_at': item.get('created_at'),
                    'updated_at': item.get('updated_at'),
                    'timestamp': item.get('timestamp')
                }
            
            logger.info(f"No pending intent found for user {user_id}")
            return None
            
        except ClientError as e:
            logger.error(f"Failed to retrieve intent for user {user_id}: {e}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error retrieving intent: {e}")
            return None
    
    def update_slots(
        self,
        user_id: str,
        timestamp: int,
        new_slots: Dict[str, Any]
    ) -> bool:
        """
        Update slots for an existing pending intent.
        
        Args:
            user_id: The user's ID
            timestamp: Timestamp of the item to update
            new_slots: New slot values to merge with existing
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            # Merge new slots with existing
            response = self.table.update_item(
                Key={
                    'user_id': user_id,
                    'timestamp': timestamp
                },
                UpdateExpression='SET slots = :slots, updated_at = :updated_at',
                ExpressionAttributeValues={
                    ':slots': new_slots,
                    ':updated_at': datetime.now().isoformat()
                },
                ReturnValues='UPDATED_NEW'
            )
            
            logger.info(f"Updated slots for user {user_id}: {new_slots}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to update slots for user {user_id}: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error updating slots: {e}")
            return False
    
    def clear_intent(self, user_id: str) -> bool:
        """
        Clear the most recent pending intent for a user.
        
        Called after successfully processing a complete query.
        
        Args:
            user_id: The user's ID
            
        Returns:
            bool: True if delete successful, False otherwise
        """
        try:
            # Get the latest intent first
            latest = self.get_latest_intent(user_id)
            
            if not latest:
                logger.info(f"No intent to clear for user {user_id}")
                return True
            
            # Delete the item
            self.table.delete_item(
                Key={
                    'user_id': user_id,
                    'timestamp': latest['timestamp']
                }
            )
            
            logger.info(f"Cleared intent for user {user_id}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to clear intent for user {user_id}: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error clearing intent: {e}")
            return False
    
    def should_save_intent(self, intent: str, slots: Dict[str, Any]) -> bool:
        """
        Determine if the intent and slots should be saved to DynamoDB.
        
        Save conditions (OR logic):
        - Intent is success_rate OR failure_rate
        - OR (domain_name OR file_name) is present in slots
        
        Args:
            intent: Extracted intent
            slots: Extracted slots
            
        Returns:
            bool: True if should save, False otherwise
        """
        logger.info(f"🔍 Checking save criteria - Intent: '{intent}', Slots: {slots}")
        
        # Check if intent is one we want to save
        valid_intents = ['success_rate', 'failure_rate']
        is_valid_intent = intent in valid_intents
        
        # Check if required slots are present
        has_domain = slots.get('domain_name') is not None and slots.get('domain_name') != ''
        has_file = slots.get('file_name') is not None and slots.get('file_name') != ''
        has_required_slot = has_domain or has_file
        
        logger.info(
            f"🔍 Criteria check - "
            f"is_valid_intent: {is_valid_intent}, "
            f"has_domain: {has_domain}, "
            f"has_file: {has_file}, "
            f"has_required_slot: {has_required_slot}"
        )
        
        # Save if EITHER condition is met (OR logic)
        should_save = is_valid_intent or has_required_slot
        
        if should_save:
            logger.info(
                f"✅ Save criteria met: "
                f"intent={intent} (valid={is_valid_intent}), "
                f"has_domain={has_domain}, has_file={has_file}"
            )
        else:
            logger.info(
                f"❌ Save criteria NOT met: "
                f"intent={intent} (valid={is_valid_intent}), "
                f"no domain_name or file_name in slots"
            )
        
        return should_save


# Singleton instance
_pending_intent_service = None


def get_pending_intent_service() -> PendingIntentService:
    """Get the singleton pending intent service instance."""
    global _pending_intent_service
    if _pending_intent_service is None:
        _pending_intent_service = PendingIntentService()
    return _pending_intent_service
