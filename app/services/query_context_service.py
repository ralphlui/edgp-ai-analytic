import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import ClientError

from app.config import (
    AWS_REGION,
    DYNAMODB_CONVERSATION_CONTEXT_TABLE,
    CONVERSATION_CONTEXT_TTL_HOURS
)

logger = logging.getLogger(__name__)


class QueryContextService:
    """
    Service for managing query context in DynamoDB.
    
    Stores query context including report type and target entities (domain/file)
    to enable multi-turn conversations and context inheritance across queries.
    """
    
    def __init__(self):
        """Initialize DynamoDB client and table."""
        self.dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
        self.dynamodb_client = boto3.client('dynamodb', region_name=AWS_REGION)
        self.table_name = DYNAMODB_CONVERSATION_CONTEXT_TABLE
        self.ttl_hours = CONVERSATION_CONTEXT_TTL_HOURS
        
        # Create table if it doesn't exist
        self._ensure_table_exists()
        
        self.table = self.dynamodb.Table(self.table_name)
        logger.info(f"QueryContextService initialized with table: {self.table_name}")
    
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
    
    def save_query_context(
        self,
        user_id: str,
        intent: str,
        slots: Dict[str, Any],
        original_prompt: str = None,
        comparison_targets: Optional[list] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Save query context (report type, slots, prompt, comparison targets) to DynamoDB.
        
        This is called when:
        - Intent is success_rate or failure_rate
        - OR domain_name OR file_name is present in slots
        
        If a record already exists for this user, the new prompt is appended to the prompts array.
        
        Args:
            user_id: The user's ID
            intent: Extracted intent (success_rate, failure_rate, etc.)
            slots: Extracted slot values (domain_name, file_name, chart_type, etc.)
            original_prompt: Optional original user prompt for context
            comparison_targets: Optional list of comparison target files
        
        Returns:
            Dict with all saved values (user_id, intent, slots, prompts, timestamps) or None if save failed
        """
        try:
            logger.info(f"========== SAVE QUERY CONTEXT START ==========")
            logger.info(f"Input parameters:")
            logger.info(f"   - intent: '{intent}'")
            logger.info(f"   - slots: {slots}")
            logger.info(f"   - original_prompt: '{original_prompt}'")
            logger.info(f"   - comparison_targets: {comparison_targets}")
            
            # Check if user already has existing context
            existing = self.get_full_context(user_id)
            
            if existing:
                logger.info(f"Existing record found:")
                logger.info(f"   - Current intent: '{existing.get('intent')}'")
                logger.info(f"   - Current slots: {existing.get('slots')}")
                logger.info(f"   - Will REPLACE with new values (REPLACE strategy)")
                
                # REPLACE strategy: Update existing record with new values
                # This also refreshes the TTL
                intent = intent if intent != "" else existing.get('intent')
                updated = self._update_existing_record(
                    user_id=user_id,
                    timestamp=existing['timestamp'],
                    new_intent=intent,
                    new_slots=slots,
                    new_prompt=original_prompt,
                    new_comparison_targets=comparison_targets
                )
                
                if updated:
                    # Return the updated record
                    final_record = self.get_full_context(user_id)
                    logger.info(f"Update successful, final record:")
                    logger.info(f"   - Stored intent: '{final_record.get('intent')}'")
                    logger.info(f"   - Stored slots: {final_record.get('slots')}")
                    logger.info(f"========== SAVE QUERY CONTEXT END ==========")
                    return final_record
                else:
                    logger.error(f"Update failed")
                    logger.info(f"========== SAVE QUERY CONTEXT END ==========")
                    return None
            
            logger.info(f"No existing record, creating new one")
            
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
            
            if comparison_targets:
                item['comparison_targets'] = comparison_targets
            
            logger.info(f"Creating new DynamoDB item:")
            logger.info(f"   - report_type: '{intent}'")
            logger.info(f"   - slots: {slots}")
            logger.info(f"   - prompts_count: {len(prompts)}")
            logger.info(f"   - comparison_targets: {comparison_targets}")
            
            # Save to DynamoDB
            self.table.put_item(Item=item)
            
            logger.info(f"Successfully created new record for user {user_id}")
            logger.info(f"========== SAVE QUERY CONTEXT END ==========")
            
            # Return the saved record with all values
            return {
                'intent': intent,
                'slots': slots,
                'prompts': prompts,
                'comparison_targets': comparison_targets
            }
            
        except ClientError as e:
            logger.error(f"DynamoDB ClientError for user {user_id}: {e}")
            logger.error(f"Error code: {e.response['Error']['Code']}")
            logger.error(f"Error message: {e.response['Error']['Message']}")
            logger.info(f"========= SAVE QUERY CONTEXT END ==========")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error saving query context: {e}")
            logger.info(f"========== SAVE QUERY CONTEXT END ==========")
            return None
    
    def _update_existing_record(
        self,
        user_id: str,
        timestamp: int,
        new_intent: str,
        new_slots: Dict[str, Any],
        new_prompt: str,
        new_comparison_targets: Optional[list] = None
    ) -> bool:
        """
        Update an existing query context record with REPLACE strategy.
        
        This replaces the intent and slots with new values and appends the prompt to history.
        TTL is also refreshed to keep active conversations alive.
        
        Args:
            user_id: The user's ID
            timestamp: Timestamp of the existing item
            new_intent: New intent to replace existing (e.g., 'success_rate')
            new_slots: New slots to replace existing
            new_prompt: New prompt to append to prompts history
            new_comparison_targets: New comparison targets to replace existing
        
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            logger.info(f"========== UPDATE EXISTING RECORD START ==========")
            logger.info(f"Update parameters:")
            logger.info(f"   - user_id: {user_id}")
            logger.info(f"   - timestamp: {timestamp}")
            logger.info(f"   - new_intent: '{new_intent}'")
            logger.info(f"   - new_slots: {new_slots}")
            logger.info(f"   - new_prompt: '{new_prompt}'")
            logger.info(f"   - new_comparison_targets: {new_comparison_targets}")
            
            if not new_prompt:
                logger.warning("No prompt to append")
                logger.info(f"========== UPDATE EXISTING RECORD END ==========")
                return False
            
            # Create new prompt entry
            new_prompt_entry = {
                'prompt': new_prompt,
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate new TTL (refresh expiry time)
            new_ttl = int((datetime.now() + timedelta(hours=self.ttl_hours)).timestamp())
            
            update_expression = ('SET report_type = :intent, '
                                'slots = :slots, '
                                'prompts = list_append(if_not_exists(prompts, :empty_list), :new_prompt), '
                                'updated_at = :updated_at, '
                                '#ttl = :ttl')
            
            expression_attribute_values = {
                ':intent': new_intent,  # REPLACE intent
                ':slots': new_slots,     # REPLACE slots
                ':empty_list': [],
                ':new_prompt': [new_prompt_entry],
                ':updated_at': datetime.now().isoformat(),
                ':ttl': new_ttl
            }
            
            if new_comparison_targets:
                update_expression += ', comparison_targets = :comparison_targets'
                expression_attribute_values[':comparison_targets'] = new_comparison_targets
            
            logger.info(f"Performing DynamoDB update:")
            logger.info(f"   - REPLACE report_type with: '{new_intent}'")
            logger.info(f"   - REPLACE slots with: {new_slots}")
            logger.info(f"   - APPEND prompt to history")
            logger.info(f"   - REFRESH TTL to: {new_ttl}")
            logger.info(f"   - REPLACE comparison_targets with: {new_comparison_targets}")
            
            # Update: REPLACE intent and slots, append prompt, refresh TTL
            # Note: 'ttl' is a reserved keyword in DynamoDB, so we use ExpressionAttributeNames
            response = self.table.update_item(
                Key={
                    'user_id': user_id,
                    'timestamp': timestamp
                },
                UpdateExpression=update_expression,
                ExpressionAttributeNames={
                    '#ttl': 'ttl'  # Handle reserved keyword
                },
                ExpressionAttributeValues=expression_attribute_values,
                ReturnValues='UPDATED_NEW'
            )
            
            logger.info(f"DynamoDB update successful!")
            logger.info(f"   - Updated attributes: {response.get('Attributes', {})}")
            logger.info(f"========== UPDATE EXISTING RECORD END ==========")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to update record for user {user_id}: {e}")
            logger.info(f"========== UPDATE EXISTING RECORD END ==========")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error updating record: {e}")
            logger.info(f"========== UPDATE EXISTING RECORD END ==========")
            return False
    
    def get_query_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the most recent query context for a user.
        
        This method checks if previous context exists and is still fresh (within TTL).
        If TTL has expired, returns None even if DynamoDB hasn't deleted the record yet.
        
        Args:
            user_id: The user's ID
            
        Returns:
            Dict with report_type, slots, updated_at, and timestamp, or None if not found/expired
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
                
                # Manual TTL validation: Check if record is still fresh
                ttl_timestamp = item.get('ttl')
                current_time = int(time.time())
                
                if ttl_timestamp and current_time >= ttl_timestamp:
                    # TTL has expired, treat as if record doesn't exist
                    logger.info(
                        f"Query context found for user {user_id} but TTL expired: "
                        f"ttl={ttl_timestamp}, now={current_time}, expired_by={current_time - ttl_timestamp}s"
                    )
                    return None
                
                logger.info(
                    f"Found query context for user {user_id}: report_type={item.get('report_type')}, "
                    f"updated_at={item.get('updated_at')}, ttl_remaining={ttl_timestamp - current_time}s"
                )
                return {
                    'report_type': item.get('report_type'),
                    'slots': item.get('slots', {}),
                    'comparison_targets': item.get('comparison_targets'),
                    'updated_at': item.get('updated_at'),
                    'timestamp': item.get('timestamp')
                }
            
            logger.info(f"No query context found for user {user_id} (expired or never existed)")
            return None
            
        except ClientError as e:
            logger.error(f"Failed to retrieve query context for user {user_id}: {e}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error retrieving query context: {e}")
            return None
    
    def get_full_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the complete query context with full history for a user.
        
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
                    f"Retrieved full context for user {user_id}: {item.get('report_type')}, "
                    f"prompts_count={len(item.get('prompts', []))}"
                )
                return {
                    'intent': item.get('report_type'),  # Changed from 'intent' to 'report_type'
                    'slots': item.get('slots', {}),
                    'comparison_targets': item.get('comparison_targets'),
                    'prompts': item.get('prompts', []),  # Array of prompts
                    'created_at': item.get('created_at'),
                    'updated_at': item.get('updated_at'),
                    'timestamp': item.get('timestamp')
                }
            
            logger.info(f"No query context found for user {user_id}")
            return None
            
        except ClientError as e:
            logger.error(f"Failed to retrieve full context for user {user_id}: {e}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error retrieving full context: {e}")
            return None
    
    def update_context_slots(
        self,
        user_id: str,
        timestamp: int,
        new_slots: Dict[str, Any]
    ) -> bool:
        """
        Update slots for an existing query context.
        
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
            
            logger.info(f"Updated context slots for user {user_id}: {new_slots}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to update context slots for user {user_id}: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error updating context slots: {e}")
            return False
    
    def clear_query_context(self, user_id: str) -> bool:
        """
        Clear the most recent query context for a user.
        
        Called after successfully processing a complete query.
        
        Args:
            user_id: The user's ID
            
        Returns:
            bool: True if delete successful, False otherwise
        """
        try:
            # Get the full context first
            latest = self.get_full_context(user_id)
            
            if not latest:
                logger.info(f"No context to clear for user {user_id}")
                return True
            
            # Delete the item
            self.table.delete_item(
                Key={
                    'user_id': user_id,
                    'timestamp': latest['timestamp']
                }
            )
            
            logger.info(f"Cleared query context for user {user_id}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to clear query context for user {user_id}: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error clearing query context: {e}")
            return False
    
    def should_save_context(self, intent: str, slots: Dict[str, Any]) -> bool:
        """
        Determine if the query context should be saved to DynamoDB.
        
        Save conditions (OR logic):
        - Intent is success_rate OR failure_rate
        - OR (domain_name OR file_name) is present in slots
        
        Args:
            intent: Extracted intent
            slots: Extracted slots
            
        Returns:
            bool: True if should save, False otherwise
        """
        logger.info(f"Checking save criteria - Intent: '{intent}', Slots: {slots}")
        
        # Check if intent is one we want to save
        valid_intents = ['success_rate', 'failure_rate']
        is_valid_intent = intent in valid_intents
        
        # Check if required slots are present
        has_domain = slots.get('domain_name') is not None and slots.get('domain_name') != ''
        has_file = slots.get('file_name') is not None and slots.get('file_name') != ''
        has_required_slot = has_domain or has_file
        
        logger.info(
            f"Criteria check - "
            f"is_valid_intent: {is_valid_intent}, "
            f"has_domain: {has_domain}, "
            f"has_file: {has_file}, "
            f"has_required_slot: {has_required_slot}"
        )
        
        # Save if EITHER condition is met (OR logic)
        should_save = is_valid_intent or has_required_slot
        
        if should_save:
            logger.info(
                f"Save criteria met: "
                f"intent={intent} (valid={is_valid_intent}), "
                f"has_domain={has_domain}, has_file={has_file}"
            )
        else:
            logger.info(
                f"Save criteria NOT met: "
                f"intent={intent} (valid={is_valid_intent}), "
                f"no domain_name or file_name in slots"
            )
        
        return should_save


# Singleton instance
_query_context_service = None


def get_query_context_service() -> QueryContextService:
    """Get the singleton query context service instance."""
    global _query_context_service
    if _query_context_service is None:
        _query_context_service = QueryContextService()
    return _query_context_service
