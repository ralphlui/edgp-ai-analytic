"""
Enhanced query coordinator with simplified reference resolution and improved security.
"""
import logging
import time
from typing import Dict, Any

from app.agents.intent_slot_agent import get_intent_slot_agent
from app.services.pending_intent_service import get_pending_intent_service
from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, ValidationError, field_validator

from app.core.analytic_service import AnalyticService
from app.services.memory import dynamo_conversation
from app.auth import validate_user_profile_with_response
from app.utils.sanitization import sanitize_text_input
from app.config import SESSION_COOKIE_MAX_AGE_HOURS

# Constants
SESSION_COOKIE_MAX_AGE = SESSION_COOKIE_MAX_AGE_HOURS * 3600  # Convert hours to seconds (unused after session removal)
SESSION_ID_PREFIX_LENGTH = 8  # For log truncation when needed

logger = logging.getLogger("analytic_agent")


class PromptRequest(BaseModel):
    prompt: str
    session_id: str | None = None

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        """Validate prompt input for basic security checks."""
        if not v or not v.strip():
            raise ValueError('Prompt cannot be empty')

        if len(v) > 5000:
            raise ValueError('Prompt too long (max 5000 characters)')

        # Check for obviously malicious patterns (before full sanitization)
        dangerous_indicators = [
            'system:', 'assistant:', 'user:', 'human:', 'ai:',
            'ignore previous', 'forget previous', 'disregard',
            'you are now', 'your role is', 'act as', 'pretend to be',
            'execute:', 'run:', 'rm -rf', '\n\n', '%0A%0A',
            '[inst]', '[/inst]', '<|', '|>', '{{', '}}'
        ]

        v_lower = v.lower()
        for indicator in dangerous_indicators:
            if indicator in v_lower:
                raise ValueError(f'Potentially malicious content detected: {indicator}')

        return v


class QueryProcessor:
    """
    Simplified processor that focuses on security and delegates
    reference resolution to memory service and LLM.
    """

    def __init__(self):
        # Stateless; conversation history is stored in DynamoDB by user_id.
        self._analytic_service = AnalyticService()

    async def query_handler(
        self,
        request: PromptRequest,
        http_request: Request,
        credentials: HTTPAuthorizationCredentials
    ) -> Dict[str, Any]:
        """
        Process analytic query with simplified reference handling.

        SECURITY FEATURES:
        - JWT enforcement (only trusts orgId from JWT claims)
        - No server-side sessions; history is retrieved by user_id from DynamoDB
        """
        user_id = None
        context_tokens = None
        req_session_id = None

        try:
            # 1) SECURITY: JWT validation and user profile verification
            auth_result = await validate_user_profile_with_response(credentials)
            if not auth_result.get("success"):
                logger.warning(f"Authentication failed: {auth_result.get('message')}")
                return auth_result  # Structured error

            user = auth_result["payload"]
            user_id = user.get("sub")
            if not user_id :
                raise ValueError("JWT missing required claims: userid")

            logger.info(f"JWT validated - user: {user_id}")
            
            # Extract intent and slots
            logger.info(f"🤖 Extracting intent and slots from prompt: '{request.prompt}'")
            agent = get_intent_slot_agent()
            result = await agent.extract_intent_and_slots(request.prompt)
            result = agent.validate_completeness(result)
            
            logger.info(f"🎯 Extracted - Intent: {result.intent}, Slots: {result.slots}, Complete: {result.is_complete}")
            
            # Save to DynamoDB if conditions are met
            # Conditions: Intent is success_rate OR failure_rate
            #            AND (domain_name OR file_name) is present
            pending_service = get_pending_intent_service()
            
            logger.info(f"🔍 Checking if should save to DynamoDB...")
            should_save = pending_service.should_save_intent(result.intent, result.slots)
            logger.info(f"💡 Should save result: {should_save}")
            
            saved_data = None
            if should_save:
                logger.info(f"💾 Attempting to save to DynamoDB for user...")
                saved_data = pending_service.save_intent_and_slots(
                    user_id=user_id,
                    intent=result.intent,
                    slots=result.slots,
                    original_prompt=request.prompt
                )
                
                if saved_data:
                    logger.info(
                        f"Saved data: {saved_data}"
                    )
                else:
                    logger.error(f"❌ Failed to save to DynamoDB for user")
            else:
                logger.info(
                    f"⏭️ Skipping save - Intent/slots do not meet criteria: "
                    f"intent={result.intent}, slots={result.slots}"
                )
            
            # Check if query is complete
            if not result.is_complete:
                logger.warning(f"Incomplete query - Missing: {result.missing_required}")
                
                # Determine what's missing for proper error messaging
                has_report_type = result.intent in ['success_rate', 'failure_rate']
                has_domain = result.slots.get('domain_name') and result.slots.get('domain_name') != ''
                has_file = result.slots.get('file_name') and result.slots.get('file_name') != ''
                has_target = has_domain or has_file
                
                # Build specific error message based on what's missing
                if not has_report_type and not has_target:
                    # Missing both report type and target
                    error_message = (
                        "⚠️ Missing Information: I need both the analysis type and target to proceed.\n\n"
                        "Please specify:\n"
                        "1. Analysis type: 'success rate' or 'failure rate'\n"
                        "2. Target: a domain name (e.g., 'example.com') or file name (e.g., 'test.py')\n\n"
                        "Example: 'Show me the success rate for example.com'"
                    )
                    missing_fields = ["report_type", "target"]
                    
                elif not has_report_type:
                    # Missing only report type (has target)
                    target = result.slots.get('domain_name') or result.slots.get('file_name')
                    target_type = "domain" if has_domain else "file"
                    error_message = (
                        f"⚠️ Missing Analysis Type: I see you want to analyze {target_type} '{target}', "
                        f"but I need to know what type of analysis.\n\n"
                        f"Please specify:\n"
                        f"- 'success rate' - to see successful operations\n"
                        f"- 'failure rate' - to see failed operations\n\n"
                        f"Example: 'Show me the success rate for {target}'"
                    )
                    missing_fields = ["report_type"]
                    
                elif not has_target:
                    # Missing only target (has report type)
                    analysis_type = result.intent.replace('_', ' ')
                    error_message = (
                        f"⚠️ Missing Target: I understand you want {analysis_type} analysis, "
                        f"but I need to know what to analyze.\n\n"
                        f"Please specify:\n"
                        f"- A domain name (e.g., 'example.com')\n"
                        f"- OR a file name (e.g., 'test.py')\n\n"
                        f"Example: 'Show me the {analysis_type} for example.com'"
                    )
                    missing_fields = ["target"]
                    
                
                return {
                    "success": False,
                    "message": error_message,
                    "chart_image": None,
                }
            
            # TODO: Route to appropriate service based on intent
            # For now, return the extracted information
            return {
                "success": True,
                "message": f"Intent detected: {result.intent}, Slots: {result.slots}",
                "chart_image": None,
            }
          





        except (ValidationError, ValueError) as e:
            logger.warning(f"Validation error: {e}")
            return self._create_error_response("Invalid request data", str(e))

        except HTTPException:
            raise

        except Exception as error:
            logger.exception(f"Query processing failed: {error}")
            safe_prompt = sanitize_text_input(request.prompt, max_length=1000)
            return self._create_error_response("Processing failed", str(error))

    
    def _create_error_response(self, error_type: str, details: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "success": False,
            "error": error_type,
            "message": f"An error occurred: {details}",
            "chart_image": None
        }
