"""
Enhanced query coordinator with simplified reference resolution and improved security.
"""
import logging
import time
from typing import Dict, Any

from app.agents.query_understanding_agent import get_query_understanding_agent
from app.services.pending_intent_service import get_pending_intent_service
from app.security.prompt_validator import validate_user_prompt, validate_llm_output
from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, ValidationError, field_validator

from app.security.auth import validate_user_profile_with_response

# Constants
SESSION_ID_PREFIX_LENGTH = 8  # For log truncation when needed

logger = logging.getLogger("analytic_agent")


class PromptRequest(BaseModel):
    prompt: str
    session_id: str | None = None

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        """Enhanced prompt validation using advanced security validator."""
        if not v or not v.strip():
            raise ValueError('Prompt cannot be empty')

        if len(v) > 5000:
            raise ValueError('Prompt too long (max 5000 characters)')

        # Use advanced regex-based security validation
        is_safe, error_msg = validate_user_prompt(v)
        if not is_safe:
            raise ValueError(error_msg)

        return v


class QueryProcessor:
    """
    Simplified processor that focuses on security and delegates
    reference resolution to memory service and LLM.
    """

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
            agent = get_query_understanding_agent()
            result = await agent.extract_intent_and_slots(request.prompt)
            result = agent.validate_completeness(result)
            
            logger.info(f"🎯 Extracted - Intent: {result.intent}, Slots: {result.slots}, Complete: {result.is_complete}")
            
             # Handle out-of-scope queries (non-analytics questions)
            if result.clarification_needed is not None:
                logger.info(f"📢 Out-of-scope query detected: '{request.prompt}'")
                return {
                    "success": False,
                    "message": result.clarification_needed or "I'm specialized in analytics. Please ask about success rates, failure rates, or data analysis.",
                    "chart_image": None,
                }
            # Smart Inheritance Logic: Try to inherit missing fields from previous context
            # This enables natural multi-turn conversations
            pending_service = get_pending_intent_service()
            
            # Check if we're missing report_type OR target (domain/file)
            has_report_type = result.intent in ['success_rate', 'failure_rate']
            has_domain = result.slots.get('domain_name') and result.slots.get('domain_name') != ''
            has_file = result.slots.get('file_name') and result.slots.get('file_name') != ''
            has_target = has_domain or has_file
            
            # Retrieve previous data first (for conflict detection and inheritance)
            previous_data = pending_service.get_pending_intent(user_id)
            
            # CONFLICT DETECTION: Check if user is switching target types
            if previous_data and has_target:
                prev_domain = previous_data.get('slots', {}).get('domain_name')
                prev_file = previous_data.get('slots', {}).get('file_name')
                
                # Detect target type conflict (switching from domain to file or vice versa)
                if (has_domain and prev_file) or (has_file and prev_domain):
                    prev_target = f"domain '{prev_domain}'" if prev_domain else f"file '{prev_file}'"
                    curr_target = f"domain '{result.slots['domain_name']}'" if has_domain else f"file '{result.slots['file_name']}'"
                    
                    logger.warning(f"⚠️ Target conflict detected: {prev_target} vs {curr_target}")
                    
                    # Save the new extraction temporarily with a special marker
                    # This allows us to retrieve it when user confirms
                    pending_service.save_intent_and_slots(
                        user_id=user_id,
                        intent=result.intent,
                        slots={**result.slots, '_conflict_pending': True},  # Add marker
                        original_prompt=f"[CONFLICT] {request.prompt}"
                    )
                    
                    logger.info(f"💾 Saved conflicting target temporarily with _conflict_pending marker")
                    
                    # Ask user to choose
                    return {
                        "success": False,
                        "message": (
                            f"⚠️ **Target Conflict Detected**\n\n"
                            f"I see you mentioned:\n"
                            f"• **Previously**: {prev_target}\n"
                            f"• **Just now**: {curr_target}\n\n"
                            f"Which target should I use?\n"
                            f"1️⃣  {curr_target} (new target)\n"
                            f"2️⃣  {prev_target} (previous target)\n\n"
                            f"💡 **Quick responses**:\n"
                            f"   • Type '1' or 'use file' for option 1\n"
                            f"   • Type '2' or 'use domain' for option 2\n"
                            f"   • Or just tell me what to analyze next!"
                        ),
                        "chart_image": None,
                    }
            
            # CONFLICT RESOLUTION: Check if user is responding to a conflict
            if previous_data and previous_data.get('slots', {}).get('_conflict_pending'):
                logger.info("🔍 Detected conflict resolution attempt")
                
                # Remove the conflict marker for comparison
                prev_slots = {k: v for k, v in previous_data.get('slots', {}).items() if k != '_conflict_pending'}
                prev_domain = prev_slots.get('domain_name')
                prev_file = prev_slots.get('file_name')
                
                # Check if user is confirming their choice
                confirmation_keywords = {
                    'use_current': ['1', 'use file', 'use current', 'new one', 'use csv', 'file'],
                    'use_previous': ['2', 'use domain', 'use previous', 'keep it', 'domain', 'keep'],
                }
                
                prompt_lower = request.prompt.lower()
                
                # Check which option user chose
                for action, keywords in confirmation_keywords.items():
                    if any(keyword in prompt_lower for keyword in keywords):
                        if action == 'use_current':
                            # User chose the new target (the one with conflict marker)
                            logger.info(f"✅ User confirmed: use new target from previous prompt")
                            # Clean up the marker and continue
                            result.slots = prev_slots
                            # Don't inherit anything else - use what's in conflict
                            break
                        elif action == 'use_previous':
                            # User chose to go back to the target before the conflict
                            logger.info(f"✅ User confirmed: revert to target before conflict")
                            
                            # Need to retrieve the record before the conflict
                            # For now, clear the conflict and ask user to re-specify
                            pending_service.clear_intent(user_id)
                            
                            return {
                                "success": False,
                                "message": (
                                    f"✅ Cleared the conflicting target.\n\n"
                                    f"Please specify what you'd like to analyze again.\n"
                                    f"For example: 'success rate for customer domain'"
                                ),
                                "chart_image": None,
                            }
            
            # If missing report_type OR target, try to inherit from previous context
            if not has_report_type or not has_target:
                logger.info(f"🔍 Missing fields detected - Checking for previous context to inherit...")
                logger.info(f"   has_report_type: {has_report_type}, has_target: {has_target}")
                
                # Use previous_data already retrieved above (for conflict detection)
                if previous_data:
                    logger.info(f"✅ Found previous context: {previous_data}")
                    
                    # Inherit missing report_type (only if previous has valid intent)
                    if not has_report_type:
                        prev_report_type = previous_data.get('report_type')
                        if prev_report_type and prev_report_type in ['success_rate', 'failure_rate']:
                            result.intent = prev_report_type
                            logger.info(
                                f"✨ Inherited report_type '{result.intent}' from previous prompt "
                                f"(last updated: {previous_data.get('updated_at')})"
                            )
                    
                    # Inherit missing target (domain or file)
                    if not has_target:
                        prev_slots = previous_data.get('slots', {})
                        if prev_slots.get('domain_name'):
                            result.slots['domain_name'] = prev_slots['domain_name']
                            logger.info(f"✨ Inherited domain_name '{result.slots['domain_name']}' from previous prompt")
                        elif prev_slots.get('file_name'):
                            result.slots['file_name'] = prev_slots['file_name']
                            logger.info(f"✨ Inherited file_name '{result.slots['file_name']}' from previous prompt")
                    
                    # Re-validate after inheritance
                    has_report_type = result.intent in ['success_rate', 'failure_rate']
                    has_domain = result.slots.get('domain_name') and result.slots.get('domain_name') != ''
                    has_file = result.slots.get('file_name') and result.slots.get('file_name') != ''
                    has_target = has_domain or has_file
                    
                    # Mark as complete if we now have both
                    if has_report_type and has_target:
                        result.is_complete = True
                        logger.info(f"✅ Query completed after inheritance: intent={result.intent}, slots={result.slots}")
                else:
                    logger.info(f"⏰ No previous context found (expired or never existed)")
            
            
            # Save to DynamoDB if conditions are met
            # Note: Check AFTER inheritance to save merged values
            # Conditions: Intent is success_rate OR failure_rate OR has target (domain/file)
            
            logger.info(f"🔍 Checking if should save to DynamoDB (after inheritance)...")
            logger.info(f"📊 VALUES TO CHECK FOR SAVE:")
            logger.info(f"   - Intent (after inheritance): '{result.intent}'")
            logger.info(f"   - Slots (after inheritance): {result.slots}")
            logger.info(f"   - Is complete (after inheritance): {result.is_complete}")
            
            should_save = pending_service.should_save_intent(result.intent, result.slots)
            logger.info(f"💡 Should save result: {should_save}")
            
            saved_data = None
            if should_save:
                logger.info(f"💾 SAVING TO DYNAMODB:")
                logger.info(f"   - user_id: {user_id}")
                logger.info(f"   - intent: '{result.intent}'")
                logger.info(f"   - slots: {result.slots}")
                logger.info(f"   - prompt: '{request.prompt}'")
                
                saved_data = pending_service.save_intent_and_slots(
                    user_id=user_id,
                    intent=result.intent,
                    slots=result.slots,
                    original_prompt=request.prompt
                )
                
                if saved_data:
                    logger.info(f"✅ SAVE SUCCESSFUL:")
                    logger.info(f"   - Saved intent: {saved_data.get('intent')}")
                    logger.info(f"   - Saved slots: {saved_data.get('slots')}")
                    logger.info(f"   - Saved prompts count: {len(saved_data.get('prompts', []))}")
                else:
                    logger.error(f"❌ Failed to save to DynamoDB for user")
            else:
                logger.info(
                    f"⏭️ SKIPPING SAVE - Intent/slots do not meet criteria:"
                )
                logger.info(f"   - Intent: '{result.intent}'")
                logger.info(f"   - Slots: {result.slots}")
            
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
                        f"- A domain name (e.g., 'customer domain')\n"
                        f"- OR a file name (e.g., 'customer.csv')\n\n"
                        f"Example: 'Show me the {analysis_type} for customer domain'"
                    )
                    missing_fields = ["target"]
                    
                
                return {
                    "success": False,
                    "message": error_message,
                    "chart_image": None,
                }
            
            # Call analytics orchestrator - coordinates tool execution, chart generation, and response
            logger.info(f"🚀 Calling analytics orchestrator")
            
            from app.agents.analytics_workflow_agent import run_analytics_query
            
            try:
                # Build extracted data for workflow
                # Note: We only pass domain_name and file_name
                # The LLM will analyze user_query to decide which tool to call
                extracted_data = {
                    "domain_name": result.slots.get("domain_name"),
                    "file_name": result.slots.get("file_name")
                }
                
                logger.info(f"📊 Workflow input - Query: '{request.prompt}'")
                logger.info(f"📊 Workflow input - Data: {extracted_data}")
                
                # Run workflow - LLM decides between success_rate and failure_rate tools
                response = await run_analytics_query(
                    user_query=request.prompt,
                    extracted_data=extracted_data
                )
                
                logger.info(f"✅ Workflow completed successfully")
                logger.info(f"📈 Response - Success: {response.get('success')}, Has chart: {response.get('chart_image') is not None}")
                
                # OUTPUT VALIDATION: Check for information leaks before returning
                is_safe_output, leak_error = validate_llm_output(response)
                if not is_safe_output:
                    logger.error(f"🚨 Blocked unsafe output for user {user_id}")
                    logger.error(f"   Leak detected: {leak_error}")
                    return {
                        "success": False,
                        "message": "I apologize, but I cannot provide that information. Please ask about analytics data only.",
                        "chart_image": None
                    }
                
                return response
                
            except Exception as e:
                logger.exception(f"❌ Analytics workflow execution failed: {e}")
                return {
                    "success": False,
                    "message": f"I encountered an error while processing your analytics request: {str(e)}",
                    "chart_image": None
                }
          





        except (ValidationError, ValueError) as e:
            logger.warning(f"Validation error: {e}")
            return self._create_error_response("Invalid request data", str(e))

        except HTTPException:
            raise

        except Exception as error:
            logger.exception(f"Query processing failed: {error}")
            return self._create_error_response("Processing failed", str(error))

    
    def _create_error_response(self, error_type: str, details: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "success": False,
            "error": error_type,
            "message": f"An error occurred: {details}",
            "chart_image": None
        }
