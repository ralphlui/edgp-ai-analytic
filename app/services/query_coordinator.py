"""
Enhanced query coordinator with simplified reference resolution and improved security.
"""
import logging
import time
from typing import Dict, Any

from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, ValidationError, field_validator

from app.core.analytic_service import AnalyticService
from app.services.memory import dynamo_conversation
from app.auth import validate_user_profile_with_response
from app.utils.sanitization import sanitize_text_input
from app.config import SESSION_COOKIE_MAX_AGE_HOURS
from app.utils.request_context import set_tenant_context, reset_tenant_context
from app.tools.session_manager import bind_session_to_tenant, unbind_session

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


class QueryCoordinator:
    """
    Simplified coordinator that focuses on security and delegates
    reference resolution to memory service and LLM.
    """

    def __init__(self):
        # Stateless; conversation history is stored in DynamoDB by user_id.
        self._analytic_service = AnalyticService()

    async def process_query(
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
            org_id = user.get("orgId")
            if not user_id or not org_id:
                raise ValueError("JWT missing required claims: sub (user_id) or orgId")

            logger.info(f"JWT validated - user..., org:...")

            # 2) Bind tenant context for downstream tools (contextvars)
            req_session_id = request.session_id or f"req-{int(time.time()*1000)}-{user_id[:8]}"
            try:
                context_tokens = set_tenant_context(org_id=org_id, user_id=user_id, session_id=req_session_id)
            except Exception as _ctx_err:
                logger.warning(f"Failed to set tenant context: {_ctx_err}")

            # Also bind session for legacy fallback used by some utilities
            try:
                bind_session_to_tenant(req_session_id, user_id, org_id)
            except Exception as _bind_err:
                logger.debug(f"Session bind fallback failed: {_bind_err}")

            # 3) Get conversation history (DynamoDB by user_id)
            conversation_history = dynamo_conversation.get_conversation_history(user_id)

            # 4) Sanitize input
            safe_prompt = sanitize_text_input(request.prompt, max_length=1000)

            # 5) Execute analytic query
            result = await self._analytic_service.process_query(
                prompt=safe_prompt,
                user_id=user_id,
                conversation_history=conversation_history
            )

            # 6) Persist interaction
            self._store_interaction_result(user_id, safe_prompt, result)

            # 7) Log for debug
            #(safe_prompt, result)

            return {
                "success": result.get("success", False),
                "message": result.get("message", ""),
                "chart_image": result.get("chart_image")
            }

        except (ValidationError, ValueError) as e:
            logger.warning(f"Validation error: {e}")
            return self._create_error_response("Invalid request data", str(e))

        except HTTPException:
            raise

        except Exception as error:
            logger.exception(f"Query processing failed: {error}")
            safe_prompt = sanitize_text_input(request.prompt, max_length=1000)
            self._store_error_interaction(user_id, safe_prompt, error)
            return self._create_error_response("Processing failed", str(error))

        finally:
            # Reset tenant context and unbind session fallback
            try:
                if context_tokens:
                    reset_tenant_context(context_tokens)
            except Exception:
                pass
            try:
                if req_session_id:
                    unbind_session(req_session_id)
            except Exception:
                pass

    def _store_interaction_result(
        self,
        user_id: str,
        original_prompt: str,
        result: Dict[str, Any]
    ) -> None:
        """Store interaction result using DynamoDB conversation storage keyed by user_id."""
        try:
            tool_name = result.get("tool") or result.get("tool_name") or "analytic_query"
            dynamo_conversation.store_interaction(user_id, original_prompt)
        except Exception as e:
            logger.warning(f"Failed to store interaction: {e}")

    def _store_error_interaction(
        self,
        user_id: str | None,
        prompt: str,
        error: Exception
    ) -> None:
        """Store error interaction using current storage type."""
        try:
            error_result = {
                "success": False,
                "error": str(error),
                "error_type": type(error).__name__
            }
            if user_id:
                dynamo_conversation.store_interaction(user_id, prompt, "error", error_result)
        except Exception as e:
            logger.warning(f"Failed to store error interaction: {e}")

    def _create_error_response(self, error_type: str, details: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "success": False,
            "error": error_type,
            "message": f"An error occurred: {details}",
            "chart_image": None
        }