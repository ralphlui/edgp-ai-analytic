"""
Enhanced query coordinator with simplified reference resolution and improved security.
"""
import logging
import time
from typing import Dict, Any, Optional

from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, ValidationError

from app.core.analytic_service import AnalyticService
from app.services.memory_service import memory_service
from app.auth import validate_jwt_token
from app.utils.request_context import (
    set_tenant_context, 
    reset_tenant_context, 
    set_current_org_id, 
    reset_current_org_id, 
    get_current_org_id
)
from app.core.tools_agent import bind_session_to_tenant, unbind_session

# Constants
SESSION_COOKIE_MAX_AGE = 3600
SESSION_ID_PREFIX_LENGTH = 8

logger = logging.getLogger("analytic_agent")

class PromptRequest(BaseModel):
    prompt: str
    session_id: str | None = None

class QueryCoordinator:
    """
    Simplified coordinator that focuses on security and delegates 
    reference resolution to memory service and LLM.
    """

    def __init__(self):
        self._analytic_service = AnalyticService()

    async def process_query(
        self,
        request: PromptRequest,
        http_request: Request,
        response: Response,
        credentials: HTTPAuthorizationCredentials
    ) -> Dict[str, Any]:
        """
        Process analytic query with simplified reference handling.
        
        SECURITY FEATURES:
        - JWT enforcement (only trusts orgId from JWT claims)
        - Secure session handling with HttpOnly+Secure+SameSite cookies
        - Session binding with tenant verification
        """

        session_id = None
        tenant_tokens = None

        try:
            # 1) SECURITY: JWT validation
            user = validate_jwt_token(credentials)
            user_id = user.get("sub")
            org_id = user.get("orgId")
            
            if not user_id or not org_id:
                raise ValueError("JWT missing required claims: sub (user_id) or orgId")
            
            logger.info(f"JWT validated - user: {user_id[:8]}..., org: {org_id[:8]}...")

            # 2) SECURITY: Resolve secure session ID
            session_id = await self._resolve_secure_session_id(request, http_request, user_id)
            
            # 3) SECURITY: Set tenant context
            tenant_tokens = set_tenant_context(org_id, user_id, session_id)
            
            # 4) SECURITY: Bind session to tenant
            bind_success = bind_session_to_tenant(session_id, user_id, org_id)
            if not bind_success:
                raise ValueError("Failed to bind session to tenant")

            # 5) Get conversation history
            conversation_history = memory_service.get_conversation_history(session_id)

            # 6) SIMPLIFIED: Extract explicit references from prompt
            memory_service.extract_and_store_references(session_id, request.prompt)

            # 7) SIMPLIFIED: Basic reference resolution (only explicit cases)
            processed_prompt = memory_service.simple_reference_resolution(session_id, request.prompt)

            # 8) Set session cookie
            self._set_session_cookie(response, session_id)

            # 9) Execute analytic query (LLM handles complex reference resolution)
            result = await self._analytic_service.process_query(
                prompt=processed_prompt,
                session_id=session_id,
                conversation_history=conversation_history
            )

            # 10) Store interaction result
            self._store_interaction_result(session_id, request.prompt, processed_prompt, result)

            # 11) Log debug information
            self._log_debug_info(session_id, request.prompt, processed_prompt, result)

            return result

        except (ValidationError, ValueError) as e:
            logger.warning(f"Validation error: {e}")
            return self._create_error_response("Invalid request data", str(e))

        except HTTPException:
            raise

        except Exception as error:
            logger.exception(f"Query processing failed: {error}")
            if session_id:
                self._store_error_interaction(session_id, request.prompt, error)
            return self._create_error_response("Processing failed", str(error))

        finally:
            # Cleanup
            if tenant_tokens:
                reset_tenant_context(tenant_tokens)
            if session_id:
                unbind_session(session_id)
                logger.debug(f"Cleaned up session {session_id[:8]}...")

    async def _resolve_secure_session_id(
        self, 
        request: PromptRequest, 
        http_request: Request, 
        user_id: str
    ) -> str:
        """Resolve and validate secure session ID."""
        # Check for existing session cookie
        session_id = http_request.cookies.get("analytic_session_id")
        
        # Generate new session if needed
        if not session_id or len(session_id) < 32:
            session_id = self._generate_secure_session_id(user_id)
            logger.info(f"Generated new session for user {user_id[:8]}...: {session_id[:8]}...")
        else:
            logger.debug(f"Using existing session: {session_id[:8]}...")
        
        # Ensure session exists in memory
        if session_id not in memory_service.sessions:
            memory_service.create_session(user_id)
            logger.info(f"Created memory session for {session_id[:8]}...")
        
        return session_id
    
    def _generate_secure_session_id(self, user_id: str) -> str:
        """Generate cryptographically secure session ID."""
        import secrets
        import hashlib
        
        random_data = secrets.token_bytes(32)
        timestamp = str(time.time()).encode()
        user_data = user_id.encode()
        
        hash_input = random_data + timestamp + user_data
        session_hash = hashlib.sha256(hash_input).hexdigest()
        
        return f"sess_{session_hash[:48]}"
    
    def _set_session_cookie(self, response: Response, session_id: str) -> None:
        """Set secure session cookie."""
        response.set_cookie(
            key="analytic_session_id",
            value=session_id,
            max_age=SESSION_COOKIE_MAX_AGE,
            httponly=True,
            secure=True,
            samesite="lax",
        )

    def _store_interaction_result(
        self, 
        session_id: str, 
        original_prompt: str, 
        processed_prompt: str,
        result: Dict[str, Any]
    ) -> None:
        """Store interaction result in memory service."""
        try:
            tool_name = result.get("tool_name", "analytic_query")
            memory_service.store_interaction(session_id, original_prompt, tool_name, result)
            logger.debug(f"Stored interaction for session {session_id[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to store interaction: {e}")

    def _store_error_interaction(
        self, 
        session_id: str, 
        prompt: str, 
        error: Exception
    ) -> None:
        """Store error interaction in memory service."""
        try:
            error_result = {
                "success": False,
                "error": str(error),
                "error_type": type(error).__name__
            }
            memory_service.store_interaction(session_id, prompt, "error", error_result)
        except Exception as e:
            logger.warning(f"Failed to store error interaction: {e}")

    def _log_debug_info(
        self, 
        session_id: str, 
        original_prompt: str, 
        processed_prompt: str,
        result: Dict[str, Any]
    ) -> None:
        """Log debug information if debug logging is enabled."""
        if logger.isEnabledFor(logging.DEBUG):
            session_prefix = session_id[:SESSION_ID_PREFIX_LENGTH]
            logger.debug(f"Session {session_prefix} - Original: '{original_prompt[:50]}...'")
            logger.debug(f"Session {session_prefix} - Processed: '{processed_prompt[:50]}...'")
            logger.debug(f"Session {session_prefix} - Success: {result.get('success', False)}")
            
            if result.get("file_name"):
                logger.debug(f"Session {session_prefix} - File: {result['file_name']}")
            if result.get("domain_name"):
                logger.debug(f"Session {session_prefix} - Domain: {result['domain_name']}")

    def _create_error_response(self, error_type: str, details: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "success": False,
            "error": error_type,
            "message": f"An error occurred: {details}",
            "chart_image": None
        }