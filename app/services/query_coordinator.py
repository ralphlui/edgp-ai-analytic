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
from app.services.memory import memory_service, memory_manager, RedisSessionStorage
from app.auth import validate_user_profile_with_response
from app.utils.sanitization import sanitize_text_input
from app.utils.request_context import (
    set_tenant_context,
    reset_tenant_context
)
from app.tools import bind_session_to_tenant, unbind_session
from app.config import USE_REDIS_SESSIONS, REDIS_URL

# Constants
SESSION_COOKIE_MAX_AGE = 3600
SESSION_ID_PREFIX_LENGTH = 8

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
            'execute:', 'run:', 'rm -rf', '\\n\\n', '%0A%0A',
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
    Now with optional Redis session storage for production scalability.
    """

    def __init__(self, redis_url: str = None):
        self._analytic_service = AnalyticService()
        
        # Initialize session storage based on configuration flag
        if USE_REDIS_SESSIONS:
            self.redis_storage = RedisSessionStorage(redis_url or REDIS_URL)
            if self.redis_storage.available:
                logger.info("QueryCoordinator initialized with Redis session storage (USE_REDIS_SESSIONS=true)")
            else:
                logger.warning("USE_REDIS_SESSIONS=true but Redis unavailable, falling back to memory")
        else:
            self.redis_storage = None
            logger.info("QueryCoordinator initialized with memory session storage (USE_REDIS_SESSIONS=false)")
    
    def _get_session_storage(self):
        """Get the appropriate session storage based on config flag and Redis availability."""
        if USE_REDIS_SESSIONS and self.redis_storage and self.redis_storage.available:
            return self.redis_storage
        else:
            return memory_service

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
            # 1) SECURITY: JWT validation and user profile verification
            auth_result = await validate_user_profile_with_response(credentials)
            if not auth_result["success"]:
                logger.warning(f"Authentication failed: {auth_result['message']}")
                return auth_result  # Return structured error response
            
            user = auth_result["payload"]
            user_id = user.get("sub")
            org_id = user.get("orgId")
            
            if not user_id or not org_id:
                raise ValueError("JWT missing required claims: sub (user_id) or orgId")
            
            logger.info(f"JWT validated and user profile verified - user: {user_id[:8]}..., org: {org_id[:8]}...")

            # 2) SECURITY: Resolve secure session ID
            session_id = await self._resolve_secure_session_id(request, http_request, user_id)
            
            # 3) SECURITY: Set tenant context
            tenant_tokens = set_tenant_context(org_id, user_id, session_id)
            
            # 4) SECURITY: Bind session to tenant
            bind_success = bind_session_to_tenant(session_id, user_id, org_id)
            if not bind_success:
                raise ValueError("Failed to bind session to tenant")

            # 5) Get conversation history
            session_storage = self._get_session_storage()
            if hasattr(session_storage, 'get_session'):
                # Using Redis storage
                session_data = session_storage.get_session(session_id)
                conversation_history = session_data.get("interactions", []) if session_data else []
            else:
                # Using memory storage
                conversation_history = session_storage.get_conversation_history(session_id)

            # 6) SECURITY: Sanitize user input to prevent prompt injection
            safe_prompt = sanitize_text_input(request.prompt, max_length=1000)
            logger.debug(f"Sanitized prompt for session {session_id[:8]}...")

            # 7) SIMPLIFIED: Extract explicit references from sanitized prompt
            session_storage = self._get_session_storage()
            if hasattr(session_storage, 'extract_and_store_references'):
                # Using memory storage
                session_storage.extract_and_store_references(session_id, safe_prompt)
            # For Redis storage, we'll handle reference extraction differently

           

            # 9) Set session cookie
            self._set_session_cookie(response, session_id)

            # 10) Execute analytic query (LLM handles complex reference resolution)
            result = await self._analytic_service.process_query(
                prompt=safe_prompt,
                session_id=session_id,
                conversation_history=conversation_history
            )

            # 11) Store interaction result (use original prompt for logging accuracy)
            self._store_interaction_result(session_id, safe_prompt, result)

            # 12) Log debug information
            self._log_debug_info(session_id, safe_prompt, result)

            return result

        except (ValidationError, ValueError) as e:
            logger.warning(f"Validation error: {e}")
            return self._create_error_response("Invalid request data", str(e))

        except HTTPException:
            raise

        except Exception as error:
            logger.exception(f"Query processing failed: {error}")
            if session_id:
                # Use sanitized prompt for error logging too
                safe_prompt = sanitize_text_input(request.prompt, max_length=1000)
                self._store_error_interaction(session_id, safe_prompt, error)
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
        """Resolve and validate secure session ID with Redis or memory storage."""
        # Check for existing session cookie
        session_id = http_request.cookies.get("analytic_session_id")
        
        # Generate new session if needed
        if not session_id or len(session_id) < 32:
            session_id = self._generate_secure_session_id(user_id)
            logger.info(f"Generated new session for user {user_id[:8]}...: {session_id[:8]}...")
        else:
            logger.debug(f"Using existing session: {session_id[:8]}...")
        
        # Ensure session exists using appropriate storage
        session_storage = self._get_session_storage()
        
        if hasattr(session_storage, 'get_session'):
            # Using Redis storage
            session_data = session_storage.get_session(session_id)
            if not session_data:
                new_session_id = session_storage.create_session(user_id)
                session_id = new_session_id
                logger.info(f"Created Redis session for {session_id[:8]}...")
            else:
                # Reset TTL on each interaction to keep active sessions alive
                session_storage.touch_session(session_id)
                logger.debug(f"Found existing Redis session for {session_id[:8]}...")
        else:
            # Using memory storage (fallback)
            session_context = session_storage.get_session_context(session_id)
            if not session_context:
                created_session_id = session_storage.create_session(user_id)
                if created_session_id != session_id:
                    session_id = created_session_id
                logger.info(f"Created memory session for {session_id[:8]}...")
            else:
                # Reset TTL on each interaction (matching Redis behavior)
                if hasattr(session_storage, 'touch_session'):
                    session_storage.touch_session(session_id)
                logger.debug(f"Found existing memory session context for {session_id[:8]}...")
        
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
        result: Dict[str, Any]
    ) -> None:
        """Store interaction result using Redis or memory storage."""
        try:
            tool_name = result.get("tool_name", "analytic_query")
            session_storage = self._get_session_storage()
            
            if hasattr(session_storage, 'get_session'):
                # Using Redis storage - need to implement interaction storage
                session_data = session_storage.get_session(session_id)
                if session_data:
                    # Add interaction to session data
                    interaction = {
                        "timestamp": time.time(),
                        "user_prompt": original_prompt,
                        "tool_used": tool_name,
                        "response_summary": {
                            "success": result.get("success", False),
                            "tool": result.get("tool"),
                            "file_name": result.get("file_name"),
                            "domain_name": result.get("domain_name"),
                            "row_count": result.get("row_count", 0),
                            "report_type": result.get("report_type", "both"),
                            "message": result.get("message", "")[:200]  # Truncate
                        }
                    }
                    
                    # Maintain history limit (10 interactions)
                    interactions = session_data.get("interactions", [])
                    interactions.append(interaction)
                    if len(interactions) > 10:
                        interactions.pop(0)
                    
                    session_data["interactions"] = interactions
                    session_storage.update_session(session_id, session_data)
                    logger.debug(f"Stored interaction in Redis for session {session_id[:8]}...")
                else:
                    logger.warning(f"Session {session_id[:8]} not found in Redis for interaction storage")
            else:
                # Using memory storage
                session_storage.store_interaction(session_id, original_prompt, tool_name, result)
                logger.debug(f"Stored interaction in memory for session {session_id[:8]}...")
            
            # Update memory stats for monitoring (regardless of storage type)
            stats = memory_manager.get_memory_stats()
            storage = self._get_session_storage()
            if hasattr(storage, 'available') and storage.available:
                # For Redis, get session info (no expensive cleanup needed)
                redis_info = storage.get_session_info()
                session_count = redis_info.get("total_sessions", 0)
                if session_count > 500:
                    logger.warning(f"High Redis session count: {session_count} sessions active")
            elif stats["total_sessions"] > 500:
                logger.warning(f"High memory session count: {stats['total_sessions']} sessions active")
                
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
        result: Dict[str, Any]
    ) -> None:
        """Log debug information if debug logging is enabled."""
        if logger.isEnabledFor(logging.DEBUG):
            session_prefix = session_id[:SESSION_ID_PREFIX_LENGTH]
            logger.debug(f"Session {session_prefix} - Original: '{original_prompt[:50]}...'")
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