import contextvars
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Context variables for tenant identity isolation (async-safe)
ORG_ID_CTX: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "org_id", default=None
)
USER_ID_CTX: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "user_id", default=None
)
SESSION_ID_CTX: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "session_id", default=None
)

# Legacy compatibility (to be removed)
_current_org_id: contextvars.ContextVar[Optional[str]] = ORG_ID_CTX

def set_tenant_context(org_id: str, user_id: str, session_id: str) -> Tuple[contextvars.Token, contextvars.Token, contextvars.Token]:
    """
    Set complete tenant context for the current request.
    
    Args:
        org_id: Organization ID from JWT claims
        user_id: User ID from JWT claims  
        session_id: Secure session identifier
        
    Returns:
        Tuple of tokens for cleanup
    """
    if not org_id or not user_id or not session_id:
        raise ValueError("All tenant context parameters are required")
    
    org_token = ORG_ID_CTX.set(org_id)
    user_token = USER_ID_CTX.set(user_id)
    session_token = SESSION_ID_CTX.set(session_id)
    
    logger.info(f"Set tenant context - org: {org_id[:8]}..., user: {user_id[:8]}..., session: {session_id[:8]}...")
    
    return org_token, user_token, session_token

def get_tenant_context() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Get complete tenant context for the current request.
    
    Returns:
        Tuple of (org_id, user_id, session_id)
    """
    return ORG_ID_CTX.get(), USER_ID_CTX.get(), SESSION_ID_CTX.get()

def get_current_org_id() -> Optional[str]:
    """Get org_id for the current request (contextvars-based)."""
    return ORG_ID_CTX.get()

def get_current_user_id() -> Optional[str]:
    """Get user_id for the current request (contextvars-based)."""
    return USER_ID_CTX.get()

def get_current_session_id() -> Optional[str]:
    """Get session_id for the current request (contextvars-based)."""
    return SESSION_ID_CTX.get()

def reset_tenant_context(tokens: Tuple[contextvars.Token, contextvars.Token, contextvars.Token]) -> None:
    """
    Reset tenant context when request ends.
    
    Args:
        tokens: Tuple of tokens returned from set_tenant_context
    """
    org_token, user_token, session_token = tokens
    ORG_ID_CTX.reset(org_token)
    USER_ID_CTX.reset(user_token)
    SESSION_ID_CTX.reset(session_token)
    
    logger.debug("Reset tenant context")

def validate_tenant_context() -> bool:
    """
    Validate that complete tenant context is available.
    
    Returns:
        True if all required context is present
    """
    org_id, user_id, session_id = get_tenant_context()
    return all([org_id, user_id, session_id])

# Legacy compatibility functions (deprecated)
def set_current_org_id(org_id: Optional[str]) -> contextvars.Token:
    """DEPRECATED: Use set_tenant_context instead."""
    logger.warning("set_current_org_id is deprecated, use set_tenant_context")
    return ORG_ID_CTX.set(org_id)

def reset_current_org_id(token: contextvars.Token) -> None:
    """DEPRECATED: Use reset_tenant_context instead."""
    logger.warning("reset_current_org_id is deprecated, use reset_tenant_context")
    ORG_ID_CTX.reset(token)