"""
Session management and tool orchestration for the analytic system.
Handles session binding, authentication context, and tool coordination.
"""
from typing import Optional, Dict, Any
import logging
from app.utils.request_context import get_current_org_id

logger = logging.getLogger("tools_agent")

# Module-level storage for current request org_id (fallback)
_current_org_id: Optional[str] = None

# Session to tenant binding storage
_session_tenant_bindings: Dict[str, Dict[str, str]] = {}


def bind_session_to_tenant(session_id: str, user_id: str, org_id: str) -> bool:
    """Bind a session to tenant context for cross-thread access."""
    try:
        _session_tenant_bindings[session_id] = {
            "user_id": user_id,
            "org_id": org_id
        }
        logger.info(f"Bound session {session_id} to org {org_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to bind session {session_id}: {e}")
        return False


def unbind_session(session_id: str) -> None:
    """Remove session binding."""
    try:
        if session_id in _session_tenant_bindings:
            del _session_tenant_bindings[session_id]
            logger.info(f"Unbound session {session_id}")
    except Exception as e:
        logger.error(f"Failed to unbind session {session_id}: {e}")


def get_session_context(session_id: str) -> Optional[Dict[str, str]]:
    """Get session context for tools."""
    return _session_tenant_bindings.get(session_id)


def set_current_org_id(org_id: str) -> None:
    """Set module-level org_id as fallback."""
    global _current_org_id
    _current_org_id = org_id
    logger.debug(f"Set current org_id to: {org_id}")


def get_current_org_id_fallback() -> Optional[str]:
    """Get module-level org_id fallback."""
    return _current_org_id


def get_active_sessions() -> Dict[str, Dict[str, str]]:
    """Get all active session bindings for debugging."""
    return _session_tenant_bindings.copy()


def cleanup_expired_sessions(active_session_ids: list[str]) -> int:
    """Remove sessions that are no longer active."""
    expired_count = 0
    sessions_to_remove = []
    
    for session_id in _session_tenant_bindings.keys():
        if session_id not in active_session_ids:
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        unbind_session(session_id)
        expired_count += 1
    
    if expired_count > 0:
        logger.info(f"Cleaned up {expired_count} expired sessions")
    
    return expired_count