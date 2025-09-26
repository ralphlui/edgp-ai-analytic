"""
Redis-based session storage for production scalability.
This is an optional upgrade from in-memory session storage.
"""
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import redis
from app.config import DEBUG

logger = logging.getLogger(__name__)

class RedisSessionStorage:
    """Production-ready session storage using Redis."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            self.available = True
            logger.info("Redis session storage initialized successfully")
        except Exception as e:
            logger.warning(f"Redis not available, falling back to memory: {e}")
            self.redis_client = None
            self.available = False
    
    def create_session(self, user_id: str) -> str:
        """Create a new session."""
        import uuid
        session_id = str(uuid.uuid4())
        
        session_data = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "interactions": [],
            "context": {
                "recent_files": [],
                "recent_domains": [],
                "last_report_type": "both",
                "preferred_chart_type": "bar",
                "session_focus": "analytic"
            }
        }
        
        if self.available:
            # Store in Redis with 24-hour expiration
            self.redis_client.setex(
                f"session:{session_id}",
                timedelta(hours=24),
                json.dumps(session_data, default=str)
            )
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        if not self.available:
            return None
        
        try:
            data = self.redis_client.get(f"session:{session_id}")
            if data:
                session_data = json.loads(data)
                # Convert created_at back to datetime
                if 'created_at' in session_data:
                    session_data['created_at'] = datetime.fromisoformat(session_data['created_at'])
                return session_data
        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
        
        return None
    
    def update_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """Update session data and reset TTL to keep active sessions alive."""
        if not self.available:
            return False
        
        try:
            # Truncate interaction history to last 20 messages for memory efficiency
            if "interactions" in session_data:
                session_data["interactions"] = session_data["interactions"][-20:]
            
            # Update session data and reset TTL to 24 hours (keep active sessions alive)
            self.redis_client.setex(
                f"session:{session_id}",
                timedelta(hours=24),  # Reset TTL on activity
                json.dumps(session_data, default=str)
            )
            return True
        except Exception as e:
            logger.error(f"Error updating session {session_id}: {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Explicitly delete a session (e.g., user logout)."""
        if not self.available:
            return False
        
        try:
            deleted_count = self.redis_client.delete(f"session:{session_id}")
            if deleted_count > 0:
                logger.info(f"Explicitly deleted session {session_id[:8]}... (user logout)")
            return deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    def touch_session(self, session_id: str) -> bool:
        """
        Reset TTL for an active session (keep alive on activity).
        Call this on each user interaction to reset the 24-hour timer.
        """
        if not self.available:
            return False
        
        try:
            # Reset TTL to 24 hours without modifying data
            result = self.redis_client.expire(f"session:{session_id}", timedelta(hours=24))
            if result:
                logger.debug(f"Reset TTL for active session {session_id[:8]}...")
            return result
        except Exception as e:
            logger.error(f"Error touching session {session_id}: {e}")
            return False
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get basic session info without complex scanning."""
        if not self.available:
            return {"available": False, "total_sessions": 0}
        
        try:
            # Simple info without expensive operations
            info = self.redis_client.info("memory")
            session_count = len(self.redis_client.keys("session:*"))
            
            return {
                "available": True,
                "total_sessions": session_count,
                "redis_memory_mb": round(info.get("used_memory", 0) / (1024 * 1024), 2),
                "connected_clients": info.get("connected_clients", 0)
            }
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return {"available": False, "error": str(e)}

# Example usage - Redis native session management
# redis_storage = RedisSessionStorage()
# 
# # 1. Sessions auto-expire after 24 hours (TTL)
# session_id = redis_storage.create_session(user_id)
# 
# # 2. Keep active sessions alive by resetting TTL on each interaction
# redis_storage.touch_session(session_id)  # Resets 24h timer
# 
# # 3. History is automatically truncated to last 20 messages
# redis_storage.update_session(session_id, session_data)  # Auto-truncates
# 
# # 4. Explicit cleanup on logout
# redis_storage.delete_session(session_id)  # Immediate cleanup