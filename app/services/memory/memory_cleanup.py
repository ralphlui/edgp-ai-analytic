"""
Memory cleanup and session management utilities.
Provides automatic session expiration and memory optimization.
"""
import asyncio
import gc
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from .memory_service import memory_service
from app.tools.session_manager import cleanup_expired_sessions

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages memory cleanup and session expiration."""
    
    def __init__(self):
        self.cleanup_interval = 300  # 5 minutes
        self.session_max_age = timedelta(hours=24)  # 24 hours
        self.max_total_sessions = 1000  # Maximum sessions in memory
        self._cleanup_task = None
    
    def start_cleanup_task(self):
        """Start the periodic cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("Started memory cleanup task")
    
    def stop_cleanup_task(self):
        """Stop the periodic cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None
            logger.info("Stopped memory cleanup task")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup task that runs every cleanup_interval seconds."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_expired_sessions()
                await self.cleanup_large_sessions()
                self.force_garbage_collection()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in periodic cleanup: {e}")
    
    async def cleanup_expired_sessions(self):
        """Remove expired sessions from memory using TTL-based approach (matching Redis)."""
        # Use the memory service's new TTL-based cleanup method
        cleaned_count = memory_service.cleanup_expired_sessions()
        
        if cleaned_count > 0:
            logger.info(f"TTL cleanup: Removed {cleaned_count} expired memory sessions")
        
        # Also cleanup session bindings for removed sessions
        cleanup_expired_sessions([sid for sid in memory_service.sessions.keys()])
    
    async def cleanup_large_sessions(self):
        """Remove oldest sessions if total count exceeds maximum."""
        if len(memory_service.sessions) <= self.max_total_sessions:
            return
        
        # Sort sessions by creation time
        sessions_by_age = sorted(
            memory_service.sessions.items(),
            key=lambda x: x[1].get("created_at", datetime.min)
        )
        
        # Remove oldest sessions
        sessions_to_remove = len(memory_service.sessions) - self.max_total_sessions
        for i in range(sessions_to_remove):
            session_id, _ = sessions_by_age[i]
            del memory_service.sessions[session_id]
            logger.info(f"Removed old session due to limit: {session_id[:8]}...")
        
        logger.info(f"Cleaned up {sessions_to_remove} sessions due to limit")
    
    def force_garbage_collection(self):
        """Force Python garbage collection to free memory."""
        collected = gc.collect()
        if collected > 0:
            logger.debug(f"Garbage collection freed {collected} objects")
    
    def get_memory_stats(self) -> Dict[str, int]:
        """Get current memory usage statistics."""
        return {
            "total_sessions": len(memory_service.sessions),
            "total_interactions": sum(
                len(session.get("interactions", []))
                for session in memory_service.sessions.values()
            ),
            "average_interactions_per_session": (
                sum(len(session.get("interactions", [])) for session in memory_service.sessions.values()) 
                // max(len(memory_service.sessions), 1)
            )
        }

# Global memory manager instance
memory_manager = MemoryManager()