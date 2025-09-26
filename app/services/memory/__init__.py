"""
Memory-based session management module.

This module provides both Redis and in-memory session storage with consistent TTL behavior:
- 24-hour default TTL with activity-based reset
- Touch-based keep-alive functionality  
- 20-message conversation history limit
- Automatic expired session cleanup
- Explicit session deletion for logout scenarios

Components:
- RedisSessionStorage: Redis-based session storage with TTL auto-expiration
- ConversationMemory: TTL-based in-memory session storage with Redis-compatible interface
- MemoryManager: Periodic cleanup and memory optimization
- memory_service: Global memory storage instance
- memory_manager: Global memory cleanup instance

Usage:
    from app.services.memory import memory_service, memory_manager, RedisSessionStorage
    
    # Redis storage
    redis_storage = RedisSessionStorage(redis_url)
    
    # Memory storage
    session_id = memory_service.create_session(user_id)
    memory_service.touch_session(session_id)  # Reset TTL
    memory_service.store_interaction(session_id, prompt, tool, response)
    
    # Cleanup management  
    memory_manager.start_cleanup_task()
    memory_manager.stop_cleanup_task()
"""

from .memory_service import ConversationMemory, memory_service
from .memory_cleanup import MemoryManager, memory_manager
from .redis_session_storage import RedisSessionStorage

__all__ = [
    'RedisSessionStorage',
    'ConversationMemory',
    'memory_service', 
    'MemoryManager',
    'memory_manager'
]