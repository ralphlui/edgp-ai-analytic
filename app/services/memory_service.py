from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import logging
import re

logger = logging.getLogger(__name__)

class ConversationMemory:
    """
    Simplified memory service that focuses on storing context and basic reference resolution.
    Complex reference resolution is delegated to the LLM.
    """
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.max_session_history = 10
    
    def create_session(self, user_id: str) -> str:
        """Create a new session for a user."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "interactions": [],
            "context": {
                "recent_files": [],  # Track multiple recent files
                "recent_domains": [],  # Track multiple recent domains
                "last_report_type": "both",
                "preferred_chart_type": "bar",
                "session_focus": "analytic"
            }
        }
        return session_id
    
    def store_interaction(self, session_id: str, user_prompt: str, 
                         tool_used: str, response: Dict[str, Any]):
        """Store an interaction in session memory."""
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id[:8]}... not found, auto-creating")
            self.sessions[session_id] = self._create_default_session()
        
        interaction = {
            "timestamp": datetime.now(),
            "user_prompt": user_prompt,
            "tool_used": tool_used,
            "response_summary": {
                "success": response.get("success", False),
                "tool": response.get("tool"),
                "file_name": response.get("file_name"),
                "domain_name": response.get("domain_name"),
                "row_count": response.get("row_count", 0),
                "report_type": response.get("report_type", "both"),
                "message": self._truncate_message(response.get("message", ""))
            }
        }
        
        # Update context with new file/domain references
        self._update_context_from_interaction(session_id, response)
        
        # Add interaction and maintain history limit
        self.sessions[session_id]["interactions"].append(interaction)
        if len(self.sessions[session_id]["interactions"]) > self.max_session_history:
            self.sessions[session_id]["interactions"].pop(0)
        
        return True
    
    def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session context for reasoning and planning."""
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id[:8]}... not found, auto-creating")
            self.sessions[session_id] = self._create_default_session()
        
        session = self.sessions[session_id]
        recent_interactions = session["interactions"][-3:]
        
        return {
            "user_id": session["user_id"],
            "session_age": (datetime.now() - session["created_at"]).seconds,
            "recent_interactions": recent_interactions,
            "context": session["context"],
            "interaction_count": len(session["interactions"])
        }
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for the session."""
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id[:8]}... not found, auto-creating")
            self.sessions[session_id] = self._create_default_session()
            return []
        return self.sessions[session_id]["interactions"]
    
    def extract_and_store_references(self, session_id: str, prompt: str):
        """Extract explicit file and domain references from prompt."""
        if session_id not in self.sessions:
            self.sessions[session_id] = self._create_default_session()
        
        # Simple, precise patterns for explicit references
        file_matches = re.findall(r"['\"]([^'\"]*\.csv)['\"]|\b([a-zA-Z0-9_-]+\.csv)\b", prompt, re.IGNORECASE)
        domain_matches = re.findall(r"['\"]([^'\"]*\.(?:com|org|net|edu|gov|mil|co\.[a-z]{2}|[a-z]{2}))['\"]|\b([a-zA-Z0-9-]+\.(?:com|org|net|edu|gov|mil|co\.[a-z]{2}|[a-z]{2}))\b", prompt, re.IGNORECASE)
        
        # Store file references
        for match in file_matches:
            file_name = match[0] or match[1]
            if file_name:
                self._add_recent_file(session_id, file_name)
                logger.info(f"Stored file reference: {file_name}")
        
        # Store domain references
        for match in domain_matches:
            domain_name = match[0] or match[1]
            if domain_name:
                self._add_recent_domain(session_id, domain_name)
                logger.info(f"Stored domain reference: {domain_name}")
    
    def get_reference_context_for_llm(self, session_id: str) -> str:
        """Get formatted reference context for LLM system message."""
        if session_id not in self.sessions:
            return ""
        
        context = self.sessions[session_id]["context"]
        recent_files = context.get("recent_files", [])
        recent_domains = context.get("recent_domains", [])
        
        context_parts = []
        
        if recent_files:
            files_str = ", ".join(f"'{f}'" for f in recent_files[-3:])  # Last 3 files
            context_parts.append(f"Recent files analyzed: {files_str}")
        
        if recent_domains:
            domains_str = ", ".join(f"'{d}'" for d in recent_domains[-3:])  # Last 3 domains
            context_parts.append(f"Recent domains analyzed: {domains_str}")
        
        if context_parts:
            return f"""
REFERENCE CONTEXT:
{chr(10).join(context_parts)}

When the user says "that file", "it", "the same data", "that domain", or similar references, 
they likely refer to the items above. Resolve these references appropriately when selecting 
tools and interpreting the user's intent.
"""
        return ""
    
    def simple_reference_resolution(self, session_id: str, prompt: str) -> str:
        """
        Perform only simple, unambiguous reference resolution.
        Complex cases are handled by the LLM.
        """
        if session_id not in self.sessions:
            return prompt
        
        context = self.sessions[session_id]["context"]
        recent_files = context.get("recent_files", [])
        recent_domains = context.get("recent_domains", [])
        
        resolved_prompt = prompt
        
        # Only handle very explicit patterns to avoid false positives
        if recent_files:
            latest_file = recent_files[-1]
            # Very specific patterns only
            patterns = [
                (r'\bthe same file\b', f"'{latest_file}'"),
                (r'\bthat exact file\b', f"'{latest_file}'"),
            ]
            for pattern, replacement in patterns:
                resolved_prompt = re.sub(pattern, replacement, resolved_prompt, flags=re.IGNORECASE)
        
        if recent_domains:
            latest_domain = recent_domains[-1]
            patterns = [
                (r'\bthe same domain\b', f"'{latest_domain}'"),
                (r'\bthat exact domain\b', f"'{latest_domain}'"),
            ]
            for pattern, replacement in patterns:
                resolved_prompt = re.sub(pattern, replacement, resolved_prompt, flags=re.IGNORECASE)
        
        return resolved_prompt
    
    def _create_default_session(self) -> Dict[str, Any]:
        """Create default session structure."""
        return {
            "user_id": "unknown",
            "created_at": datetime.now(),
            "interactions": [],
            "context": {
                "recent_files": [],
                "recent_domains": [],
                "last_report_type": "both",
                "preferred_chart_type": "bar",
                "session_focus": "analytic"
            }
        }
    
    def _update_context_from_interaction(self, session_id: str, response: Dict[str, Any]):
        """Update session context based on interaction response."""
        if response.get("file_name"):
            self._add_recent_file(session_id, response["file_name"])
        
        if response.get("domain_name"):
            self._add_recent_domain(session_id, response["domain_name"])
        
        if response.get("report_type"):
            self.sessions[session_id]["context"]["last_report_type"] = response["report_type"]
    
    def _add_recent_file(self, session_id: str, file_name: str):
        """Add file to recent files list, maintaining order and uniqueness."""
        clean_name = file_name.strip("'\"")
        recent_files = self.sessions[session_id]["context"]["recent_files"]
        
        # Remove if already exists to avoid duplicates
        if clean_name in recent_files:
            recent_files.remove(clean_name)
        
        # Add to end and limit to 5 recent files
        recent_files.append(clean_name)
        if len(recent_files) > 5:
            recent_files.pop(0)
    
    def _add_recent_domain(self, session_id: str, domain_name: str):
        """Add domain to recent domains list, maintaining order and uniqueness."""
        clean_name = domain_name.strip("'\"")
        recent_domains = self.sessions[session_id]["context"]["recent_domains"]
        
        # Remove if already exists to avoid duplicates
        if clean_name in recent_domains:
            recent_domains.remove(clean_name)
        
        # Add to end and limit to 5 recent domains
        recent_domains.append(clean_name)
        if len(recent_domains) > 5:
            recent_domains.pop(0)
    
    def _truncate_message(self, message: str) -> str:
        """Truncate message for storage."""
        if len(message) > 200:
            return message[:200] + "..."
        return message

# Initialize memory service
memory_service = ConversationMemory()