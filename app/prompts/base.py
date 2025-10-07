"""
Base classes for prompt management with versioning support.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum


class PromptVersion(Enum):
    """Prompt version enumeration for tracking changes."""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"


@dataclass
class PromptTemplate:
    """
    Base template for all prompts with metadata and versioning.
    
    Attributes:
        content: The actual prompt text
        version: Version identifier for tracking
        description: Human-readable description
        created_at: When this version was created
        tags: Optional tags for categorization
        variables: Variables that can be injected into the prompt
    """
    content: str
    version: PromptVersion = PromptVersion.V1_0
    description: str = ""
    created_at: datetime = None
    tags: Optional[list] = None
    variables: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.tags is None:
            self.tags = []
        if self.variables is None:
            self.variables = {}
    
    def format(self, **kwargs) -> str:
        """
        Format the prompt template with provided variables.
        
        Args:
            **kwargs: Variables to inject into the template
            
        Returns:
            Formatted prompt string
        """
        # Merge default variables with provided ones
        format_vars = {**self.variables, **kwargs}
        return self.content.format(**format_vars)
    
    def with_context(self, context: Dict[str, Any]) -> str:
        """
        Format prompt with context dictionary.
        
        Args:
            context: Dictionary containing context variables
            
        Returns:
            Formatted prompt string
        """
        return self.format(**context)


class PromptRegistry:
    """Registry for managing multiple prompt versions."""
    
    def __init__(self):
        self._prompts: Dict[str, Dict[str, PromptTemplate]] = {}
    
    def register(self, name: str, prompt: PromptTemplate):
        """Register a prompt template."""
        if name not in self._prompts:
            self._prompts[name] = {}
        self._prompts[name][prompt.version.value] = prompt
    
    def get(self, name: str, version: Optional[str] = None) -> Optional[PromptTemplate]:
        """Get a prompt by name and optional version."""
        if name not in self._prompts:
            return None
        
        if version:
            return self._prompts[name].get(version)
        
        # Return latest version if no version specified
        versions = sorted(self._prompts[name].keys(), reverse=True)
        return self._prompts[name][versions[0]] if versions else None
    
    def list_versions(self, name: str) -> list:
        """List all versions of a prompt."""
        return list(self._prompts.get(name, {}).keys())


# Global prompt registry instance
prompt_registry = PromptRegistry()
