"""
Prompts module for analytics agent.

This module provides structured prompt templates including:
- System prompts
- ReAct (Reasoning, Acting, Observing) patterns
- Plan-&-Execute templates
- Tool-specific prompts
"""

from .base import PromptTemplate, PromptVersion
from .system_prompts import SystemPrompts
from .react_prompts import ReActPrompts
from .plan_execute_prompts import PlanExecutePrompts
from .tool_prompts import ToolPrompts

__all__ = [
    "PromptTemplate",
    "PromptVersion", 
    "SystemPrompts",
    "ReActPrompts",
    "PlanExecutePrompts",
    "ToolPrompts",
]
