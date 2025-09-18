"""
Simplified analytic query processing service where LLM handles date extraction.
Main entry point that imports from organized modules.
"""
import logging
from app.core.analytic_service import AnalyticService

# Setup logger
logger = logging.getLogger(__name__)

# Re-export the main service for backward compatibility
__all__ = ['AnalyticService']