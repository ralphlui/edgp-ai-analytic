"""
Repositories package for data access layer.
"""

from app.repositories.analytics_repository import (
    AnalyticsRepository,
    get_analytics_repository
)

__all__ = [
    'AnalyticsRepository',
    'get_analytics_repository'
]
