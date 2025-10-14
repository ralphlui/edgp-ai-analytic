"""
Logging Configuration Module

Centralized logging configuration with PII redaction for the analytics agent.
This module sets up logging with automatic PII filtering to ensure compliance
with data protection regulations (GDPR, CCPA).
"""

import logging
import sys
from typing import Optional
from app.security.pii_redactor import PIIRedactionFilter


def setup_logging(
    log_level: str = "INFO",
    log_format: Optional[str] = None,
    enable_pii_redaction: bool = True
) -> logging.Logger:
    """
    Configure application logging with PII redaction.
    
    This function should be called once at application startup to configure
    the logging system. It sets up:
    - Console output with structured formatting
    - Automatic PII redaction (if enabled)
    - Consistent log levels across the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom format string. If None, uses default structured format.
        enable_pii_redaction: Whether to enable automatic PII redaction (default: True)
        
    Returns:
        Configured root logger instance
        
    Example:
        >>> # At application startup (e.g., in main.py or analytic_api.py)
        >>> logger = setup_logging(log_level="INFO", enable_pii_redaction=True)
        >>> logger.info("Logging configured successfully")
    """
    # Default structured format with timestamp, level, module, and message
    if log_format is None:
        log_format = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(message)s'
        )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Create formatter
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    
    # Add PII redaction filter if enabled
    if enable_pii_redaction:
        pii_filter = PIIRedactionFilter()
        console_handler.addFilter(pii_filter)
        root_logger.info("PII redaction filter enabled for all logs")
    
    # Add handler to root logger
    root_logger.addHandler(console_handler)
    
    # Configure specific logger for the application
    app_logger = logging.getLogger("analytic_agent")
    app_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    return root_logger


def get_logger(name: str = "analytic_agent") -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    This is a convenience function that returns a logger. The logger will
    automatically inherit the PII redaction filter from the root logger if
    setup_logging() was called.
    
    Args:
        name: Logger name (default: "analytic_agent")
        
    Returns:
        Logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("User email: user@example.com")  # Auto-redacted to [EMAIL_REDACTED]
    """
    return logging.getLogger(name)


def add_pii_filter_to_existing_loggers() -> None:
    """
    Add PII redaction filter to all existing loggers and handlers.
    
    This function can be used to retrofit PII redaction onto an existing
    logging setup without reconfiguring everything. It finds all active
    loggers and adds the PII filter to their handlers.
    
    Note: This is mainly for compatibility with existing setups. 
    Prefer using setup_logging() for new applications.
    """
    pii_filter = PIIRedactionFilter()
    
    # Get all existing loggers
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers.append(logging.getLogger())  # Include root logger
    
    # Add filter to all handlers
    for logger in loggers:
        for handler in logger.handlers:
            if not any(isinstance(f, PIIRedactionFilter) for f in handler.filters):
                handler.addFilter(pii_filter)


def disable_pii_redaction() -> None:
    """
    Disable PII redaction by removing all PIIRedactionFilter instances.
    
    This is useful for development/debugging environments where you need
    to see the actual data in logs. Should NEVER be used in production.
    
    Warning: This will expose PII in logs. Only use in secure development environments.
    """
    # Get all existing loggers
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers.append(logging.getLogger())  # Include root logger
    
    # Remove PII filters from all handlers
    for logger in loggers:
        for handler in logger.handlers:
            handler.filters = [
                f for f in handler.filters 
                if not isinstance(f, PIIRedactionFilter)
            ]
    
    logging.warning("⚠️  PII redaction has been DISABLED - do not use in production!")
