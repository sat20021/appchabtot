"""
Structured Logging Configuration

Configures structured JSON logging with correlation IDs and context.
"""

import logging
import sys
from typing import Any, Dict
from datetime import datetime

import structlog
from structlog.types import EventDict, Processor


def add_timestamp(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add ISO timestamp to log entries"""
    event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return event_dict


def add_log_level(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add log level to event dict"""
    event_dict["level"] = method_name.upper()
    return event_dict


def add_logger_name(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add logger name to event dict"""
    if hasattr(logger, "name"):
        event_dict["logger"] = logger.name
    return event_dict


def configure_structured_logging(
    log_level: str = "INFO",
    json_logs: bool = True,
    include_stdlib: bool = True
) -> None:
    """
    Configure structured logging for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: Whether to output JSON format (True) or console format (False)
        include_stdlib: Whether to configure standard library logging
    """
    
    # Shared processors for all loggers
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        add_timestamp,
        add_log_level,
        add_logger_name,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if json_logs:
        # JSON output for production
        processors = shared_processors + [
            structlog.processors.JSONRenderer()
        ]
    else:
        # Console output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging to use structlog
    if include_stdlib:
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, log_level),
        )
        
        # Redirect standard logging to structlog
        logging.root.handlers = []
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logging.root.addHandler(handler)
        logging.root.setLevel(getattr(logging, log_level))


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance
    
    Usage:
        from core.logging_config import get_logger
        
        logger = get_logger(__name__)
        logger.info("User logged in", user_id="123", email="user@example.com")
    """
    return structlog.get_logger(name)


# Context management for request-scoped logging
def bind_context(**kwargs: Any) -> None:
    """
    Bind context variables to the current request
    
    Usage:
        from core.logging_config import bind_context
        
        bind_context(
            correlation_id=request.state.correlation_id,
            user_id=user.user_id
        )
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """
    Unbind specific context variables
    
    Usage:
        unbind_context("user_id", "session_id")
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all context variables"""
    structlog.contextvars.clear_contextvars()


# Example usage in FastAPI middleware
class LoggingMiddleware:
    """
    Middleware to add request context to logs
    
    Usage:
        from core.logging_config import LoggingMiddleware
        
        app.add_middleware(LoggingMiddleware)
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Bind request context
            bind_context(
                path=scope["path"],
                method=scope["method"],
                client=scope.get("client", ["unknown", 0])[0]
            )
        
        try:
            await self.app(scope, receive, send)
        finally:
            # Clear context after request
            clear_context()
