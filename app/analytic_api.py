"""
Clean FastAPI application with separated concerns.
Main.py only handles routing and basic app setup.
"""
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Depends, Request, Response, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import ValidationError

from app.security.auth import bearer_scheme, validate_jwt_token
from app.services.query_processor import QueryProcessor, PromptRequest
from app.services.audit_sqs_service import get_audit_sqs_service
from app.logging_config import setup_logging, get_logger

# Setup logging with PII redaction
setup_logging(log_level="INFO", enable_pii_redaction=True)
logger = get_logger("analytic_agent")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events (stateless)."""
    logger.info("Starting Analytic Agent API (stateless)...")
    yield
    logger.info("Shutdown complete")

# Create FastAPI app with lifespan handler
app = FastAPI(
    title="Analytic Agent API",
    description="Scalable analytic agent (stateless, DynamoDB conversation history)",
    version="2.0.0",
    lifespan=lifespan
)

# Initialize query coordinator
query_processor = QueryProcessor()

@app.post("/api/analytics/report", response_model=Dict[str, Any])
async def receive_userprompt(
    http_request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
) -> Dict[str, Any]:
    """
    Process analytic user prompt with authentication and audit logging.
    Includes AWS SQS audit logging before sending response.
    """
    user_id = None
    username = None
    prompt = ""
    result = None
    
    try:
        # Extract user info from JWT for audit logging
        try:
            jwt_payload = validate_jwt_token(credentials)
            user_id = jwt_payload.get("sub")
            username = jwt_payload.get("userName") or jwt_payload.get("email", "").split("@")[0]
        except Exception as jwt_error:
            logger.warning(f"Failed to extract user info for audit: {jwt_error}")
        
        # Parse and validate request
        request_data = await http_request.json()
        request = PromptRequest(**request_data)
        prompt = request.prompt
        
        # Process the request
        result = await query_processor.query_handler(request, http_request, credentials)
        
        # Send audit log before returning response
        audit_service = get_audit_sqs_service()
        
        audit_service.send_analytics_query_audit(
            statusCode=200,
            user_id=user_id,
            username=username or "unknown",
            prompt=prompt,
            success=result.get("success", False),
            message=result.get("message") if not result.get("success") else None
        )
        
        return result
        
    except ValidationError as e:
        # Extract validation error message
        error_msg = ""
        if e.errors():
            error_detail = e.errors()[0]
            if "Potentially malicious content detected" in error_detail.get("msg", ""):
                error_msg = "Request blocked for security reasons"
            elif "Prompt cannot be empty" in error_detail.get("msg", ""):
                error_msg = "Prompt cannot be empty"
            elif "Prompt too long" in error_detail.get("msg", ""):
                error_msg = "Prompt exceeds maximum length"
            else:
                error_msg = "Invalid request format"
        
        logger.warning(f"Validation failed: {error_msg}")
        
        # Send audit log for validation failure
        audit_service = get_audit_sqs_service()
        audit_service.send_analytics_query_audit(
            statusCode=400,
            user_id=user_id,
            username=username or "unknown",
            prompt=prompt,
            success=False,
            message=error_msg
        )
        
        return {
            "success": False,
            "message": error_msg,
            "chart_image": None
        }
        
    except Exception as e:
        logger.exception(f"Unexpected error in API endpoint: {e}")
        
        # Send audit log for unexpected errors
        audit_service = get_audit_sqs_service()
        audit_service.send_analytics_query_audit(
            statusCode=500,
            user_id=user_id,
            username=username or "unknown",
            prompt=prompt,
            success=False,
            message=str(e)
        )
        
        return {
            "success": False,
            "message": "An unexpected error occurred",
            "chart_image": None
        }