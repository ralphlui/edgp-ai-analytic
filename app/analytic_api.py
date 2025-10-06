"""
Clean FastAPI application with separated concerns.
Main.py only handles routing and basic app setup.
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Depends, Request, Response, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import ValidationError

from app.auth import bearer_scheme, validate_jwt_token
from app.services.query_coordinator import QueryCoordinator, PromptRequest
from app.services.audit_sqs_service import get_audit_sqs_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analytic_agent")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events (stateless)."""
    logger.info("� Starting Analytic Agent API (stateless)...")
    yield
    logger.info("✅ Shutdown complete")

# Create FastAPI app with lifespan handler
app = FastAPI(
    title="Analytic Agent API",
    description="Scalable analytic agent (stateless, DynamoDB conversation history)",
    version="2.0.0",
    lifespan=lifespan
)

# Initialize query coordinator
coordinator = QueryCoordinator()

@app.get("/api/health/audit-sqs")
async def audit_sqs_health():
    """Health check endpoint for SQS audit service."""
    if not ENABLE_SQS_AUDIT_LOGGING:
        return {
            "enabled": False,
            "message": "SQS audit logging is disabled"
        }
    
    audit_service = get_audit_sqs_service()
    health_info = audit_service.health_check()
    
    return {
        "enabled": True,
        "sqs_health": health_info
    }

@app.post("/api/analytics/query", response_model=Dict[str, Any])
async def receive_prompt(
    http_request: Request,
    response: Response,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
) -> Dict[str, Any]:
    """
    Process analytic query with authentication and session management.
    Custom validation error handling to return structured responses.
    Includes AWS SQS audit logging before sending response.
    """
    start_time = time.time()
    user_id = None
    username = None
    prompt = ""
    result = None
    
    try:
        # Extract user info from JWT for audit logging
        #if ENABLE_SQS_AUDIT_LOGGING:
        try:
                jwt_payload = validate_jwt_token(credentials)
                user_id = jwt_payload.get("sub")
                username = jwt_payload.get("userName") or jwt_payload.get("email", "").split("@")[0]
        except Exception as jwt_error:
                logger.warning(f"Failed to extract user info for audit: {jwt_error}")
        
        # Parse request body manually to catch validation errors
        request_data = await http_request.json()
        
        # Validate the request data
        request = PromptRequest(**request_data)
        prompt = request.prompt
        
        # Process the validated request
        result = await coordinator.process_query(request, http_request, response, credentials)
        
        # Send audit log before returning response
        #if ENABLE_SQS_AUDIT_LOGGING and user_id:
        processing_time_ms = int((time.time() - start_time) * 1000)
        audit_service = get_audit_sqs_service()
            
        audit_service.send_analytics_query_audit(
                statusCode=200,
                user_id=user_id,
                username=username or "unknown",
                prompt=prompt,
                success=result.get("success", False),
                processing_time_ms=processing_time_ms,
                error_message=result.get("message") if not result.get("success") else None
            )
        
        return result
        
    except ValidationError as e:
        # Extract the validation error message
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
        #if ENABLE_SQS_AUDIT_LOGGING and user_id:
        processing_time_ms = int((time.time() - start_time) * 1000)
        audit_service = get_audit_sqs_service()
        audit_service.send_analytics_query_audit(
                user_id=user_id,
                username=username or "unknown",
                prompt=prompt,
                success=False,
                processing_time_ms=processing_time_ms,
                error_message=error_msg
            )
        
        return {
            "success": False,
            "message": error_msg,
            "chart_image": None
        }
        
    except Exception as e:
        logger.exception(f"Unexpected error in API endpoint: {e}")
        
        # Send audit log for unexpected errors
        #if ENABLE_SQS_AUDIT_LOGGING and user_id:
        processing_time_ms = int((time.time() - start_time) * 1000)
        audit_service = get_audit_sqs_service()
        audit_service.send_analytics_query_audit(
                statusCode=500,
                user_id=user_id,
                username=username or "unknown", 
                prompt=prompt,
                success=False,
                processing_time_ms=processing_time_ms,
                error_message=str(e)
            )
        
        return {
            "success": False,
            "message": "An unexpected error occurred",
            "chart_image": None
        }