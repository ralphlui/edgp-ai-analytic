import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Depends, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import ValidationError

from app.security.auth import bearer_scheme, validate_jwt_token
from app.services.query_processor import QueryProcessor, PromptRequest
from app.services.audit_sqs_service import get_audit_sqs_service
from app.logging_config import setup_logging, get_logger
from app.config import (
    CORS_ORIGINS,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_HEADERS,
    CORS_MAX_AGE
)

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

logger.info(f"🔒 CORS:mode - Allowing origins: {CORS_ORIGINS}")
app.add_middleware(
            CORSMiddleware,
            allow_origins=CORS_ORIGINS,
            allow_credentials=CORS_ALLOW_CREDENTIALS,
            allow_methods=CORS_ALLOW_METHODS,
            allow_headers=CORS_ALLOW_HEADERS,
            max_age=CORS_MAX_AGE,
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


@app.delete("/api/analytics/conversation/clear")
async def clear_conversation_history(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
) -> Dict[str, Any]:
    """
    Clear conversation history (context) for the authenticated user.
    
    This endpoint removes all stored conversation context including:
    - report_type (success_rate/failure_rate)
    - domain_name and file_name
    - comparison_targets
    - prompt history
    
    Use this when:
    - User wants to start a fresh conversation
    - Context is incorrect or stale
    - User is switching to a different analysis topic
    
    Returns:
        Dict containing success status and message
        
    Example:
        DELETE /api/analytics/conversation/clear
        Headers: Authorization: Bearer <token>
        
        Response:
        {
            "success": true,
            "message": "Conversation history cleared successfully"
        }
    """
    try:
        # Validate JWT and extract user info
        jwt_payload = validate_jwt_token(credentials)
        user_id = jwt_payload.get("sub")
        username = jwt_payload.get("userName") or jwt_payload.get("email", "").split("@")[0]
        
        if not user_id:
            logger.warning("User ID not found in JWT token")
            return {
                "success": False,
                "message": "Invalid authentication token"
            }
        
        # Get query context service
        from app.services.query_context_service import QueryContextService
        context_service = QueryContextService()
        
        # Clear the conversation context
        success = context_service.clear_query_context(user_id)
        
        if success:
            logger.info(f"Conversation history cleared for user: {username} (ID: {user_id})")
            
            # Send audit log
            audit_service = get_audit_sqs_service()
            audit_service.send_analytics_query_audit(
                statusCode=200,
                user_id=user_id,
                username=username,
                prompt="[CLEAR_CONVERSATION_HISTORY]",
                success=True,
                message="Conversation history cleared"
            )
            
            return {
                "success": True,
                "message": "Conversation history cleared successfully"
            }
        else:
            logger.warning(f"Failed to clear conversation history for user: {username} (ID: {user_id})")
            return {
                "success": False,
                "message": "Failed to clear conversation history"
            }
    
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions (like 401 from validate_jwt_token)
        raise http_exc
        
    except Exception as e:
        logger.exception(f"Unexpected error clearing conversation history: {e}")
        
        # Try to send audit log
        try:
            user_id = jwt_payload.get("sub") if 'jwt_payload' in locals() else None
            username = jwt_payload.get("userName") if 'jwt_payload' in locals() else "unknown"
            
            audit_service = get_audit_sqs_service()
            audit_service.send_analytics_query_audit(
                statusCode=500,
                user_id=user_id or "unknown",
                username=username,
                prompt="[CLEAR_CONVERSATION_HISTORY]",
                success=False,
                message=str(e)
            )
        except:
            pass
        
        return {
            "success": False,
            "message": "An unexpected error occurred"
        }