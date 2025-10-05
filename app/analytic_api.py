"""
Clean FastAPI application with separated concerns.
Main.py only handles routing and basic app setup.
"""
import logging
import signal
import os
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Depends, Request, Response, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import ValidationError

from app.auth import bearer_scheme
from app.services.query_coordinator import QueryCoordinator, PromptRequest

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

@app.post("/api/analytics/query", response_model=Dict[str, Any])
async def receive_prompt(
    http_request: Request,
    response: Response,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
) -> Dict[str, Any]:
    """
    Process analytic query with authentication and session management.
    Custom validation error handling to return structured responses.
    """
    try:
        # Parse request body manually to catch validation errors
        request_data = await http_request.json()
        
        # Validate the request data
        request = PromptRequest(**request_data)
        
        # Process the validated request
        return await coordinator.process_query(request, http_request, response, credentials)
        
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
        return {
            "success": False,
            "message": error_msg,
            "chart_image": None
        }
        
    except Exception as e:
        logger.exception(f"Unexpected error in API endpoint: {e}")
        return {
            "success": False,
            "message": "An unexpected error occurred",
            "chart_image": None
        }