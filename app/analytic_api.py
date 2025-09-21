"""
Clean FastAPI application with separated concerns.
Main.py only handles routing and basic app setup.
"""
import logging
from typing import Dict, Any

from fastapi import FastAPI, Depends, Request, Response
from fastapi.security import HTTPAuthorizationCredentials

from app.auth import bearer_scheme
from app.services.query_coordinator import QueryCoordinator, PromptRequest

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analytic_agent")

# Create FastAPI app
app = FastAPI(
    title="Analytic Agent API",
    description="Scalable analytic agent with session management",
    version="2.0.0"
)

# Initialize query coordinator
coordinator = QueryCoordinator()

@app.post("/query", response_model=Dict[str, Any])
async def receive_prompt(
    request: PromptRequest,
    http_request: Request,
    response: Response,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
) -> Dict[str, Any]:
    """
    Process analytic query with authentication and session management.
    """
    return await coordinator.process_query(request, http_request, response, credentials)

