#!/usr/bin/env python3
"""
Main entry point for the Analytic Agent API.
Handles server startup with environment-based configuration.
"""
import uvicorn
from config import APP_PORT

if __name__ == "__main__":
    print(f"Starting Analytic Agent API on port {APP_PORT}")
    uvicorn.run(
        "app.analytic_api:app",
        host="0.0.0.0",
        port=APP_PORT,
        reload=True,
        log_level="info"
    )