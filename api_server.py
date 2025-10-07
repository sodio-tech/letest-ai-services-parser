"""
FastAPI Server for AI CSV Parser Agent
Provides REST API endpoints for CSV processing operations
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from api.routes import router
from api.models import ErrorResponse
from src.utils.config import ConfigManager
from src.utils.env_manager import EnvManager
from src.utils.logger import LoggerSetup

# Initialize configuration and logging
config_manager = ConfigManager()
env_manager = EnvManager()
logger_setup = LoggerSetup(config_manager, env_manager)
logger = logger_setup.get_logger("api_server")

# Create FastAPI app
app = FastAPI(
    title="AI CSV Parser Agent API",
    description="Intelligent CSV to JSON conversion using GPT-4o-mini",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure this properly for production
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["csv-parser"])

# Create uploads directory
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            details={"path": str(request.url)}
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            details={"path": str(request.url)}
        ).dict()
    )


@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("AI CSV Parser Agent API starting up...")
    
    # Check OpenAI configuration
    if not env_manager.is_openai_configured():
        logger.error("OpenAI API key not configured. Please set OPENAI_API_KEY in .env file")
        raise RuntimeError("OpenAI API key not configured")
    
    logger.info("API server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("AI CSV Parser Agent API shutting down...")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI CSV Parser Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get("/api/v1/")
async def api_root():
    """API root endpoint"""
    return {
        "message": "AI CSV Parser Agent API v1",
        "endpoints": {
            "health": "/api/v1/health",
            "upload": "/api/v1/upload",
            "process": "/api/v1/process",
            "process_csv": "/api/v1/process-csv",
            "files": "/api/v1/files",
            "config": "/api/v1/config"
        },
        "documentation": "/docs"
    }


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    return app


if __name__ == "__main__":
    # Get configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    logger.info(f"Starting API server on {host}:{port}")
    
    # Run the server
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
