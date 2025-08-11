"""
FastAPI Application Entry Point
Location: backend/main.py
Phase: 3 (FastAPI app initialization, middleware, CORS setup)
"""

import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn

from .config import get_settings, Settings
from .api_routes import router
from .schemas import ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application state
app_state = {
    "startup_time": None,
    "request_count": 0,
    "health_checks": 0
}


class AppConfig:
    """Application configuration management"""
    
    def __init__(self):
        self.settings = get_settings()
        self.startup_time = None
    
    def get_app_info(self) -> Dict[str, Any]:
        """Get application information"""
        return {
            "name": self.settings.app_name,
            "version": self.settings.app_version,
            "debug": self.settings.debug,
            "startup_time": self.startup_time,
            "uptime_seconds": time.time() - self.startup_time if self.startup_time else None
        }


# Global config instance
config = AppConfig()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting RAG Desktop API...")
    config.startup_time = time.time()
    app_state["startup_time"] = config.startup_time
    
    # Verify upload directory exists
    import os
    if not os.path.exists(config.settings.upload_dir):
        os.makedirs(config.settings.upload_dir, exist_ok=True)
        logger.info(f"Created upload directory: {config.settings.upload_dir}")
    
    # Log configuration
    logger.info(f"Upload directory: {config.settings.upload_dir}")
    logger.info(f"Max file size: {config.settings.max_file_size_mb}MB")
    logger.info(f"Allowed extensions: {config.settings.allowed_extensions}")
    logger.info(f"Database URL: {config.settings.postgres_host}:{config.settings.postgres_port}")
    logger.info(f"Qdrant URL: {config.settings.qdrant_url}")
    logger.info(f"Ollama URL: {config.settings.ollama_url}")
    
    logger.info("RAG Desktop API startup complete!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Desktop API...")
    logger.info(f"Total requests handled: {app_state.get('request_count', 0)}")
    logger.info("RAG Desktop API shutdown complete!")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="RAG Desktop Application Backend API",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        debug=settings.debug
    )
    
    # Setup middleware
    setup_middleware(app, settings)
    
    # Setup CORS
    setup_cors(app, settings)
    
    # Setup exception handlers
    setup_exception_handlers(app)
    
    # Include routers
    app.include_router(router, prefix="/api/v1")
    
    return app


def setup_middleware(app: FastAPI, settings: Settings):
    """Setup application middleware"""
    
    # Request tracking middleware
    @app.middleware("http")
    async def track_requests(request: Request, call_next):
        start_time = time.time()
        app_state["request_count"] += 1
        
        # Process request
        response = await call_next(request)
        
        # Log request details
        process_time = time.time() - start_time
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        
        # Add response headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = f"req_{app_state['request_count']}"
        
        return response
    
    # Security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0", "*"]
    )


def setup_cors(app: FastAPI, settings: Settings):
    """Setup CORS middleware"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time", "X-Request-ID"]
    )


def setup_exception_handlers(app: FastAPI):
    """Setup global exception handlers"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error="http_error",
                message=exc.detail,
                details={"status_code": exc.status_code}
            ).dict()
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors"""
        logger.warning(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="validation_error",
                message="Request validation failed",
                details={"errors": exc.errors()}
            ).dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions"""
        logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_server_error",
                message="An unexpected error occurred",
                details={"type": type(exc).__name__} if get_settings().debug else None
            ).dict()
        )


def get_app_info() -> Dict[str, Any]:
    """Get application information"""
    return config.get_app_info()


# Startup and shutdown events (legacy support)
async def startup_event():
    """Application startup event"""
    logger.info("Legacy startup event triggered")


async def shutdown_event():
    """Application shutdown event"""
    logger.info("Legacy shutdown event triggered")


# Create app instance
app = create_app()

# Add legacy event handlers for compatibility
app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)


# Additional endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Desktop API",
        "version": get_settings().app_version,
        "docs": "/docs",
        "health": "/api/v1/health",
        "upload": "/api/v1/upload"
    }


@app.get("/info")
async def app_info():
    """Get application information and statistics"""
    info = get_app_info()
    info.update({
        "total_requests": app_state.get("request_count", 0),
        "health_checks": app_state.get("health_checks", 0),
        "status": "running"
    })
    return info


if __name__ == "__main__":
    # Run application directly
    settings = get_settings()
    
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Listening on {settings.host}:{settings.port}")
    
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers,
        log_level="info" if not settings.debug else "debug",
        access_log=True
    )