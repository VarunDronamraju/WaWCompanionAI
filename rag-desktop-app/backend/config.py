"""
Environment Configuration Management
Location: backend/config.py
Phase: 3 (Foundation for all backend modules)
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import field_validator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application settings
    app_name: str = "RAG Desktop API"
    app_version: str = "1.0.0"
    debug: bool = False
    api_prefix: str = "/api/v1"
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Database settings
    database_url: str = "postgresql://postgres:qwerty12345@localhost:5433/ragbot"
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "ragbot"
    postgres_user: str = "postgres"
    postgres_password: str = "qwerty12345"
    
    # Qdrant settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "documents"
    
    # Redis settings
    redis_url: str = "redis://:redispassword123@localhost:6379"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = "redispassword123"
    
    # Ollama settings
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma:2b"
    
    # OAuth settings (for future phases)
    google_client_id: str = "your_google_client_id_here"
    google_client_secret: str = "your_google_client_secret_here"
    google_redirect_uri: str = "http://localhost:8000/auth/google/callback"
    
    # JWT settings (for future phases)
    jwt_secret_key: str = "your_super_secret_jwt_key_change_in_production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 hours
    
    # ML model settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # TAVILY settings (for future phases)
    tavily_api_key: str = "your_tavily_api_key_here"
    
    # File upload settings
    max_file_size_mb: int = 50
    upload_dir: str = "uploads"
    allowed_extensions: list = [".pdf", ".docx", ".txt", ".md"]
    
    # CORS settings
    cors_origins: list = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ]
    
    @field_validator("upload_dir")
    @classmethod
    def create_upload_dir(cls, v):
        """Ensure upload directory exists"""
        if not os.path.exists(v):
            os.makedirs(v, exist_ok=True)
            logger.info(f"Created upload directory: {v}")
        return v
    
    @field_validator("max_file_size_mb")
    @classmethod
    def validate_file_size(cls, v):
        """Validate file size is reasonable"""
        if v <= 0 or v > 500:
            raise ValueError("File size must be between 1MB and 500MB")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


def get_settings() -> Settings:
    """Get application settings instance"""
    return Settings()


def validate_settings(settings: Settings) -> bool:
    """Validate all required settings are present and valid"""
    try:
        # Check database connection string
        if not settings.database_url:
            logger.error("Database URL not configured")
            return False
            
        # Check Qdrant URL
        if not settings.qdrant_url:
            logger.error("Qdrant URL not configured")
            return False
            
        # Check upload directory is writable
        if not os.access(settings.upload_dir, os.W_OK):
            logger.error(f"Upload directory not writable: {settings.upload_dir}")
            return False
            
        logger.info("All settings validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Settings validation failed: {e}")
        return False


# Global settings instance
settings = get_settings()

# Validate on import
if not validate_settings(settings):
    logger.warning("Some settings validation failed - check configuration")