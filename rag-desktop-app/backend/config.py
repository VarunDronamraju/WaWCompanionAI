"""
Enhanced Configuration for RAG Desktop Application
Phase 4: Added Chunking and Processing Settings

This module provides comprehensive configuration management with
Pydantic v2 settings for all application components.
"""

import os
import logging
from typing import Optional, List
from pathlib import Path

from pydantic import Field, field_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Setup logging
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    Enhanced for Phase 4 with chunking and processing configuration.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ========================================================================
    # Application Settings
    # ========================================================================
    
    app_name: str = Field(
        default="RAG Desktop Application",
        description="Application name"
    )
    
    app_version: str = Field(
        default="1.0.0-phase4",
        description="Application version"
    )
    
    debug: bool = Field(
        default=False,
        description="Debug mode flag"
    )
    
    environment: str = Field(
        default="development",
        description="Environment (development/staging/production)"
    )
    
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    # ========================================================================
    # Database Settings (PostgreSQL)
    # ========================================================================
    
    postgres_host: str = Field(
        default="localhost",
        description="PostgreSQL host"
    )
    
    postgres_port: int = Field(
        default=5432,
        ge=1,
        le=65535,
        description="PostgreSQL port"
    )
    
    postgres_db: str = Field(
        default="rag_desktop",
        description="PostgreSQL database name"
    )
    
    postgres_user: str = Field(
        default="rag_user",
        description="PostgreSQL username"
    )
    
    postgres_password: str = Field(
        default="rag_password",
        description="PostgreSQL password"
    )
    
    database_pool_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Database connection pool size"
    )
    
    database_pool_overflow: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Database connection pool overflow"
    )
    
    # ========================================================================
    # Qdrant Vector Store Settings
    # ========================================================================
    
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL"
    )
    
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API key (optional)"
    )
    
    qdrant_collection_name: str = Field(
        default="rag_documents",
        description="Default Qdrant collection name"
    )
    
    qdrant_vector_size: int = Field(
        default=384,
        ge=128,
        le=2048,
        description="Vector dimension size"
    )
    
    qdrant_distance_metric: str = Field(
        default="Cosine",
        description="Distance metric for similarity search"
    )
    
    # ========================================================================
    # OAuth Settings (Google)
    # ========================================================================
    
    google_client_id: str = Field(
        default="",
        description="Google OAuth client ID"
    )
    
    google_client_secret: str = Field(
        default="",
        description="Google OAuth client secret"
    )
    
    google_redirect_uri: str = Field(
        default="http://localhost:8000/auth/google/callback",
        description="Google OAuth redirect URI"
    )
    
    # ========================================================================
    # JWT Settings
    # ========================================================================
    
    jwt_secret_key: str = Field(
        default="your-super-secret-jwt-key-change-in-production",
        min_length=32,
        description="JWT secret key"
    )
    
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )
    
    jwt_expire_minutes: int = Field(
        default=60 * 24,  # 24 hours
        ge=15,
        le=60 * 24 * 7,  # Max 7 days
        description="JWT token expiration in minutes"
    )
    
    jwt_refresh_expire_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="JWT refresh token expiration in days"
    )
    
    # ========================================================================
    # ML Model Settings
    # ========================================================================
    
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    
    embedding_model_cache_dir: str = Field(
        default="./models/sentence-transformers",
        description="Cache directory for embedding models"
    )
    
    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Batch size for embedding generation"
    )
    
    embedding_max_length: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Maximum token length for embeddings"
    )
    
    # ========================================================================
    # Ollama LLM Settings
    # ========================================================================
    
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
    )
    
    ollama_model: str = Field(
        default="gemma:1b-it-q8_0",
        description="Ollama model for text generation"
    )
    
    ollama_timeout: int = Field(
        default=120,
        ge=30,
        le=600,
        description="Ollama request timeout in seconds"
    )
    
    ollama_max_tokens: int = Field(
        default=2048,
        ge=256,
        le=8192,
        description="Maximum tokens for generation"
    )
    
    ollama_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Generation temperature"
    )
    
    # ========================================================================
    # TAVILY Web Search Settings
    # ========================================================================
    
    tavily_api_key: str = Field(
        default="",
        description="TAVILY API key for web search fallback"
    )
    
    tavily_max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum web search results"
    )
    
    tavily_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Web search timeout in seconds"
    )
    
    # ========================================================================
    # Document Processing Settings (New for Phase 4)
    # ========================================================================
    
    # File Upload Settings
    max_file_size_mb: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum file size in megabytes"
    )
    
    upload_dir: str = Field(
        default="./uploads",
        description="Directory for uploaded files"
    )
    
    allowed_extensions: set = Field(
        default={'.pdf', '.docx', '.txt', '.md', '.rtf'},
        description="Allowed file extensions for upload"
    )
    
    temp_dir: str = Field(
        default="./temp",
        description="Temporary processing directory"
    )
    
    # Chunking Settings
    chunk_strategy: str = Field(
        default="adaptive",
        description="Default chunking strategy (adaptive/semantic/paragraph/fixed)"
    )
    
    min_chunk_size: int = Field(
        default=500,
        ge=100,
        le=1000,
        description="Minimum chunk size in characters"
    )
    
    max_chunk_size: int = Field(
        default=1200,
        ge=800,
        le=3000,
        description="Maximum chunk size in characters"
    )
    
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        le=500,
        description="Overlap between chunks in characters"
    )
    
    preserve_sentence_boundaries: bool = Field(
        default=True,
        description="Preserve sentence boundaries during chunking"
    )
    
    preserve_paragraph_boundaries: bool = Field(
        default=True,
        description="Preserve paragraph boundaries during chunking"
    )
    
    # Processing Settings
    max_concurrent_processes: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum concurrent document processing tasks"
    )
    
    processing_timeout_minutes: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Document processing timeout in minutes"
    )
    
    enable_background_processing: bool = Field(
        default=True,
        description="Enable background document processing"
    )
    
    # Quality Control Settings
    min_meaningful_content_ratio: float = Field(
        default=0.3,
        ge=0.1,
        le=1.0,
        description="Minimum ratio of meaningful content in documents"
    )
    
    max_empty_chunks_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Maximum ratio of empty/low-quality chunks allowed"
    )
    
    enable_content_validation: bool = Field(
        default=True,
        description="Enable content quality validation"
    )
    
    # ========================================================================
    # Search and Retrieval Settings
    # ========================================================================
    
    default_search_limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Default search result limit"
    )
    
    max_search_limit: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Maximum search result limit"
    )
    
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold for search results"
    )
    
    enable_query_expansion: bool = Field(
        default=True,
        description="Enable automatic query expansion"
    )
    
    enable_semantic_reranking: bool = Field(
        default=True,
        description="Enable semantic reranking of search results"
    )
    
    # ========================================================================
    # Cache Settings
    # ========================================================================
    
    enable_caching: bool = Field(
        default=True,
        description="Enable application caching"
    )
    
    cache_ttl_seconds: int = Field(
        default=3600,  # 1 hour
        ge=300,  # 5 minutes
        le=86400,  # 24 hours
        description="Cache time-to-live in seconds"
    )
    
    max_cache_size_mb: int = Field(
        default=500,
        ge=50,
        le=5000,
        description="Maximum cache size in megabytes"
    )
    
    cache_embedding_results: bool = Field(
        default=True,
        description="Cache embedding computation results"
    )
    
    cache_search_results: bool = Field(
        default=True,
        description="Cache search results"
    )
    
    # ========================================================================
    # Performance Settings
    # ========================================================================
    
    max_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Maximum worker threads for async operations"
    )
    
    request_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="API request timeout in seconds"
    )
    
    enable_request_compression: bool = Field(
        default=True,
        description="Enable gzip compression for API responses"
    )
    
    max_request_size_mb: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum request size in megabytes"
    )
    
    # ========================================================================
    # Security Settings
    # ========================================================================
    
    enable_cors: bool = Field(
        default=True,
        description="Enable CORS middleware"
    )
    
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )
    
    cors_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed CORS methods"
    )
    
    enable_rate_limiting: bool = Field(
        default=True,
        description="Enable API rate limiting"
    )
    
    rate_limit_requests_per_minute: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Rate limit: requests per minute per user"
    )
    
    enable_request_logging: bool = Field(
        default=True,
        description="Enable detailed request logging"
    )
    
    log_sensitive_data: bool = Field(
        default=False,
        description="Include sensitive data in logs (not recommended for production)"
    )
    
    # ========================================================================
    # Monitoring and Health Check Settings
    # ========================================================================
    
    enable_health_checks: bool = Field(
        default=True,
        description="Enable health check endpoints"
    )
    
    health_check_interval_seconds: int = Field(
        default=60,
        ge=10,
        le=3600,
        description="Health check interval in seconds"
    )
    
    enable_metrics_collection: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    
    metrics_retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Metrics retention period in days"
    )
    
    # ========================================================================
    # Computed Properties
    # ========================================================================
    
    @computed_field
    @property
    def database_url(self) -> str:
        """Construct PostgreSQL database URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @computed_field
    @property
    def async_database_url(self) -> str:
        """Construct async PostgreSQL database URL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @computed_field
    @property
    def max_file_size_bytes(self) -> int:
        """Convert max file size to bytes."""
        return self.max_file_size_mb * 1024 * 1024
    
    @computed_field
    @property
    def processing_timeout_seconds(self) -> int:
        """Convert processing timeout to seconds."""
        return self.processing_timeout_minutes * 60
    
    @computed_field
    @property
    def cache_size_bytes(self) -> int:
        """Convert cache size to bytes."""
        return self.max_cache_size_mb * 1024 * 1024
    
    @computed_field
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    @computed_field
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"
    
    # ========================================================================
    # Validation Methods
    # ========================================================================
    
    @field_validator('chunk_strategy')
    @classmethod
    def validate_chunk_strategy(cls, v: str) -> str:
        """Validate chunking strategy."""
        valid_strategies = {'adaptive', 'semantic', 'paragraph', 'fixed', 'auto'}
        if v.lower() not in valid_strategies:
            raise ValueError(f'chunk_strategy must be one of: {valid_strategies}')
        return v.lower()
    
    @field_validator('max_chunk_size')
    @classmethod
    def validate_max_chunk_size(cls, v: int, info) -> int:
        """Ensure max_chunk_size is greater than min_chunk_size."""
        if hasattr(info, 'data') and 'min_chunk_size' in info.data:
            min_size = info.data['min_chunk_size']
            if v <= min_size:
                raise ValueError('max_chunk_size must be greater than min_chunk_size')
        return v
    
    @field_validator('chunk_overlap')
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        """Ensure chunk_overlap is reasonable."""
        if hasattr(info, 'data') and 'min_chunk_size' in info.data:
            min_size = info.data['min_chunk_size']
            if v >= min_size:
                raise ValueError('chunk_overlap must be less than min_chunk_size')
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f'log_level must be one of: {valid_levels}')
        return v_upper
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        valid_envs = {'development', 'staging', 'production', 'testing'}
        v_lower = v.lower()
        if v_lower not in valid_envs:
            raise ValueError(f'environment must be one of: {valid_envs}')
        return v_lower
    
    @field_validator('qdrant_distance_metric')
    @classmethod
    def validate_distance_metric(cls, v: str) -> str:
        """Validate Qdrant distance metric."""
        valid_metrics = {'Cosine', 'Euclidean', 'Dot'}
        if v not in valid_metrics:
            raise ValueError(f'qdrant_distance_metric must be one of: {valid_metrics}')
        return v
    
    @field_validator('upload_dir', 'temp_dir', 'embedding_model_cache_dir')
    @classmethod
    def validate_directories(cls, v: str) -> str:
        """Validate and create directories if they don't exist."""
        path = Path(v)
        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ensured: {path.absolute()}")
        except Exception as e:
            logger.warning(f"Could not create directory {path}: {e}")
        return str(path)
    
    @field_validator('cors_origins')
    @classmethod
    def validate_cors_origins(cls, v: List[str]) -> List[str]:
        """Validate CORS origins format."""
        import re
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)', re.IGNORECASE)
        
        valid_origins = []
        for origin in v:
            if origin == "*" or url_pattern.match(origin):
                valid_origins.append(origin)
            else:
                logger.warning(f"Invalid CORS origin skipped: {origin}")
        
        return valid_origins
    
    # ========================================================================
    # Configuration Validation
    # ========================================================================
    
    def validate_complete_config(self) -> bool:
        """
        Validate complete configuration for production readiness.
        
        Returns:
            True if configuration is valid for production
        """
        issues = []
        
        # Check critical security settings
        if self.is_production:
            if self.jwt_secret_key == "your-super-secret-jwt-key-change-in-production":
                issues.append("JWT secret key must be changed for production")
            
            if not self.google_client_id or not self.google_client_secret:
                issues.append("Google OAuth credentials required for production")
            
            if self.debug:
                issues.append("Debug mode should be disabled in production")
            
            if self.log_sensitive_data:
                issues.append("Sensitive data logging should be disabled in production")
        
        # Check required directories
        required_dirs = [self.upload_dir, self.temp_dir, self.embedding_model_cache_dir]
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                issues.append(f"Required directory does not exist: {dir_path}")
        
        # Check model availability
        if not self.embedding_model:
            issues.append("Embedding model must be specified")
        
        # Check database settings
        if not all([self.postgres_host, self.postgres_user, self.postgres_password, self.postgres_db]):
            issues.append("Incomplete database configuration")
        
        # Log issues
        if issues:
            logger.error("Configuration validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def get_chunking_config(self) -> dict:
        """Get chunking configuration as dictionary."""
        return {
            'strategy': self.chunk_strategy,
            'min_size': self.min_chunk_size,
            'max_size': self.max_chunk_size,
            'overlap': self.chunk_overlap,
            'preserve_sentences': self.preserve_sentence_boundaries,
            'preserve_paragraphs': self.preserve_paragraph_boundaries
        }
    
    def get_embedding_config(self) -> dict:
        """Get embedding configuration as dictionary."""
        return {
            'model': self.embedding_model,
            'cache_dir': self.embedding_model_cache_dir,
            'batch_size': self.embedding_batch_size,
            'max_length': self.embedding_max_length,
            'vector_size': self.qdrant_vector_size
        }
    
    def get_search_config(self) -> dict:
        """Get search configuration as dictionary."""
        return {
            'default_limit': self.default_search_limit,
            'max_limit': self.max_search_limit,
            'similarity_threshold': self.similarity_threshold,
            'enable_query_expansion': self.enable_query_expansion,
            'enable_reranking': self.enable_semantic_reranking
        }


# ============================================================================
# Global Settings Instance
# ============================================================================

_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """
    Get global settings instance (singleton pattern).
    
    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
        
        # Validate configuration on first load
        if not _settings.validate_complete_config():
            logger.warning("Configuration validation failed - some features may not work correctly")
    
    return _settings

def reload_settings() -> Settings:
    """
    Reload settings from environment (useful for testing).
    
    Returns:
        New settings instance
    """
    global _settings
    _settings = None
    return get_settings()

def validate_settings(settings: Settings) -> bool:
    """
    Validate settings configuration.
    
    Args:
        settings: Settings instance to validate
        
    Returns:
        True if settings are valid
    """
    return settings.validate_complete_config()


# ============================================================================
# Configuration Utilities
# ============================================================================

def setup_logging(settings: Settings):
    """
    Setup application logging based on settings.
    
    Args:
        settings: Application settings
    """
    import logging.config
    
    log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': settings.log_level,
                'formatter': 'detailed' if settings.debug else 'simple',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'filename': 'rag_desktop.log',
                'mode': 'a'
            }
        },
        'root': {
            'level': settings.log_level,
            'handlers': ['console']
        },
        'loggers': {
            'rag_desktop': {
                'level': settings.log_level,
                'handlers': ['console', 'file'] if not settings.is_development else ['console'],
                'propagate': False
            }
        }
    }
    
    logging.config.dictConfig(log_config)
    logger.info(f"Logging configured - Level: {settings.log_level}")

def get_environment_info() -> dict:
    """
    Get current environment information.
    
    Returns:
        Dictionary with environment details
    """
    settings = get_settings()
    
    return {
        'app_name': settings.app_name,
        'app_version': settings.app_version,
        'environment': settings.environment,
        'debug_mode': settings.debug,
        'is_production': settings.is_production,
        'python_version': os.sys.version,
        'platform': os.name
    }


# ============================================================================
# Export key components
# ============================================================================

__all__ = [
    'Settings',
    'get_settings',
    'reload_settings', 
    'validate_settings',
    'setup_logging',
    'get_environment_info'
]