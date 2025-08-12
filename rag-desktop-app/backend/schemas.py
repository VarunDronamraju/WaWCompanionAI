"""
Pydantic Data Models for RAG Desktop Application
Phase 4: Enhanced with Chunking Support

This module defines all request/response models with Pydantic v2 compatibility
and comprehensive validation for document chunking functionality.
"""

from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings

# ============================================================================
# Base Configuration
# ============================================================================

class BaseModelConfig(BaseModel):
    """Base model with common configuration."""
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True
    )


# ============================================================================
# Enums for Type Safety
# ============================================================================

class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    ADAPTIVE = "adaptive"
    SEMANTIC = "semantic"
    PARAGRAPH = "paragraph"
    FIXED = "fixed"
    AUTO = "auto"


class ProcessingStatus(str, Enum):
    """Document processing status options."""
    PENDING = "pending"
    PROCESSING = "processing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    COMPLETED = "completed"
    FAILED = "failed"


class FileType(str, Enum):
    """Supported file types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "md"
    RTF = "rtf"


# ============================================================================
# User Schemas
# ============================================================================

class UserBase(BaseModelConfig):
    """Base user model."""
    email: str = Field(..., description="User email address")
    name: str = Field(..., min_length=1, max_length=100, description="User full name")

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate email format."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, v):
            raise ValueError('Invalid email format')
        return v.lower()


class UserCreate(UserBase):
    """User creation model."""
    google_id: str = Field(..., description="Google OAuth identifier")


class UserResponse(UserBase):
    """User response model."""
    id: str = Field(..., description="User unique identifier")
    created_at: datetime = Field(..., description="Account creation timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")


# ============================================================================
# Document Schemas
# ============================================================================

class DocumentBase(BaseModelConfig):
    """Base document model."""
    title: str = Field(..., min_length=1, max_length=255, description="Document title")
    file_type: FileType = Field(..., description="Document file type")

    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Sanitize document title."""
        # Remove invalid filename characters
        import re
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', v)
        return sanitized.strip()


class DocumentCreate(DocumentBase):
    """Document creation model."""
    file_path: str = Field(..., description="Path to uploaded file")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    user_id: str = Field(..., description="Owner user ID")

    @field_validator('file_size')
    @classmethod
    def validate_file_size(cls, v: int) -> int:
        """Validate file size limits."""
        max_size = 50 * 1024 * 1024  # 50MB
        if v > max_size:
            raise ValueError(f'File size exceeds maximum limit of {max_size} bytes')
        return v


class DocumentResponse(DocumentBase):
    """Document response model."""
    id: str = Field(..., description="Document unique identifier")
    upload_time: datetime = Field(..., description="Upload timestamp")
    chunk_count: int = Field(0, ge=0, description="Number of text chunks")
    processing_status: ProcessingStatus = Field(..., description="Processing status")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate chunk metadata."""
        if v is None:
            return {}
        return v


class ChunkCreate(BaseModelConfig):
    """Chunk creation model."""
    document_id: str = Field(..., description="Parent document ID")
    chunk_text: str = Field(..., min_length=10, description="Chunk text content")
    chunk_index: int = Field(..., ge=0, description="Chunk order index")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChunkUpdate(BaseModelConfig):
    """Chunk update model."""
    chunk_text: Optional[str] = Field(None, min_length=10)
    metadata: Optional[Dict[str, Any]] = Field(None)


class DocumentChunk(BaseModelConfig):
    """Document chunk model."""
    id: str = Field(..., description="Chunk unique identifier")
    document_id: str = Field(..., description="Parent document ID")
    chunk_text: str = Field(..., min_length=1, description="Chunk text content")
    chunk_index: int = Field(..., ge=0, description="Chunk order index")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Chunk metadata")

    @field_validator('chunk_text')
    @classmethod
    def validate_chunk_text(cls, v: str) -> str:
        """Validate chunk text content."""
        if len(v.strip()) < 10:
            raise ValueError('Chunk text must be at least 10 characters')
        return v.strip()

    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate chunk metadata."""
        if v is not None and not isinstance(v, dict):
            raise ValueError('Metadata must be a dictionary')
        return v


class ChunkResponse(DocumentChunk):
    """Enhanced chunk response with additional computed fields."""
    word_count: Optional[int] = Field(None, description="Number of words in chunk")
    character_count: Optional[int] = Field(None, description="Number of characters")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")

    def __init__(self, **data):
        super().__init__(**data)
        # Compute derived fields
        if self.chunk_text:
            self.word_count = len(self.chunk_text.split())
            self.character_count = len(self.chunk_text)


# ============================================================================
# Chunking Configuration Schemas
# ============================================================================

class ChunkingConfig(BaseModelConfig):
    """Configuration for chunking operations."""
    strategy: ChunkingStrategy = Field(ChunkingStrategy.ADAPTIVE, description="Chunking strategy")
    min_size: int = Field(500, ge=100, le=1000, description="Minimum chunk size")
    max_size: int = Field(1200, ge=800, le=3000, description="Maximum chunk size")
    overlap: int = Field(100, ge=0, le=500, description="Overlap between chunks")
    preserve_sentences: bool = Field(True, description="Preserve sentence boundaries")
    preserve_paragraphs: bool = Field(True, description="Preserve paragraph boundaries")

    @field_validator('max_size')
    @classmethod
    def validate_max_size(cls, v: int, info) -> int:
        """Ensure max_size is greater than min_size."""
        if hasattr(info, 'data') and 'min_size' in info.data:
            min_size = info.data['min_size']
            if v <= min_size:
                raise ValueError('max_size must be greater than min_size')
        return v

    @field_validator('overlap')
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is reasonable compared to chunk sizes."""
        if hasattr(info, 'data') and 'min_size' in info.data:
            min_size = info.data['min_size']
            if v >= min_size:
                raise ValueError('overlap must be less than min_size')
        return v


class ChunkingRequest(BaseModelConfig):
    """Request model for chunking operations."""
    document_id: str = Field(..., description="Document ID to chunk")
    config: Optional[ChunkingConfig] = Field(None, description="Chunking configuration")
    force_rechunk: bool = Field(False, description="Force rechunking if already processed")


class ChunkingResponse(BaseModelConfig):
    """Response model for chunking operations."""
    document_id: str = Field(..., description="Processed document ID")
    chunk_count: int = Field(..., ge=0, description="Number of chunks created")
    strategy_used: ChunkingStrategy = Field(..., description="Chunking strategy applied")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    quality_metrics: Optional[Dict[str, float]] = Field(None, description="Quality assessment metrics")


# ============================================================================
# Query and Search Schemas
# ============================================================================

class QueryRequest(BaseModelConfig):
    """RAG query request model."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    max_results: int = Field(5, ge=1, le=20, description="Maximum results to return")
    include_chunks: bool = Field(True, description="Include chunk details in response")
    document_ids: Optional[List[str]] = Field(None, description="Limit search to specific documents")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate and clean query text."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError('Query cannot be empty')
        return cleaned


class SearchRequest(BaseModelConfig):
    """Semantic search request model."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    document_ids: Optional[List[str]] = Field(None, description="Specific documents to search")
    limit: int = Field(10, ge=1, le=50, description="Maximum results")
    offset: int = Field(0, ge=0, description="Result offset for pagination")
    include_metadata: bool = Field(True, description="Include chunk metadata")

    @field_validator('document_ids')
    @classmethod
    def validate_document_ids(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate document ID format."""
        if v is None:
            return None
        
        # Basic UUID format validation
        import re
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        
        if isinstance(v, list):
            for doc_id in v:
                if not re.match(uuid_pattern, doc_id, re.IGNORECASE):
                    raise ValueError(f'Invalid document ID format: {doc_id}')
        
        return v





# ============================================================================
# Document Chunk Schemas
# ============================================================================


class ChunkSearchRequest(BaseModelConfig):
    """Chunk-level search request model."""
    query: str = Field(..., min_length=1, description="Search query")
    document_id: str = Field(..., description="Document to search within")
    limit: int = Field(5, ge=1, le=20, description="Maximum chunks to return")
    min_score: float = Field(0.5, ge=0.0, le=1.0, description="Minimum relevance score")


class SearchResult(BaseModelConfig):
    """Individual search result model."""
    chunk_id: str = Field(..., description="Chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    chunk_text: str = Field(..., description="Matching chunk text")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    chunk_index: int = Field(..., ge=0, description="Chunk position in document")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Chunk metadata")
    highlights: Optional[List[str]] = Field(None, description="Query highlights in text")


class SearchResponse(BaseModelConfig):
    """Search results response model."""
    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., ge=0, description="Total matching results")
    results: List[SearchResult] = Field(..., description="Search results")
    processing_time: float = Field(..., ge=0, description="Search processing time")
    used_fallback: bool = Field(False, description="Whether web fallback was used")


# ============================================================================
# Chat and Conversation Schemas
# ============================================================================

class ChatMessage(BaseModelConfig):
    """Chat message model."""
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., min_length=1, description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Message metadata")

    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate message role."""
        valid_roles = {'user', 'assistant', 'system'}
        if v not in valid_roles:
            raise ValueError(f'Role must be one of: {valid_roles}')
        return v


class ChatRequest(BaseModelConfig):
    """Chat request model."""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    session_id: Optional[str] = Field(None, description="Chat session ID")
    stream: bool = Field(True, description="Enable response streaming")
    context_length: int = Field(2000, ge=500, le=4000, description="Context window size")

    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate and clean message content."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError('Message cannot be empty')
        return cleaned


class RagChatRequest(ChatRequest):
    """RAG-enhanced chat request model."""
    use_fallback: bool = Field(True, description="Enable web search fallback")
    max_context_length: int = Field(2000, ge=1000, le=4000, description="Maximum context length")
    document_filter: Optional[List[str]] = Field(None, description="Filter by document IDs")
    include_sources: bool = Field(True, description="Include source references")


class ChatSession(BaseModelConfig):
    """Chat session model."""
    id: str = Field(..., description="Session unique identifier")
    user_id: str = Field(..., description="Session owner ID")
    created_at: datetime = Field(..., description="Session creation time")
    updated_at: datetime = Field(..., description="Last update time")
    messages: List[ChatMessage] = Field(default_factory=list, description="Session messages")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Session metadata")

    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v: List[ChatMessage]) -> List[ChatMessage]:
        """Validate message order and content."""
        if len(v) > 100:  # Reasonable limit for UI performance
            raise ValueError('Too many messages in session')
        return v


class ChatResponse(BaseModelConfig):
    """Chat response model."""
    message: str = Field(..., description="Assistant response")
    session_id: str = Field(..., description="Chat session ID")
    sources: Optional[List[SearchResult]] = Field(None, description="Source chunks used")
    processing_time: float = Field(..., ge=0, description="Response generation time")
    token_count: Optional[int] = Field(None, description="Response token count")
    used_fallback: bool = Field(False, description="Whether web fallback was used")


# ============================================================================
# Authentication Schemas
# ============================================================================

class TokenResponse(BaseModelConfig):
    """Token response model."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., gt=0, description="Token expiration in seconds")


class GoogleAuthRequest(BaseModelConfig):
    """Google OAuth request model."""
    code: str = Field(..., description="OAuth authorization code")
    state: Optional[str] = Field(None, description="OAuth state parameter")

    @field_validator('code')
    @classmethod
    def validate_code(cls, v: str) -> str:
        """Validate OAuth code format."""
        if not v or len(v) < 10:
            raise ValueError('Invalid OAuth code')
        return v


class LoginRequest(BaseModelConfig):
    """Login request model."""
    email: str = Field(..., description="User email")
    password: str = Field(..., min_length=8, description="User password")

    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v


# ============================================================================
# System and Configuration Schemas
# ============================================================================

class SystemSettings(BaseModelConfig):
    """System configuration model."""
    embedding_model: Optional[str] = Field(None, description="Embedding model name")
    chunk_size: Optional[int] = Field(None, ge=100, le=2000, description="Default chunk size")
    chunk_overlap: Optional[int] = Field(None, ge=0, le=500, description="Default chunk overlap")
    max_retrieval_results: Optional[int] = Field(None, ge=1, le=50, description="Max retrieval results")
    enable_web_fallback: Optional[bool] = Field(None, description="Enable web search fallback")


class HealthResponse(BaseModelConfig):
    """Health check response model."""
    status: str = Field(..., description="Overall system status")
    database: str = Field(..., description="Database connection status")
    qdrant: str = Field(..., description="Qdrant vector store status")
    ollama: str = Field(..., description="Ollama LLM service status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: Optional[str] = Field(None, description="Application version")
    uptime: Optional[float] = Field(None, description="System uptime in seconds")


class UsageStats(BaseModelConfig):
    """Usage statistics model."""
    user_id: str = Field(..., description="User identifier")
    document_count: int = Field(..., ge=0, description="Total documents uploaded")
    chunk_count: int = Field(..., ge=0, description="Total chunks processed")
    query_count: int = Field(..., ge=0, description="Total queries made")
    storage_used: int = Field(..., ge=0, description="Storage used in bytes")
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")


# ============================================================================
# File Upload Schemas
# ============================================================================

class FileUploadResponse(BaseModelConfig):
    """File upload response model."""
    file_id: str = Field(..., description="Uploaded file identifier")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    file_type: FileType = Field(..., description="Detected file type")
    upload_time: datetime = Field(..., description="Upload timestamp")
    processing_status: ProcessingStatus = Field(..., description="Initial processing status")


class UploadConfig(BaseModelConfig):
    """Upload configuration model."""
    max_file_size: int = Field(50 * 1024 * 1024, description="Maximum file size in bytes")
    allowed_types: List[FileType] = Field(
        default_factory=lambda: [FileType.PDF, FileType.DOCX, FileType.TXT, FileType.MARKDOWN],
        description="Allowed file types"
    )
    auto_process: bool = Field(True, description="Automatically process after upload")
    chunking_strategy: ChunkingStrategy = Field(ChunkingStrategy.AUTO, description="Default chunking strategy")


# ============================================================================
# Pagination Schemas
# ============================================================================

class PaginationParams(BaseModelConfig):
    """Pagination parameters model."""
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(10, ge=1, le=100, description="Page size")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: Optional[str] = Field("desc", description="Sort order (asc/desc)")

    @field_validator('sort_order')
    @classmethod
    def validate_sort_order(cls, v: Optional[str]) -> Optional[str]:
        """Validate sort order."""
        if v is not None and v not in ['asc', 'desc']:
            raise ValueError('sort_order must be "asc" or "desc"')
        return v


class PaginatedResponse(BaseModelConfig):
    """Generic paginated response model."""
    items: List[Any] = Field(..., description="Response items")
    total: int = Field(..., ge=0, description="Total items available")
    page: int = Field(..., ge=1, description="Current page")
    size: int = Field(..., ge=1, description="Page size")
    pages: int = Field(..., ge=1, description="Total pages")
    has_next: bool = Field(..., description="Has next page")
    has_prev: bool = Field(..., description="Has previous page")


# ============================================================================
# Error Response Schemas
# ============================================================================

class ErrorDetail(BaseModelConfig):
    """Error detail model."""
    type: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    field: Optional[str] = Field(None, description="Field that caused error")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseModelConfig):
    """Error response model."""
    error: str = Field(..., description="Error summary")
    details: Optional[List[ErrorDetail]] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")


# ============================================================================
# Export Models for Easy Import
# ============================================================================

__all__ = [
    # Enums
    'ChunkingStrategy',
    'ProcessingStatus',
    'FileType',
    
    # User models
    'UserBase',
    'UserCreate', 
    'UserResponse',
    
    # Document models
    'DocumentBase',
    'DocumentCreate',
    'DocumentResponse',
    'DocumentUpdate',
    
    # Chunk models
    'DocumentChunk',
    'ChunkCreate',
    'ChunkUpdate',
    'ChunkResponse',
    
    # Chunking models
    'ChunkingConfig',
    'ChunkingRequest',
    'ChunkingResponse',
    
    # Query models
    'QueryRequest',
    'SearchRequest',
    'ChunkSearchRequest',
    'SearchResult',
    'SearchResponse',
    
    # Chat models
    'ChatMessage',
    'ChatRequest',
    'RagChatRequest',
    'ChatSession',
    'ChatResponse',
    
    # Auth models
    'TokenResponse',
    'GoogleAuthRequest',
    'LoginRequest',
    
    # System models
    'SystemSettings',
    'HealthResponse',
    'UsageStats',
    
    # File models
    'FileUploadResponse',
    'UploadConfig',
    
    # Utility models
    'PaginationParams',
    'PaginatedResponse',
    'ErrorDetail',
    'ErrorResponse'
]


class DocumentUpdate(BaseModelConfig):
    """Document update model."""
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    metadata: Optional[Dict[str, Any]] = Field(None)
