"""
Pydantic Data Models for Request/Response Validation
Location: backend/schemas.py
Phase: 3 (Request/Response models for API endpoints)
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class ProcessingStatus(str, Enum):
    """Document processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileType(str, Enum):
    """Supported file types"""
    PDF = "pdf"
    DOCX = "docx" 
    TXT = "txt"
    MARKDOWN = "md"


# Base model configuration
class BaseModel(BaseModel):
    """Base model with common configuration"""
    
    class Config:
        from_attributes = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Document schemas
class DocumentBase(BaseModel):
    """Base document schema"""
    title: str = Field(..., min_length=1, max_length=255, description="Document title")
    file_type: FileType = Field(..., description="File type")


class DocumentCreate(DocumentBase):
    """Schema for creating a new document"""
    file_path: str = Field(..., description="Path to uploaded file")
    size: int = Field(..., gt=0, description="File size in bytes")
    original_filename: str = Field(..., description="Original filename")
    
    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v):
        """Validate file path is not empty"""
        if not v or not v.strip():
            raise ValueError("File path cannot be empty")
        return v.strip()


class DocumentResponse(DocumentBase):
    """Schema for document response"""
    id: str = Field(..., description="Unique document identifier")
    upload_time: datetime = Field(..., description="Upload timestamp")
    chunk_count: int = Field(default=0, ge=0, description="Number of text chunks")
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    size: int = Field(..., gt=0, description="File size in bytes")
    original_filename: str = Field(..., description="Original filename")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")


class DocumentList(BaseModel):
    """Schema for paginated document list"""
    documents: List[DocumentResponse] = Field(default=[], description="List of documents")
    total: int = Field(default=0, ge=0, description="Total number of documents")
    page: int = Field(default=1, ge=1, description="Current page number")
    size: int = Field(default=10, ge=1, le=100, description="Page size")
    has_next: bool = Field(default=False, description="Whether there are more pages")


class DocumentChunk(BaseModel):
    """Schema for document text chunks"""
    id: str = Field(..., description="Chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    chunk_text: str = Field(..., min_length=1, description="Text content")
    chunk_index: int = Field(..., ge=0, description="Chunk position in document")
    chunk_metadata: Dict[str, Any] = Field(default={}, description="Additional chunk metadata")
    created_at: datetime = Field(..., description="Creation timestamp")


# Upload schemas
class UploadResponse(BaseModel):
    """Schema for file upload response"""
    document_id: str = Field(..., description="Created document ID")
    filename: str = Field(..., description="Uploaded filename")
    size: int = Field(..., gt=0, description="File size in bytes")
    file_type: FileType = Field(..., description="Detected file type")
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    message: str = Field(default="File uploaded successfully")


# Error schemas
class ErrorResponse(BaseModel):
    """Schema for error responses"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ValidationError(BaseModel):
    """Schema for validation errors"""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    value: Optional[Any] = Field(None, description="Invalid value")


# Health check schemas
class ServiceHealth(BaseModel):
    """Schema for individual service health"""
    status: str = Field(..., description="Service status (healthy/unhealthy)")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")
    error: Optional[str] = Field(None, description="Error message if unhealthy")


class HealthResponse(BaseModel):
    """Schema for comprehensive health check"""
    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, ServiceHealth] = Field(..., description="Individual service health")
    version: str = Field(..., description="API version")
    uptime_seconds: Optional[float] = Field(None, description="System uptime in seconds")


# API Info schemas
class APIInfo(BaseModel):
    """Schema for API information"""
    name: str = Field(..., description="API name")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="API description")
    docs_url: str = Field(..., description="Documentation URL")
    health_url: str = Field(..., description="Health check URL")
    supported_formats: List[str] = Field(..., description="Supported file formats")
    max_file_size_mb: int = Field(..., description="Maximum file size in MB")


# Query and search schemas (for future phases)
class QueryRequest(BaseModel):
    """Schema for RAG query requests"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum number of results")
    include_chunks: bool = Field(default=True, description="Include text chunks in response")
    
    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        """Validate query is not just whitespace"""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty or just whitespace")
        return v.strip()


class SearchRequest(BaseModel):
    """Schema for semantic search requests"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    document_ids: Optional[List[str]] = Field(None, description="Specific document IDs to search")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of results")
    
    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(cls, v):
        """Validate document IDs if provided"""
        if v is not None and len(v) == 0:
            raise ValueError("Document IDs list cannot be empty if provided")
        return v


# Pagination schemas
class PaginationParams(BaseModel):
    """Schema for pagination parameters"""
    page: int = Field(default=1, ge=1, description="Page number")
    size: int = Field(default=10, ge=1, le=100, description="Page size")
    
    @property
    def offset(self) -> int:
        """Calculate offset for database queries"""
        return (self.page - 1) * self.size


# System schemas
class SystemStats(BaseModel):
    """Schema for system statistics"""
    total_documents: int = Field(default=0, ge=0)
    total_chunks: int = Field(default=0, ge=0)
    storage_used_mb: float = Field(default=0.0, ge=0.0)
    avg_processing_time_seconds: Optional[float] = Field(None, ge=0.0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)