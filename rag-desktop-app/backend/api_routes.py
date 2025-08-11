"""
REST API Endpoints for Document Management
Location: backend/api_routes.py
Phase: 3 (Core API endpoints with upload functionality)
"""

import os
import time
import asyncio
from typing import List, Optional
from pathlib import Path
import logging
import aiofiles
import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse

from .config import get_settings, Settings
from .schemas import (
    DocumentResponse, DocumentCreate, DocumentList, UploadResponse,
    ErrorResponse, HealthResponse, ServiceHealth, APIInfo,
    ProcessingStatus, FileType, PaginationParams
)
from .utils import (
    detect_file_type, validate_file_size, sanitize_filename,
    generate_secure_id, extract_text_from_file, get_utc_now,
    FileProcessingError, UnsupportedFileTypeError
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# In-memory storage for Phase 3 (will be replaced with database in Phase 10)
documents_store: dict = {}
processing_queue: list = []


# Dependencies
def get_settings_dependency() -> Settings:
    """Get settings dependency"""
    return get_settings()


async def verify_services_health() -> dict:
    """Verify all external services are healthy"""
    settings = get_settings()
    services = {}
    
    # Check PostgreSQL
    try:
        start_time = time.time()
        # Simple connection test - will be enhanced in Phase 10
        import psycopg2
        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password
        )
        conn.close()
        response_time = (time.time() - start_time) * 1000
        services['postgresql'] = ServiceHealth(
            status='healthy',
            response_time_ms=response_time
        )
    except Exception as e:
        services['postgresql'] = ServiceHealth(
            status='unhealthy',
            error=str(e)
        )
    
    # Check Qdrant
    try:
        start_time = time.time()
        client = QdrantClient(url=settings.qdrant_url)
        collections = client.get_collections()
        response_time = (time.time() - start_time) * 1000
        services['qdrant'] = ServiceHealth(
            status='healthy',
            response_time_ms=response_time
        )
    except Exception as e:
        services['qdrant'] = ServiceHealth(
            status='unhealthy',
            error=str(e)
        )
    
    # Check Redis
    try:
        start_time = time.time()
        import redis
        r = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            decode_responses=True
        )
        r.ping()
        response_time = (time.time() - start_time) * 1000
        services['redis'] = ServiceHealth(
            status='healthy',
            response_time_ms=response_time
        )
    except Exception as e:
        services['redis'] = ServiceHealth(
            status='unhealthy',
            error=str(e)
        )
    
    # Check Ollama
    try:
        start_time = time.time()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.ollama_url}/api/tags")
            response.raise_for_status()
        response_time = (time.time() - start_time) * 1000
        services['ollama'] = ServiceHealth(
            status='healthy',
            response_time_ms=response_time
        )
    except Exception as e:
        services['ollama'] = ServiceHealth(
            status='unhealthy',
            error=str(e)
        )
    
    return services


async def process_document_background(doc_id: str, file_path: str, settings: Settings):
    """Background task to process uploaded document"""
    try:
        logger.info(f"Starting background processing for document {doc_id}")
        
        # Update status to processing
        if doc_id in documents_store:
            documents_store[doc_id]['processing_status'] = ProcessingStatus.PROCESSING
        
        # Extract text from file
        text_content = extract_text_from_file(file_path)
        
        # Simulate processing time (remove in later phases)
        await asyncio.sleep(2)
        
        # Update document with extracted content
        if doc_id in documents_store:
            documents_store[doc_id].update({
                'processing_status': ProcessingStatus.COMPLETED,
                'text_content': text_content,
                'chunk_count': len(text_content.split()) // 100  # Rough chunk estimate
            })
        
        logger.info(f"Successfully processed document {doc_id}")
        
    except Exception as e:
        logger.error(f"Error processing document {doc_id}: {e}")
        if doc_id in documents_store:
            documents_store[doc_id].update({
                'processing_status': ProcessingStatus.FAILED,
                'error_message': str(e)
            })


# API Endpoints

@router.get("/", response_model=APIInfo)
async def get_api_info(settings: Settings = Depends(get_settings_dependency)):
    """Get API information and capabilities"""
    return APIInfo(
        name=settings.app_name,
        version=settings.app_version,
        description="RAG Desktop Application API for document upload and processing",
        docs_url="/docs",
        health_url="/health",
        supported_formats=["pdf", "docx", "txt", "md"],
        max_file_size_mb=settings.max_file_size_mb
    )


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings_dependency)):
    """Comprehensive health check for all services"""
    try:
        services = await verify_services_health()
        
        # Determine overall status
        unhealthy_services = [name for name, health in services.items() if health.status != 'healthy']
        overall_status = "healthy" if not unhealthy_services else "degraded"
        
        if len(unhealthy_services) == len(services):
            overall_status = "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            services=services,
            version=settings.app_version
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            services={},
            version=settings.app_version
        )


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings_dependency)
):
    """
    Upload and process a document
    
    Supports: PDF, DOCX, TXT, Markdown files
    Max size: 50MB (configurable)
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Generate secure document ID
        doc_id = generate_secure_id()
        
        # Sanitize filename
        safe_filename = sanitize_filename(file.filename)
        
        # Create upload directory if it doesn't exist
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(exist_ok=True)
        
        # Create unique file path
        file_extension = Path(safe_filename).suffix
        stored_filename = f"{doc_id}{file_extension}"
        file_path = upload_dir / stored_filename
        
        # Save uploaded file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Validate file size
        try:
            validate_file_size(str(file_path), settings.max_file_size_mb)
        except FileProcessingError as e:
            # Clean up uploaded file
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=413, detail=str(e))
        
        # Detect file type
        try:
            file_type = detect_file_type(str(file_path))
        except UnsupportedFileTypeError as e:
            # Clean up uploaded file
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=415, detail=str(e))
        
        # Get file size
        file_size = file_path.stat().st_size
        
        # Create document record
        document_data = {
            'id': doc_id,
            'title': Path(safe_filename).stem,
            'file_type': file_type,
            'file_path': str(file_path),
            'size': file_size,
            'original_filename': file.filename,
            'upload_time': get_utc_now(),
            'processing_status': ProcessingStatus.PENDING,
            'chunk_count': 0,
            'error_message': None
        }
        
        # Store document (in-memory for Phase 3)
        documents_store[doc_id] = document_data
        
        # Start background processing
        background_tasks.add_task(
            process_document_background, 
            doc_id, 
            str(file_path), 
            settings
        )
        
        logger.info(f"Successfully uploaded document {doc_id}: {safe_filename}")
        
        return UploadResponse(
            document_id=doc_id,
            filename=safe_filename,
            size=file_size,
            file_type=FileType(file_type),
            processing_status=ProcessingStatus.PENDING,
            message="File uploaded successfully and processing started"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/documents", response_model=DocumentList)
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Page size"),
    settings: Settings = Depends(get_settings_dependency)
):
    """
    List uploaded documents with pagination
    """
    try:
        # Get all documents
        all_docs = list(documents_store.values())
        total = len(all_docs)
        
        # Sort by upload time (newest first)
        all_docs.sort(key=lambda x: x['upload_time'], reverse=True)
        
        # Paginate
        offset = (page - 1) * size
        paginated_docs = all_docs[offset:offset + size]
        
        # Convert to response models
        documents = []
        for doc_data in paginated_docs:
            doc_response = DocumentResponse(
                id=doc_data['id'],
                title=doc_data['title'],
                file_type=FileType(doc_data['file_type']),
                upload_time=doc_data['upload_time'],
                chunk_count=doc_data['chunk_count'],
                processing_status=ProcessingStatus(doc_data['processing_status']),
                size=doc_data['size'],
                original_filename=doc_data['original_filename'],
                error_message=doc_data.get('error_message')
            )
            documents.append(doc_response)
        
        has_next = offset + size < total
        
        return DocumentList(
            documents=documents,
            total=total,
            page=page,
            size=size,
            has_next=has_next
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")


@router.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: str,
    settings: Settings = Depends(get_settings_dependency)
):
    """
    Get specific document by ID
    """
    try:
        if doc_id not in documents_store:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc_data = documents_store[doc_id]
        
        return DocumentResponse(
            id=doc_data['id'],
            title=doc_data['title'],
            file_type=FileType(doc_data['file_type']),
            upload_time=doc_data['upload_time'],
            chunk_count=doc_data['chunk_count'],
            processing_status=ProcessingStatus(doc_data['processing_status']),
            size=doc_data['size'],
            original_filename=doc_data['original_filename'],
            error_message=doc_data.get('error_message')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document")


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    settings: Settings = Depends(get_settings_dependency)
):
    """
    Delete document and associated files
    """
    try:
        if doc_id not in documents_store:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc_data = documents_store[doc_id]
        
        # Delete physical file
        file_path = Path(doc_data['file_path'])
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted file: {file_path}")
        
        # Remove from store
        del documents_store[doc_id]
        
        logger.info(f"Successfully deleted document {doc_id}")
        
        return {"message": "Document deleted successfully", "document_id": doc_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")


@router.post("/documents/{doc_id}/reprocess")
async def reprocess_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings_dependency)
):
    """
    Reprocess a document (re-extract text and generate embeddings)
    """
    try:
        if doc_id not in documents_store:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc_data = documents_store[doc_id]
        
        # Check if file still exists
        file_path = Path(doc_data['file_path'])
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Document file not found")
        
        # Reset processing status
        documents_store[doc_id]['processing_status'] = ProcessingStatus.PENDING
        documents_store[doc_id]['error_message'] = None
        
        # Start background processing
        background_tasks.add_task(
            process_document_background,
            doc_id,
            str(file_path),
            settings
        )
        
        logger.info(f"Started reprocessing for document {doc_id}")
        
        return {
            "message": "Document reprocessing started",
            "document_id": doc_id,
            "status": ProcessingStatus.PENDING
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to reprocess document")


@router.get("/documents/{doc_id}/metadata")
async def get_document_metadata(
    doc_id: str,
    settings: Settings = Depends(get_settings_dependency)
):
    """
    Get document metadata and processing information
    """
    try:
        if doc_id not in documents_store:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc_data = documents_store[doc_id]
        
        # Get file stats
        file_path = Path(doc_data['file_path'])
        file_exists = file_path.exists()
        
        metadata = {
            "document_id": doc_id,
            "title": doc_data['title'],
            "original_filename": doc_data['original_filename'],
            "file_type": doc_data['file_type'],
            "file_size_bytes": doc_data['size'],
            "file_size_mb": round(doc_data['size'] / (1024 * 1024), 2),
            "upload_time": doc_data['upload_time'],
            "processing_status": doc_data['processing_status'],
            "chunk_count": doc_data['chunk_count'],
            "file_exists": file_exists,
            "error_message": doc_data.get('error_message'),
            "text_length": len(doc_data.get('text_content', '')),
        }
        
        return metadata
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metadata for document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get document metadata")


# Note: Exception handlers should be registered on the FastAPI app, not the router
# These will be moved to main.py