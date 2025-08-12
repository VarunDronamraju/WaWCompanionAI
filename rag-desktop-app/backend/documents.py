"""
Document Processing Pipeline for RAG Desktop Application
Phase 4: Advanced Chunking Integration

This module handles document processing, chunking, and preparation for embedding.
Integrates with the existing Phase 3 infrastructure seamlessly.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import uuid
import json
from enum import Enum

from pydantic import BaseModel, Field
from fastapi import HTTPException, UploadFile

from .config import get_settings
from .schemas import DocumentResponse, DocumentChunk, ChunkingStrategy
from .utils import (
    extract_text_from_pdf,
    extract_text_from_docx, 
    extract_text_from_txt,
    extract_text_from_markdown,
    detect_file_type,
    validate_file_size,
    clean_text,
    chunk_text_with_strategy,
    get_optimal_chunking_strategy,
    assess_chunk_quality,
    generate_secure_id
)

# Setup logging
logger = logging.getLogger(__name__)

# Constants
PROCESSING_STATUS = {
    'PENDING': 'pending',
    'PROCESSING': 'processing', 
    'CHUNKING': 'chunking',
    'COMPLETED': 'completed',
    'FAILED': 'failed'
}

SUPPORTED_FILE_TYPES = {
    'pdf': extract_text_from_pdf,
    'docx': extract_text_from_docx,
    'txt': extract_text_from_txt,
    'md': extract_text_from_markdown
}

# ============================================================================
# Core Processing Classes
# ============================================================================

class DocumentProcessor:
    """
    Main document processing class that handles the complete pipeline
    from file upload to chunked content ready for embedding.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize document processor.
        
        Args:
            storage_path: Base path for document storage
        """
        self.settings = get_settings()
        self.storage_path = Path(storage_path or "uploads")
        self.storage_path.mkdir(exist_ok=True)
        self.chunking_engine = ChunkingEngine()
        
        logger.info(f"DocumentProcessor initialized with storage: {self.storage_path}")
    
    async def process_document(
        self, 
        file_path: str, 
        user_id: str,
        chunking_strategy: Optional[str] = None,
        chunk_params: Optional[Dict[str, Any]] = None
    ) -> DocumentResponse:
        """
        Complete document processing pipeline.
        
        Args:
            file_path: Path to uploaded file
            user_id: User identifier
            chunking_strategy: Override default chunking strategy
            chunk_params: Custom chunking parameters
            
        Returns:
            Document response with processing results
            
        Raises:
            HTTPException: If processing fails
        """
        file_path = Path(file_path)
        doc_id = generate_secure_id()
        
        try:
            logger.info(f"Starting document processing: {file_path} for user {user_id}")
            
            # Validate file
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="File not found")
            
            if not validate_file_size(str(file_path)):
                raise HTTPException(status_code=413, detail="File too large")
            
            # Detect file type
            file_type = detect_file_type(str(file_path))
            if file_type == 'unknown':
                raise HTTPException(status_code=415, detail="Unsupported file type")
            
            # Extract text
            text_content = await self._extract_text_async(file_path, file_type)
            if not text_content or len(text_content.strip()) < 50:
                raise HTTPException(status_code=422, detail="No meaningful content extracted")
            
            # Update processing status
            await self._update_processing_status(doc_id, PROCESSING_STATUS['CHUNKING'])
            
            # Process chunks
            chunks = await self._process_chunks(
                text_content, 
                doc_id, 
                file_type,
                chunking_strategy,
                chunk_params
            )
            
            # Create document response
            document_response = DocumentResponse(
                id=doc_id,
                title=file_path.name,
                file_type=file_type,
                upload_time=datetime.utcnow(),
                chunk_count=len(chunks),
                processing_status=PROCESSING_STATUS['COMPLETED'],
                file_size=file_path.stat().st_size,
                metadata={
                    'original_filename': file_path.name,
                    'user_id': user_id,
                    'text_length': len(text_content),
                    'chunking_strategy': chunking_strategy or 'auto',
                    'quality_metrics': assess_chunk_quality([c.chunk_text for c in chunks])
                }
            )
            
            logger.info(f"Document processing completed: {doc_id} with {len(chunks)} chunks")
            return document_response
            
        except HTTPException:
            await self._update_processing_status(doc_id, PROCESSING_STATUS['FAILED'])
            raise
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            await self._update_processing_status(doc_id, PROCESSING_STATUS['FAILED'])
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    async def reprocess_document(
        self, 
        doc_id: str,
        chunking_strategy: str,
        chunk_params: Optional[Dict[str, Any]] = None
    ) -> DocumentResponse:
        """
        Reprocess existing document with new chunking strategy.
        
        Args:
            doc_id: Document identifier
            chunking_strategy: New chunking strategy
            chunk_params: Custom chunking parameters
            
        Returns:
            Updated document response
        """
        try:
            logger.info(f"Reprocessing document {doc_id} with strategy {chunking_strategy}")
            
            # In a real implementation, you'd retrieve the original text from storage
            # For now, we'll simulate the reprocessing
            await self._update_processing_status(doc_id, PROCESSING_STATUS['PROCESSING'])
            
            # This would normally retrieve stored text and metadata
            # Then reprocess with new strategy
            
            await self._update_processing_status(doc_id, PROCESSING_STATUS['COMPLETED'])
            
            # Return updated document response
            # This is a placeholder - real implementation would reconstruct from storage
            return DocumentResponse(
                id=doc_id,
                title="Reprocessed Document",
                file_type="pdf",
                upload_time=datetime.utcnow(),
                chunk_count=0,
                processing_status=PROCESSING_STATUS['COMPLETED']
            )
            
        except Exception as e:
            logger.error(f"Document reprocessing failed: {e}")
            await self._update_processing_status(doc_id, PROCESSING_STATUS['FAILED'])
            raise HTTPException(status_code=500, detail=f"Reprocessing failed: {str(e)}")
    
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete document and all associated data.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if deletion successful
        """
        try:
            logger.info(f"Deleting document: {doc_id}")
            
            # In real implementation, this would:
            # 1. Delete file from storage
            # 2. Remove database records
            # 3. Delete vector embeddings
            # 4. Clean up any cached data
            
            return True
            
        except Exception as e:
            logger.error(f"Document deletion failed: {e}")
            return False
    
    async def _extract_text_async(self, file_path: Path, file_type: str) -> str:
        """
        Extract text from file asynchronously.
        
        Args:
            file_path: Path to file
            file_type: Type of file
            
        Returns:
            Extracted text content
        """
        extraction_func = SUPPORTED_FILE_TYPES.get(file_type)
        if not extraction_func:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Run extraction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, extraction_func, str(file_path))
        
        return clean_text(text)
    
    async def _process_chunks(
        self,
        text_content: str,
        doc_id: str,
        file_type: str,
        chunking_strategy: Optional[str] = None,
        chunk_params: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Process text into chunks using specified or optimal strategy.
        
        Args:
            text_content: Text to chunk
            doc_id: Document identifier
            file_type: Type of source file
            chunking_strategy: Chunking strategy to use
            chunk_params: Custom parameters
            
        Returns:
            List of document chunks
        """
        # Determine chunking strategy
        if not chunking_strategy:
            chunking_strategy = get_optimal_chunking_strategy(text_content, file_type)
        
        # Get default parameters
        default_params = {
            'min_size': self.settings.min_chunk_size,
            'max_size': self.settings.max_chunk_size,
            'overlap': self.settings.chunk_overlap
        }
        
        # Merge with custom parameters
        if chunk_params:
            default_params.update(chunk_params)
        
        # Perform chunking
        chunks = await self.chunking_engine.chunk_document(
            text_content,
            chunking_strategy,
            default_params
        )
        
        # Create DocumentChunk objects
        document_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk = DocumentChunk(
                id=generate_secure_id(),
                document_id=doc_id,
                chunk_text=chunk_text,
                chunk_index=i,
                metadata={
                    'strategy': chunking_strategy,
                    'length': len(chunk_text),
                    'word_count': len(chunk_text.split()),
                    'created_at': datetime.utcnow().isoformat()
                }
            )
            document_chunks.append(chunk)
        
        logger.info(f"Created {len(document_chunks)} chunks for document {doc_id}")
        return document_chunks
    
    async def _update_processing_status(self, doc_id: str, status: str):
        """
        Update document processing status.
        
        Args:
            doc_id: Document identifier
            status: New processing status
        """
        logger.info(f"Document {doc_id} status: {status}")
        # In real implementation, this would update database record


class ChunkingEngine:
    """
    Advanced chunking engine with multiple strategies and optimization.
    """
    
    def __init__(self):
        """Initialize chunking engine with default settings."""
        self.settings = get_settings()
        logger.info("ChunkingEngine initialized")
    
    async def chunk_document(
        self,
        text: str,
        strategy: str = "adaptive",
        params: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Chunk document using specified strategy.
        
        Args:
            text: Text to chunk
            strategy: Chunking strategy
            params: Strategy parameters
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        params = params or {}
        
        logger.info(f"Chunking text ({len(text)} chars) with strategy: {strategy}")
        
        try:
            # Run chunking in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(
                None,
                chunk_text_with_strategy,
                text,
                strategy,
                **params
            )
            
            # Validate and optimize chunks
            chunks = await self._optimize_chunks(chunks)
            
            logger.info(f"Chunking completed: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            # Fallback to simple chunking
            return await self._fallback_chunking(text)
    
    def get_optimal_chunk_size(self, text: str) -> int:
        """
        Determine optimal chunk size based on text characteristics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Recommended chunk size
        """
        text_length = len(text)
        
        # Dynamic sizing based on content length
        if text_length < 1000:
            return min(text_length, 500)
        elif text_length < 5000:
            return 800
        elif text_length < 20000:
            return 1000
        else:
            return 1200
    
    async def _optimize_chunks(self, chunks: List[str]) -> List[str]:
        """
        Optimize chunks for quality and coherence.
        
        Args:
            chunks: Raw chunks
            
        Returns:
            Optimized chunks
        """
        if not chunks:
            return []
        
        optimized = []
        min_size = self.settings.min_chunk_size
        
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            
            # Ensure minimum size
            if len(chunk) < min_size:
                # Try to merge with previous chunk if possible
                if optimized and len(optimized[-1]) + len(chunk) < self.settings.max_chunk_size:
                    optimized[-1] = optimized[-1] + " " + chunk
                else:
                    optimized.append(chunk)
            else:
                optimized.append(chunk)
        
        return optimized
    
    async def _fallback_chunking(self, text: str) -> List[str]:
        """
        Fallback chunking method for error cases.
        
        Args:
            text: Text to chunk
            
        Returns:
            Simple chunks
        """
        logger.warning("Using fallback chunking method")
        
        chunk_size = self.settings.max_chunk_size
        chunks = []
        
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks


# ============================================================================
# Document Management Functions
# ============================================================================

async def upload_and_process(
    file: UploadFile, 
    user_id: str,
    storage_path: Optional[str] = None,
    chunking_strategy: Optional[str] = None
) -> DocumentResponse:
    """
    Handle file upload and processing pipeline.
    
    Args:
        file: Uploaded file
        user_id: User identifier
        storage_path: Custom storage path
        chunking_strategy: Override chunking strategy
        
    Returns:
        Document processing result
    """
    # Create unique filename
    file_id = generate_secure_id()
    file_extension = Path(file.filename).suffix.lower()
    safe_filename = f"{file_id}{file_extension}"
    
    # Setup storage
    storage_dir = Path(storage_path or "uploads")
    storage_dir.mkdir(exist_ok=True)
    file_path = storage_dir / safe_filename
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File saved: {file_path}")
        
        # Process document
        processor = DocumentProcessor(str(storage_dir))
        result = await processor.process_document(
            str(file_path),
            user_id,
            chunking_strategy
        )
        
        return result
        
    except Exception as e:
        # Clean up file on error
        if file_path.exists():
            file_path.unlink()
        raise


async def extract_document_text(file_path: str) -> str:
    """
    Extract text from document file.
    
    Args:
        file_path: Path to document
        
    Returns:
        Extracted text
    """
    file_type = detect_file_type(file_path)
    extraction_func = SUPPORTED_FILE_TYPES.get(file_type)
    
    if not extraction_func:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    # Run extraction asynchronously
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, extraction_func, file_path)
    
    return clean_text(text)


async def chunk_and_embed_document(
    text: str, 
    doc_id: str,
    strategy: str = "adaptive"
) -> List[DocumentChunk]:
    """
    Chunk document text for embedding.
    
    Args:
        text: Document text
        doc_id: Document identifier
        strategy: Chunking strategy
        
    Returns:
        List of document chunks
    """
    chunking_engine = ChunkingEngine()
    chunks = await chunking_engine.chunk_document(text, strategy)
    
    # Convert to DocumentChunk objects
    document_chunks = []
    for i, chunk_text in enumerate(chunks):
        chunk = DocumentChunk(
            id=generate_secure_id(),
            document_id=doc_id,
            chunk_text=chunk_text,
            chunk_index=i,
            metadata={
                'strategy': strategy,
                'length': len(chunk_text),
                'created_at': datetime.utcnow().isoformat()
            }
        )
        document_chunks.append(chunk)
    
    return document_chunks


async def store_document_metadata(
    doc_data: Dict[str, Any], 
    user_id: str
) -> str:
    """
    Store document metadata in database.
    
    Args:
        doc_data: Document metadata
        user_id: User identifier
        
    Returns:
        Document ID
    """
    doc_id = generate_secure_id()
    
    # In real implementation, this would store in database
    logger.info(f"Storing metadata for document {doc_id}")
    
    return doc_id


async def update_processing_status(doc_id: str, status: str):
    """
    Update document processing status.
    
    Args:
        doc_id: Document identifier
        status: New status
    """
    logger.info(f"Document {doc_id} status updated to: {status}")
    # In real implementation, this would update database


# ============================================================================
# Document Query Functions  
# ============================================================================

async def get_user_documents(
    user_id: str, 
    skip: int = 0, 
    limit: int = 10
) -> List[DocumentResponse]:
    """
    Get user's documents with pagination.
    
    Args:
        user_id: User identifier
        skip: Number of documents to skip
        limit: Maximum documents to return
        
    Returns:
        List of user documents
    """
    # Placeholder implementation
    logger.info(f"Fetching documents for user {user_id}, skip={skip}, limit={limit}")
    return []


async def get_document_by_id(
    doc_id: str, 
    user_id: str
) -> Optional[DocumentResponse]:
    """
    Get specific document by ID.
    
    Args:
        doc_id: Document identifier
        user_id: User identifier
        
    Returns:
        Document if found, None otherwise
    """
    logger.info(f"Fetching document {doc_id} for user {user_id}")
    return None


async def get_document_chunks_paginated(
    doc_id: str, 
    page: int = 1, 
    size: int = 10
) -> List[DocumentChunk]:
    """
    Get document chunks with pagination.
    
    Args:
        doc_id: Document identifier
        page: Page number
        size: Page size
        
    Returns:
        List of document chunks
    """
    logger.info(f"Fetching chunks for document {doc_id}, page={page}, size={size}")
    return []


async def delete_document_and_embeddings(doc_id: str) -> bool:
    """
    Delete document and all associated data.
    
    Args:
        doc_id: Document identifier
        
    Returns:
        True if successful
    """
    try:
        logger.info(f"Deleting document {doc_id} and embeddings")
        
        # In real implementation, this would:
        # 1. Delete from database
        # 2. Remove file from storage
        # 3. Delete vector embeddings
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete document {doc_id}: {e}")
        return False


# ============================================================================
# File Format Handlers
# ============================================================================

def handle_pdf_upload(file_path: str) -> str:
    """Handle PDF file upload and text extraction."""
    return extract_text_from_pdf(file_path)


def handle_docx_upload(file_path: str) -> str:
    """Handle DOCX file upload and text extraction."""
    return extract_text_from_docx(file_path)


def handle_txt_upload(file_path: str) -> str:
    """Handle TXT file upload and text extraction."""
    return extract_text_from_txt(file_path)


def handle_markdown_upload(file_path: str) -> str:
    """Handle Markdown file upload and text extraction."""
    return extract_text_from_markdown(file_path)


# ============================================================================
# Chunk Processing Functions
# ============================================================================

def create_chunk_metadata(
    doc_id: str, 
    chunk_index: int, 
    source_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create comprehensive metadata for document chunk.
    
    Args:
        doc_id: Document identifier
        chunk_index: Index of chunk in document
        source_info: Source document information
        
    Returns:
        Chunk metadata dictionary
    """
    return {
        'document_id': doc_id,
        'chunk_index': chunk_index,
        'created_at': datetime.utcnow().isoformat(),
        'source_file': source_info.get('filename', ''),
        'file_type': source_info.get('file_type', ''),
        'processing_version': '1.0'
    }


def validate_chunk_quality(chunk_text: str) -> bool:
    """
    Validate chunk quality based on content analysis.
    
    Args:
        chunk_text: Text chunk to validate
        
    Returns:
        True if chunk meets quality standards
    """
    if not chunk_text or len(chunk_text.strip()) < 20:
        return False
    
    # Check for meaningful content
    words = chunk_text.split()
    if len(words) < 5:
        return False
    
    # Check for complete sentences
    sentence_endings = ['.', '!', '?']
    has_complete_sentence = any(ending in chunk_text for ending in sentence_endings)
    
    return has_complete_sentence


def merge_overlapping_chunks(chunks: List[str]) -> List[str]:
    """
    Merge chunks that have significant overlap.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        List of merged chunks
    """
    if len(chunks) <= 1:
        return chunks
    
    merged = [chunks[0]]
    
    for chunk in chunks[1:]:
        # Simple overlap detection - can be enhanced
        last_chunk = merged[-1]
        
        # Check if there's significant overlap
        overlap_threshold = 0.3  # 30% overlap
        if _calculate_text_overlap(last_chunk, chunk) > overlap_threshold:
            # Merge chunks
            merged[-1] = last_chunk + " " + chunk
        else:
            merged.append(chunk)
    
    return merged


def _calculate_text_overlap(text1: str, text2: str) -> float:
    """
    Calculate overlap percentage between two text chunks.
    
    Args:
        text1: First text chunk
        text2: Second text chunk
        
    Returns:
        Overlap percentage (0.0 to 1.0)
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


# ============================================================================
# Async Context Managers
# ============================================================================

class DocumentProcessingContext:
    """
    Context manager for document processing operations.
    """
    
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.start_time = None
    
    async def __aenter__(self):
        self.start_time = datetime.utcnow()
        await update_processing_status(self.doc_id, PROCESSING_STATUS['PROCESSING'])
        logger.info(f"Started processing document {self.doc_id}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        if exc_type is None:
            await update_processing_status(self.doc_id, PROCESSING_STATUS['COMPLETED'])
            logger.info(f"Completed processing document {self.doc_id} in {duration:.2f}s")
        else:
            await update_processing_status(self.doc_id, PROCESSING_STATUS['FAILED'])
            logger.error(f"Failed processing document {self.doc_id} after {duration:.2f}s: {exc_val}")
        
        return False  # Don't suppress exceptions