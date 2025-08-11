"""
File Processing and Utility Functions
Location: backend/utils.py
Phase: 3 (File handling, text extraction, validation)
"""

import os
import uuid
import hashlib
import mimetypes
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import logging

# File processing imports
import PyPDF2
import docx
from docx import Document

# Try to import magic, but handle case where it's not available on Windows
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("python-magic not available - using file extension detection only")

logger = logging.getLogger(__name__)


class FileProcessingError(Exception):
    """Custom exception for file processing errors"""
    pass


class UnsupportedFileTypeError(Exception):
    """Custom exception for unsupported file types"""
    pass


# File type detection
def detect_file_type(file_path: str) -> str:
    """
    Detect file type using multiple methods for accuracy
    
    Args:
        file_path: Path to the file
        
    Returns:
        Detected file extension (pdf, docx, txt, md)
        
    Raises:
        UnsupportedFileTypeError: If file type is not supported
    """
    try:
        # Get file extension
        file_ext = Path(file_path).suffix.lower()
        
        # Use python-magic for MIME type detection (if available)
        mime_type = None
        if MAGIC_AVAILABLE:
            try:
                mime_type = magic.from_file(file_path, mime=True)
                logger.debug(f"Detected MIME type: {mime_type} for {file_path}")
            except Exception as e:
                logger.warning(f"Could not detect MIME type for {file_path}: {e}")
                mime_type = None
        else:
            logger.debug("Skipping MIME type detection - magic not available")
        
        # Map extensions to our supported types
        ext_mapping = {
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.doc': 'docx',  # Treat .doc as docx for now
            '.txt': 'txt',
            '.md': 'md',
            '.markdown': 'md'
        }
        
        if file_ext in ext_mapping:
            detected_type = ext_mapping[file_ext]
            logger.info(f"Detected file type: {detected_type} for {file_path}")
            return detected_type
        
        # Fallback to MIME type detection
        if mime_type:
            mime_mapping = {
                'application/pdf': 'pdf',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
                'application/msword': 'docx',
                'text/plain': 'txt',
                'text/markdown': 'md'
            }
            
            if mime_type in mime_mapping:
                detected_type = mime_mapping[mime_type]
                logger.info(f"Detected file type from MIME: {detected_type}")
                return detected_type
        
        raise UnsupportedFileTypeError(f"Unsupported file type: {file_ext}")
        
    except Exception as e:
        logger.error(f"Error detecting file type for {file_path}: {e}")
        raise UnsupportedFileTypeError(f"Could not determine file type: {str(e)}")


def validate_file_size(file_path: str, max_size_mb: int = 50) -> bool:
    """
    Validate file size is within limits
    
    Args:
        file_path: Path to file
        max_size_mb: Maximum allowed size in MB
        
    Returns:
        True if file size is valid
        
    Raises:
        FileProcessingError: If file is too large or doesn't exist
    """
    try:
        if not os.path.exists(file_path):
            raise FileProcessingError(f"File does not exist: {file_path}")
        
        file_size = os.path.getsize(file_path)
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if file_size > max_size_bytes:
            raise FileProcessingError(
                f"File too large: {file_size} bytes (max: {max_size_bytes} bytes)"
            )
        
        logger.info(f"File size validation passed: {file_size} bytes")
        return True
        
    except Exception as e:
        logger.error(f"File size validation failed: {e}")
        raise


def generate_secure_id() -> str:
    """Generate a secure unique identifier"""
    return str(uuid.uuid4())


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem
    """
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove multiple spaces and underscores
    filename = re.sub(r'[_\s]+', '_', filename)
    
    # Ensure reasonable length
    name, ext = os.path.splitext(filename)
    if len(name) > 200:
        name = name[:200]
    
    return f"{name}{ext}".strip('_')


def calculate_file_hash(file_path: str) -> str:
    """
    Calculate SHA-256 hash of file for deduplication
    
    Args:
        file_path: Path to file
        
    Returns:
        Hexadecimal hash string
    """
    sha256_hash = hashlib.sha256()
    
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        file_hash = sha256_hash.hexdigest()
        logger.debug(f"Calculated hash for {file_path}: {file_hash[:16]}...")
        return file_hash
        
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        raise FileProcessingError(f"Could not calculate file hash: {str(e)}")


# Text extraction functions
def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from PDF file
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Extracted text content
        
    Raises:
        FileProcessingError: If PDF processing fails
    """
    try:
        text_content = ""
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Check if PDF is encrypted
            if pdf_reader.is_encrypted:
                logger.warning(f"PDF is encrypted: {file_path}")
                # Try to decrypt with empty password
                try:
                    pdf_reader.decrypt("")
                except:
                    raise FileProcessingError("PDF is password protected")
            
            # Extract text from all pages
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
                except Exception as e:
                    logger.warning(f"Could not extract text from page {page_num + 1}: {e}")
                    continue
        
        if not text_content.strip():
            raise FileProcessingError("No text content found in PDF")
        
        logger.info(f"Successfully extracted {len(text_content)} characters from PDF")
        return clean_text(text_content)
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {e}")
        raise FileProcessingError(f"PDF text extraction failed: {str(e)}")


def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from DOCX file
    
    Args:
        file_path: Path to DOCX file
        
    Returns:
        Extracted text content
        
    Raises:
        FileProcessingError: If DOCX processing fails
    """
    try:
        doc = Document(file_path)
        text_content = ""
        
        # Extract text from paragraphs
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content += paragraph.text + "\n"
        
        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    text_content += " | ".join(row_text) + "\n"
        
        if not text_content.strip():
            raise FileProcessingError("No text content found in DOCX")
        
        logger.info(f"Successfully extracted {len(text_content)} characters from DOCX")
        return clean_text(text_content)
        
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {file_path}: {e}")
        raise FileProcessingError(f"DOCX text extraction failed: {str(e)}")


def extract_text_from_txt(file_path: str) -> str:
    """
    Extract text from TXT file with encoding detection
    
    Args:
        file_path: Path to TXT file
        
    Returns:
        Text content
        
    Raises:
        FileProcessingError: If text file processing fails
    """
    try:
        # Try common encodings
        encodings = ['utf-8', 'utf-16', 'ascii', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    text_content = file.read()
                    
                if text_content.strip():
                    logger.info(f"Successfully read TXT file with {encoding} encoding")
                    return clean_text(text_content)
                    
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Error reading with {encoding}: {e}")
                continue
        
        raise FileProcessingError("Could not decode text file with any supported encoding")
        
    except Exception as e:
        logger.error(f"Error extracting text from TXT {file_path}: {e}")
        raise FileProcessingError(f"TXT text extraction failed: {str(e)}")


def extract_text_from_markdown(file_path: str) -> str:
    """
    Extract text from Markdown file
    
    Args:
        file_path: Path to Markdown file
        
    Returns:
        Text content with basic markdown preserved
        
    Raises:
        FileProcessingError: If markdown processing fails
    """
    try:
        # Markdown files are essentially text files
        text_content = extract_text_from_txt(file_path)
        
        # Optional: Could add markdown parsing here in future
        # For now, preserve the markdown formatting
        
        logger.info(f"Successfully extracted {len(text_content)} characters from Markdown")
        return text_content
        
    except Exception as e:
        logger.error(f"Error extracting text from Markdown {file_path}: {e}")
        raise FileProcessingError(f"Markdown text extraction failed: {str(e)}")


# Text processing functions
def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Normalize whitespace
    text = normalize_whitespace(text)
    
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    if not text:
        return ""
    
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    return text


def remove_special_characters(text: str, keep_punctuation: bool = True) -> str:
    """
    Remove special characters from text
    
    Args:
        text: Input text
        keep_punctuation: Whether to keep basic punctuation
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    if keep_punctuation:
        # Keep alphanumeric, basic punctuation, and whitespace
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\']+', ' ', text)
    else:
        # Keep only alphanumeric and whitespace
        text = re.sub(r'[^\w\s]+', ' ', text)
    
    return normalize_whitespace(text)


# Validation functions
def validate_email(email: str) -> bool:
    """Validate email format"""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email))


def validate_url(url: str) -> bool:
    """Validate URL format"""
    url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(url_pattern, url))


# Date/time utilities
def get_utc_now() -> datetime:
    """Get current UTC datetime"""
    return datetime.utcnow()


def format_timestamp(dt: datetime) -> str:
    """Format datetime as ISO string"""
    return dt.isoformat()


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse ISO timestamp string to datetime"""
    return datetime.fromisoformat(timestamp_str)


# Security functions
def hash_password(password: str) -> str:
    """Hash password with salt (placeholder for future use)"""
    import hashlib
    import secrets
    
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return f"{salt}:{pwd_hash.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash (placeholder for future use)"""
    import hashlib
    
    try:
        salt, pwd_hash = hashed.split(':')
        computed_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return computed_hash.hex() == pwd_hash
    except:
        return False


def generate_api_key() -> str:
    """Generate API key for future use"""
    import secrets
    return secrets.token_urlsafe(32)


# File processing dispatcher
def extract_text_from_file(file_path: str, file_type: Optional[str] = None) -> str:
    """
    Extract text from file based on type
    
    Args:
        file_path: Path to file
        file_type: File type (if known)
        
    Returns:
        Extracted text content
        
    Raises:
        FileProcessingError: If extraction fails
        UnsupportedFileTypeError: If file type not supported
    """
    if not file_type:
        file_type = detect_file_type(file_path)
    
    extractors = {
        'pdf': extract_text_from_pdf,
        'docx': extract_text_from_docx,
        'txt': extract_text_from_txt,
        'md': extract_text_from_markdown
    }
    
    if file_type not in extractors:
        raise UnsupportedFileTypeError(f"Unsupported file type: {file_type}")
    
    try:
        text = extractors[file_type](file_path)
        if not text or not text.strip():
            raise FileProcessingError("No text content extracted from file")
        
        return text
        
    except Exception as e:
        logger.error(f"Text extraction failed for {file_path}: {e}")
        raise