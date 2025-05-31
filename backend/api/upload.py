"""
Upload API - Endpoints for PDF file upload and processing
Production-ready with comprehensive validation and error handling
"""

import logging
from typing import Optional, List
from datetime import datetime
import asyncio

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import the main app's chat engine getter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import get_chat_engine

logger = logging.getLogger(__name__)

# Create router
upload_router = APIRouter()

# Configuration for Back4app Container constraints
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_MIME_TYPES = [
    "application/pdf",
    "application/x-pdf",
    "application/acrobat",
    "applications/vnd.pdf",
    "text/pdf",
    "text/x-pdf"
]
ALLOWED_EXTENSIONS = [".pdf"]

# Pydantic models
class UploadResponse(BaseModel):
    """Response model for successful upload"""
    document_id: str
    filename: str
    status: str = Field(..., description="Processing status: 'success', 'processing', 'already_exists'")
    message: str
    chunk_count: Optional[int] = None
    processing_time_ms: Optional[int] = None
    uploaded_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class ProcessingStatus(BaseModel):
    """Model for processing status"""
    document_id: str
    status: str = Field(..., description="Status: 'processing', 'completed', 'failed'")
    progress: float = Field(..., ge=0.0, le=100.0, description="Processing progress percentage")
    message: str
    estimated_completion: Optional[str] = None

class UploadError(BaseModel):
    """Error response model"""
    error: str
    detail: str
    error_code: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# In-memory processing status tracking (use Redis in production)
processing_status: dict = {}

def validate_file(file: UploadFile) -> None:
    """
    Comprehensive file validation for security and compatibility
    
    Args:
        file: Uploaded file object
        
    Raises:
        HTTPException: If file validation fails
    """
    # Check file size
    if hasattr(file, 'size') and file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Check filename
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )
    
    # Check file extension
    file_extension = os.path.splitext(file.filename.lower())[1]
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check MIME type if available
    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        logger.warning(f"⚠️ Unexpected MIME type: {file.content_type} for {file.filename}")
        # Don't fail on MIME type mismatch, just log warning
    
    # Sanitize filename
    if len(file.filename) > 255:
        raise HTTPException(
            status_code=400,
            detail="Filename too long (max 255 characters)"
        )
    
    # Check for dangerous characters
    dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\0']
    if any(char in file.filename for char in dangerous_chars):
        raise HTTPException(
            status_code=400,
            detail="Filename contains invalid characters"
        )

@upload_router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload and process"),
    process_async: bool = Form(default=False, description="Process file asynchronously"),
    chat_engine = Depends(get_chat_engine)
):
    """
    Upload and process a PDF file for chat functionality
    
    Args:
        file: PDF file to upload
        process_async: Whether to process file in background (for large files)
        
    Returns:
        UploadResponse with document ID and processing status
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"📤 Upload request: {file.filename} ({file.content_type})")
        
        # Validate file
        validate_file(file)
        
        # Read file content
        try:
            file_content = await file.read()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read file: {str(e)}"
            )
        
        # Validate actual file size after reading
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        if len(file_content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Validate PDF format
        if not _is_valid_pdf(file_content):
            raise HTTPException(
                status_code=400,
                detail="Invalid PDF file format"
            )
        
        # Process file based on mode
        if process_async:
            # Background processing for large files
            doc_id = await _start_background_processing(
                background_tasks, chat_engine, file_content, file.filename
            )
            
            return UploadResponse(
                document_id=doc_id,
                filename=file.filename,
                status="processing",
                message="File uploaded successfully. Processing in background.",
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
        else:
            # Synchronous processing
            result = await chat_engine.process_document(file_content, file.filename)
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            logger.info(f"✅ Document processed successfully in {processing_time}ms")
            
            return UploadResponse(
                document_id=result["document_id"],
                filename=file.filename,
                status=result["status"],
                message="Document processed successfully" if result["status"] == "success" else result.get("message", ""),
                chunk_count=result.get("chunk_count"),
                processing_time_ms=processing_time
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload error for {file.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the file: {str(e)}"
        )

@upload_router.post("/upload-batch")
async def upload_multiple_pdfs(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Multiple PDF files to upload"),
    chat_engine = Depends(get_chat_engine)
):
    """
    Upload and process multiple PDF files
    
    Args:
        files: List of PDF files to upload
        
    Returns:
        Batch processing results
    """
    try:
        logger.info(f"📤 Batch upload request: {len(files)} files")
        
        # Validate batch size
        if len(files) > 10:  # Limit for free tier
            raise HTTPException(
                status_code=400,
                detail="Too many files. Maximum 10 files per batch"
            )
        
        results = []
        
        for file in files:
            try:
                # Validate each file
                validate_file(file)
                
                # Read and validate content
                file_content = await file.read()
                
                if not _is_valid_pdf(file_content):
                    results.append({
                        "filename": file.filename,
                        "status": "failed",
                        "error": "Invalid PDF format"
                    })
                    continue
                
                # Start background processing for each file
                doc_id = await _start_background_processing(
                    background_tasks, chat_engine, file_content, file.filename
                )
                
                results.append({
                    "filename": file.filename,
                    "document_id": doc_id,
                    "status": "processing"
                })
                
            except Exception as e:
                logger.error(f"❌ Error processing file {file.filename}: {e}")
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": str(e)
                })
        
        return {
            "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "total_files": len(files),
            "results": results,
            "message": f"Batch upload initiated for {len(files)} files"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Batch upload error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during batch upload: {str(e)}"
        )

@upload_router.get("/processing-status/{doc_id}", response_model=ProcessingStatus)
async def get_processing_status(doc_id: str):
    """
    Get processing status for a document
    
    Args:
        doc_id: Document identifier
        
    Returns:
        Processing status information
    """
    try:
        if doc_id not in processing_status:
            raise HTTPException(
                status_code=404,
                detail=f"Processing status not found for document {doc_id}"
            )
        
        status_info = processing_status[doc_id]
        
        return ProcessingStatus(
            document_id=doc_id,
            status=status_info["status"],
            progress=status_info["progress"],
            message=status_info["message"],
            estimated_completion=status_info.get("estimated_completion")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting processing status for {doc_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while checking processing status"
        )

@upload_router.delete("/processing-status/{doc_id}")
async def cancel_processing(doc_id: str):
    """
    Cancel background processing for a document
    
    Args:
        doc_id: Document identifier
        
    Returns:
        Cancellation confirmation
    """
    try:
        if doc_id not in processing_status:
            raise HTTPException(
                status_code=404,
                detail=f"Processing status not found for document {doc_id}"
            )
        
        # Mark as cancelled
        processing_status[doc_id]["status"] = "cancelled"
        processing_status[doc_id]["message"] = "Processing cancelled by user"
        
        logger.info(f"🚫 Processing cancelled for document {doc_id}")
        
        return {
            "message": f"Processing cancelled for document {doc_id}",
            "document_id": doc_id,
            "cancelled_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error cancelling processing for {doc_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while cancelling processing"
        )

@upload_router.get("/upload-limits")
async def get_upload_limits():
    """Get current upload limits and configuration"""
    return {
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "max_file_size_bytes": MAX_FILE_SIZE,
        "allowed_extensions": ALLOWED_EXTENSIONS,
        "allowed_mime_types": ALLOWED_MIME_TYPES,
        "max_batch_size": 10,
        "max_filename_length": 255
    }

# Helper functions
def _is_valid_pdf(content: bytes) -> bool:
    """
    Validate PDF file format by checking magic bytes and basic structure
    
    Args:
        content: File content as bytes
        
    Returns:
        True if valid PDF, False otherwise
    """
    try:
        # Check minimum length
        if len(content) < 8:
            return False
        
        # Check PDF magic bytes
        if not content.startswith(b'%PDF-'):
            return False
        
        # Check for EOF marker (basic structure validation)
        content_str = content[-1024:].decode('latin-1', errors='ignore')
        if '%%EOF' not in content_str:
            logger.warning("⚠️ PDF missing EOF marker, but proceeding")
        
        # Additional validation could be added here
        # For now, basic magic byte check is sufficient
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PDF validation error: {e}")
        return False

async def _start_background_processing(
    background_tasks: BackgroundTasks,
    chat_engine,
    file_content: bytes,
    filename: str
) -> str:
    """
    Start background processing for a document
    
    Args:
        background_tasks: FastAPI background tasks
        chat_engine: Chat engine instance
        file_content: PDF file content
        filename: Original filename
        
    Returns:
        Document ID for tracking
    """
    # Generate temporary document ID
    doc_id = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(filename) % 10000}"
    
    # Initialize processing status
    processing_status[doc_id] = {
        "status": "processing",
        "progress": 0.0,
        "message": "Starting document processing...",
        "started_at": datetime.now().isoformat()
    }
    
    # Add background task
    background_tasks.add_task(
        _process_document_background,
        chat_engine,
        doc_id,
        file_content,
        filename
    )
    
    return doc_id

async def _process_document_background(
    chat_engine,
    temp_doc_id: str,
    file_content: bytes,
    filename: str
):
    """
    Background task for document processing
    
    Args:
        chat_engine: Chat engine instance
        temp_doc_id: Temporary document ID
        file_content: PDF file content
        filename: Original filename
    """
    try:
        logger.info(f"🔄 Starting background processing for {filename}")
        
        # Update status
        processing_status[temp_doc_id].update({
            "progress": 10.0,
            "message": "Extracting text from PDF..."
        })
        
        # Process document
        result = await chat_engine.process_document(file_content, filename)
        
        # Update status with completion
        processing_status[temp_doc_id].update({
            "status": "completed",
            "progress": 100.0,
            "message": "Document processing completed successfully",
            "document_id": result["document_id"],
            "chunk_count": result.get("chunk_count"),
            "completed_at": datetime.now().isoformat()
        })
        
        logger.info(f"✅ Background processing completed for {filename}")
        
    except Exception as e:
        logger.error(f"❌ Background processing failed for {filename}: {e}")
        
        # Update status with error
        processing_status[temp_doc_id].update({
            "status": "failed",
            "progress": 0.0,
            "message": f"Processing failed: {str(e)}",
            "failed_at": datetime.now().isoformat()
        })

@upload_router.get("/health")
async def upload_health_check():
    """Health check for upload service"""
    try:
        return {
            "status": "healthy",
            "upload_service": "operational",
            "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
            "active_processing_jobs": len([
                doc_id for doc_id, status in processing_status.items()
                if status["status"] == "processing"
            ]),
            "supported_formats": ALLOWED_EXTENSIONS
        }
    except Exception as e:
        logger.error(f"❌ Upload health check failed: {e}")
        raise HTTPException(status_code=503, detail="Upload service unhealthy")