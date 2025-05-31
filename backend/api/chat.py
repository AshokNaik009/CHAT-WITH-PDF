"""
Chat API - Endpoints for document querying and chat functionality
Production-ready with comprehensive error handling and validation
"""

import logging
from typing import Optional, List, Dict
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field, validator

# Import the main app's chat engine getter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import get_chat_engine

logger = logging.getLogger(__name__)

# Create router
chat_router = APIRouter()

# Pydantic models for request/response validation
class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    use_llm: bool = Field(default=True, description="Whether to use LLM for enhanced responses")
    max_chunks: int = Field(default=5, ge=1, le=10, description="Maximum number of chunks to return")
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()

class SourceChunk(BaseModel):
    """Model for source chunk information"""
    chunk_id: str
    text: str
    similarity: float = Field(..., ge=0.0, le=1.0)

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str
    sources: List[SourceChunk]
    confidence: float = Field(..., ge=0.0, le=1.0)
    response_type: str = Field(..., description="Type: 'llm_enhanced', 'semantic_search', or 'no_results'")
    document_id: str
    processing_time_ms: Optional[int] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class DocumentInfo(BaseModel):
    """Model for document information"""
    document_id: str
    filename: str
    processed_at: str
    chunk_count: int
    text_length: int

class DocumentListResponse(BaseModel):
    """Response model for document listing"""
    documents: List[DocumentInfo]
    total_count: int

class ChatHistoryItem(BaseModel):
    """Model for chat history item"""
    question: str
    answer: str
    timestamp: str
    confidence: float
    response_type: str

# In-memory chat history (for demo purposes)
# In production, use Redis or database
chat_history: Dict[str, List[ChatHistoryItem]] = {}

@chat_router.post("/chat/{doc_id}", response_model=ChatResponse)
async def chat_with_document(
    doc_id: str,
    request: ChatRequest,
    chat_engine = Depends(get_chat_engine)
):
    """
    Chat with a specific document using semantic search and optional LLM enhancement
    
    Args:
        doc_id: Document identifier
        request: Chat request with question and options
        
    Returns:
        ChatResponse with answer and sources
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"💬 Chat request for doc {doc_id}: {request.question[:100]}...")
        
        # Validate document ID format
        if not doc_id or len(doc_id) < 5:
            raise HTTPException(
                status_code=400, 
                detail="Invalid document ID format"
            )
        
        # Process the chat request
        result = await chat_engine.chat_with_document(
            doc_id=doc_id,
            question=request.question,
            use_llm=request.use_llm
        )
        
        # Calculate processing time
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Create response
        response = ChatResponse(
            answer=result["answer"],
            sources=[
                SourceChunk(
                    chunk_id=source["chunk_id"],
                    text=source["text"],
                    similarity=source["similarity"]
                )
                for source in result["sources"][:request.max_chunks]
            ],
            confidence=result["confidence"],
            response_type=result["type"],
            document_id=result["document_id"],
            processing_time_ms=processing_time
        )
        
        # Store in chat history
        _add_to_chat_history(doc_id, request.question, result["answer"], 
                           result["confidence"], result["type"])
        
        logger.info(f"✅ Chat completed in {processing_time}ms with confidence {result['confidence']}")
        
        return response
        
    except ValueError as e:
        logger.warning(f"⚠️ Invalid request for doc {doc_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"❌ Chat error for doc {doc_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while processing your question: {str(e)}"
        )

@chat_router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of documents to return"),
    chat_engine = Depends(get_chat_engine)
):
    """
    Get list of all processed documents
    
    Args:
        limit: Maximum number of documents to return
        
    Returns:
        List of document information
    """
    try:
        logger.info("📋 Fetching document list...")
        
        documents = await chat_engine.list_documents()
        
        # Convert to response format
        document_list = [
            DocumentInfo(
                document_id=doc["doc_id"],
                filename=doc["filename"],
                processed_at=doc["processed_at"],
                chunk_count=doc["chunk_count"],
                text_length=doc["text_length"]
            )
            for doc in documents[:limit]
        ]
        
        response = DocumentListResponse(
            documents=document_list,
            total_count=len(documents)
        )
        
        logger.info(f"✅ Returned {len(document_list)} documents")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error listing documents: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while fetching documents"
        )

@chat_router.get("/documents/{doc_id}/info")
async def get_document_info(
    doc_id: str,
    chat_engine = Depends(get_chat_engine)
):
    """
    Get detailed information about a specific document
    
    Args:
        doc_id: Document identifier
        
    Returns:
        Document information and statistics
    """
    try:
        logger.info(f"ℹ️ Fetching info for document {doc_id}")
        
        documents = await chat_engine.list_documents()
        document = next((doc for doc in documents if doc["doc_id"] == doc_id), None)
        
        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document {doc_id} not found"
            )
        
        # Get chat history for this document
        history = chat_history.get(doc_id, [])
        
        return {
            "document": DocumentInfo(
                document_id=document["doc_id"],
                filename=document["filename"],
                processed_at=document["processed_at"],
                chunk_count=document["chunk_count"],
                text_length=document["text_length"]
            ),
            "chat_history_count": len(history),
            "last_accessed": history[-1]["timestamp"] if history else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching document info {doc_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while fetching document information"
        )

@chat_router.get("/documents/{doc_id}/history")
async def get_chat_history(
    doc_id: str,
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of history items")
):
    """
    Get chat history for a specific document
    
    Args:
        doc_id: Document identifier
        limit: Maximum number of history items to return
        
    Returns:
        List of chat history items
    """
    try:
        logger.info(f"📚 Fetching chat history for document {doc_id}")
        
        history = chat_history.get(doc_id, [])
        
        # Return most recent items first
        recent_history = list(reversed(history))[:limit]
        
        return {
            "document_id": doc_id,
            "history": recent_history,
            "total_count": len(history)
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching chat history for {doc_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while fetching chat history"
        )

@chat_router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    chat_engine = Depends(get_chat_engine)
):
    """
    Delete a document and its associated data
    
    Args:
        doc_id: Document identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        logger.info(f"🗑️ Deleting document {doc_id}")
        
        success = await chat_engine.delete_document(doc_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Document {doc_id} not found"
            )
        
        # Clear chat history
        if doc_id in chat_history:
            del chat_history[doc_id]
        
        logger.info(f"✅ Document {doc_id} deleted successfully")
        
        return {
            "message": f"Document {doc_id} deleted successfully",
            "document_id": doc_id,
            "deleted_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting document {doc_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while deleting the document"
        )

@chat_router.get("/health")
async def chat_health_check(chat_engine = Depends(get_chat_engine)):
    """Health check endpoint for chat functionality"""
    try:
        is_healthy = await chat_engine.health_check()
        
        if not is_healthy:
            raise HTTPException(status_code=503, detail="Chat engine not ready")
        
        # Get vector store stats
        if hasattr(chat_engine, 'vector_store') and chat_engine.vector_store:
            stats = chat_engine.vector_store.get_stats()
        else:
            stats = {}
        
        return {
            "status": "healthy",
            "chat_engine": "operational",
            "vector_store_stats": stats,
            "total_documents": len(await chat_engine.list_documents()),
            "total_chat_sessions": len(chat_history)
        }
        
    except Exception as e:
        logger.error(f"❌ Chat health check failed: {e}")
        raise HTTPException(status_code=503, detail="Chat service unhealthy")

# Helper functions
def _add_to_chat_history(doc_id: str, question: str, answer: str, confidence: float, response_type: str):
    """Add item to chat history"""
    try:
        if doc_id not in chat_history:
            chat_history[doc_id] = []
        
        # Limit history size per document
        max_history = 50
        if len(chat_history[doc_id]) >= max_history:
            chat_history[doc_id] = chat_history[doc_id][-(max_history-1):]
        
        chat_history[doc_id].append(ChatHistoryItem(
            question=question,
            answer=answer[:500] + "..." if len(answer) > 500 else answer,  # Truncate long answers
            timestamp=datetime.now().isoformat(),
            confidence=confidence,
            response_type=response_type
        ))
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to add to chat history: {e}")

@chat_router.post("/documents/{doc_id}/clear-history")
async def clear_chat_history(doc_id: str):
    """Clear chat history for a specific document"""
    try:
        if doc_id in chat_history:
            del chat_history[doc_id]
            logger.info(f"🧹 Cleared chat history for document {doc_id}")
        
        return {
            "message": f"Chat history cleared for document {doc_id}",
            "document_id": doc_id,
            "cleared_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error clearing chat history for {doc_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while clearing chat history"
        )