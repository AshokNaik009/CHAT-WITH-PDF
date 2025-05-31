"""
Chat Engine - Core orchestrator for PDF processing and chat functionality
Optimized for production deployment on Back4app Containers
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from .pdf_processor import PDFProcessor
from .vector_store import VectorStore
from utils.embeddings import EmbeddingManager
from utils.huggingface import HuggingFaceClient

logger = logging.getLogger(__name__)

class ChatEngine:
    """
    Main orchestrator for Chat with PDF functionality
    Handles document processing, storage, and query responses
    """
    
    def __init__(self):
        self.pdf_processor = None
        self.vector_store = None
        self.embedding_manager = None
        self.hf_client = None
        self.documents = {}  # In-memory document metadata
        self.initialized = False
        
        # Configuration
        self.max_chunk_size = 500
        self.overlap_size = 50
        self.max_documents = 100  # Limit for free tier
        
    async def initialize(self):
        """Initialize all components asynchronously"""
        try:
            logger.info("🔧 Initializing ChatEngine components...")
            
            # Initialize components
            self.pdf_processor = PDFProcessor()
            self.vector_store = VectorStore()
            self.embedding_manager = EmbeddingManager()
            self.hf_client = HuggingFaceClient()
            
            # Load embedding model
            await self.embedding_manager.initialize()
            
            # Initialize vector store
            await self.vector_store.initialize()
            
            self.initialized = True
            logger.info("✅ ChatEngine initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChatEngine: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if all components are healthy"""
        try:
            if not self.initialized:
                return False
                
            # Check if embedding model is loaded
            if not self.embedding_manager.is_ready():
                return False
                
            # Check vector store
            if not self.vector_store.is_ready():
                return False
                
            return True
        except Exception:
            return False
    
    async def process_document(self, file_content: bytes, filename: str) -> Dict:
        """
        Process uploaded PDF and create searchable index
        
        Args:
            file_content: PDF file bytes
            filename: Original filename
            
        Returns:
            Dict with document_id and processing status
        """
        try:
            # Check document limit
            if len(self.documents) >= self.max_documents:
                raise ValueError("Document limit reached. Please delete some documents first.")
            
            # Generate document ID
            doc_id = self._generate_doc_id(filename, file_content)
            
            # Check if document already exists
            if doc_id in self.documents:
                logger.info(f"📄 Document {filename} already processed")
                return {
                    "document_id": doc_id,
                    "status": "already_exists",
                    "message": "Document already processed"
                }
            
            logger.info(f"📤 Processing document: {filename}")
            
            # Extract text from PDF
            text_content = await self.pdf_processor.extract_text(file_content)
            
            if not text_content.strip():
                raise ValueError("No text content found in PDF")
            
            # Create chunks
            chunks = self._create_chunks(text_content)
            logger.info(f"📝 Created {len(chunks)} chunks")
            
            # Generate embeddings
            embeddings = await self.embedding_manager.encode_texts(chunks)
            logger.info(f"🧮 Generated embeddings: {len(embeddings)} vectors")
            
            # Store in vector database
            await self.vector_store.add_document(doc_id, chunks, embeddings)
            
            # Store document metadata
            self.documents[doc_id] = {
                "filename": filename,
                "doc_id": doc_id,
                "processed_at": datetime.now().isoformat(),
                "chunk_count": len(chunks),
                "text_length": len(text_content)
            }
            
            logger.info(f"✅ Document {filename} processed successfully")
            
            return {
                "document_id": doc_id,
                "status": "success",
                "chunk_count": len(chunks),
                "filename": filename
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing document {filename}: {e}")
            raise
    
    async def chat_with_document(self, doc_id: str, question: str, use_llm: bool = True) -> Dict:
        """
        Process user question and return relevant answers
        
        Args:
            doc_id: Document identifier
            question: User question
            use_llm: Whether to use LLM for enhanced responses
            
        Returns:
            Dict with answer and sources
        """
        try:
            # Validate document exists
            if doc_id not in self.documents:
                raise ValueError(f"Document {doc_id} not found")
            
            logger.info(f"💬 Processing question for doc {doc_id}: {question[:100]}...")
            
            # Generate question embedding
            question_embedding = await self.embedding_manager.encode_texts([question])
            
            # Search for relevant chunks
            relevant_chunks = await self.vector_store.search(
                doc_id, 
                question_embedding[0], 
                top_k=5
            )
            
            if not relevant_chunks:
                return {
                    "answer": "I couldn't find relevant information in the document to answer your question.",
                    "sources": [],
                    "confidence": 0.0,
                    "type": "no_results"
                }
            
            # Determine response strategy
            max_similarity = max(chunk["similarity"] for chunk in relevant_chunks)
            
            if use_llm and max_similarity > 0.3:  # Threshold for LLM usage
                # Enhanced response with HuggingFace
                return await self._generate_llm_response(relevant_chunks, question, doc_id)
            else:
                # Basic semantic search response
                return self._generate_basic_response(relevant_chunks, doc_id)
                
        except Exception as e:
            logger.error(f"❌ Error in chat processing: {e}")
            raise
    
    async def _generate_llm_response(self, chunks: List[Dict], question: str, doc_id: str) -> Dict:
        """Generate enhanced response using HuggingFace LLM"""
        try:
            # Prepare context from chunks
            context = "\n\n".join([
                f"[Chunk {i+1}]: {chunk['text']}"
                for i, chunk in enumerate(chunks[:3])  # Use top 3 chunks
            ])
            
            # Generate response using HuggingFace
            llm_response = await self.hf_client.generate_response(question, context)
            
            return {
                "answer": llm_response,
                "sources": [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                        "similarity": round(chunk["similarity"], 3)
                    }
                    for chunk in chunks
                ],
                "confidence": round(max(chunk["similarity"] for chunk in chunks), 3),
                "type": "llm_enhanced",
                "document_id": doc_id
            }
            
        except Exception as e:
            logger.warning(f"⚠️ LLM generation failed, falling back to basic response: {e}")
            return self._generate_basic_response(chunks, doc_id)
    
    def _generate_basic_response(self, chunks: List[Dict], doc_id: str) -> Dict:
        """Generate basic response from semantic search results"""
        best_chunk = chunks[0]
        
        return {
            "answer": best_chunk["text"],
            "sources": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "similarity": round(chunk["similarity"], 3)
                }
                for chunk in chunks
            ],
            "confidence": round(best_chunk["similarity"], 3),
            "type": "semantic_search",
            "document_id": doc_id
        }
    
    def _create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks for better context"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.max_chunk_size - self.overlap_size):
            chunk_words = words[i:i + self.max_chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if len(chunk_text.strip()) > 50:  # Minimum chunk size
                chunks.append(chunk_text)
        
        return chunks
    
    def _generate_doc_id(self, filename: str, content: bytes) -> str:
        """Generate unique document ID from filename and content hash"""
        content_hash = hashlib.md5(content).hexdigest()[:8]
        filename_clean = "".join(c for c in filename if c.isalnum() or c in "._-")[:20]
        return f"{filename_clean}_{content_hash}"
    
    async def list_documents(self) -> List[Dict]:
        """Get list of processed documents"""
        return list(self.documents.values())
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete document and its embeddings"""
        try:
            if doc_id in self.documents:
                await self.vector_store.delete_document(doc_id)
                del self.documents[doc_id]
                logger.info(f"🗑️ Deleted document {doc_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error deleting document {doc_id}: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources on shutdown"""
        try:
            if self.vector_store:
                await self.vector_store.cleanup()
            logger.info("🧹 ChatEngine cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")