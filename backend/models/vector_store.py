"""
Vector Store - FAISS-based vector database for semantic search
Optimized for Back4app Container deployment with memory constraints
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
import faiss
import pickle
from pathlib import Path
import tempfile
import os

logger = logging.getLogger(__name__)

class VectorStore:
    """
    FAISS-based vector database for efficient similarity search
    Designed for production deployment with memory optimization
    """
    
    def __init__(self):
        self.index = None
        self.document_chunks = {}  # doc_id -> {chunk_id: text}
        self.chunk_mappings = {}   # global_id -> (doc_id, chunk_id)
        self.dimension = 384       # all-MiniLM-L6-v2 embedding size
        self.next_id = 0
        self.initialized = False
        
        # Configuration for Back4app constraints
        self.max_vectors = 10000   # Limit for memory management
        self.use_gpu = False       # CPU-only for containers
        
    async def initialize(self):
        """Initialize FAISS index"""
        try:
            logger.info("🗂️ Initializing FAISS vector store...")
            
            # Create FAISS index (CPU-optimized)
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._create_index
            )
            
            self.initialized = True
            logger.info("✅ Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}")
            raise
    
    def _create_index(self):
        """Create optimized FAISS index for production"""
        try:
            # Use IndexFlatIP for small datasets (best for containers)
            # Inner product similarity (cosine similarity for normalized vectors)
            self.index = faiss.IndexFlatIP(self.dimension)
            
            # Alternative: Use IndexIVFFlat for larger datasets
            # if self.max_vectors > 5000:
            #     quantizer = faiss.IndexFlatIP(self.dimension)
            #     self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            
            logger.info(f"📊 Created FAISS index with dimension {self.dimension}")
            
        except Exception as e:
            logger.error(f"❌ Index creation failed: {e}")
            raise
    
    async def add_document(self, doc_id: str, chunks: List[str], embeddings: np.ndarray):
        """
        Add document chunks and embeddings to the vector store
        
        Args:
            doc_id: Unique document identifier
            chunks: List of text chunks
            embeddings: Numpy array of embeddings (shape: [num_chunks, dimension])
        """
        try:
            # Validate inputs
            if len(chunks) != len(embeddings):
                raise ValueError("Number of chunks and embeddings must match")
            
            # Check capacity
            if self.index.ntotal + len(embeddings) > self.max_vectors:
                raise ValueError(f"Vector store capacity exceeded. Max: {self.max_vectors}")
            
            logger.info(f"➕ Adding {len(chunks)} chunks for document {doc_id}")
            
            # Run in executor for CPU-intensive operations
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._add_vectors_sync,
                doc_id, chunks, embeddings
            )
            
            logger.info(f"✅ Added document {doc_id} to vector store")
            
        except Exception as e:
            logger.error(f"❌ Error adding document {doc_id}: {e}")
            raise
    
    def _add_vectors_sync(self, doc_id: str, chunks: List[str], embeddings: np.ndarray):
        """Synchronously add vectors to FAISS index"""
        try:
            # Normalize embeddings for cosine similarity
            normalized_embeddings = self._normalize_embeddings(embeddings)
            
            # Store document chunks
            self.document_chunks[doc_id] = {}
            
            # Add vectors and create mappings
            start_id = self.next_id
            
            for i, (chunk, embedding) in enumerate(zip(chunks, normalized_embeddings)):
                chunk_id = f"{doc_id}_chunk_{i}"
                global_id = self.next_id
                
                # Store chunk text
                self.document_chunks[doc_id][chunk_id] = chunk
                
                # Store mapping
                self.chunk_mappings[global_id] = (doc_id, chunk_id)
                
                self.next_id += 1
            
            # Add all embeddings to FAISS index at once
            self.index.add(normalized_embeddings)
            
            logger.info(f"📊 Index now contains {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"❌ Sync vector addition failed: {e}")
            raise
    
    async def search(self, doc_id: str, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks in a specific document
        
        Args:
            doc_id: Document to search in
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of matching chunks with similarity scores
        """
        try:
            if not self.initialized or self.index.ntotal == 0:
                return []
            
            # Run search in executor
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                self._search_sync,
                doc_id, query_embedding, top_k
            )
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def _search_sync(self, doc_id: str, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Synchronous similarity search"""
        try:
            # Normalize query embedding
            query_normalized = self._normalize_embeddings(query_embedding.reshape(1, -1))
            
            # Search in FAISS index
            # Get more results to filter by document
            search_k = min(top_k * 3, self.index.ntotal)
            similarities, indices = self.index.search(query_normalized, search_k)
            
            results = []
            similarities = similarities[0]  # First query
            indices = indices[0]
            
            for similarity, idx in zip(similarities, indices):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                # Get document and chunk info
                if idx in self.chunk_mappings:
                    mapped_doc_id, chunk_id = self.chunk_mappings[idx]
                    
                    # Filter by document
                    if mapped_doc_id == doc_id:
                        chunk_text = self.document_chunks[mapped_doc_id][chunk_id]
                        
                        results.append({
                            "chunk_id": chunk_id,
                            "text": chunk_text,
                            "similarity": float(similarity),
                            "document_id": mapped_doc_id
                        })
                        
                        if len(results) >= top_k:
                            break
            
            # Sort by similarity (descending)
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            logger.info(f"🔍 Found {len(results)} relevant chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"❌ Sync search failed: {e}")
            return []
    
    async def delete_document(self, doc_id: str):
        """
        Delete document from vector store
        Note: FAISS doesn't support efficient deletion, so we mark as deleted
        """
        try:
            if doc_id in self.document_chunks:
                # Remove from our mappings
                chunk_ids_to_remove = list(self.document_chunks[doc_id].keys())
                
                # Find and mark global IDs as deleted
                global_ids_to_remove = []
                for global_id, (mapped_doc_id, chunk_id) in self.chunk_mappings.items():
                    if mapped_doc_id == doc_id:
                        global_ids_to_remove.append(global_id)
                
                # Remove from mappings
                for global_id in global_ids_to_remove:
                    del self.chunk_mappings[global_id]
                
                # Remove document chunks
                del self.document_chunks[doc_id]
                
                logger.info(f"🗑️ Deleted {len(chunk_ids_to_remove)} chunks for document {doc_id}")
                
        except Exception as e:
            logger.error(f"❌ Error deleting document {doc_id}: {e}")
            raise
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity using inner product"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms
    
    def is_ready(self) -> bool:
        """Check if vector store is ready for operations"""
        return self.initialized and self.index is not None
    
    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_documents": len(self.document_chunks),
            "dimension": self.dimension,
            "max_capacity": self.max_vectors,
            "memory_usage_mb": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        if not self.index:
            return 0.0
        
        # Rough estimation
        vectors_size = self.index.ntotal * self.dimension * 4  # float32
        metadata_size = len(self.chunk_mappings) * 100  # rough estimate
        chunks_size = sum(
            len(chunk.encode('utf-8')) 
            for doc_chunks in self.document_chunks.values() 
            for chunk in doc_chunks.values()
        )
        
        total_bytes = vectors_size + metadata_size + chunks_size
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.index = None
            self.document_chunks.clear()
            self.chunk_mappings.clear()
            self.next_id = 0
            self.initialized = False
            logger.info("🧹 Vector store cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during vector store cleanup: {e}")