"""
Embedding Manager - Handles text embeddings using sentence-transformers
Optimized for production deployment with caching and memory management
"""

import asyncio
import logging
import numpy as np
from typing import List, Optional, Dict, Union
import torch
from sentence_transformers import SentenceTransformer
import os
from functools import lru_cache
import hashlib
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Production-ready embedding manager with caching and optimization
    Uses sentence-transformers for high-quality semantic embeddings
    """
    
    def __init__(self):
        # Model configuration
        self.model_name = "all-MiniLM-L6-v2"  # Best balance of speed/quality
        self.model = None
        self.device = None
        self.max_seq_length = 256  # Optimize for speed
        self.dimension = 384       # Model output dimension
        
        # Performance settings for Back4app containers
        self.batch_size = 32       # Optimal for 256MB RAM limit
        self.max_cache_size = 1000 # Limit memory usage
        self.enable_caching = True
        
        # Cache for embeddings (memory-based for containers)
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Status tracking
        self.initialized = False
        self.model_loading = False
        
    async def initialize(self):
        """
        Initialize the embedding model asynchronously
        Downloads model on first run, cached afterwards
        """
        if self.initialized:
            logger.info("📊 Embedding manager already initialized")
            return
        
        if self.model_loading:
            logger.info("⏳ Waiting for model to load...")
            while self.model_loading:
                await asyncio.sleep(0.1)
            return
        
        try:
            self.model_loading = True
            logger.info(f"🤖 Initializing embedding model: {self.model_name}")
            
            # Load model in thread pool to avoid blocking
            model = await asyncio.get_event_loop().run_in_executor(
                None,
                self._load_model_sync
            )
            
            self.model = model
            self.initialized = True
            self.model_loading = False
            
            logger.info(f"✅ Embedding model loaded successfully")
            logger.info(f"📊 Model info: {self.dimension}D embeddings, max_seq_length: {self.max_seq_length}")
            
        except Exception as e:
            self.model_loading = False
            logger.error(f"❌ Failed to initialize embedding model: {e}")
            raise
    
    def _load_model_sync(self) -> SentenceTransformer:
        """
        Synchronously load the sentence transformer model
        Optimized for container deployment
        """
        try:
            # Set device (CPU for containers)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🔧 Using device: {self.device}")
            
            # Load model with optimizations
            model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=os.getenv("MODEL_CACHE_DIR", "/tmp/sentence_transformers")
            )
            
            # Optimize for inference
            model.eval()
            
            # Set max sequence length for efficiency
            if hasattr(model, 'max_seq_length'):
                model.max_seq_length = self.max_seq_length
            
            # Disable gradients for inference
            for param in model.parameters():
                param.requires_grad = False
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise
    
    async def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            NumPy array of embeddings with shape (len(texts), dimension)
        """
        if not self.initialized:
            await self.initialize()
        
        if not texts:
            return np.array([]).reshape(0, self.dimension)
        
        try:
            logger.info(f"🧮 Encoding {len(texts)} texts...")
            
            # Check cache first
            cached_embeddings, uncached_texts, cache_indices = self._check_cache(texts)
            
            if uncached_texts:
                # Generate embeddings for uncached texts
                new_embeddings = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._encode_batch_sync,
                    uncached_texts
                )
                
                # Update cache
                self._update_cache(uncached_texts, new_embeddings)
            else:
                new_embeddings = np.array([]).reshape(0, self.dimension)
            
            # Combine cached and new embeddings
            final_embeddings = self._combine_embeddings(
                cached_embeddings, new_embeddings, cache_indices
            )
            
            logger.info(f"✅ Generated {len(final_embeddings)} embeddings")
            logger.info(f"📊 Cache stats - Hits: {self.cache_hits}, Misses: {self.cache_misses}")
            
            return final_embeddings
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            raise
    
    def _encode_batch_sync(self, texts: List[str]) -> np.ndarray:
        """
        Synchronously encode texts in batches for memory efficiency
        
        Args:
            texts: List of texts to encode
            
        Returns:
            NumPy array of embeddings
        """
        try:
            if not texts:
                return np.array([]).reshape(0, self.dimension)
            
            # Preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Encode in batches to manage memory
            all_embeddings = []
            
            for i in range(0, len(processed_texts), self.batch_size):
                batch = processed_texts[i:i + self.batch_size]
                
                # Generate embeddings
                with torch.no_grad():
                    batch_embeddings = self.model.encode(
                        batch,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=len(batch)
                    )
                
                all_embeddings.append(batch_embeddings)
            
            # Concatenate all batches
            final_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([]).reshape(0, self.dimension)
            
            # Validate output shape
            if final_embeddings.shape[1] != self.dimension:
                raise ValueError(f"Unexpected embedding dimension: {final_embeddings.shape[1]} vs {self.dimension}")
            
            return final_embeddings
            
        except Exception as e:
            logger.error(f"❌ Batch encoding failed: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for optimal embedding generation
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Truncate if too long (leave room for special tokens)
        if len(text) > (self.max_seq_length - 10) * 4:  # Rough character to token ratio
            text = text[:self.max_seq_length * 4]
            logger.debug(f"📏 Truncated long text to {len(text)} characters")
        
        return text
    
    def _check_cache(self, texts: List[str]) -> tuple:
        """
        Check cache for existing embeddings
        
        Args:
            texts: List of texts to check
            
        Returns:
            Tuple of (cached_embeddings, uncached_texts, cache_indices)
        """
        if not self.enable_caching:
            return {}, texts, list(range(len(texts)))
        
        cached_embeddings = {}
        uncached_texts = []
        cache_indices = []
        
        for i, text in enumerate(texts):
            text_hash = self._get_text_hash(text)
            
            if text_hash in self.embedding_cache:
                cached_embeddings[i] = self.embedding_cache[text_hash]
                self.cache_hits += 1
            else:
                uncached_texts.append(text)
                cache_indices.append(i)
                self.cache_misses += 1
        
        return cached_embeddings, uncached_texts, cache_indices
    
    def _update_cache(self, texts: List[str], embeddings: np.ndarray):
        """
        Update embedding cache with new embeddings
        
        Args:
            texts: List of texts
            embeddings: Corresponding embeddings
        """
        if not self.enable_caching or len(texts) != len(embeddings):
            return
        
        # Implement LRU-style cache management
        if len(self.embedding_cache) >= self.max_cache_size:
            # Remove oldest entries (simple FIFO for now)
            keys_to_remove = list(self.embedding_cache.keys())[:len(texts)]
            for key in keys_to_remove:
                del self.embedding_cache[key]
        
        # Add new embeddings to cache
        for text, embedding in zip(texts, embeddings):
            text_hash = self._get_text_hash(text)
            self.embedding_cache[text_hash] = embedding
    
    def _combine_embeddings(self, cached: Dict[int, np.ndarray], new: np.ndarray, cache_indices: List[int]) -> np.ndarray:
        """
        Combine cached and newly generated embeddings in correct order
        
        Args:
            cached: Dictionary of cached embeddings by index
            new: Array of new embeddings
            cache_indices: Indices of texts that needed new embeddings
            
        Returns:
            Combined embeddings array
        """
        total_count = len(cached) + len(cache_indices)
        result = np.zeros((total_count, self.dimension))
        
        # Fill cached embeddings
        for idx, embedding in cached.items():
            result[idx] = embedding
        
        # Fill new embeddings
        for i, idx in enumerate(cache_indices):
            result[idx] = new[i]
        
        return result
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text caching"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    async def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text (convenience method)
        
        Args:
            text: Text to encode
            
        Returns:
            Single embedding as 1D numpy array
        """
        embeddings = await self.encode_texts([text])
        return embeddings[0] if len(embeddings) > 0 else np.zeros(self.dimension)
    
    def get_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"❌ Similarity calculation failed: {e}")
            return 0.0
    
    def is_ready(self) -> bool:
        """Check if embedding manager is ready for use"""
        return self.initialized and self.model is not None
    
    def get_stats(self) -> Dict:
        """Get embedding manager statistics"""
        return {
            "initialized": self.initialized,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "cache_size": len(self.embedding_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            "max_seq_length": self.max_seq_length,
            "batch_size": self.batch_size
        }
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.embedding_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("🧹 Embedding cache cleared")
    
    async def warmup(self, sample_texts: Optional[List[str]] = None):
        """
        Warm up the model with sample texts for faster first requests
        
        Args:
            sample_texts: Optional list of texts to use for warmup
        """
        try:
            if not self.initialized:
                await self.initialize()
            
            warmup_texts = sample_texts or [
                "This is a sample document for testing purposes.",
                "Machine learning and artificial intelligence are transforming industries.",
                "Natural language processing enables computers to understand human text."
            ]
            
            logger.info("🔥 Warming up embedding model...")
            
            # Generate embeddings for warmup (discard results)
            await self.encode_texts(warmup_texts)
            
            logger.info("✅ Model warmup completed")
            
        except Exception as e:
            logger.warning(f"⚠️ Model warmup failed: {e}")
    
    async def cleanup(self):
        """Clean up resources and memory"""
        try:
            # Clear cache
            self.clear_cache()
            
            # Clear model from memory
            if self.model is not None:
                del self.model
                self.model = None
            
            # Clear torch cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.initialized = False
            logger.info("🧹 Embedding manager cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during embedding manager cleanup: {e}")
    
    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """
        Validate that an embedding has the correct format
        
        Args:
            embedding: Embedding to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not isinstance(embedding, np.ndarray):
                return False
            
            if embedding.shape != (self.dimension,):
                return False
            
            if not np.isfinite(embedding).all():
                return False
            
            return True
            
        except Exception:
            return False
    
    async def get_embedding_quality_score(self, text: str) -> float:
        """
        Get a quality score for text embedding (experimental)
        Based on text length and content diversity
        
        Args:
            text: Input text
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            if not text or len(text.strip()) < 10:
                return 0.1
            
            # Basic quality metrics
            word_count = len(text.split())
            char_count = len(text)
            unique_words = len(set(text.lower().split()))
            
            # Calculate diversity score
            diversity_score = unique_words / max(word_count, 1)
            
            # Length score (optimal around 50-200 words)
            optimal_length = 100
            length_score = 1.0 - abs(word_count - optimal_length) / optimal_length
            length_score = max(0.2, min(1.0, length_score))
            
            # Character to word ratio (detect very short or very long words)
            avg_word_length = char_count / max(word_count, 1)
            word_length_score = 1.0 if 3 <= avg_word_length <= 8 else 0.7
            
            # Combined score
            quality_score = (diversity_score * 0.4 + length_score * 0.4 + word_length_score * 0.2)
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.warning(f"⚠️ Quality score calculation failed: {e}")
            return 0.5  # Default moderate quality