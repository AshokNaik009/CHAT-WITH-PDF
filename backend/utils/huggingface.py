"""
HuggingFace API Client - Free inference API integration
Optimized for production with rate limiting and fallbacks
"""

import asyncio
import logging
import aiohttp
import json
from typing import Optional, Dict, List
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class HuggingFaceClient:
    """
    Production-ready HuggingFace Inference API client
    Uses free tier with intelligent rate limiting and fallbacks
    """
    
    def __init__(self):
        self.api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        self.base_url = "https://api-inference.huggingface.co/models"
        
        # Model configuration (free tier optimized)
        self.primary_model = "microsoft/DialoGPT-medium"
        self.fallback_models = [
            "facebook/blenderbot-400M-distill",
            "microsoft/DialoGPT-small"
        ]
        
        # Rate limiting for free tier
        self.max_requests_per_minute = 30
        self.request_history = []
        
        # Request configuration
        self.timeout = 30
        self.max_retries = 3
        self.retry_delay = 2
        
        # Session management
        self.session = None
        
    async def initialize(self):
        """Initialize HTTP session"""
        if not self.session:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self._get_headers()
            )
    
    async def generate_response(self, question: str, context: str, max_length: int = 512) -> str:
        """
        Generate enhanced response using HuggingFace model
        
        Args:
            question: User question
            context: Relevant document context
            max_length: Maximum response length
            
        Returns:
            Generated response text
        """
        try:
            # Check rate limits
            if not self._check_rate_limit():
                logger.warning("⚠️ Rate limit exceeded, using basic response")
                return self._create_basic_response(question, context)
            
            # Initialize session if needed
            await self.initialize()
            
            # Try primary model first
            response = await self._try_generate(
                self.primary_model, question, context, max_length
            )
            
            if response:
                return response
            
            # Try fallback models
            for model in self.fallback_models:
                response = await self._try_generate(
                    model, question, context, max_length
                )
                if response:
                    return response
            
            # All models failed, return basic response
            logger.warning("⚠️ All HF models failed, using basic response")
            return self._create_basic_response(question, context)
            
        except Exception as e:
            logger.error(f"❌ HuggingFace generation failed: {e}")
            return self._create_basic_response(question, context)
    
    async def _try_generate(self, model: str, question: str, context: str, max_length: int) -> Optional[str]:
        """Try to generate response with specific model"""
        try:
            # Prepare prompt
            prompt = self._create_prompt(question, context)
            
            # Make API request
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": max_length,
                    "temperature": 0.7,
                    "do_sample": True,
                    "top_p": 0.9,
                    "return_full_text": False
                }
            }
            
            url = f"{self.base_url}/{model}"
            
            for attempt in range(self.max_retries):
                try:
                    async with self.session.post(url, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            return self._extract_response(result, question)
                        
                        elif response.status == 503:  # Model loading
                            if attempt < self.max_retries - 1:
                                wait_time = self.retry_delay * (attempt + 1)
                                logger.info(f"⏳ Model loading, waiting {wait_time}s...")
                                await asyncio.sleep(wait_time)
                                continue
                        
                        elif response.status == 429:  # Rate limited
                            logger.warning(f"⚠️ Rate limited for model {model}")
                            return None
                        
                        else:
                            logger.warning(f"⚠️ API error {response.status} for model {model}")
                            return None
                
                except asyncio.TimeoutError:
                    logger.warning(f"⚠️ Timeout for model {model}, attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
                        continue
                    return None
                
                except Exception as e:
                    logger.warning(f"⚠️ Request failed for model {model}: {e}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Model {model} generation failed: {e}")
            return None
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create optimized prompt for the model"""
        # Truncate context if too long
        max_context_length = 800
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        prompt = f"""Based on the following document content, please answer the question.

Document Content:
{context}

Question: {question}

Answer:"""
        
        return prompt
    
    def _extract_response(self, result: Dict, question: str) -> str:
        """Extract and clean response from API result"""
        try:
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                generated_text = result.get("generated_text", "")
            else:
                return ""
            
            # Clean the response
            response = generated_text.strip()
            
            # Remove the original prompt if it's included
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
            
            # Basic quality checks
            if len(response) < 10:
                return ""
            
            if response.lower().startswith(question.lower()[:20]):
                return ""
            
            # Truncate if too long
            if len(response) > 500:
                response = response[:500] + "..."
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Response extraction failed: {e}")
            return ""
    
    def _create_basic_response(self, question: str, context: str) -> str:
        """Create basic response when LLM fails"""
        # Extract the most relevant sentence from context
        sentences = context.split('.')
        
        # Simple keyword matching
        question_words = set(question.lower().split())
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences[:5]:  # Check first 5 sentences
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
                
            sentence_words = set(sentence.lower().split())
            score = len(question_words.intersection(sentence_words))
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        if best_sentence:
            return f"Based on the document: {best_sentence}."
        else:
            return "I found relevant information in the document, but cannot provide a specific answer to your question."
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        
        # Remove old requests
        self.request_history = [
            req_time for req_time in self.request_history 
            if req_time > cutoff
        ]
        
        # Check if we can make another request
        if len(self.request_history) >= self.max_requests_per_minute:
            return False
        
        # Add current request
        self.request_history.append(now)
        return True
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ChatWithPDF/1.0"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    async def health_check(self) -> bool:
        """Check if HuggingFace API is accessible"""
        try:
            await self.initialize()
            
            # Simple test request
            url = f"{self.base_url}/{self.fallback_models[-1]}"
            payload = {"inputs": "Hello", "parameters": {"max_length": 10}}
            
            async with self.session.post(url, json=payload) as response:
                return response.status in [200, 503]  # 503 = model loading
                
        except Exception:
            return False
    
    def get_stats(self) -> Dict:
        """Get client statistics"""
        return {
            "primary_model": self.primary_model,
            "fallback_models": self.fallback_models,
            "requests_last_minute": len(self.request_history),
            "rate_limit": self.max_requests_per_minute,
            "has_api_key": bool(self.api_key),
            "session_active": self.session is not None
        }
    
    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
            self.session = None
        
        self.request_history.clear()
        logger.info("🧹 HuggingFace client cleanup completed")