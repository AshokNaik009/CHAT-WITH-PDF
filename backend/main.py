"""
Chat with PDF - FastAPI Backend
Production-ready implementation for Back4app Container deployment
"""

import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules (now should work)
try:
    from api.upload import upload_router
    from api.chat import chat_router
    from models.chat_engine import ChatEngine
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all __init__.py files exist and code files are in correct directories")
    sys.exit(1)
    

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global chat engine instance
chat_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events
    """
    global chat_engine
    
    # Startup
    logger.info("🚀 Starting Chat with PDF service...")
    try:
        chat_engine = ChatEngine()
        await chat_engine.initialize()
        logger.info("✅ Chat engine initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize chat engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Chat with PDF service...")
    if chat_engine:
        await chat_engine.cleanup()

# Create FastAPI app with lifespan management
app = FastAPI(
    title="Chat with PDF API",
    description="Semantic search and chat interface for PDF documents",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload_router, prefix="/api/v1", tags=["upload"])
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Chat with PDF API is running!",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check for monitoring"""
    try:
        global chat_engine
        if chat_engine and await chat_engine.health_check():
            return {
                "status": "healthy",
                "services": {
                    "chat_engine": "operational",
                    "embeddings": "loaded",
                    "vector_store": "ready"
                }
            }
        else:
            raise HTTPException(status_code=503, detail="Chat engine not ready")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for production error management"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": "An unexpected error occurred"
        }
    )

# Make chat_engine available to other modules
def get_chat_engine():
    global chat_engine
    if chat_engine is None:
        raise HTTPException(status_code=503, detail="Chat engine not initialized")
    return chat_engine

if __name__ == "__main__":
    # Back4app Container configuration
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🌐 Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        workers=1,     # Single worker for Back4app free tier
        log_level="info"
    )