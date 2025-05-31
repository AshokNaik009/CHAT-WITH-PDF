"""
PDF Processor - Advanced text extraction using PyMuPDF
Extends your existing OCR capabilities for robust PDF processing
"""

import asyncio
import logging
from typing import Optional, Dict
import fitz  # PyMuPDF
from io import BytesIO

logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Advanced PDF text extraction with error handling and optimization
    Building on your existing OCR system
    """
    
    def __init__(self):
        self.max_file_size = 50 * 1024 * 1024  # 50MB limit for Back4app
        self.supported_languages = ["eng", "hin", "ara", "spa"]  # From your existing system
        
    async def extract_text(self, pdf_content: bytes) -> str:
        """
        Extract text from PDF bytes with comprehensive error handling
        
        Args:
            pdf_content: PDF file as bytes
            
        Returns:
            Extracted text content
        """
        try:
            # Validate file size
            if len(pdf_content) > self.max_file_size:
                raise ValueError(f"PDF file too large. Maximum size: {self.max_file_size // (1024*1024)}MB")
            
            # Run CPU-intensive task in thread pool
            text = await asyncio.get_event_loop().run_in_executor(
                None, 
                self._extract_text_sync, 
                pdf_content
            )
            
            if not text.strip():
                raise ValueError("No text content found in PDF")
                
            return text
            
        except Exception as e:
            logger.error(f"❌ PDF extraction failed: {e}")
            raise
    
    def _extract_text_sync(self, pdf_content: bytes) -> str:
        """
        Synchronous text extraction with advanced PyMuPDF features
        Leverages your existing OCR knowledge
        """
        try:
            # Open PDF from bytes
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
            
            if pdf_document.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported")
            
            extracted_text = []
            total_pages = len(pdf_document)
            
            # Limit pages for free tier performance
            max_pages = min(total_pages, 100)
            
            logger.info(f"📄 Processing {max_pages} pages...")
            
            for page_num in range(max_pages):
                try:
                    page = pdf_document[page_num]
                    
                    # Extract text with advanced options
                    text = page.get_text(
                        "text",
                        flags=fitz.TEXTFLAGS_TEXT & ~fitz.TEXT_PRESERVE_LIGATURES
                    )
                    
                    # Clean and validate text
                    cleaned_text = self._clean_text(text)
                    
                    if cleaned_text:
                        # Add page marker for better context
                        extracted_text.append(f"[Page {page_num + 1}]\n{cleaned_text}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error processing page {page_num + 1}: {e}")
                    continue
            
            pdf_document.close()
            
            final_text = "\n\n".join(extracted_text)
            
            # Log extraction statistics
            word_count = len(final_text.split())
            logger.info(f"✅ Extracted {word_count} words from {len(extracted_text)} pages")
            
            return final_text
            
        except Exception as e:
            logger.error(f"❌ Sync extraction failed: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        Enhanced version of your existing text processing
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = []
        for line in text.split('\n'):
            cleaned_line = ' '.join(line.split())
            if cleaned_line and len(cleaned_line) > 2:  # Skip very short lines
                lines.append(cleaned_line)
        
        # Join with single newlines
        cleaned = '\n'.join(lines)
        
        # Remove excessive newlines
        while '\n\n\n' in cleaned:
            cleaned = cleaned.replace('\n\n\n', '\n\n')
        
        return cleaned.strip()
    
    async def get_pdf_metadata(self, pdf_content: bytes) -> Dict:
        """
        Extract PDF metadata for additional context
        """
        try:
            metadata = await asyncio.get_event_loop().run_in_executor(
                None,
                self._get_metadata_sync,
                pdf_content
            )
            return metadata
            
        except Exception as e:
            logger.warning(f"⚠️ Could not extract metadata: {e}")
            return {}
    
    def _get_metadata_sync(self, pdf_content: bytes) -> Dict:
        """Extract PDF metadata synchronously"""
        try:
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
            
            metadata = {
                "page_count": len(pdf_document),
                "title": pdf_document.metadata.get("title", ""),
                "author": pdf_document.metadata.get("author", ""),
                "subject": pdf_document.metadata.get("subject", ""),
                "creator": pdf_document.metadata.get("creator", ""),
                "producer": pdf_document.metadata.get("producer", ""),
                "creation_date": pdf_document.metadata.get("creationDate", ""),
                "modification_date": pdf_document.metadata.get("modDate", ""),
                "is_pdf": True,
                "is_encrypted": pdf_document.is_encrypted
            }
            
            pdf_document.close()
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Metadata extraction failed: {e}")
            return {}
    
    def validate_pdf(self, pdf_content: bytes) -> bool:
        """
        Validate if the file is a proper PDF
        """
        try:
            if len(pdf_content) < 4:
                return False
                
            # Check PDF magic number
            if not pdf_content.startswith(b'%PDF'):
                return False
                
            # Try to open with PyMuPDF
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
            is_valid = len(pdf_document) > 0
            pdf_document.close()
            
            return is_valid
            
        except Exception:
            return False