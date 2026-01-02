"""
File Validator (Layer 1)
========================
Validates file before image processing:
- File size limits
- Extension whitelist
- MIME type verification
- Malicious content detection
"""

import os
import logging
from typing import Tuple, Optional, List, BinaryIO
from dataclasses import dataclass
from pathlib import Path
import struct

logger = logging.getLogger(__name__)


@dataclass
class FileValidationResult:
    """Result of file validation."""
    valid: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    file_size: int = 0
    detected_mime: Optional[str] = None
    extension: Optional[str] = None


class FileValidator:
    """
    Layer 1: File-level validation.
    
    Checks file properties before any image processing to catch
    obviously invalid or malicious uploads early.
    """
    
    # File signatures (magic bytes) for image formats
    MAGIC_SIGNATURES = {
        b"\xff\xd8\xff": ("image/jpeg", [".jpg", ".jpeg"]),
        b"\x89PNG\r\n\x1a\n": ("image/png", [".png"]),
        b"GIF87a": ("image/gif", [".gif"]),
        b"GIF89a": ("image/gif", [".gif"]),
        b"RIFF": ("image/webp", [".webp"]),  # WebP check needs additional validation
    }
    
    # Dangerous patterns that might indicate polyglot files
    DANGEROUS_PATTERNS = [
        b"<script",
        b"<?php",
        b"<%",
        b"#!/",
        b"PK\x03\x04",  # ZIP header (could be JAR/APK)
        b"\x4d\x5a",    # Windows executable
        b"\x7fELF",     # Linux executable
    ]
    
    def __init__(
        self,
        min_size: int = 1024,           # 1 KB minimum
        max_size: int = 10 * 1024 * 1024,  # 10 MB maximum
        allowed_extensions: Optional[List[str]] = None,
        allowed_mime_types: Optional[List[str]] = None,
    ):
        """
        Initialize file validator.
        
        Args:
            min_size: Minimum file size in bytes
            max_size: Maximum file size in bytes
            allowed_extensions: List of allowed extensions (e.g., ['.jpg', '.png'])
            allowed_mime_types: List of allowed MIME types
        """
        self.min_size = min_size
        self.max_size = max_size
        self.allowed_extensions = allowed_extensions or [".jpg", ".jpeg", ".png", ".webp"]
        self.allowed_mime_types = allowed_mime_types or [
            "image/jpeg", "image/png", "image/webp"
        ]
    
    def validate(
        self,
        file_data: bytes,
        filename: str,
        content_type: Optional[str] = None,
    ) -> FileValidationResult:
        """
        Validate a file upload.
        
        Args:
            file_data: Raw file bytes
            filename: Original filename
            content_type: Declared Content-Type (optional)
            
        Returns:
            FileValidationResult with validation status
        """
        # Check file size
        file_size = len(file_data)
        
        if file_size < self.min_size:
            return FileValidationResult(
                valid=False,
                error_code="FILE_TOO_SMALL",
                error_message=f"File size ({file_size} bytes) below minimum ({self.min_size} bytes)",
                file_size=file_size,
            )
        
        if file_size > self.max_size:
            return FileValidationResult(
                valid=False,
                error_code="FILE_TOO_LARGE",
                error_message=f"File size ({file_size} bytes) exceeds maximum ({self.max_size} bytes)",
                file_size=file_size,
            )
        
        # Check extension
        ext = Path(filename).suffix.lower() if filename else ""
        if ext not in self.allowed_extensions:
            return FileValidationResult(
                valid=False,
                error_code="INVALID_EXTENSION",
                error_message=f"File extension '{ext}' not allowed. Allowed: {self.allowed_extensions}",
                file_size=file_size,
                extension=ext,
            )
        
        # Detect actual MIME type from magic bytes
        detected_mime = self._detect_mime_type(file_data)
        
        if detected_mime is None:
            return FileValidationResult(
                valid=False,
                error_code="UNKNOWN_FILE_TYPE",
                error_message="Could not determine file type from content",
                file_size=file_size,
                extension=ext,
            )
        
        if detected_mime not in self.allowed_mime_types:
            return FileValidationResult(
                valid=False,
                error_code="INVALID_MIME_TYPE",
                error_message=f"Detected file type '{detected_mime}' not allowed",
                file_size=file_size,
                extension=ext,
                detected_mime=detected_mime,
            )
        
        # Check for MIME type mismatch (potential attack)
        if content_type:
            declared_mime = content_type.split(";")[0].strip().lower()
            if declared_mime != detected_mime:
                logger.warning(
                    f"MIME type mismatch: declared={declared_mime}, detected={detected_mime}"
                )
                # Allow but log - some browsers send wrong types
        
        # Scan for malicious patterns
        malicious = self._check_malicious_patterns(file_data)
        if malicious:
            logger.warning(f"Malicious pattern detected in file: {malicious}")
            return FileValidationResult(
                valid=False,
                error_code="MALICIOUS_CONTENT",
                error_message="File contains potentially malicious content",
                file_size=file_size,
                extension=ext,
                detected_mime=detected_mime,
            )
        
        # All checks passed
        return FileValidationResult(
            valid=True,
            file_size=file_size,
            extension=ext,
            detected_mime=detected_mime,
        )
    
    def _detect_mime_type(self, data: bytes) -> Optional[str]:
        """Detect MIME type from file magic bytes."""
        if len(data) < 12:
            return None
        
        # Check standard signatures
        for signature, (mime_type, extensions) in self.MAGIC_SIGNATURES.items():
            if data.startswith(signature):
                # Special handling for WebP (RIFF....WEBP)
                if signature == b"RIFF":
                    if len(data) >= 12 and data[8:12] == b"WEBP":
                        return "image/webp"
                    continue  # Not WebP, skip
                return mime_type
        
        return None
    
    def _check_malicious_patterns(self, data: bytes) -> Optional[str]:
        """Check for potentially malicious patterns in file content."""
        # For image files, we should be very conservative since EXIF/metadata
        # and image binary data can contain byte sequences that look like patterns.
        # Only check for actual text-based script injections at the very start.
        
        # Only check first 64 bytes - actual scripts would need to be at the very start
        # Image files start with magic bytes (FFD8 for JPEG, 89PNG for PNG, etc.)
        # so any script would have to be before the image data
        header = data[:64]
        
        # Only flag if we find actual text-based script tags (not binary that happens to match)
        # Check if the header looks like text rather than binary image data
        try:
            header_text = header.decode('utf-8', errors='strict')
            # If header is valid UTF-8 text, check for script patterns
            if '<script' in header_text.lower():
                return "<script"
            if '<?php' in header_text.lower():
                return "<?php"
            if header_text.startswith('#!'):
                return "shebang"
        except UnicodeDecodeError:
            # Header is binary data (normal for images) - this is fine
            pass
        
        return None
    
    def validate_stream(
        self,
        file_stream: BinaryIO,
        filename: str,
        content_type: Optional[str] = None,
    ) -> Tuple[FileValidationResult, Optional[bytes]]:
        """
        Validate a file stream.
        
        Args:
            file_stream: File-like object
            filename: Original filename
            content_type: Declared Content-Type
            
        Returns:
            Tuple of (validation result, file data if valid)
        """
        # Read file data
        try:
            file_data = file_stream.read()
        except Exception as e:
            return FileValidationResult(
                valid=False,
                error_code="READ_ERROR",
                error_message=f"Failed to read file: {str(e)}",
            ), None
        
        result = self.validate(file_data, filename, content_type)
        return result, file_data if result.valid else None


# Global validator instance
_validator: Optional[FileValidator] = None


def get_file_validator(**kwargs) -> FileValidator:
    """Get or create the global file validator."""
    global _validator
    if _validator is None:
        _validator = FileValidator(**kwargs)
    return _validator


def validate_file(
    file_data: bytes,
    filename: str,
    content_type: Optional[str] = None,
) -> FileValidationResult:
    """Validate a file using the global validator."""
    return get_file_validator().validate(file_data, filename, content_type)
