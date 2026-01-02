"""
Input Sanitization
==================
Secure input validation and sanitization to prevent attacks.
"""

import re
import os
import unicodedata
from typing import Any, Dict, Optional, List, Union
from pathlib import Path, PurePath
import logging
import json

logger = logging.getLogger(__name__)


class SanitizationError(Exception):
    """Raised when sanitization fails due to malicious input."""
    pass


class InputSanitizer:
    """Comprehensive input sanitization for security."""
    
    # Dangerous patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"\.\.",
        r"%2e%2e",
        r"%252e%252e",
        r"\.%00",
        r"%00",
    ]
    
    # Allowed characters for filenames
    SAFE_FILENAME_CHARS = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    
    # Maximum lengths
    MAX_FILENAME_LENGTH = 255
    MAX_PATH_LENGTH = 4096
    MAX_JSON_DEPTH = 10
    MAX_STRING_LENGTH = 10000
    
    def __init__(self):
        self._path_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.PATH_TRAVERSAL_PATTERNS
        ]
    
    def sanitize_filename(
        self,
        filename: str,
        max_length: int = MAX_FILENAME_LENGTH,
        allowed_extensions: Optional[List[str]] = None,
    ) -> str:
        """
        Sanitize a filename to prevent attacks.
        
        Args:
            filename: The filename to sanitize
            max_length: Maximum allowed length
            allowed_extensions: List of allowed extensions (e.g., ['.jpg', '.png'])
            
        Returns:
            Sanitized filename
            
        Raises:
            SanitizationError: If filename is malicious or invalid
        """
        if not filename:
            raise SanitizationError("Empty filename")
        
        # Normalize unicode
        filename = unicodedata.normalize("NFKC", filename)
        
        # Remove null bytes
        filename = filename.replace("\x00", "")
        
        # Get just the filename, not path
        filename = os.path.basename(filename)
        filename = PurePath(filename).name
        
        # Check for path traversal attempts
        for pattern in self._path_patterns:
            if pattern.search(filename):
                logger.warning(f"Path traversal attempt detected: {filename[:50]}")
                raise SanitizationError("Path traversal detected in filename")
        
        # Remove leading dots (hidden files)
        while filename.startswith("."):
            filename = filename[1:]
        
        # Split name and extension
        if "." in filename:
            name, ext = filename.rsplit(".", 1)
            ext = f".{ext.lower()}"
        else:
            name = filename
            ext = ""
        
        # Check extension
        if allowed_extensions and ext:
            if ext.lower() not in [e.lower() for e in allowed_extensions]:
                raise SanitizationError(f"File extension not allowed: {ext}")
        
        # Sanitize name - keep only safe characters
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
        safe_name = re.sub(r"_+", "_", safe_name)  # Collapse multiple underscores
        safe_name = safe_name.strip("_")
        
        if not safe_name:
            safe_name = "file"
        
        # Reconstruct filename
        result = f"{safe_name}{ext}"
        
        # Truncate if too long
        if len(result) > max_length:
            # Keep extension, truncate name
            max_name_len = max_length - len(ext)
            result = f"{safe_name[:max_name_len]}{ext}"
        
        return result
    
    def sanitize_path(
        self,
        path: str,
        base_dir: Optional[Path] = None,
    ) -> Path:
        """
        Sanitize a file path to prevent directory traversal.
        
        Args:
            path: The path to sanitize
            base_dir: Base directory to resolve against (for containment check)
            
        Returns:
            Sanitized Path object
            
        Raises:
            SanitizationError: If path is malicious
        """
        if not path:
            raise SanitizationError("Empty path")
        
        # Normalize unicode
        path = unicodedata.normalize("NFKC", path)
        
        # Remove null bytes
        path = path.replace("\x00", "")
        
        # Check for path traversal patterns
        for pattern in self._path_patterns:
            if pattern.search(path):
                logger.warning(f"Path traversal attempt: {path[:100]}")
                raise SanitizationError("Path traversal detected")
        
        # Convert to Path object
        try:
            sanitized = Path(path)
        except Exception as e:
            raise SanitizationError(f"Invalid path: {e}")
        
        # If base_dir provided, ensure path stays within it
        if base_dir:
            try:
                base_dir = Path(base_dir).resolve()
                full_path = (base_dir / sanitized).resolve()
                
                # Check if resolved path is within base_dir
                if not str(full_path).startswith(str(base_dir)):
                    logger.warning(f"Path escape attempt: {path} -> {full_path}")
                    raise SanitizationError("Path escapes base directory")
                
                return full_path
            except Exception as e:
                raise SanitizationError(f"Path resolution failed: {e}")
        
        return sanitized
    
    def sanitize_json(
        self,
        data: Any,
        max_depth: int = MAX_JSON_DEPTH,
        max_string_length: int = MAX_STRING_LENGTH,
        _current_depth: int = 0,
    ) -> Any:
        """
        Sanitize JSON data to prevent attacks.
        
        Args:
            data: The JSON data to sanitize
            max_depth: Maximum nesting depth
            max_string_length: Maximum string length
            
        Returns:
            Sanitized data
            
        Raises:
            SanitizationError: If data is malicious or too nested
        """
        if _current_depth > max_depth:
            raise SanitizationError(f"JSON nesting too deep (>{max_depth})")
        
        if isinstance(data, dict):
            return {
                self.sanitize_string(str(k), max_string_length): 
                self.sanitize_json(v, max_depth, max_string_length, _current_depth + 1)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [
                self.sanitize_json(item, max_depth, max_string_length, _current_depth + 1)
                for item in data
            ]
        elif isinstance(data, str):
            return self.sanitize_string(data, max_string_length)
        else:
            return data
    
    def sanitize_string(
        self,
        text: str,
        max_length: int = MAX_STRING_LENGTH,
    ) -> str:
        """
        Sanitize a string value.
        
        Args:
            text: The string to sanitize
            max_length: Maximum length
            
        Returns:
            Sanitized string
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)
        
        # Remove null bytes
        text = text.replace("\x00", "")
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]
        
        return text
    
    def validate_content_type(
        self,
        content_type: str,
        allowed_types: List[str],
    ) -> bool:
        """
        Validate Content-Type header.
        
        Args:
            content_type: The Content-Type header value
            allowed_types: List of allowed content types
            
        Returns:
            True if valid
        """
        if not content_type:
            return False
        
        # Extract main type (ignore charset, boundary, etc.)
        main_type = content_type.split(";")[0].strip().lower()
        
        return main_type in [t.lower() for t in allowed_types]
    
    def check_file_magic(self, file_data: bytes) -> Optional[str]:
        """
        Check file magic bytes to determine actual file type.
        
        Args:
            file_data: First bytes of file
            
        Returns:
            Detected MIME type or None
        """
        magic_bytes = {
            b"\xff\xd8\xff": "image/jpeg",
            b"\x89PNG\r\n\x1a\n": "image/png",
            b"RIFF": "image/webp",  # WebP starts with RIFF
            b"GIF87a": "image/gif",
            b"GIF89a": "image/gif",
        }
        
        for magic, mime_type in magic_bytes.items():
            if file_data.startswith(magic):
                return mime_type
        
        # Special check for WebP (RIFF....WEBP)
        if file_data.startswith(b"RIFF") and len(file_data) >= 12:
            if file_data[8:12] == b"WEBP":
                return "image/webp"
        
        return None


# Global sanitizer instance
_sanitizer = InputSanitizer()


def sanitize_filename(
    filename: str,
    allowed_extensions: Optional[List[str]] = None,
) -> str:
    """Sanitize a filename."""
    return _sanitizer.sanitize_filename(filename, allowed_extensions=allowed_extensions)


def sanitize_path(path: str, base_dir: Optional[Path] = None) -> Path:
    """Sanitize a file path."""
    return _sanitizer.sanitize_path(path, base_dir)


def get_sanitizer() -> InputSanitizer:
    """Get the global sanitizer instance."""
    return _sanitizer
