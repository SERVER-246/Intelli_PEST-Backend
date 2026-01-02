"""
Security Headers
================
HTTP security headers for protection against common attacks.
"""

from typing import Dict, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class SecurityHeaders:
    """Security headers configuration."""
    
    # Prevent MIME type sniffing
    x_content_type_options: str = "nosniff"
    
    # Prevent clickjacking
    x_frame_options: str = "DENY"
    
    # XSS protection (legacy, but still useful)
    x_xss_protection: str = "1; mode=block"
    
    # HTTPS enforcement
    strict_transport_security: str = "max-age=31536000; includeSubDomains"
    
    # Content Security Policy
    content_security_policy: str = "default-src 'self'"
    
    # Referrer Policy
    referrer_policy: str = "strict-origin-when-cross-origin"
    
    # Permissions Policy (Feature Policy replacement)
    permissions_policy: str = "geolocation=(), microphone=(), camera=()"
    
    # Cache control for sensitive data
    cache_control: str = "no-store, no-cache, must-revalidate, private"
    
    # Pragma (legacy cache control)
    pragma: str = "no-cache"
    
    # Custom headers to remove
    headers_to_remove: list = field(default_factory=lambda: [
        "Server",
        "X-Powered-By",
        "X-AspNet-Version",
    ])
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for response headers."""
        return {
            "X-Content-Type-Options": self.x_content_type_options,
            "X-Frame-Options": self.x_frame_options,
            "X-XSS-Protection": self.x_xss_protection,
            "Strict-Transport-Security": self.strict_transport_security,
            "Content-Security-Policy": self.content_security_policy,
            "Referrer-Policy": self.referrer_policy,
            "Permissions-Policy": self.permissions_policy,
            "Cache-Control": self.cache_control,
            "Pragma": self.pragma,
        }
    
    @classmethod
    def for_api(cls) -> "SecurityHeaders":
        """Get security headers optimized for API responses."""
        return cls(
            # APIs typically don't need strict CSP
            content_security_policy="default-src 'none'",
            # Allow caching for some API responses
            cache_control="no-store, max-age=0",
        )
    
    @classmethod
    def for_static(cls) -> "SecurityHeaders":
        """Get security headers for static files."""
        return cls(
            # Allow caching for static files
            cache_control="public, max-age=86400",
        )


# Global headers instance
_security_headers: Optional[SecurityHeaders] = None


def get_security_headers(api_mode: bool = True) -> SecurityHeaders:
    """Get the global security headers instance."""
    global _security_headers
    if _security_headers is None:
        _security_headers = SecurityHeaders.for_api() if api_mode else SecurityHeaders()
    return _security_headers


def apply_security_headers(response, headers: Optional[SecurityHeaders] = None):
    """
    Apply security headers to a response object.
    Works with both Flask and generic response objects.
    
    Args:
        response: Response object with headers attribute
        headers: SecurityHeaders instance (uses default if None)
        
    Returns:
        Modified response object
    """
    if headers is None:
        headers = get_security_headers()
    
    # Add security headers
    for header_name, header_value in headers.to_dict().items():
        response.headers[header_name] = header_value
    
    # Remove sensitive headers
    for header_name in headers.headers_to_remove:
        response.headers.pop(header_name, None)
    
    return response
