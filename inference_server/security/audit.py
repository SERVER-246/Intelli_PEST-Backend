"""
Audit Logging
=============
Comprehensive audit logging for security monitoring and compliance.
"""

import json
import logging
import hashlib
import time
from datetime import datetime
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import threading
from queue import Queue
import atexit

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    REQUEST = "request"
    RESPONSE = "response"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    RATE_LIMIT = "rate_limit"
    VALIDATION_ERROR = "validation_error"
    INFERENCE = "inference"
    ERROR = "error"
    SECURITY_ALERT = "security_alert"
    ADMIN_ACTION = "admin_action"


@dataclass
class AuditEvent:
    """Represents an audit event."""
    event_type: AuditEventType
    timestamp: datetime
    request_id: str
    
    # Request info
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    
    # Authentication
    api_key_hash: Optional[str] = None
    api_key_tier: Optional[str] = None
    
    # Details
    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Error info
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["timestamp"] = self.timestamp.isoformat()
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """
    Audit logger with async writing and sensitive data masking.
    """
    
    # Fields to mask in logs
    SENSITIVE_FIELDS = {
        "api_key", "password", "token", "secret", "authorization",
        "x-api-key", "cookie", "session",
    }
    
    def __init__(
        self,
        log_dir: Optional[Path] = None,
        log_to_file: bool = True,
        async_write: bool = True,
        retention_days: int = 90,
    ):
        """
        Initialize audit logger.
        
        Args:
            log_dir: Directory for audit log files
            log_to_file: Whether to write to file
            async_write: Whether to use async writing
            retention_days: Days to retain logs
        """
        self.log_dir = log_dir
        self.log_to_file = log_to_file
        self.async_write = async_write
        self.retention_days = retention_days
        
        self._queue: Queue = Queue() if async_write else None
        self._writer_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Statistics
        self._stats = {
            "total_events": 0,
            "events_by_type": {},
            "auth_failures": 0,
            "rate_limits": 0,
            "errors": 0,
        }
        
        if log_to_file and log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            
        if async_write:
            self._start_writer()
    
    def _start_writer(self):
        """Start async writer thread."""
        self._running = True
        self._writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._writer_thread.start()
        atexit.register(self._stop_writer)
    
    def _stop_writer(self):
        """Stop async writer thread."""
        self._running = False
        if self._queue:
            self._queue.put(None)  # Signal to stop
        if self._writer_thread:
            self._writer_thread.join(timeout=5)
    
    def _write_loop(self):
        """Async write loop."""
        while self._running:
            try:
                event = self._queue.get(timeout=1)
                if event is None:
                    break
                self._write_event(event)
            except Exception:
                continue
    
    def _write_event(self, event: AuditEvent):
        """Write event to file."""
        if not self.log_to_file or not self.log_dir:
            return
        
        try:
            # Daily log file
            date_str = event.timestamp.strftime("%Y-%m-%d")
            log_file = self.log_dir / f"audit_{date_str}.jsonl"
            
            with open(log_file, "a") as f:
                f.write(event.to_json() + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def _mask_sensitive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive fields in data."""
        masked = {}
        for key, value in data.items():
            if key.lower() in self.SENSITIVE_FIELDS:
                if isinstance(value, str) and len(value) > 8:
                    masked[key] = f"{value[:4]}...{value[-4:]}"
                else:
                    masked[key] = "***MASKED***"
            elif isinstance(value, dict):
                masked[key] = self._mask_sensitive(value)
            else:
                masked[key] = value
        return masked
    
    def _update_stats(self, event: AuditEvent):
        """Update internal statistics."""
        self._stats["total_events"] += 1
        
        event_type = event.event_type.value
        self._stats["events_by_type"][event_type] = (
            self._stats["events_by_type"].get(event_type, 0) + 1
        )
        
        if event.event_type == AuditEventType.AUTH_FAILURE:
            self._stats["auth_failures"] += 1
        elif event.event_type == AuditEventType.RATE_LIMIT:
            self._stats["rate_limits"] += 1
        elif event.event_type == AuditEventType.ERROR:
            self._stats["errors"] += 1
    
    def log(
        self,
        event_type: AuditEventType,
        request_id: str,
        **kwargs,
    ) -> AuditEvent:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            request_id: Unique request identifier
            **kwargs: Additional event fields
            
        Returns:
            The created AuditEvent
        """
        # Mask sensitive data in details
        if "details" in kwargs:
            kwargs["details"] = self._mask_sensitive(kwargs["details"])
        
        # Create event
        event = AuditEvent(
            event_type=event_type,
            timestamp=datetime.utcnow(),
            request_id=request_id,
            **kwargs,
        )
        
        # Update statistics
        self._update_stats(event)
        
        # Log to standard logger
        log_level = logging.WARNING if event_type in [
            AuditEventType.AUTH_FAILURE,
            AuditEventType.RATE_LIMIT,
            AuditEventType.SECURITY_ALERT,
            AuditEventType.ERROR,
        ] else logging.INFO
        
        logger.log(log_level, f"AUDIT: {event.to_json()}")
        
        # Write to file
        if self.async_write and self._queue:
            self._queue.put(event)
        elif self.log_to_file:
            self._write_event(event)
        
        return event
    
    def log_request(
        self,
        request_id: str,
        ip_address: str,
        endpoint: str,
        method: str,
        user_agent: Optional[str] = None,
        api_key_hash: Optional[str] = None,
        details: Optional[Dict] = None,
    ) -> AuditEvent:
        """Log an incoming request."""
        return self.log(
            AuditEventType.REQUEST,
            request_id=request_id,
            ip_address=ip_address,
            endpoint=endpoint,
            method=method,
            user_agent=user_agent,
            api_key_hash=api_key_hash[:16] + "..." if api_key_hash else None,
            details=details or {},
        )
    
    def log_response(
        self,
        request_id: str,
        status_code: int,
        response_time_ms: float,
        details: Optional[Dict] = None,
    ) -> AuditEvent:
        """Log a response."""
        return self.log(
            AuditEventType.RESPONSE,
            request_id=request_id,
            status_code=status_code,
            response_time_ms=response_time_ms,
            details=details or {},
        )
    
    def log_auth_failure(
        self,
        request_id: str,
        ip_address: str,
        reason: str,
        details: Optional[Dict] = None,
    ) -> AuditEvent:
        """Log authentication failure."""
        return self.log(
            AuditEventType.AUTH_FAILURE,
            request_id=request_id,
            ip_address=ip_address,
            details={"reason": reason, **(details or {})},
        )
    
    def log_rate_limit(
        self,
        request_id: str,
        ip_address: str,
        api_key_hash: Optional[str] = None,
        retry_after: int = 0,
    ) -> AuditEvent:
        """Log rate limit event."""
        return self.log(
            AuditEventType.RATE_LIMIT,
            request_id=request_id,
            ip_address=ip_address,
            api_key_hash=api_key_hash[:16] + "..." if api_key_hash else None,
            details={"retry_after": retry_after},
        )
    
    def log_security_alert(
        self,
        request_id: str,
        alert_type: str,
        ip_address: str,
        details: Dict,
    ) -> AuditEvent:
        """Log security alert."""
        logger.warning(f"SECURITY ALERT: {alert_type} from {ip_address}")
        return self.log(
            AuditEventType.SECURITY_ALERT,
            request_id=request_id,
            ip_address=ip_address,
            details={"alert_type": alert_type, **details},
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get audit statistics."""
        return self._stats.copy()


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(log_dir: Optional[Path] = None) -> AuditLogger:
    """Get or create the global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(log_dir=log_dir)
    return _audit_logger


def audit_log(event_type: AuditEventType, request_id: str, **kwargs) -> AuditEvent:
    """Log an audit event using the global logger."""
    return get_audit_logger().log(event_type, request_id, **kwargs)
