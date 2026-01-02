"""
Logging Configuration
=====================
Centralized logging setup for the inference server.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record):
        # Add color to level name
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    format_string: Optional[str] = None,
    colored: bool = True,
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Specific log file path
        log_dir: Directory for log files (auto-named by date)
        format_string: Custom format string
        colored: Use colored console output
        
    Returns:
        Root logger
    """
    # Get log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    if colored and sys.stdout.isatty():
        console_formatter = ColoredFormatter(format_string)
    else:
        console_formatter = logging.Formatter(format_string)
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file or log_dir:
        if log_file:
            file_path = Path(log_file)
        else:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            date_str = datetime.now().strftime("%Y-%m-%d")
            file_path = log_dir / f"inference_server_{date_str}.log"
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        root_logger.info(f"Logging to file: {file_path}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class RequestLogger:
    """
    Logger for API requests with structured output.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize request logger.
        
        Args:
            logger: Logger instance to use
        """
        self.logger = logger or logging.getLogger("request")
    
    def log_request(
        self,
        request_id: str,
        method: str,
        path: str,
        ip: str,
        user_agent: Optional[str] = None,
    ):
        """Log incoming request."""
        self.logger.info(
            f"REQUEST {request_id} | {method} {path} | IP: {ip} | UA: {user_agent or 'N/A'}"
        )
    
    def log_response(
        self,
        request_id: str,
        status_code: int,
        response_time_ms: float,
    ):
        """Log outgoing response."""
        level = logging.INFO if status_code < 400 else logging.WARNING
        self.logger.log(
            level,
            f"RESPONSE {request_id} | Status: {status_code} | Time: {response_time_ms:.2f}ms"
        )
    
    def log_error(
        self,
        request_id: str,
        error_type: str,
        error_message: str,
    ):
        """Log error."""
        self.logger.error(
            f"ERROR {request_id} | {error_type}: {error_message}"
        )
    
    def log_inference(
        self,
        request_id: str,
        model: str,
        inference_time_ms: float,
        predicted_class: str,
        confidence: float,
    ):
        """Log inference result."""
        self.logger.info(
            f"INFERENCE {request_id} | Model: {model} | "
            f"Time: {inference_time_ms:.2f}ms | "
            f"Prediction: {predicted_class} ({confidence:.2%})"
        )


# Global request logger
_request_logger: Optional[RequestLogger] = None


def get_request_logger() -> RequestLogger:
    """Get the global request logger."""
    global _request_logger
    if _request_logger is None:
        _request_logger = RequestLogger()
    return _request_logger
