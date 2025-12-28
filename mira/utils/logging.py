"""
Logging utilities for the MIRA framework.
"""

import logging
import sys
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "mira",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up and configure a logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to log to
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "mira") -> logging.Logger:
    """Get an existing logger or create a new one."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


class ProgressLogger:
    """Simple progress logger for long-running operations."""
    
    def __init__(self, total: int, name: str = "Progress", update_freq: int = 10):
        self.total = total
        self.name = name
        self.update_freq = update_freq
        self.current = 0
        self.start_time = datetime.now()
        self.logger = get_logger("mira.progress")
    
    def update(self, n: int = 1, message: str = "") -> None:
        """Update progress."""
        self.current += n
        
        if self.current % self.update_freq == 0 or self.current == self.total:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0
            
            progress = self.current / self.total * 100
            
            msg = f"{self.name}: {self.current}/{self.total} ({progress:.1f}%)"
            msg += f" - {rate:.1f} it/s - ETA: {eta:.1f}s"
            if message:
                msg += f" - {message}"
            
            self.logger.info(msg)
    
    def finish(self) -> None:
        """Log completion."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(f"{self.name} completed in {elapsed:.1f}s")
