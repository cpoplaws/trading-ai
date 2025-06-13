"""
Logging utility for the trading AI system.
"""
import logging
import os
from datetime import datetime
from typing import Optional
import sys

def setup_logger(name: str = "trading_ai", log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with both console and file output.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    # Create logger
    trading_logger = logging.getLogger(name)
    trading_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    trading_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    trading_logger.addHandler(console_handler)
    
    # File handler (if specified or use default)
    if log_file is None:
        log_dir = './logs/'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{datetime.today().strftime('%Y-%m-%d')}.log")
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    trading_logger.addHandler(file_handler)
    
    return trading_logger

# Default logger setup
default_logger = setup_logger()

if __name__ == "__main__":
    # Test logging
    test_logger = setup_logger("test", "DEBUG", "./logs/test.log")
    test_logger.info("Logger test successful")