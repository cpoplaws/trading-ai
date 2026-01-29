"""
Logging utility for the trading AI system.
"""
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str = "trading_ai",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
    reset_handlers: bool = False,
    propagate: bool = True,
) -> logging.Logger:
    """
    Set up a logger with both console and rotating file output.

    Args:
        name: Logger name.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional log file path.
        max_bytes: Maximum size in bytes before rotating the log file.
        backup_count: Number of backup files to keep.
        reset_handlers: Whether to clear existing handlers before adding new ones.
        propagate: Whether to propagate logs to parent loggers.

    Returns:
        Configured logger instance.
    """
    level = getattr(logging, log_level.upper())
    if log_file is None:
        log_dir = Path("./logs/")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{datetime.today().strftime('%Y-%m-%d')}.log"
    else:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    trading_logger = logging.getLogger(name)
    trading_logger.setLevel(level)
    trading_logger.propagate = propagate
    if reset_handlers:
        for handler in list(trading_logger.handlers):
            trading_logger.removeHandler(handler)
    elif trading_logger.handlers:
        existing_rotating = next((h for h in trading_logger.handlers if isinstance(h, RotatingFileHandler)), None)
        if existing_rotating:
            same_file = Path(existing_rotating.baseFilename).resolve() == log_path.resolve()
            same_limits = getattr(existing_rotating, "maxBytes", None) == max_bytes and getattr(existing_rotating, "backupCount", None) == backup_count
            if same_file and same_limits and trading_logger.level == level:
                return trading_logger
        for handler in list(trading_logger.handlers):
            trading_logger.removeHandler(handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    trading_logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    trading_logger.addHandler(file_handler)

    return trading_logger

# Default logger setup
default_logger = setup_logger()

if __name__ == "__main__":
    # Test logging
    test_logger = setup_logger("test", "DEBUG", "./logs/test.log")
    test_logger.info("Logger test successful")
