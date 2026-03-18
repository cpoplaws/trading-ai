"""
Log Aggregator - Centralized log collection and forwarding

Features:
- Centralized log collection to external services
- Structured log format (JSON)
- Log filtering and routing
- Error alerting
- Log retention policies
"""

import logging
import json
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
import threading
import queue
import time
from enum import Enum

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """Structured log entry."""
    level: str
    message: str
    service: str
    timestamp: str
    logger_name: str
    function: Optional[str] = None
    line_number: Optional[int] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = None
    traceback: Optional[str] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary."""
        return {
            "level": self.level,
            "message": self.message,
            "service": self.service,
            "timestamp": self.timestamp,
            "logger": self.logger_name,
            "function": self.function,
            "line_number": self.line_number,
            "file_path": self.file_path,
            "metadata": self.metadata,
            "traceback": self.traceback,
            "request_id": self.request_id,
            "user_id": self.user_id
        }

    def to_json(self) -> str:
        """Convert log entry to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class LogBackend:
    """Abstract base class for log backends."""

    def send(self, entry: LogEntry) -> bool:
        """Send log entry to backend."""
        raise NotImplementedError


class ConsoleBackend(LogBackend):
    """Console output backend for development."""

    def __init__(self, color_output: bool = True):
        self.color_output = color_output
        self.colors = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
            "RESET": "\033[0m",
        }

    def send(self, entry: LogEntry) -> bool:
        """Print log entry to console."""
        try:
            timestamp = entry.timestamp[:19]  # Truncate milliseconds
            level = entry.level

            if self.color_output and sys.stdout.isatty():
                color = self.colors.get(level, "")
                reset = self.colors["RESET"]
                colored_level = f"{color}{level}{reset}"
            else:
                colored_level = level

            log_line = f"[{timestamp}] {colored_level:8} [{entry.service}] {entry.message}"

            if entry.traceback:
                log_line += f"\n{entry.traceback}"

            if entry.metadata:
                log_line += f" | metadata: {json.dumps(entry.metadata, default=str)}"

            print(log_line)
            return True
        except Exception as e:
            print(f"Failed to log to console: {e}", file=sys.stderr)
            return False


class FileBackend(LogBackend):
    """File output backend."""

    def __init__(self, filepath: str, max_size_mb: int = 100, backup_count: int = 5):
        self.filepath = filepath
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.backup_count = backup_count
        self._lock = threading.Lock()

        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    def send(self, entry: LogEntry) -> bool:
        """Write log entry to file."""
        try:
            with self._lock:
                # Check rotation
                if self._should_rotate():
                    self._rotate()

                # Write entry
                with open(self.filepath, 'a', encoding='utf-8') as f:
                    f.write(entry.to_json() + '\n')

            return True
        except Exception as e:
            print(f"Failed to log to file: {e}", file=sys.stderr)
            return False

    def _should_rotate(self) -> bool:
        """Check if file should be rotated."""
        try:
            return os.path.getsize(self.filepath) >= self.max_size_bytes
        except OSError:
            return False

    def _rotate(self) -> None:
        """Rotate log files."""
        import os
        import shutil

        # Rotate existing backups
        for i in range(self.backup_count - 1, 0, -1):
            old_file = f"{self.filepath}.{i}"
            new_file = f"{self.filepath}.{i + 1}"
            if os.path.exists(old_file):
                shutil.move(old_file, new_file)

        # Move current file to .1
        if os.path.exists(self.filepath):
            shutil.move(self.filepath, f"{self.filepath}.1")


class HTTPBackend(LogBackend):
    """HTTP-based log backend (e.g., ELK, Splunk, Loggly)."""

    def __init__(self, endpoint: str, api_key: str = None, timeout: float = 1.0):
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.session = None

        if api_key:
            self.headers = {
                "X-API-Key": api_key,
                "Content-Type": "application/json"
            }
        else:
            self.headers = {
                "Content-Type": "application/json"
            }

    def _get_session(self):
        """Lazy session initialization."""
        if self.session is None:
            try:
                import requests
                self.session = requests.Session()
                self.session.headers.update(self.headers)
            except ImportError:
                logger.warning("requests not installed, HTTP logging disabled")
        return self.session

    def send(self, entry: LogEntry) -> bool:
        """Send log entry via HTTP POST."""
        session = self._get_session()
        if session is None:
            return False

        try:
            response = session.post(
                self.endpoint,
                json=entry.to_dict(),
                timeout=self.timeout
            )
            return response.status_code < 400
        except Exception:
            # Don't fail if logging fails
            return False


class SlackBackend(LogBackend):
    """Slack webhook backend for critical alerts."""

    def __init__(self, webhook_url: str, channel: str = "#alerts", username: str = "Trading AI"):
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username
        self.session = None

    def _get_session(self):
        """Lazy session initialization."""
        if self.session is None:
            try:
                import requests
                self.session = requests.Session()
            except ImportError:
                logger.warning("requests not installed, Slack logging disabled")
        return self.session

    def send(self, entry: LogEntry) -> bool:
        """Send alert to Slack webhook."""
        session = self._get_session()
        if session is None:
            return False

        # Only send ERROR and CRITICAL to Slack
        if entry.level not in ["ERROR", "CRITICAL"]:
            return True

        try:
            # Format message
            color = {
                "ERROR": "#ff0000",
                "CRITICAL": "#ff00ff"
            }.get(entry.level, "#ffff00")

            attachment = {
                "color": color,
                "fields": [
                    {"title": "Level", "value": entry.level, "short": True},
                    {"title": "Service", "value": entry.service, "short": True},
                    {"title": "Timestamp", "value": entry.timestamp, "short": True},
                    {"title": "Logger", "value": entry.logger_name, "short": True},
                ],
                "text": entry.message
            }

            if entry.traceback:
                attachment["fields"].append({
                    "title": "Traceback",
                    "value": f"```{entry.traceback[:500]}```",
                    "short": False
                })

            payload = {
                "channel": self.channel,
                "username": self.username,
                "attachments": [attachment]
            }

            response = session.post(self.webhook_url, json=payload, timeout=5)
            return response.status_code < 400
        except Exception:
            # Don't fail if Slack is unavailable
            return False


class LogAggregator:
    """
    Centralized log aggregator with multiple backends.

    Supports:
    - Multiple backends (console, file, HTTP, Slack)
    - Async log delivery
    - Log filtering and routing
    - Structured log format
    - Error alerting
    """

    def __init__(
        self,
        service_name: str = "trading-ai",
        backends: Optional[List[LogBackend]] = None,
        min_level: LogLevel = LogLevel.INFO,
        enable_async: bool = True
    ):
        """
        Initialize log aggregator.

        Args:
            service_name: Name of the service
            backends: List of log backends (defaults to console)
            min_level: Minimum log level to forward
            enable_async: Enable async log delivery
        """
        self.service_name = service_name
        self.backends = backends or [ConsoleBackend()]
        self.min_level = min_level
        self.enable_async = enable_async

        # Async queue
        self._queue = queue.Queue(maxsize=10000)
        self._running = False
        self._worker_thread = None

        # Start worker thread if async enabled
        if enable_async:
            self._start_worker()

    def _start_worker(self) -> None:
        """Start background worker thread."""
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Worker thread for async log delivery."""
        while self._running:
            try:
                entry = self._queue.get(timeout=0.1)
                if entry is None:
                    continue

                for backend in self.backends:
                    backend.send(entry)

                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Log worker error: {e}", file=sys.stderr)

    def log(
        self,
        level: LogLevel,
        message: str,
        logger_name: str = "",
        function: str = None,
        line_number: int = None,
        file_path: str = None,
        metadata: Dict[str, Any] = None,
        traceback: str = None,
        request_id: str = None,
        user_id: str = None
    ) -> None:
        """
        Log a message.

        Args:
            level: Log level
            message: Log message
            logger_name: Logger name
            function: Function name where log was called
            line_number: Line number
            file_path: File path
            metadata: Additional metadata
            traceback: Exception traceback
            request_id: Request ID for tracing
            user_id: User ID for tracing
        """
        # Filter by minimum level
        level_priority = {
            LogLevel.DEBUG: 10,
            LogLevel.INFO: 20,
            LogLevel.WARNING: 30,
            LogLevel.ERROR: 40,
            LogLevel.CRITICAL: 50
        }

        if level_priority.get(level, 0) < level_priority.get(self.min_level, 0):
            return

        # Create log entry
        entry = LogEntry(
            level=level.value,
            message=message,
            service=self.service_name,
            timestamp=datetime.utcnow().isoformat() + "Z",
            logger_name=logger_name,
            function=function,
            line_number=line_number,
            file_path=file_path,
            metadata=metadata or {},
            traceback=traceback,
            request_id=request_id,
            user_id=user_id
        )

        # Send to backends
        if self.enable_async:
            try:
                self._queue.put_nowait(entry)
            except queue.Full:
                # Drop logs if queue full
                print("Log queue full, dropping log entry", file=sys.stderr)
        else:
            for backend in self.backends:
                backend.send(entry)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.log(LogLevel.CRITICAL, message, **kwargs)

    def exception(self, message: str, exc: Exception, **kwargs) -> None:
        """Log exception with traceback."""
        import traceback as tb_module

        kwargs["traceback"] = "".join(tb_module.format_exception(
            type(exc), exc, exc.__traceback__
        ))
        self.log(LogLevel.ERROR, message, **kwargs)

    def close(self) -> None:
        """Close log aggregator and flush queue."""
        self._running = False

        if self._worker_thread:
            # Wait for queue to empty
            self._queue.join()
            self._worker_thread.join(timeout=5)


# Global instance
_aggregator: Optional[LogAggregator] = None


def get_aggregator() -> LogAggregator:
    """Get or create global log aggregator instance."""
    global _aggregator

    if _aggregator is None:
        # Initialize from environment variables
        service_name = os.getenv('SERVICE_NAME', 'trading-ai')
        backends = [ConsoleBackend()]

        # Add file backend if configured
        log_file = os.getenv('LOG_FILE')
        if log_file:
            backends.append(FileBackend(log_file))

        # Add HTTP backend if configured
        log_endpoint = os.getenv('LOG_ENDPOINT')
        log_api_key = os.getenv('LOG_API_KEY')
        if log_endpoint:
            backends.append(HTTPBackend(log_endpoint, log_api_key))

        # Add Slack backend for alerts if configured
        slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
        if slack_webhook:
            backends.append(SlackBackend(slack_webhook))

        # Determine minimum level
        min_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
        min_level = LogLevel[min_level_str]

        _aggregator = LogAggregator(
            service_name=service_name,
            backends=backends,
            min_level=min_level
        )

    return _aggregator


def configure_logging(
    service_name: str = "trading-ai",
    log_file: str = None,
    log_endpoint: str = None,
    log_api_key: str = None,
    slack_webhook: str = None,
    log_level: str = "INFO"
) -> LogAggregator:
    """
    Configure centralized logging.

    Args:
        service_name: Name of the service
        log_file: Path to log file (optional)
        log_endpoint: HTTP endpoint for logs (optional)
        log_api_key: API key for log endpoint (optional)
        slack_webhook: Slack webhook URL for alerts (optional)
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured LogAggregator instance
    """
    global _aggregator

    backends = [ConsoleBackend()]

    # Add configured backends
    if log_file:
        backends.append(FileBackend(log_file))

    if log_endpoint:
        backends.append(HTTPBackend(log_endpoint, log_api_key))

    if slack_webhook:
        backends.append(SlackBackend(slack_webhook))

    min_level = LogLevel[log_level.upper()]

    _aggregator = LogAggregator(
        service_name=service_name,
        backends=backends,
        min_level=min_level
    )

    return _aggregator


# Example usage
if __name__ == "__main__":
    import os

    # Configure logging
    log = configure_logging(
        service_name="trading-ai",
        log_file="logs/trading.jsonl",
        log_endpoint=os.getenv("LOG_ENDPOINT"),
        slack_webhook=os.getenv("SLACK_WEBHOOK_URL"),
        log_level="INFO"
    )

    # Test logging
    log.debug("This is a debug message")
    log.info("Application started successfully")
    log.warning("Configuration warning: using default values")
    log.error("Connection failed to exchange API", metadata={"exchange": "binance", "retry": 3})
    log.critical("CRITICAL: Trading operations halted due to error")

    # Test exception logging
    try:
        raise ValueError("Test exception")
    except Exception as e:
        log.exception("An error occurred", exc=e)

    # Close aggregator
    log.close()
