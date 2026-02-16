"""
Structured Logging

Provides JSON-formatted logging for centralized aggregation and analysis.
"""
import logging
import json
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import os


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs logs in JSON format for easy parsing by log aggregation tools.
    """

    def __init__(
        self,
        service_name: str = "trading-ai",
        environment: str = "development",
        include_trace: bool = True
    ):
        """
        Initialize structured formatter.

        Args:
            service_name: Name of the service
            environment: Environment (development, staging, production)
            include_trace: Include stack trace for errors
        """
        super().__init__()
        self.service_name = service_name
        self.environment = environment
        self.include_trace = include_trace
        self.hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record

        Returns:
            JSON-formatted log string
        """
        # Base log structure
        log_data = {
            # Timestamp
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'time_ms': int(record.created * 1000),

            # Log level and message
            'level': record.levelname,
            'message': record.getMessage(),

            # Service identification
            'service': self.service_name,
            'environment': self.environment,
            'hostname': self.hostname,

            # Source information
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': record.process,
            'process_name': record.processName
        }

        # Add exception info if present
        if record.exc_info and self.include_trace:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        # Add custom fields from extra
        if hasattr(record, 'extra_fields'):
            log_data['extra'] = record.extra_fields

        # Add common custom fields
        for attr in ['agent_id', 'trade_id', 'order_id', 'symbol', 'user_id']:
            if hasattr(record, attr):
                log_data[attr] = getattr(record, attr)

        return json.dumps(log_data)


class StructuredLogger:
    """
    Enhanced logger with structured logging support.

    Provides convenient methods for logging with additional context.
    """

    def __init__(
        self,
        name: str,
        service_name: str = "trading-ai",
        environment: str = "development",
        level: int = logging.INFO
    ):
        """
        Initialize structured logger.

        Args:
            name: Logger name
            service_name: Service name for identification
            environment: Environment name
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Remove existing handlers
        self.logger.handlers.clear()

        # Add JSON handler for stdout
        json_handler = logging.StreamHandler(sys.stdout)
        json_handler.setFormatter(
            StructuredFormatter(service_name, environment)
        )
        self.logger.addHandler(json_handler)

    def log(
        self,
        level: int,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Log message with extra context.

        Args:
            level: Log level
            message: Log message
            extra: Extra fields to include
            **kwargs: Additional context fields
        """
        # Combine extra and kwargs
        all_extra = {**(extra or {}), **kwargs}

        # Add to log record
        self.logger.log(
            level,
            message,
            extra={'extra_fields': all_extra} if all_extra else {}
        )

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self.log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.log(logging.CRITICAL, message, **kwargs)

    def trade_executed(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        **kwargs
    ):
        """Log trade execution with structured data."""
        self.info(
            f"Trade executed: {side} {quantity} {symbol} @ ${price}",
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            value=quantity * price,
            event_type='trade_executed',
            **kwargs
        )

    def order_placed(
        self,
        order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        **kwargs
    ):
        """Log order placement with structured data."""
        self.info(
            f"Order placed: {order_type} {side} {quantity} {symbol}",
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            event_type='order_placed',
            **kwargs
        )

    def agent_state_change(
        self,
        agent_id: str,
        old_state: str,
        new_state: str,
        **kwargs
    ):
        """Log agent state change."""
        self.info(
            f"Agent state changed: {old_state} → {new_state}",
            agent_id=agent_id,
            old_state=old_state,
            new_state=new_state,
            event_type='agent_state_change',
            **kwargs
        )

    def performance_metric(
        self,
        metric_name: str,
        value: float,
        **kwargs
    ):
        """Log performance metric."""
        self.info(
            f"Metric: {metric_name} = {value}",
            metric_name=metric_name,
            metric_value=value,
            event_type='performance_metric',
            **kwargs
        )


def setup_structured_logging(
    service_name: str = "trading-ai",
    environment: str = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None
):
    """
    Configure structured logging for the application.

    Args:
        service_name: Service name
        environment: Environment (auto-detected if not provided)
        level: Logging level
        log_file: Optional log file path
    """
    # Auto-detect environment
    if not environment:
        environment = os.getenv('ENVIRONMENT', 'development')

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # JSON formatter
    formatter = StructuredFormatter(service_name, environment)

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logging.info(
        f"Structured logging configured: {service_name} ({environment})",
        extra={'extra_fields': {'service': service_name, 'environment': environment}}
    )


def get_structured_logger(
    name: str,
    service_name: str = "trading-ai",
    environment: str = None
) -> StructuredLogger:
    """
    Get structured logger instance.

    Args:
        name: Logger name
        service_name: Service name
        environment: Environment

    Returns:
        Structured logger
    """
    if not environment:
        environment = os.getenv('ENVIRONMENT', 'development')

    return StructuredLogger(name, service_name, environment)


if __name__ == "__main__":
    print("=" * 70)
    print("STRUCTURED LOGGING TEST")
    print("=" * 70)

    # Setup logging
    setup_structured_logging(
        service_name="test-service",
        environment="development",
        level=logging.DEBUG
    )

    # Get logger
    logger = get_structured_logger(__name__)

    print("\n1. Basic logging:")
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    print("\n2. Logging with extra context:")
    logger.info(
        "Request processed",
        request_id="req-12345",
        duration_ms=150,
        status_code=200
    )

    print("\n3. Trade execution log:")
    logger.trade_executed(
        trade_id="trade-001",
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.1,
        price=45000.0,
        strategy="dca_bot"
    )

    print("\n4. Order placement log:")
    logger.order_placed(
        order_id="order-123",
        symbol="ETHUSDT",
        side="SELL",
        order_type="MARKET",
        quantity=2.0
    )

    print("\n5. Agent state change log:")
    logger.agent_state_change(
        agent_id="agent-001",
        old_state="RUNNING",
        new_state="PAUSED",
        reason="manual_pause"
    )

    print("\n6. Performance metric log:")
    logger.performance_metric(
        metric_name="portfolio_value",
        value=125000.50,
        agent_id="agent-001"
    )

    print("\n7. Error with exception:")
    try:
        1 / 0
    except Exception as e:
        logger.error("Division by zero", exception=str(e))

    print("\n✅ Structured logging test complete!")
    print("\nNote: Check stdout for JSON-formatted logs")
