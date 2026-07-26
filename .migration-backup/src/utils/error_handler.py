"""
Error Handler - Production-grade error handling and retry logic

Features:
- Exponential backoff retry
- Circuit breaker pattern
- Error classification and routing
- Automatic recovery strategies
- Error metrics and logging
"""

import logging
import time
import functools
from typing import Callable, Optional, Tuple, Type, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import traceback

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    RECOVERABLE = "recoverable"      # Retry automatically
    DEGRADED = "degraded"            # Continue with reduced functionality
    FATAL = "fatal"                  # Stop immediately


class ErrorCategory(Enum):
    """Error categories"""
    NETWORK = "network"              # API, RPC, network issues
    VALIDATION = "validation"        # Invalid input, schema errors
    INSUFFICIENT_FUNDS = "insufficient_funds"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    INTERNAL = "internal"            # Code bugs, unexpected errors
    EXTERNAL = "external"            # Third-party service issues


@dataclass
class ErrorRecord:
    """Error tracking record"""
    timestamp: datetime
    error_type: str
    error_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    stack_trace: str
    context: dict = field(default_factory=dict)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state"""
    is_open: bool = False
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    open_until: Optional[datetime] = None


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents repeated calls to failing operations.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        half_open_attempts: int = 1
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            timeout_seconds: Time to wait before half-open
            half_open_attempts: Attempts in half-open state
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_attempts = half_open_attempts
        self.state = CircuitBreakerState()
        self.half_open_attempt_count = 0

    def record_success(self) -> None:
        """Record successful operation."""
        self.state.failure_count = 0
        self.state.is_open = False
        self.half_open_attempt_count = 0
        logger.debug("Circuit breaker: success recorded, circuit closed")

    def record_failure(self) -> None:
        """Record failed operation."""
        self.state.failure_count += 1
        self.state.last_failure_time = datetime.now()

        if self.state.failure_count >= self.failure_threshold:
            self.state.is_open = True
            self.state.open_until = datetime.now() + timedelta(seconds=self.timeout_seconds)
            logger.warning(
                f"Circuit breaker OPENED after {self.state.failure_count} failures. "
                f"Will retry at {self.state.open_until}"
            )

    def is_available(self) -> Tuple[bool, str]:
        """
        Check if circuit allows calls.

        Returns:
            (is_available, reason)
        """
        if not self.state.is_open:
            return True, "Circuit closed"

        # Check if timeout expired (enter half-open state)
        if self.state.open_until and datetime.now() >= self.state.open_until:
            if self.half_open_attempt_count < self.half_open_attempts:
                self.half_open_attempt_count += 1
                logger.info("Circuit breaker: entering HALF-OPEN state")
                return True, "Circuit half-open, attempting recovery"
            else:
                # Reset to allow more attempts
                self.state.open_until = datetime.now() + timedelta(seconds=self.timeout_seconds)
                self.half_open_attempt_count = 0

        return False, f"Circuit open until {self.state.open_until}"


class ErrorHandler:
    """
    Production error handling system.

    Features:
    - Automatic retry with exponential backoff
    - Circuit breaker integration
    - Error classification
    - Recovery strategies
    - Metrics tracking
    """

    def __init__(self, alert_manager=None):
        """
        Initialize error handler.

        Args:
            alert_manager: AlertManager instance for notifications
        """
        self.alert_manager = alert_manager
        self.error_history: List[ErrorRecord] = []
        self.max_history = 1000

        # Circuit breakers per service
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

        # Metrics
        self.metrics = {
            "total_errors": 0,
            "errors_by_category": {cat: 0 for cat in ErrorCategory},
            "errors_by_severity": {sev: 0 for sev in ErrorSeverity},
            "retries_attempted": 0,
            "retries_succeeded": 0,
        }

        logger.info("Error Handler initialized")

    def classify_error(self, error: Exception) -> Tuple[ErrorCategory, ErrorSeverity]:
        """
        Classify error by category and severity.

        Args:
            error: Exception to classify

        Returns:
            (category, severity)
        """
        error_str = str(error).lower()
        error_type = type(error).__name__

        # Network/API errors
        if "timeout" in error_str or "connection" in error_str:
            return ErrorCategory.NETWORK, ErrorSeverity.RECOVERABLE
        if "rate limit" in error_str or "too many requests" in error_str:
            return ErrorCategory.RATE_LIMIT, ErrorSeverity.RECOVERABLE

        # Authentication
        if "auth" in error_str or "unauthorized" in error_str or "forbidden" in error_str:
            return ErrorCategory.AUTHENTICATION, ErrorSeverity.FATAL

        # Insufficient funds
        if "insufficient" in error_str or "balance" in error_str:
            return ErrorCategory.INSUFFICIENT_FUNDS, ErrorSeverity.DEGRADED

        # Validation
        if isinstance(error, (ValueError, TypeError, KeyError)):
            return ErrorCategory.VALIDATION, ErrorSeverity.FATAL

        # Default
        return ErrorCategory.INTERNAL, ErrorSeverity.RECOVERABLE

    def record_error(
        self,
        error: Exception,
        context: dict = None,
        category: ErrorCategory = None,
        severity: ErrorSeverity = None
    ) -> ErrorRecord:
        """Record error for tracking."""
        # Auto-classify if not provided
        if category is None or severity is None:
            auto_cat, auto_sev = self.classify_error(error)
            category = category or auto_cat
            severity = severity or auto_sev

        record = ErrorRecord(
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            category=category,
            severity=severity,
            stack_trace=traceback.format_exc(),
            context=context or {}
        )

        self.error_history.append(record)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)

        # Update metrics
        self.metrics["total_errors"] += 1
        self.metrics["errors_by_category"][category] += 1
        self.metrics["errors_by_severity"][severity] += 1

        # Alert if severe
        if severity == ErrorSeverity.FATAL and self.alert_manager:
            self.alert_manager.alert_api_error(
                api=context.get("service", "Unknown"),
                error=str(error)
            )

        return record

    def get_circuit_breaker(self, service: str) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        if service not in self.circuit_breakers:
            self.circuit_breakers[service] = CircuitBreaker()
        return self.circuit_breakers[service]


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
    circuit_breaker_service: Optional[str] = None
):
    """
    Decorator for automatic retry with exponential backoff.

    Args:
        max_retries: Maximum retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        retry_on: Exception types to retry on
        circuit_breaker_service: Service name for circuit breaker

    Example:
        @retry_with_backoff(max_retries=3, initial_delay=1.0)
        def fetch_data():
            return api.get_data()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            error_handler = kwargs.pop('_error_handler', None)

            # Check circuit breaker if specified
            if circuit_breaker_service and error_handler:
                breaker = error_handler.get_circuit_breaker(circuit_breaker_service)
                is_available, reason = breaker.is_available()

                if not is_available:
                    raise RuntimeError(f"Circuit breaker open for {circuit_breaker_service}: {reason}")

            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)

                    # Record success in circuit breaker
                    if circuit_breaker_service and error_handler:
                        breaker = error_handler.get_circuit_breaker(circuit_breaker_service)
                        breaker.record_success()

                    if attempt > 0:
                        logger.info(f"Retry succeeded on attempt {attempt + 1}")
                        if error_handler:
                            error_handler.metrics["retries_succeeded"] += 1

                    return result

                except retry_on as e:
                    last_exception = e

                    # Record failure in circuit breaker
                    if circuit_breaker_service and error_handler:
                        breaker = error_handler.get_circuit_breaker(circuit_breaker_service)
                        breaker.record_failure()

                    if attempt < max_retries:
                        if error_handler:
                            error_handler.metrics["retries_attempted"] += 1

                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay = min(delay * exponential_base, max_delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")

            # All retries exhausted
            if error_handler:
                error_handler.record_error(
                    last_exception,
                    context={
                        "function": func.__name__,
                        "attempts": max_retries + 1,
                        "service": circuit_breaker_service
                    }
                )

            raise last_exception

        return wrapper
    return decorator


def handle_errors(
    error_handler: ErrorHandler,
    context: dict = None,
    reraise: bool = True,
    default_return: Any = None
):
    """
    Decorator for automatic error handling.

    Args:
        error_handler: ErrorHandler instance
        context: Context information
        reraise: Whether to reraise exception
        default_return: Default value to return on error

    Example:
        @handle_errors(error_handler, context={"operation": "trade"})
        def execute_trade():
            # Trade logic
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Record error
                error_handler.record_error(e, context=context)

                # Log
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)

                if reraise:
                    raise
                else:
                    return default_return

        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*70)
    print("ERROR HANDLER TEST")
    print("="*70)

    # Initialize error handler
    handler = ErrorHandler()

    # Test 1: Error classification
    print("\n--- Test 1: Error Classification ---")
    test_errors = [
        ConnectionError("Connection timeout"),
        ValueError("Invalid input"),
        RuntimeError("Insufficient balance"),
        Exception("Rate limit exceeded"),
    ]

    for error in test_errors:
        category, severity = handler.classify_error(error)
        print(f"   {type(error).__name__}: {category.value} ({severity.value})")

    # Test 2: Retry with backoff
    print("\n--- Test 2: Retry with Backoff ---")

    attempt_count = [0]

    @retry_with_backoff(max_retries=3, initial_delay=0.5)
    def flaky_function():
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise ConnectionError(f"Attempt {attempt_count[0]} failed")
        return "Success!"

    try:
        result = flaky_function()
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Failed: {e}")

    # Test 3: Circuit breaker
    print("\n--- Test 3: Circuit Breaker ---")

    breaker = CircuitBreaker(failure_threshold=3, timeout_seconds=2)

    # Simulate failures
    for i in range(5):
        breaker.record_failure()
        is_available, reason = breaker.is_available()
        status = "OPEN" if not is_available else "CLOSED"
        print(f"   Failure {i+1}: Circuit {status}")

    # Wait and try recovery
    print("   Waiting 2 seconds for recovery...")
    time.sleep(2)
    is_available, reason = breaker.is_available()
    print(f"   After timeout: {reason}")

    # Simulate success
    breaker.record_success()
    is_available, reason = breaker.is_available()
    print(f"   After success: {reason}")

    # Test 4: Metrics
    print("\n--- Test 4: Error Metrics ---")
    metrics = handler.metrics
    print(f"   Total errors: {metrics['total_errors']}")
    print(f"   Retries attempted: {metrics['retries_attempted']}")
    print(f"   Retries succeeded: {metrics['retries_succeeded']}")

    print("\n" + "="*70)
    print("✅ Error Handler ready!")
    print("="*70)
