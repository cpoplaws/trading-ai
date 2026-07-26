"""
Retry and Error Recovery Utilities

Provides robust retry mechanisms with exponential backoff, error classification,
and recovery strategies for the trading system.
"""
import time
import functools
import logging
from typing import Callable, Optional, Tuple, Type, List
from enum import Enum
import asyncio
import random

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity classification."""
    TRANSIENT = "transient"  # Temporary, retry immediately
    RECOVERABLE = "recoverable"  # Can recover with backoff
    PERMANENT = "permanent"  # Cannot recover, fail fast
    CRITICAL = "critical"  # System-wide issue, alert


class RetryStrategy(Enum):
    """Retry strategy types."""
    EXPONENTIAL = "exponential"  # 1s, 2s, 4s, 8s...
    LINEAR = "linear"  # 1s, 2s, 3s, 4s...
    FIXED = "fixed"  # 1s, 1s, 1s, 1s...
    JITTERED = "jittered"  # Exponential with random jitter


def classify_error(exception: Exception) -> ErrorSeverity:
    """
    Classify error severity.

    Args:
        exception: The exception to classify

    Returns:
        Error severity level
    """
    error_msg = str(exception).lower()

    # Transient errors (retry immediately)
    transient_patterns = [
        'timeout',
        'connection reset',
        'temporary',
        'try again'
    ]

    # Recoverable errors (retry with backoff)
    recoverable_patterns = [
        'rate limit',
        'too many requests',
        'service unavailable',
        '503',
        '429',
        'connection refused',
        'connection error'
    ]

    # Permanent errors (don't retry)
    permanent_patterns = [
        'unauthorized',
        'forbidden',
        '401',
        '403',
        'invalid',
        'not found',
        '404',
        'bad request',
        '400'
    ]

    # Critical errors (alert immediately)
    critical_patterns = [
        'out of memory',
        'disk full',
        'database corruption',
        'critical'
    ]

    # Check patterns
    for pattern in critical_patterns:
        if pattern in error_msg:
            return ErrorSeverity.CRITICAL

    for pattern in permanent_patterns:
        if pattern in error_msg:
            return ErrorSeverity.PERMANENT

    for pattern in recoverable_patterns:
        if pattern in error_msg:
            return ErrorSeverity.RECOVERABLE

    for pattern in transient_patterns:
        if pattern in error_msg:
            return ErrorSeverity.TRANSIENT

    # Default to recoverable for unknown errors
    return ErrorSeverity.RECOVERABLE


def calculate_backoff(
    attempt: int,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    base_delay: float = 1.0,
    max_delay: float = 60.0
) -> float:
    """
    Calculate backoff delay.

    Args:
        attempt: Retry attempt number (0-indexed)
        strategy: Retry strategy
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Delay in seconds
    """
    if strategy == RetryStrategy.EXPONENTIAL:
        delay = base_delay * (2 ** attempt)

    elif strategy == RetryStrategy.LINEAR:
        delay = base_delay * (attempt + 1)

    elif strategy == RetryStrategy.FIXED:
        delay = base_delay

    elif strategy == RetryStrategy.JITTERED:
        # Exponential with jitter
        exp_delay = base_delay * (2 ** attempt)
        jitter = random.uniform(0, exp_delay * 0.1)
        delay = exp_delay + jitter

    else:
        delay = base_delay

    return min(delay, max_delay)


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None,
    on_failure: Optional[Callable] = None
):
    """
    Decorator for automatic retry with exponential backoff.

    Args:
        max_attempts: Maximum retry attempts
        base_delay: Base delay between retries (seconds)
        max_delay: Maximum delay (seconds)
        strategy: Retry strategy
        exceptions: Exceptions to catch and retry
        on_retry: Callback on retry (called with attempt number and exception)
        on_failure: Callback on final failure (called with exception)

    Example:
        @retry(max_attempts=3, base_delay=1.0)
        def fetch_data():
            return api.get_data()
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e
                    severity = classify_error(e)

                    # Don't retry permanent or critical errors
                    if severity in [ErrorSeverity.PERMANENT, ErrorSeverity.CRITICAL]:
                        logger.error(f"{func.__name__} failed with {severity.value} error: {e}")
                        if on_failure:
                            on_failure(e)
                        raise

                    # Calculate backoff
                    if attempt < max_attempts - 1:
                        delay = calculate_backoff(attempt, strategy, base_delay, max_delay)

                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )

                        if on_retry:
                            on_retry(attempt, e)

                        time.sleep(delay)

            # All attempts failed
            logger.error(f"{func.__name__} failed after {max_attempts} attempts: {last_exception}")
            if on_failure:
                on_failure(last_exception)

            raise last_exception

        return wrapper
    return decorator


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None,
    on_failure: Optional[Callable] = None
):
    """
    Async decorator for automatic retry with exponential backoff.

    Args:
        Same as retry() decorator

    Example:
        @async_retry(max_attempts=3)
        async def fetch_data():
            return await api.get_data()
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e
                    severity = classify_error(e)

                    # Don't retry permanent or critical errors
                    if severity in [ErrorSeverity.PERMANENT, ErrorSeverity.CRITICAL]:
                        logger.error(f"{func.__name__} failed with {severity.value} error: {e}")
                        if on_failure:
                            if asyncio.iscoroutinefunction(on_failure):
                                await on_failure(e)
                            else:
                                on_failure(e)
                        raise

                    # Calculate backoff
                    if attempt < max_attempts - 1:
                        delay = calculate_backoff(attempt, strategy, base_delay, max_delay)

                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )

                        if on_retry:
                            if asyncio.iscoroutinefunction(on_retry):
                                await on_retry(attempt, e)
                            else:
                                on_retry(attempt, e)

                        await asyncio.sleep(delay)

            # All attempts failed
            logger.error(f"{func.__name__} failed after {max_attempts} attempts: {last_exception}")
            if on_failure:
                if asyncio.iscoroutinefunction(on_failure):
                    await on_failure(last_exception)
                else:
                    on_failure(last_exception)

            raise last_exception

        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by stopping requests to failing services.

    States:
    - CLOSED: Normal operation
    - OPEN: Failing, reject requests immediately
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Time before trying half-open (seconds)
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker."""
        if self.state == "OPEN":
            # Check if recovery timeout passed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info(f"Circuit breaker for {func.__name__} entering HALF_OPEN state")
            else:
                raise Exception(f"Circuit breaker OPEN for {func.__name__}")

        try:
            result = func(*args, **kwargs)

            # Success - reset or close circuit
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info(f"Circuit breaker for {func.__name__} CLOSED")

            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(
                    f"Circuit breaker OPENED for {func.__name__} "
                    f"after {self.failure_count} failures"
                )

            raise


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("RETRY AND ERROR RECOVERY TEST")
    print("=" * 70)

    # Test retry decorator
    attempt_count = 0

    @retry(max_attempts=3, base_delay=0.5)
    def flaky_function():
        global attempt_count
        attempt_count += 1
        print(f"  Attempt {attempt_count}")

        if attempt_count < 3:
            raise ConnectionError("Temporary failure")

        return "Success!"

    print("\n1. Testing retry with recovery:")
    result = flaky_function()
    print(f"Result: {result}")

    # Test error classification
    print("\n2. Testing error classification:")
    errors = [
        ConnectionError("Connection timeout"),
        ValueError("Rate limit exceeded"),
        PermissionError("Unauthorized access"),
        RuntimeError("Out of memory - critical")
    ]

    for error in errors:
        severity = classify_error(error)
        print(f"  {error.__class__.__name__}: {severity.value}")

    # Test backoff calculation
    print("\n3. Testing backoff strategies:")
    for strategy in RetryStrategy:
        print(f"  {strategy.value}:")
        delays = [calculate_backoff(i, strategy, base_delay=1.0) for i in range(5)]
        print(f"    Delays: {[f'{d:.1f}s' for d in delays]}")

    # Test circuit breaker
    print("\n4. Testing circuit breaker:")
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=2.0)

    def failing_service():
        raise ConnectionError("Service unavailable")

    for i in range(5):
        try:
            breaker.call(failing_service)
        except Exception as e:
            print(f"  Attempt {i+1}: {e}")

    print("\n✅ Retry system test complete!")
