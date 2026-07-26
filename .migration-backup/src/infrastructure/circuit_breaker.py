"""
Circuit Breaker Pattern
Prevents cascading failures by failing fast when a service is unavailable.
"""
import sys
from pathlib import Path
from typing import Callable, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import time
import logging
from functools import wraps
from collections import deque

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, rejecting all requests
    - HALF_OPEN: Testing if service recovered

    Transitions:
    - CLOSED → OPEN: After threshold failures
    - OPEN → HALF_OPEN: After timeout period
    - HALF_OPEN → CLOSED: After successful test requests
    - HALF_OPEN → OPEN: If test requests fail
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
        name: str = "default"
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening
            recovery_timeout: Seconds to wait before trying half-open
            expected_exception: Exception type to count as failure
            name: Circuit breaker name for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._success_count = 0

        # Stats
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0

        logger.info(f"Circuit breaker '{name}' initialized")

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for timeout transitions."""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._state = CircuitState.HALF_OPEN
                logger.info(f"Circuit '{self.name}' transitioning to HALF_OPEN")

        return self._state

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try recovery."""
        if self._last_failure_time is None:
            return False

        return (datetime.now() - self._last_failure_time).total_seconds() >= self.recovery_timeout

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args, **kwargs: Function arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        self._total_calls += 1

        # Check circuit state
        if self.state == CircuitState.OPEN:
            logger.warning(f"Circuit '{self.name}' is OPEN, rejecting call")
            raise CircuitBreakerError(f"Circuit breaker '{self.name}' is open")

        try:
            # Execute function
            result = func(*args, **kwargs)

            # Success
            self._on_success()

            return result

        except self.expected_exception as e:
            # Failure
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call."""
        self._total_successes += 1

        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1

            # If enough successful calls in half-open, close circuit
            if self._success_count >= 3:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                logger.info(f"Circuit '{self.name}' transitioning to CLOSED")

        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    def _on_failure(self):
        """Handle failed call."""
        self._total_failures += 1
        self._failure_count += 1
        self._last_failure_time = datetime.now()

        if self._state == CircuitState.HALF_OPEN:
            # Failure in half-open, reopen circuit
            self._state = CircuitState.OPEN
            self._success_count = 0
            logger.warning(f"Circuit '{self.name}' transitioning back to OPEN")

        elif self._state == CircuitState.CLOSED:
            # Check if threshold reached
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.error(f"Circuit '{self.name}' OPENED after {self._failure_count} failures")

    def reset(self):
        """Manually reset circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        logger.info(f"Circuit '{self.name}' manually reset to CLOSED")

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            'name': self.name,
            'state': self._state.value,
            'failure_count': self._failure_count,
            'total_calls': self._total_calls,
            'total_successes': self._total_successes,
            'total_failures': self._total_failures,
            'success_rate': (self._total_successes / self._total_calls * 100) if self._total_calls > 0 else 0,
            'last_failure_time': self._last_failure_time.isoformat() if self._last_failure_time else None
        }


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: type = Exception,
    name: Optional[str] = None
):
    """
    Decorator for circuit breaker protection.

    Args:
        failure_threshold: Failures before opening
        recovery_timeout: Seconds before trying recovery
        expected_exception: Exception type to catch
        name: Circuit breaker name

    Example:
        @circuit_breaker(failure_threshold=3, recovery_timeout=30)
        def call_external_api():
            return requests.get('https://api.example.com')
    """
    def decorator(func):
        breaker_name = name or func.__name__
        breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            name=breaker_name
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)

        wrapper.circuit_breaker = breaker
        return wrapper

    return decorator


class RateLimiter:
    """
    Token bucket rate limiter.

    Limits the rate of requests using a token bucket algorithm.
    """

    def __init__(self, max_calls: int, time_window: int):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window

        self.calls = deque()

    def is_allowed(self) -> bool:
        """Check if request is allowed."""
        now = time.time()

        # Remove old calls outside time window
        while self.calls and now - self.calls[0] > self.time_window:
            self.calls.popleft()

        # Check if under limit
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True

        return False

    def wait_time(self) -> float:
        """Get time to wait before next allowed call."""
        if len(self.calls) < self.max_calls:
            return 0.0

        oldest_call = self.calls[0]
        wait_time = self.time_window - (time.time() - oldest_call)

        return max(0.0, wait_time)

    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        now = time.time()
        recent_calls = sum(1 for call_time in self.calls if now - call_time <= self.time_window)

        return {
            'max_calls': self.max_calls,
            'time_window': self.time_window,
            'recent_calls': recent_calls,
            'remaining': max(0, self.max_calls - recent_calls),
            'wait_time': self.wait_time()
        }


def rate_limit(max_calls: int, time_window: int):
    """
    Decorator for rate limiting.

    Args:
        max_calls: Maximum calls in time window
        time_window: Time window in seconds

    Example:
        @rate_limit(max_calls=10, time_window=60)
        def api_call():
            return fetch_data()
    """
    limiter = RateLimiter(max_calls, time_window)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not limiter.is_allowed():
                wait_time = limiter.wait_time()
                raise Exception(f"Rate limit exceeded. Wait {wait_time:.2f}s")

            return func(*args, **kwargs)

        wrapper.rate_limiter = limiter
        return wrapper

    return decorator


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("⚡ Circuit Breaker & Rate Limiter Demo")
    print("=" * 60)

    # Test circuit breaker
    print("\n🔌 Testing Circuit Breaker...")

    call_count = 0

    @circuit_breaker(failure_threshold=3, recovery_timeout=5, name="test_service")
    def unstable_service():
        global call_count
        call_count += 1

        # Fail first 5 calls
        if call_count <= 5:
            raise Exception("Service temporarily unavailable")

        return "Success!"

    for i in range(10):
        try:
            result = unstable_service()
            print(f"   Call {i+1}: {result}")
        except CircuitBreakerError as e:
            print(f"   Call {i+1}: ⛔ Circuit OPEN")
        except Exception as e:
            print(f"   Call {i+1}: ❌ Failed - {e}")

        time.sleep(1)

    # Stats
    stats = unstable_service.circuit_breaker.get_stats()
    print(f"\n📊 Circuit Stats:")
    print(f"   State: {stats['state'].upper()}")
    print(f"   Total Calls: {stats['total_calls']}")
    print(f"   Successes: {stats['total_successes']}")
    print(f"   Failures: {stats['total_failures']}")
    print(f"   Success Rate: {stats['success_rate']:.1f}%")

    # Test rate limiter
    print("\n⏱️  Testing Rate Limiter...")

    @rate_limit(max_calls=5, time_window=10)
    def limited_api():
        return "API Response"

    for i in range(8):
        try:
            result = limited_api()
            print(f"   Call {i+1}: ✅ {result}")
        except Exception as e:
            print(f"   Call {i+1}: ⛔ {e}")

        time.sleep(0.5)

    # Stats
    rate_stats = limited_api.rate_limiter.get_stats()
    print(f"\n📊 Rate Limiter Stats:")
    print(f"   Max Calls: {rate_stats['max_calls']}/{rate_stats['time_window']}s")
    print(f"   Recent Calls: {rate_stats['recent_calls']}")
    print(f"   Remaining: {rate_stats['remaining']}")

    print("\n✅ Circuit breaker & rate limiter demo complete!")
