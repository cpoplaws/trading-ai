"""
Circuit Breaker Manager

Centralized circuit breaker management with monitoring and alerting.
"""
import time
import logging
from typing import Dict, Optional, Callable, List
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import threading

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitMetrics:
    """Circuit breaker metrics."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changes: int = 0
    total_downtime: float = 0.0
    last_state_change: Optional[float] = None


@dataclass
class CircuitConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout: float = 60.0  # Seconds before trying half-open
    expected_exception: type = Exception


class ManagedCircuitBreaker:
    """
    Circuit breaker with enhanced monitoring.

    Prevents cascading failures by stopping requests to failing services.
    """

    def __init__(self, name: str, config: CircuitConfig):
        """
        Initialize circuit breaker.

        Args:
            name: Circuit breaker identifier
            config: Configuration
        """
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.metrics = CircuitMetrics()
        self.state_opened_at: Optional[float] = None
        self.lock = threading.Lock()

        logger.info(f"Circuit breaker created: {name}")

    def call(self, func: Callable, *args, **kwargs):
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        with self.lock:
            self.metrics.total_calls += 1

            # Check circuit state
            if self.state == CircuitState.OPEN:
                # Check if timeout passed
                if time.time() - self.state_opened_at >= self.config.timeout:
                    self._transition_to_half_open()
                else:
                    self.metrics.rejected_calls += 1
                    raise Exception(f"Circuit breaker OPEN: {self.name}")

        # Execute function
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result

        except self.config.expected_exception as e:
            self._record_failure(e)
            raise

    def _record_success(self):
        """Record successful call."""
        with self.lock:
            self.metrics.successful_calls += 1
            self.metrics.last_success_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1

                # Check if we can close the circuit
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()

    def _record_failure(self, exception: Exception):
        """Record failed call."""
        with self.lock:
            self.metrics.failed_calls += 1
            self.metrics.last_failure_time = time.time()

            if self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]:
                self.failure_count += 1

                # Check if we should open the circuit
                if self.failure_count >= self.config.failure_threshold:
                    self._transition_to_open()

    def _transition_to_open(self):
        """Transition to OPEN state."""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.state_opened_at = time.time()
        self.failure_count = 0
        self.success_count = 0

        self._record_state_change(old_state, CircuitState.OPEN)

        logger.error(
            f"Circuit breaker OPENED: {self.name} "
            f"(failures: {self.metrics.failed_calls})"
        )

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.failure_count = 0
        self.success_count = 0

        self._record_state_change(old_state, CircuitState.HALF_OPEN)

        logger.info(f"Circuit breaker HALF_OPEN: {self.name}")

    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0

        # Record downtime
        if self.state_opened_at:
            downtime = time.time() - self.state_opened_at
            self.metrics.total_downtime += downtime
            self.state_opened_at = None

        self._record_state_change(old_state, CircuitState.CLOSED)

        logger.info(f"Circuit breaker CLOSED: {self.name}")

    def _record_state_change(self, old_state: CircuitState, new_state: CircuitState):
        """Record state change."""
        self.metrics.state_changes += 1
        self.metrics.last_state_change = time.time()

    def get_state(self) -> CircuitState:
        """Get current state."""
        return self.state

    def get_metrics(self) -> CircuitMetrics:
        """Get metrics."""
        return self.metrics

    def reset(self):
        """Reset circuit breaker."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.state_opened_at = None
            logger.info(f"Circuit breaker reset: {self.name}")

    def is_available(self) -> bool:
        """Check if circuit is available for requests."""
        return self.state != CircuitState.OPEN


class CircuitBreakerManager:
    """
    Centralized circuit breaker management.

    Features:
    - Multiple circuit breakers
    - Global monitoring
    - Alert triggers
    - Health checks
    """

    def __init__(self):
        self.breakers: Dict[str, ManagedCircuitBreaker] = {}
        self.alert_callbacks: List[Callable] = []
        self.lock = threading.Lock()

    def create_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        expected_exception: type = Exception
    ) -> ManagedCircuitBreaker:
        """
        Create or get circuit breaker.

        Args:
            name: Circuit breaker identifier
            failure_threshold: Failures before opening
            success_threshold: Successes to close
            timeout: Recovery timeout (seconds)
            expected_exception: Exception type to catch

        Returns:
            Circuit breaker instance
        """
        with self.lock:
            if name in self.breakers:
                return self.breakers[name]

            config = CircuitConfig(
                failure_threshold=failure_threshold,
                success_threshold=success_threshold,
                timeout=timeout,
                expected_exception=expected_exception
            )

            breaker = ManagedCircuitBreaker(name, config)
            self.breakers[name] = breaker

            logger.info(f"Circuit breaker registered: {name}")

            return breaker

    def get_breaker(self, name: str) -> Optional[ManagedCircuitBreaker]:
        """Get circuit breaker by name."""
        return self.breakers.get(name)

    def call_through_breaker(
        self,
        name: str,
        func: Callable,
        *args,
        **kwargs
    ):
        """
        Execute function through circuit breaker.

        Args:
            name: Circuit breaker name
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        breaker = self.get_breaker(name)
        if not breaker:
            raise ValueError(f"Circuit breaker not found: {name}")

        try:
            result = breaker.call(func, *args, **kwargs)
            return result

        except Exception as e:
            # Trigger alerts if circuit opened
            if breaker.get_state() == CircuitState.OPEN:
                self._trigger_alerts(name, breaker)
            raise

    def get_all_metrics(self) -> Dict[str, Dict]:
        """Get metrics for all circuit breakers."""
        metrics = {}

        for name, breaker in self.breakers.items():
            m = breaker.get_metrics()
            metrics[name] = {
                'state': breaker.get_state().value,
                'total_calls': m.total_calls,
                'successful_calls': m.successful_calls,
                'failed_calls': m.failed_calls,
                'rejected_calls': m.rejected_calls,
                'success_rate': (
                    m.successful_calls / m.total_calls * 100
                    if m.total_calls > 0 else 0
                ),
                'state_changes': m.state_changes,
                'total_downtime': m.total_downtime,
                'available': breaker.is_available()
            }

        return metrics

    def get_health_summary(self) -> Dict:
        """Get overall health summary."""
        total_breakers = len(self.breakers)
        open_breakers = sum(
            1 for b in self.breakers.values()
            if b.get_state() == CircuitState.OPEN
        )
        half_open_breakers = sum(
            1 for b in self.breakers.values()
            if b.get_state() == CircuitState.HALF_OPEN
        )

        return {
            'total_breakers': total_breakers,
            'open_breakers': open_breakers,
            'half_open_breakers': half_open_breakers,
            'closed_breakers': total_breakers - open_breakers - half_open_breakers,
            'health_percentage': (
                (total_breakers - open_breakers) / total_breakers * 100
                if total_breakers > 0 else 100
            )
        }

    def register_alert_callback(self, callback: Callable):
        """
        Register callback for circuit breaker alerts.

        Callback signature: callback(breaker_name: str, metrics: CircuitMetrics)
        """
        self.alert_callbacks.append(callback)

    def _trigger_alerts(self, name: str, breaker: ManagedCircuitBreaker):
        """Trigger alerts for circuit breaker trip."""
        metrics = breaker.get_metrics()

        for callback in self.alert_callbacks:
            try:
                callback(name, metrics)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def reset_breaker(self, name: str) -> bool:
        """Reset specific circuit breaker."""
        breaker = self.get_breaker(name)
        if breaker:
            breaker.reset()
            return True
        return False

    def reset_all_breakers(self):
        """Reset all circuit breakers."""
        for breaker in self.breakers.values():
            breaker.reset()

        logger.info("All circuit breakers reset")


# Global circuit breaker manager
_global_cb_manager = CircuitBreakerManager()


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get global circuit breaker manager."""
    return _global_cb_manager


def with_circuit_breaker(name: str, **breaker_kwargs):
    """
    Decorator for circuit breaker protection.

    Args:
        name: Circuit breaker name
        **breaker_kwargs: Circuit breaker configuration

    Usage:
        @with_circuit_breaker("binance_api", failure_threshold=5, timeout=60.0)
        def fetch_data():
            return api.get_data()
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_circuit_breaker_manager()

            # Create breaker if doesn't exist
            breaker = manager.create_breaker(name, **breaker_kwargs)

            # Call through breaker
            return breaker.call(func, *args, **kwargs)

        return wrapper
    return decorator


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("CIRCUIT BREAKER MANAGER TEST")
    print("=" * 70)

    # Create manager
    manager = CircuitBreakerManager()

    # Create circuit breakers
    print("\n1. Creating circuit breakers:")
    manager.create_breaker("binance_api", failure_threshold=3, timeout=5.0)
    manager.create_breaker("database", failure_threshold=5, timeout=10.0)
    print("   Created breakers")

    # Register alert callback
    print("\n2. Registering alert callback:")

    def alert_callback(name, metrics):
        print(f"   🚨 ALERT: Circuit breaker '{name}' opened!")
        print(f"      Failed calls: {metrics.failed_calls}")

    manager.register_alert_callback(alert_callback)
    print("   Callback registered")

    # Test circuit breaker
    print("\n3. Testing circuit breaker:")

    call_count = 0

    def flaky_api_call():
        global call_count
        call_count += 1
        if call_count < 4:
            raise Exception("API failed")
        return "Success"

    # Call until circuit opens
    for i in range(6):
        try:
            result = manager.call_through_breaker(
                "binance_api",
                flaky_api_call
            )
            print(f"   Call {i+1}: {result}")
        except Exception as e:
            print(f"   Call {i+1}: Failed - {e}")

    # Check metrics
    print("\n4. Circuit breaker metrics:")
    metrics = manager.get_all_metrics()

    for name, data in metrics.items():
        print(f"\n   {name}:")
        print(f"     State: {data['state']}")
        print(f"     Total calls: {data['total_calls']}")
        print(f"     Success rate: {data['success_rate']:.1f}%")
        print(f"     Failed: {data['failed_calls']}")
        print(f"     Rejected: {data['rejected_calls']}")
        print(f"     Available: {data['available']}")

    # Health summary
    print("\n5. Health summary:")
    health = manager.get_health_summary()
    print(f"   Total breakers: {health['total_breakers']}")
    print(f"   Open: {health['open_breakers']}")
    print(f"   Half-open: {health['half_open_breakers']}")
    print(f"   Closed: {health['closed_breakers']}")
    print(f"   Health: {health['health_percentage']:.1f}%")

    # Wait for timeout
    print("\n6. Waiting for circuit recovery (5s)...")
    time.sleep(5.5)

    # Try again (should be half-open)
    print("\n7. Testing recovery:")
    try:
        result = manager.call_through_breaker(
            "binance_api",
            flaky_api_call
        )
        print(f"   Result: {result}")
        print(f"   State: {manager.get_breaker('binance_api').get_state().value}")
    except Exception as e:
        print(f"   Failed: {e}")

    # Test decorator
    print("\n8. Testing decorator:")

    @with_circuit_breaker("test_api", failure_threshold=2)
    def test_function():
        return "Decorator works!"

    result = test_function()
    print(f"   {result}")

    # Reset breakers
    print("\n9. Resetting all breakers:")
    manager.reset_all_breakers()

    health = manager.get_health_summary()
    print(f"   Health: {health['health_percentage']:.1f}%")

    print("\n✅ Circuit breaker manager test complete!")
