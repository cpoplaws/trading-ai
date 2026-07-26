#!/usr/bin/env python3
"""
Rate Limiting and Circuit Breaker Integration

Demonstrates how to use rate limiting and circuit breakers together
for robust API access.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.rate_limiter import RateLimiter, rate_limited, RequestQueue
from utils.circuit_breaker_manager import (
    CircuitBreakerManager,
    with_circuit_breaker
)
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Example 1: API Client with Both Protections
# ============================================================

class ProtectedAPIClient:
    """
    API client with rate limiting and circuit breaker protection.
    """

    def __init__(self):
        # Initialize rate limiter
        self.rate_limiter = RateLimiter()
        self.rate_limiter.configure(
            endpoint="public_api",
            requests_per_second=10.0,
            burst_size=20
        )
        self.rate_limiter.configure(
            endpoint="trading_api",
            requests_per_second=5.0,
            burst_size=10
        )

        # Initialize circuit breaker manager
        self.cb_manager = CircuitBreakerManager()
        self.cb_manager.create_breaker(
            "public_api",
            failure_threshold=5,
            timeout=30.0
        )
        self.cb_manager.create_breaker(
            "trading_api",
            failure_threshold=3,
            timeout=60.0
        )

        # Register alert callback
        self.cb_manager.register_alert_callback(self._handle_circuit_open)

    def _handle_circuit_open(self, breaker_name, metrics):
        """Handle circuit breaker opening."""
        logger.critical(
            f"🚨 Circuit breaker OPENED: {breaker_name}\n"
            f"   Failed calls: {metrics.failed_calls}\n"
            f"   Last failure: {metrics.last_failure_time}"
        )
        # In production: send alert via Slack/PagerDuty

    def get_ticker_price(self, symbol: str) -> float:
        """
        Get ticker price with rate limiting and circuit breaker.

        Args:
            symbol: Trading symbol

        Returns:
            Price
        """
        # Rate limit
        self.rate_limiter.wait_for_token("public_api")

        # Circuit breaker
        def _fetch():
            logger.info(f"Fetching price for {symbol}")
            # Simulate API call
            return 45000.0 + (hash(symbol) % 1000)

        return self.cb_manager.call_through_breaker("public_api", _fetch)

    def place_order(self, symbol: str, side: str, quantity: float) -> dict:
        """
        Place order with protections.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity

        Returns:
            Order result
        """
        # Rate limit (trading is more restricted)
        self.rate_limiter.wait_for_token("trading_api")

        # Circuit breaker
        def _execute():
            logger.info(f"Placing order: {side} {quantity} {symbol}")
            # Simulate order placement
            return {
                'order_id': '12345',
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'status': 'FILLED'
            }

        return self.cb_manager.call_through_breaker("trading_api", _execute)

    def get_metrics(self) -> dict:
        """Get protection metrics."""
        return {
            'rate_limiting': self.rate_limiter.get_metrics(),
            'circuit_breakers': self.cb_manager.get_all_metrics(),
            'health': self.cb_manager.get_health_summary()
        }


# ============================================================
# Example 2: Decorator-Based Protection
# ============================================================

# Configure global instances
rate_limiter = RateLimiter()
rate_limiter.configure("api_v1", requests_per_second=20.0, burst_size=40)

cb_manager = CircuitBreakerManager()
cb_manager.create_breaker("api_v1", failure_threshold=5, timeout=30.0)


@rate_limited("api_v1", tokens=1)
@with_circuit_breaker("api_v1")
def fetch_market_data(symbol: str) -> dict:
    """
    Fetch market data with both protections.

    Decorators are applied in order:
    1. Rate limiting (outer)
    2. Circuit breaker (inner)
    """
    logger.info(f"Fetching market data for {symbol}")
    # Simulate API call
    return {
        'symbol': symbol,
        'price': 45000.0,
        'volume': 1000000.0
    }


# ============================================================
# Example 3: Request Queue with Both Protections
# ============================================================

class ProtectedRequestQueue:
    """Request queue with rate limiting and circuit breaker."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

        # Set up rate limiter
        self.rate_limiter = RateLimiter()
        self.rate_limiter.configure(
            endpoint=endpoint,
            requests_per_second=10.0,
            burst_size=20
        )

        # Set up circuit breaker
        self.cb_manager = CircuitBreakerManager()
        self.cb_manager.create_breaker(
            endpoint,
            failure_threshold=5,
            timeout=30.0
        )

        # Create request queue
        self.queue = RequestQueue(self.rate_limiter, endpoint)

    def enqueue_protected(self, func, *args, **kwargs):
        """Enqueue request with circuit breaker protection."""
        def protected_func():
            return self.cb_manager.call_through_breaker(
                self.endpoint,
                func,
                *args,
                **kwargs
            )

        self.queue.enqueue(protected_func)

    def start(self):
        """Start processing."""
        self.queue.start()

    def stop(self):
        """Stop processing."""
        self.queue.stop()


# ============================================================
# Example 4: Multi-Service Protection Manager
# ============================================================

class ServiceProtectionManager:
    """
    Centralized protection for multiple services.

    Provides unified interface for rate limiting and circuit breaking
    across multiple external services.
    """

    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.cb_manager = CircuitBreakerManager()
        self.services = {}

    def register_service(
        self,
        name: str,
        rate_limit_rps: float,
        rate_limit_burst: int,
        cb_failure_threshold: int = 5,
        cb_timeout: float = 60.0
    ):
        """
        Register service with protection configuration.

        Args:
            name: Service identifier
            rate_limit_rps: Requests per second
            rate_limit_burst: Burst size
            cb_failure_threshold: Circuit breaker failure threshold
            cb_timeout: Circuit breaker timeout
        """
        # Configure rate limiting
        self.rate_limiter.configure(
            endpoint=name,
            requests_per_second=rate_limit_rps,
            burst_size=rate_limit_burst
        )

        # Configure circuit breaker
        self.cb_manager.create_breaker(
            name,
            failure_threshold=cb_failure_threshold,
            timeout=cb_timeout
        )

        self.services[name] = {
            'rate_limit_rps': rate_limit_rps,
            'rate_limit_burst': rate_limit_burst,
            'cb_failure_threshold': cb_failure_threshold,
            'cb_timeout': cb_timeout
        }

        logger.info(f"Service registered: {name}")

    def call(self, service: str, func, *args, **kwargs):
        """
        Call service with all protections.

        Args:
            service: Service name
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        if service not in self.services:
            raise ValueError(f"Service not registered: {service}")

        # Apply rate limiting
        self.rate_limiter.wait_for_token(service)

        # Apply circuit breaker
        return self.cb_manager.call_through_breaker(service, func, *args, **kwargs)

    def get_service_health(self) -> dict:
        """Get health status of all services."""
        health = {}

        for service_name in self.services:
            # Rate limit metrics
            rl_metrics = self.rate_limiter.get_metrics(service_name)

            # Circuit breaker metrics
            cb_metrics = self.cb_manager.get_all_metrics().get(service_name, {})

            # Calculate health score (0-100)
            success_rate = cb_metrics.get('success_rate', 100)
            is_available = cb_metrics.get('available', True)
            rate_utilization = self.rate_limiter.get_utilization(service_name)

            health_score = success_rate if is_available else 0

            health[service_name] = {
                'healthy': is_available and success_rate > 80,
                'health_score': health_score,
                'available': is_available,
                'success_rate': success_rate,
                'rate_utilization': rate_utilization,
                'circuit_state': cb_metrics.get('state', 'unknown'),
                'total_calls': cb_metrics.get('total_calls', 0)
            }

        return health


# ============================================================
# Demo
# ============================================================

def main():
    """Demonstrate rate limiting and circuit breaker integration."""
    print("=" * 70)
    print("RATE LIMITING + CIRCUIT BREAKER INTEGRATION")
    print("=" * 70)

    # Example 1: Protected API client
    print("\n1. Protected API Client:")
    client = ProtectedAPIClient()

    # Make some calls
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        try:
            price = client.get_ticker_price(symbol)
            print(f"   {symbol}: ${price:,.2f}")
        except Exception as e:
            print(f"   {symbol}: Failed - {e}")

    # Show metrics
    metrics = client.get_metrics()
    print(f"\n   Rate limit summary:")
    print(f"     Total requests: {metrics['rate_limiting']['summary']['total_requests']}")
    print(f"     Health: {metrics['health']['health_percentage']:.1f}%")

    # Example 2: Decorator-based
    print("\n2. Decorator-Based Protection:")
    data = fetch_market_data('BTCUSDT')
    print(f"   {data}")

    # Example 3: Protected queue
    print("\n3. Protected Request Queue:")

    def mock_api_call(symbol):
        time.sleep(0.05)
        return f"Fetched {symbol}"

    queue = ProtectedRequestQueue("queue_api")
    queue.start()

    for symbol in ['BTC', 'ETH', 'SOL']:
        queue.enqueue_protected(mock_api_call, symbol)

    time.sleep(1.0)
    queue.stop()
    print("   Queue processed")

    # Example 4: Multi-service manager
    print("\n4. Multi-Service Protection Manager:")

    manager = ServiceProtectionManager()

    # Register services
    manager.register_service(
        "binance",
        rate_limit_rps=10.0,
        rate_limit_burst=20,
        cb_failure_threshold=5,
        cb_timeout=30.0
    )

    manager.register_service(
        "coinbase",
        rate_limit_rps=5.0,
        rate_limit_burst=10,
        cb_failure_threshold=3,
        cb_timeout=60.0
    )

    # Make calls through manager
    def binance_api_call():
        return {"exchange": "binance", "status": "ok"}

    def coinbase_api_call():
        return {"exchange": "coinbase", "status": "ok"}

    result1 = manager.call("binance", binance_api_call)
    result2 = manager.call("coinbase", coinbase_api_call)

    print(f"   Binance: {result1}")
    print(f"   Coinbase: {result2}")

    # Service health
    print("\n5. Service Health:")
    health = manager.get_service_health()

    for service, status in health.items():
        health_indicator = "✅" if status['healthy'] else "❌"
        print(f"\n   {health_indicator} {service}:")
        print(f"      Health score: {status['health_score']:.1f}")
        print(f"      Success rate: {status['success_rate']:.1f}%")
        print(f"      Circuit state: {status['circuit_state']}")
        print(f"      Rate utilization: {status['rate_utilization']:.1f}%")

    print("\n✅ Integration example complete!")


if __name__ == '__main__':
    main()
