"""
Advanced Rate Limiting

Implements token bucket algorithm and request queuing for API rate limiting.
"""
import time
import asyncio
import threading
from typing import Dict, Optional, Callable
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_second: float
    burst_size: int  # Maximum tokens that can accumulate
    endpoint: Optional[str] = None


class TokenBucket:
    """
    Token bucket algorithm implementation.

    Allows burst traffic while maintaining average rate.
    """

    def __init__(
        self,
        rate: float,  # Tokens per second
        capacity: int,  # Bucket capacity (burst size)
        initial_tokens: Optional[int] = None
    ):
        """
        Initialize token bucket.

        Args:
            rate: Token refill rate (per second)
            capacity: Maximum tokens (burst size)
            initial_tokens: Initial token count (defaults to capacity)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = initial_tokens if initial_tokens is not None else capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        with self.lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    def wait_time(self, tokens: int = 1) -> float:
        """
        Calculate wait time until tokens are available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Wait time in seconds
        """
        with self.lock:
            self._refill()

            if self.tokens >= tokens:
                return 0.0

            tokens_needed = tokens - self.tokens
            return tokens_needed / self.rate

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        # Add tokens based on elapsed time
        new_tokens = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def get_tokens(self) -> float:
        """Get current token count."""
        with self.lock:
            self._refill()
            return self.tokens


class RateLimiter:
    """
    Rate limiter with token bucket algorithm.

    Features:
    - Per-endpoint rate limiting
    - Burst traffic support
    - Request queuing
    - Metrics tracking
    """

    def __init__(self):
        self.buckets: Dict[str, TokenBucket] = {}
        self.metrics: Dict[str, Dict] = {}
        self.lock = threading.Lock()

    def configure(
        self,
        endpoint: str,
        requests_per_second: float,
        burst_size: int
    ):
        """
        Configure rate limit for endpoint.

        Args:
            endpoint: API endpoint identifier
            requests_per_second: Request rate limit
            burst_size: Maximum burst size
        """
        with self.lock:
            self.buckets[endpoint] = TokenBucket(
                rate=requests_per_second,
                capacity=burst_size
            )
            self.metrics[endpoint] = {
                'total_requests': 0,
                'throttled_requests': 0,
                'total_wait_time': 0.0,
                'last_request_time': None
            }

        logger.info(
            f"Configured rate limit for {endpoint}: "
            f"{requests_per_second} req/s, burst {burst_size}"
        )

    def allow_request(self, endpoint: str, tokens: int = 1) -> bool:
        """
        Check if request is allowed (non-blocking).

        Args:
            endpoint: API endpoint identifier
            tokens: Number of tokens to consume

        Returns:
            True if request is allowed
        """
        bucket = self._get_bucket(endpoint)
        if not bucket:
            return True  # No limit configured

        allowed = bucket.consume(tokens)

        # Update metrics
        self.metrics[endpoint]['total_requests'] += 1
        if not allowed:
            self.metrics[endpoint]['throttled_requests'] += 1
        self.metrics[endpoint]['last_request_time'] = datetime.utcnow()

        return allowed

    def wait_for_token(self, endpoint: str, tokens: int = 1, timeout: Optional[float] = None):
        """
        Wait until request is allowed (blocking).

        Args:
            endpoint: API endpoint identifier
            tokens: Number of tokens needed
            timeout: Maximum wait time (seconds)

        Raises:
            TimeoutError: If timeout is reached
        """
        bucket = self._get_bucket(endpoint)
        if not bucket:
            return  # No limit configured

        start_time = time.time()

        while True:
            if bucket.consume(tokens):
                # Update metrics
                wait_time = time.time() - start_time
                self.metrics[endpoint]['total_wait_time'] += wait_time
                return

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Rate limit timeout for {endpoint}")

            # Wait for next token
            wait_time = bucket.wait_time(tokens)
            time.sleep(min(wait_time, 0.1))  # Sleep in small increments

    async def wait_for_token_async(
        self,
        endpoint: str,
        tokens: int = 1,
        timeout: Optional[float] = None
    ):
        """Async version of wait_for_token."""
        bucket = self._get_bucket(endpoint)
        if not bucket:
            return

        start_time = time.time()

        while True:
            if bucket.consume(tokens):
                wait_time = time.time() - start_time
                self.metrics[endpoint]['total_wait_time'] += wait_time
                return

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Rate limit timeout for {endpoint}")

            wait_time = bucket.wait_time(tokens)
            await asyncio.sleep(min(wait_time, 0.1))

    def get_metrics(self, endpoint: Optional[str] = None) -> Dict:
        """
        Get rate limit metrics.

        Args:
            endpoint: Specific endpoint or None for all

        Returns:
            Metrics dictionary
        """
        if endpoint:
            return self.metrics.get(endpoint, {})

        # Return all metrics
        return {
            'endpoints': self.metrics,
            'summary': {
                'total_requests': sum(m['total_requests'] for m in self.metrics.values()),
                'total_throttled': sum(m['throttled_requests'] for m in self.metrics.values()),
                'total_wait_time': sum(m['total_wait_time'] for m in self.metrics.values())
            }
        }

    def get_utilization(self, endpoint: str) -> float:
        """
        Get rate limit utilization percentage.

        Args:
            endpoint: API endpoint identifier

        Returns:
            Utilization percentage (0-100)
        """
        bucket = self._get_bucket(endpoint)
        if not bucket:
            return 0.0

        return ((bucket.capacity - bucket.get_tokens()) / bucket.capacity) * 100

    def reset_metrics(self, endpoint: Optional[str] = None):
        """Reset metrics for endpoint(s)."""
        if endpoint:
            if endpoint in self.metrics:
                self.metrics[endpoint] = {
                    'total_requests': 0,
                    'throttled_requests': 0,
                    'total_wait_time': 0.0,
                    'last_request_time': None
                }
        else:
            for endpoint in self.metrics:
                self.reset_metrics(endpoint)

    def _get_bucket(self, endpoint: str) -> Optional[TokenBucket]:
        """Get token bucket for endpoint."""
        return self.buckets.get(endpoint)


class RequestQueue:
    """
    Request queue with rate limiting and prioritization.

    Features:
    - FIFO queue with priority support
    - Automatic rate limiting
    - Request batching
    - Timeout handling
    """

    def __init__(self, rate_limiter: RateLimiter, endpoint: str):
        """
        Initialize request queue.

        Args:
            rate_limiter: Rate limiter instance
            endpoint: API endpoint identifier
        """
        self.rate_limiter = rate_limiter
        self.endpoint = endpoint
        self.queue = deque()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker_thread = None

    def enqueue(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        priority: int = 0,
        callback: Optional[Callable] = None
    ):
        """
        Add request to queue.

        Args:
            func: Function to call
            args: Function arguments
            kwargs: Function keyword arguments
            priority: Request priority (higher = more urgent)
            callback: Callback for result
        """
        kwargs = kwargs or {}

        request = {
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'priority': priority,
            'callback': callback,
            'enqueue_time': time.time()
        }

        with self.lock:
            # Insert based on priority
            inserted = False
            for i, req in enumerate(self.queue):
                if priority > req['priority']:
                    self.queue.insert(i, request)
                    inserted = True
                    break

            if not inserted:
                self.queue.append(request)

    def start(self):
        """Start processing queue."""
        if self.worker_thread and self.worker_thread.is_alive():
            return

        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()

        logger.info(f"Request queue started for {self.endpoint}")

    def stop(self):
        """Stop processing queue."""
        self.stop_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)

        logger.info(f"Request queue stopped for {self.endpoint}")

    def _process_queue(self):
        """Process queued requests."""
        while not self.stop_event.is_set():
            request = None

            with self.lock:
                if self.queue:
                    request = self.queue.popleft()

            if request:
                try:
                    # Wait for rate limit
                    self.rate_limiter.wait_for_token(self.endpoint)

                    # Execute request
                    result = request['func'](*request['args'], **request['kwargs'])

                    # Call callback if provided
                    if request['callback']:
                        request['callback'](result)

                    # Log queue time
                    queue_time = time.time() - request['enqueue_time']
                    if queue_time > 1.0:
                        logger.warning(
                            f"Request queued for {queue_time:.1f}s: {self.endpoint}"
                        )

                except Exception as e:
                    logger.error(f"Request execution failed: {e}")

            else:
                # No requests, sleep briefly
                time.sleep(0.01)


# Global rate limiter instance
_global_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    return _global_rate_limiter


def rate_limited(endpoint: str, tokens: int = 1):
    """
    Decorator for rate-limited functions.

    Args:
        endpoint: API endpoint identifier
        tokens: Number of tokens to consume

    Usage:
        @rate_limited("binance_orders", tokens=1)
        def place_order(symbol, side, quantity):
            return exchange.place_order(...)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            limiter = get_rate_limiter()
            limiter.wait_for_token(endpoint, tokens)
            return func(*args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("RATE LIMITER TEST")
    print("=" * 70)

    # Create rate limiter
    limiter = RateLimiter()

    # Configure endpoints
    print("\n1. Configuring rate limits:")
    limiter.configure("binance_api", requests_per_second=10.0, burst_size=20)
    limiter.configure("binance_orders", requests_per_second=5.0, burst_size=10)
    print("   Configured endpoints")

    # Test token bucket
    print("\n2. Testing token bucket (10 req/s, burst 20):")
    allowed_count = 0
    throttled_count = 0

    # Burst test
    for i in range(25):
        if limiter.allow_request("binance_api"):
            allowed_count += 1
        else:
            throttled_count += 1

    print(f"   Allowed: {allowed_count} (expected ~20 for burst)")
    print(f"   Throttled: {throttled_count} (expected ~5)")

    # Test wait_for_token
    print("\n3. Testing blocking wait:")
    start = time.time()
    limiter.wait_for_token("binance_api")
    elapsed = time.time() - start
    print(f"   Waited {elapsed:.3f}s for token")

    # Test metrics
    print("\n4. Rate limit metrics:")
    metrics = limiter.get_metrics()
    print(f"   Total requests: {metrics['summary']['total_requests']}")
    print(f"   Total throttled: {metrics['summary']['total_throttled']}")
    print(f"   Total wait time: {metrics['summary']['total_wait_time']:.2f}s")

    for endpoint, data in metrics['endpoints'].items():
        utilization = limiter.get_utilization(endpoint)
        print(f"\n   {endpoint}:")
        print(f"     Requests: {data['total_requests']}")
        print(f"     Throttled: {data['throttled_requests']}")
        print(f"     Utilization: {utilization:.1f}%")

    # Test rate-limited decorator
    print("\n5. Testing decorator:")

    @rate_limited("binance_api", tokens=1)
    def fetch_price(symbol):
        return f"Price for {symbol}: $45000"

    result = fetch_price("BTC")
    print(f"   {result}")

    # Test request queue
    print("\n6. Testing request queue:")

    def mock_api_call(symbol):
        time.sleep(0.05)  # Simulate API call
        return f"Fetched {symbol}"

    queue = RequestQueue(limiter, "binance_api")
    queue.start()

    # Enqueue requests
    results = []

    def collect_result(result):
        results.append(result)

    for symbol in ['BTC', 'ETH', 'SOL']:
        queue.enqueue(mock_api_call, args=(symbol,), callback=collect_result)

    # Wait for processing
    time.sleep(2.0)
    queue.stop()

    print(f"   Processed: {len(results)} requests")
    for result in results:
        print(f"     {result}")

    print("\n✅ Rate limiter test complete!")
