"""Rate limiter with Redis backend and in-memory fallback.

The Redis backend is required for multi-worker / multi-instance deployments where
per-process dicts would let a client's quota reset on every worker. When no
``REDIS_URL`` is configured — or Redis is unreachable — we transparently fall
back to an in-process sliding window so single-worker dev still works.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from typing import Deque, Optional, Protocol

logger = logging.getLogger(__name__)


class RateLimiter(Protocol):
    async def allow(self, key: str, limit: int, window_seconds: int = 60) -> bool: ...


class InMemoryRateLimiter:
    """Sliding-window counter keyed by client id. Process-local."""

    def __init__(self) -> None:
        self._buckets: dict[str, Deque[float]] = defaultdict(deque)

    async def allow(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        now = time.monotonic()
        window = self._buckets[key]
        cutoff = now - window_seconds
        while window and window[0] < cutoff:
            window.popleft()

        if not window:
            self._buckets.pop(key, None)
            window = self._buckets[key]

        if len(window) >= limit:
            return False
        window.append(now)
        return True


class RedisRateLimiter:
    """Sliding-window counter backed by a Redis sorted set (ZSET)."""

    def __init__(self, client) -> None:
        self._client = client

    async def allow(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - window_seconds * 1000
        bucket = f"ratelimit:{key}"

        pipe = self._client.pipeline()
        pipe.zremrangebyscore(bucket, 0, cutoff)
        pipe.zcard(bucket)
        pipe.zadd(bucket, {str(now_ms): now_ms})
        pipe.expire(bucket, window_seconds)
        results = await pipe.execute()
        count_before_add = results[1]
        return count_before_add < limit


async def build_rate_limiter(redis_url: Optional[str] = None) -> RateLimiter:
    url = redis_url or os.getenv("REDIS_URL")
    if not url:
        logger.info("REDIS_URL not configured; using in-memory rate limiter")
        return InMemoryRateLimiter()

    try:
        import redis.asyncio as redis
    except ImportError:
        logger.warning("redis package unavailable; falling back to in-memory rate limiter")
        return InMemoryRateLimiter()

    try:
        client = redis.from_url(url, decode_responses=True)
        await client.ping()
    except Exception as exc:
        logger.warning("Redis unreachable (%s); falling back to in-memory rate limiter", exc)
        return InMemoryRateLimiter()

    logger.info("Using Redis-backed rate limiter at %s", url)
    return RedisRateLimiter(client)
