"""
Token-bucket rate limiter.

Maps to OWASP ASVS 11.1 (Business Logic Security):
- 11.1.4 Resource consumption limits enforced
- 11.1.5 Anti-automation controls on sensitive operations

This is an in-process implementation suitable for single-replica deployments.
For multi-replica deployments, swap the backing store for Redis with
`SET key value EX ttl NX` semantics.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass

from chromatic.exceptions import RateLimitExceededError


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class TokenBucketRateLimiter:
    """Per-principal token-bucket rate limiter.

    Each principal gets a bucket that refills at `rate_per_minute / 60` tokens
    per second, up to `burst` tokens. A request consumes one token; if no
    tokens are available, the request is rejected.

    Thread-safe via a single lock; for high-throughput multi-process deployments,
    move state to Redis.
    """

    def __init__(self, rate_per_minute: int, burst: int | None = None) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        self._rate_per_second = rate_per_minute / 60.0
        self._capacity = float(burst if burst is not None else rate_per_minute)
        self._buckets: dict[str, _Bucket] = defaultdict(
            lambda: _Bucket(tokens=self._capacity, last_refill=time.monotonic())
        )
        self._lock = threading.Lock()

    def acquire(self, principal_id: str) -> None:
        """Consume one token from the principal's bucket.

        Raises:
            RateLimitExceededError: No tokens available.
        """
        with self._lock:
            # Capture `now` *inside* the lock so it's always >= the timestamp
            # of any bucket created by the defaultdict factory in the same
            # critical section. Capturing it outside the lock can yield a
            # negative `elapsed` and incorrectly drain the bucket.
            now = time.monotonic()
            bucket = self._buckets[principal_id]
            elapsed = max(0.0, now - bucket.last_refill)
            bucket.tokens = min(
                self._capacity, bucket.tokens + elapsed * self._rate_per_second
            )
            bucket.last_refill = now
            if bucket.tokens < 1.0:
                raise RateLimitExceededError(
                    "rate limit exceeded"  # deliberately generic
                )
            bucket.tokens -= 1.0
