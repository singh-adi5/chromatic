"""Token-bucket rate limiter tests."""

from __future__ import annotations

import time

import pytest

from chromatic.exceptions import RateLimitExceededError
from chromatic.security import TokenBucketRateLimiter

pytestmark = pytest.mark.security


def test_allows_within_burst() -> None:
    limiter = TokenBucketRateLimiter(rate_per_minute=60, burst=60)
    for _ in range(60):
        limiter.acquire("alice")


def test_rejects_over_burst() -> None:
    limiter = TokenBucketRateLimiter(rate_per_minute=60, burst=3)
    limiter.acquire("alice")
    limiter.acquire("alice")
    limiter.acquire("alice")
    with pytest.raises(RateLimitExceededError):
        limiter.acquire("alice")


def test_independent_principals() -> None:
    """Per-principal isolation: one user does not affect another.

    Use a slow refill rate (6/min = 1/10s) so the burst stays exhausted
    for the duration of the test.
    """
    limiter = TokenBucketRateLimiter(rate_per_minute=6, burst=1)
    limiter.acquire("alice")
    with pytest.raises(RateLimitExceededError):
        limiter.acquire("alice")
    limiter.acquire("bob")
    with pytest.raises(RateLimitExceededError):
        limiter.acquire("bob")


def test_bucket_refills_over_time() -> None:
    """At 600/min the bucket refills 10 tokens per second."""
    limiter = TokenBucketRateLimiter(rate_per_minute=600, burst=1)
    limiter.acquire("alice")
    with pytest.raises(RateLimitExceededError):
        limiter.acquire("alice")
    time.sleep(0.25)  # > 100 ms — refill should produce a token
    limiter.acquire("alice")


def test_rejects_invalid_rate() -> None:
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(rate_per_minute=0)
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(rate_per_minute=-1)
