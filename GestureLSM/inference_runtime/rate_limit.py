"""Rate limiting and circuit breaker utilities for the HTTP server."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any


class RateLimiter:
    """Sliding-window rate limiter keyed by client identifier.

    Tracks request timestamps per key and rejects when the count within
    the window exceeds ``max_requests``.  Thread-safe.
    """

    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            cutoff = now - self.window
            while hits and hits[0] < cutoff:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return False
            hits.append(now)
            return True


class CircuitBreaker:
    """Fail-fast circuit breaker for external service calls.

    States:
      - **closed**: calls pass through; failures are counted.
      - **open**: calls fail immediately until ``reset_timeout`` elapses.
      - **half_open**: one trial call is allowed; success closes the circuit,
        failure re-opens it.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        expected_exception: type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.expected_exception = expected_exception
        self._state = self.CLOSED
        self._failures = 0
        self._last_failure_time: float = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        return self._state

    def _can_attempt(self) -> bool:
        if self._state == self.CLOSED:
            return True
        if self._state == self.OPEN:
            if time.monotonic() - self._last_failure_time >= self.reset_timeout:
                self._state = self.HALF_OPEN
                return True
            return False
        return True  # half_open

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            if not self._can_attempt():
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is open (failures={self._failures})"
                )
        try:
            result = func(*args, **kwargs)
        except self.expected_exception:
            self._record_failure()
            raise
        self._record_success()
        return result

    def _record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = self.CLOSED

    def _record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.monotonic()
            if self._failures >= self.failure_threshold:
                self._state = self.OPEN


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is open and rejects a call."""
