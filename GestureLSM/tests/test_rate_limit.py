"""Tests for rate limiting and circuit breaker utilities."""

import time

import pytest

from inference_runtime.rate_limit import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    RateLimiter,
)


class TestRateLimiter:
    def test_allows_under_limit(self):
        limiter = RateLimiter(max_requests=5, window_seconds=1)
        for _ in range(5):
            assert limiter.allow("client1") is True

    def test_blocks_over_limit(self):
        limiter = RateLimiter(max_requests=3, window_seconds=10)
        for _ in range(3):
            limiter.allow("client1")
        assert limiter.allow("client1") is False

    def test_separate_keys(self):
        limiter = RateLimiter(max_requests=2, window_seconds=10)
        assert limiter.allow("a") is True
        assert limiter.allow("a") is True
        assert limiter.allow("a") is False
        assert limiter.allow("b") is True
        assert limiter.allow("b") is True
        assert limiter.allow("b") is False

    def test_window_expires(self):
        limiter = RateLimiter(max_requests=2, window_seconds=0.1)
        assert limiter.allow("x") is True
        assert limiter.allow("x") is True
        assert limiter.allow("x") is False
        time.sleep(0.15)
        assert limiter.allow("x") is True

    def test_thread_safety(self):
        import threading

        limiter = RateLimiter(max_requests=100, window_seconds=10)
        results = []
        lock = threading.Lock()

        def worker():
            for _ in range(50):
                allowed = limiter.allow("shared")
                with lock:
                    results.append(allowed)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sum(results) == 100  # exactly 100 allowed


class TestCircuitBreaker:
    def test_closed_allows_calls(self):
        breaker = CircuitBreaker(failure_threshold=3, reset_timeout=0.5)
        assert breaker.state == "closed"
        result = breaker.call(lambda: "ok")
        assert result == "ok"
        assert breaker.state == "closed"

    def test_opens_after_threshold(self):
        breaker = CircuitBreaker(failure_threshold=3, reset_timeout=0.5)
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert breaker.state == "open"

    def test_open_rejects_calls(self):
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout=5)
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert breaker.state == "open"
        with pytest.raises(CircuitBreakerOpenError):
            breaker.call(lambda: "ok")

    def test_half_open_after_timeout(self):
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert breaker.state == "open"
        time.sleep(0.15)
        # Next call should transition to half_open and succeed
        result = breaker.call(lambda: "ok")
        assert result == "ok"
        assert breaker.state == "closed"

    def test_half_open_failure_reopens(self):
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        time.sleep(0.15)
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert breaker.state == "open"

    def test_success_resets_failures(self):
        breaker = CircuitBreaker(failure_threshold=5, reset_timeout=5)
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        breaker.call(lambda: "ok")
        assert breaker.state == "closed"
        # Should need 5 more failures to open
        for _ in range(4):
            with pytest.raises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert breaker.state == "closed"

    def test_only_catches_expected_exception(self):
        breaker = CircuitBreaker(
            failure_threshold=1,
            reset_timeout=5,
            expected_exception=ValueError,
        )
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("other")))
        assert breaker.state == "closed"  # RuntimeError not counted

    def test_call_with_args(self):
        breaker = CircuitBreaker(failure_threshold=3, reset_timeout=5)
        result = breaker.call(lambda x, y: x + y, 3, 4)
        assert result == 7

    def test_call_with_kwargs(self):
        breaker = CircuitBreaker(failure_threshold=3, reset_timeout=5)
        result = breaker.call(lambda x=1, y=2: x * y, x=5, y=6)
        assert result == 30
