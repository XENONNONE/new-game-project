"""Local HTTP bridge for Qwen, TTS, and WAV-to-gesture inference.

Production-hardened version with:
  - Structured logging via :mod:`inference_runtime.logging_config`
  - YAML configuration with env-var overrides (:mod:`inference_runtime.config`)
  - Per-client rate limiting (:class:`inference_runtime.rate_limit.RateLimiter`)
  - Circuit breakers for Qwen and TTS (:class:`inference_runtime.rate_limit.CircuitBreaker`)
  - Retry with exponential backoff for external calls
  - Sanitized error responses (no stack traces leaked to clients)
  - Prometheus-style metrics endpoint
  - Graceful shutdown on SIGTERM/SIGINT
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import signal
import threading
import time
import uuid
import wave
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.request import Request, urlopen

from .config import CONFIG, PROJECT
from .conversation import parse_speech_plan, qwen_system_prompt
from .llm import qwen_chat
from .logging_config import get_logger
from .pipeline import GesturePipeline
from .rate_limit import CircuitBreaker, CircuitBreakerOpenError, RateLimiter
from .tts import synthesize

logger = get_logger("server")

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_metrics_lock = threading.Lock()
_metrics: dict[str, float | int] = {
    "requests_total": 0,
    "requests_4xx": 0,
    "requests_5xx": 0,
    "infer_requests_total": 0,
    "stream_requests_total": 0,
    "chat_requests_total": 0,
    "speak_requests_total": 0,
    "inference_errors_total": 0,
    "tts_errors_total": 0,
    "llm_errors_total": 0,
    "rate_limited_total": 0,
    "uptime_seconds": 0.0,
    "inference_seconds_sum": 0.0,
    "inference_seconds_count": 0,
    "stream_seconds_sum": 0.0,
    "stream_seconds_count": 0,
    "tts_seconds_sum": 0.0,
    "tts_seconds_count": 0,
    "llm_seconds_sum": 0.0,
    "llm_seconds_count": 0,
}
_start_time = time.monotonic()


def _metric_inc(name: str, value: float = 1.0) -> None:
    with _metrics_lock:
        _metrics[name] = _metrics.get(name, 0) + value


def _metric_set(name: str, value: float) -> None:
    with _metrics_lock:
        _metrics[name] = value


def _metrics_snapshot() -> dict[str, float | int]:
    with _metrics_lock:
        snapshot = dict(_metrics)
    snapshot["uptime_seconds"] = time.monotonic() - _start_time
    return snapshot


# ---------------------------------------------------------------------------
# Circuit breakers and rate limiter
# ---------------------------------------------------------------------------

_qwen_breaker = CircuitBreaker(
    failure_threshold=CONFIG.llm.circuit_breaker_failures,
    reset_timeout=CONFIG.llm.circuit_breaker_timeout,
)
_tts_breaker = CircuitBreaker(
    failure_threshold=CONFIG.tts.max_retries + 2,
    reset_timeout=CONFIG.tts.retry_backoff * 5,
)
_rate_limiter = RateLimiter(
    max_requests=CONFIG.server.rate_limit_per_minute,
    window_seconds=60,
)


def _retry_with_backoff(
    func: Callable[[], Any],
    max_retries: int,
    backoff: float,
    breaker: CircuitBreaker | None = None,
) -> Any:
    """Call *func* with exponential backoff and optional circuit breaker."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            if breaker is not None:
                return breaker.call(func)
            return func()
        except CircuitBreakerOpenError:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = backoff**attempt
                logger.warning(
                    "Retry %d/%d after %.1fs: %s",
                    attempt + 1,
                    max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    pipeline: GesturePipeline
    test_result: dict | None = None
    inference_lock = threading.Lock()
    server_version = "GestureLSM-Avatar/2.0"
    timeout = CONFIG.server.request_timeout

    # -- helpers -----------------------------------------------------------

    def _client_id(self) -> str:
        return self.client_address[0] if self.client_address else "unknown"

    def _request_id(self) -> str:
        return self.headers.get("X-Request-ID", uuid.uuid4().hex[:12])

    def _check_rate_limit(self) -> bool:
        return _rate_limiter.allow(self._client_id())

    def _security_headers(self) -> None:
        """Add security and CORS headers to the response."""
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("X-XSS-Protection", "1; mode=block")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Content-Security-Policy", "default-src 'none'")
        origin = self.headers.get("Origin", "")
        if origin and self._cors_allowed(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Request-ID")

    def _cors_allowed(self, origin: str) -> bool:
        allowed = CONFIG.server.cors_origins
        if not allowed:
            return False
        if allowed == "*":
            return True
        return origin in (o.strip() for o in allowed.split(",") if o.strip())

    def reply(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Request-ID", self._request_id())
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str, log_detail: str = "") -> None:
        """Send a sanitized error response; log the detail server-side only."""
        if log_detail:
            logger.error("HTTP %d: %s", status, log_detail)
        else:
            logger.error("HTTP %d: %s", status, message)
        self.reply(status, {"error": message})

    def _read_body(self, max_size: int) -> bytes:
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= max_size:
            raise ValueError(f"Body must be 1 byte to {max_size} bytes")
        return self.rfile.read(size)

    # -- routes ------------------------------------------------------------

    def do_GET(self) -> None:
        _metric_inc("requests_total")
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/metrics":
            self._handle_metrics()
        elif self.path == "/ready":
            self._handle_ready()
        else:
            _metric_inc("requests_4xx")
            self._error(404, "Not found")

    def do_OPTIONS(self) -> None:
        _metric_inc("requests_total")
        self.send_response(204)
        self._security_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        _metric_inc("requests_total")
        if not self._check_rate_limit():
            _metric_inc("rate_limited_total")
            self._error(429, "Rate limit exceeded")
            return
        try:
            if self.path == "/infer_test":
                self._handle_infer_test()
            elif self.path in ("/speak", "/chat"):
                self._handle_speak_or_chat()
            elif self.path == "/infer":
                self._handle_infer()
            elif self.path == "/infer_stream":
                self._handle_infer_stream()
            else:
                _metric_inc("requests_4xx")
                self._error(400, "Use POST /infer, /infer_stream, /infer_test, /speak, or /chat")
        except ValueError as exc:
            _metric_inc("requests_4xx")
            self._error(400, str(exc))
        except CircuitBreakerOpenError:
            _metric_inc("requests_5xx")
            self._error(503, "Service temporarily unavailable")
        except Exception as exc:
            _metric_inc("requests_5xx")
            self._error(500, "Internal server error", log_detail=f"{type(exc).__name__}: {exc}")

    # -- route handlers ----------------------------------------------------

    def _handle_health(self) -> None:
        payload: dict[str, Any] = {"ok": True}
        if CONFIG.server.enable_health_details:
            payload.update(
                {
                    "model": "GestureLSM MeanFlow",
                    "fps": CONFIG.streaming.stream_fps,
                    "streaming": {
                        "enabled": CONFIG.streaming.enabled,
                        "window_samples": CONFIG.streaming.window_samples,
                        "overlap_frames": CONFIG.streaming.overlap_frames,
                        "ladder_step": CONFIG.streaming.ladder_step,
                        "ladder_strategy": CONFIG.streaming.ladder_strategy,
                    },
                    "qwen_system_prompt": qwen_system_prompt(),
                    "metrics": _metrics_snapshot() if CONFIG.server.enable_metrics else None,
                }
            )
            payload["dependencies"] = self._check_dependencies()
        self.reply(200, payload)

    def _check_dependencies(self) -> dict[str, Any]:
        """Check connectivity to external services (Qwen LLM, TTS)."""
        deps: dict[str, Any] = {}
        deps["llm"] = self._check_llm()
        deps["tts"] = self._check_tts()
        deps["pipeline"] = self.pipeline is not None
        return deps

    def _check_llm(self) -> dict[str, Any]:
        if _qwen_breaker.state == CircuitBreaker.OPEN:
            return {"status": "open", "available": False}
        try:
            req = Request(
                CONFIG.llm.chat_url.replace("/chat/completions", "/models"),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=5) as resp:
                ok = resp.status < 500
        except Exception:
            return {"status": "unreachable", "available": False}
        return {"status": "ok" if ok else "error", "available": ok}

    def _check_tts(self) -> dict[str, Any]:
        if _tts_breaker.state == CircuitBreaker.OPEN:
            return {"status": "open", "available": False}
        backend = CONFIG.tts.backend.lower()
        if backend == "command":
            available = bool(CONFIG.tts.command)
            return {"status": "ok" if available else "unconfigured", "available": available}
        try:
            if backend == "kokoro":
                import kokoro  # noqa: F401
            elif backend == "kitten":
                import kittentts  # noqa: F401
        except ImportError:
            return {"status": "import_error", "available": False}
        return {"status": "ok", "available": True}

    def _handle_metrics(self) -> None:
        if not CONFIG.server.enable_metrics:
            self._error(404, "Metrics disabled")
            return
        lines = []
        for key, value in _metrics_snapshot().items():
            if isinstance(value, (int, float)):
                lines.append(f"# TYPE {key} gauge\n{key} {value}")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self._security_headers()
        body = "\n".join(lines).encode()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_ready(self) -> None:
        ready = self.pipeline is not None
        self.reply(200 if ready else 503, {"ready": ready})

    def _handle_infer_test(self) -> None:
        _metric_inc("infer_requests_total")
        data = (PROJECT / "test.wav").read_bytes()
        if self.test_result is None:
            with self.inference_lock:
                if self.test_result is None:
                    started = time.perf_counter()
                    self.test_result = self.pipeline.infer_wav(data)
                    elapsed = time.perf_counter() - started
                    _metric_set("inference_seconds_sum", elapsed)
                    _metric_inc("inference_seconds_count")
        self.reply(200, self.test_result)

    def _handle_infer(self) -> None:
        _metric_inc("infer_requests_total")
        data = self._read_body(CONFIG.server.max_request_bytes)
        started = time.perf_counter()
        with self.inference_lock:
            result = self.pipeline.infer_wav(data)
        elapsed = time.perf_counter() - started
        _metric_set("inference_seconds_sum", elapsed)
        _metric_inc("inference_seconds_count")
        self.reply(200, result)

    def _handle_infer_stream(self) -> None:
        """Stream gesture frames as Server-Sent Events (SSE).

        Accepts raw WAV bytes in the POST body.  Each event is a JSON dict
        with ``type`` ("info", "frames", or "done").  Clients should listen
        for ``done`` to know when the stream is complete.
        """
        _metric_inc("stream_requests_total")
        if not CONFIG.streaming.enabled:
            self._error(503, "Streaming inference is disabled")
            return
        data = self._read_body(CONFIG.server.max_request_bytes)
        request_id = self._request_id()
        logger.info(
            "Streaming inference started (request_id=%s, bytes=%d)",
            request_id,
            len(data),
        )
        started = time.perf_counter()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Request-ID", request_id)
        self._security_headers()
        self.end_headers()

        def _send_event(event: dict) -> None:
            payload = json.dumps(event, separators=(",", ":"))
            self.wfile.write(f"data: {payload}\n\n".encode())
            self.wfile.flush()

        try:
            with self.inference_lock:
                for event in self.pipeline.infer_stream_wav(data):
                    _send_event(event)
                    if event.get("type") == "done":
                        break
        except BrokenPipeError:
            logger.info("Client disconnected during streaming (request_id=%s)", request_id)
        except Exception:
            _metric_inc("inference_errors_total")
            logger.exception("Streaming inference failed (request_id=%s)", request_id)
            error_payload = json.dumps({"type": "error", "error": "Internal server error"})
            try:
                self.wfile.write(f"data: {error_payload}\n\n".encode())
                self.wfile.flush()
            except BrokenPipeError:
                pass
            return
        elapsed = time.perf_counter() - started
        _metric_set("stream_seconds_sum", elapsed)
        _metric_inc("stream_seconds_count")
        logger.info(
            "Streaming inference complete (request_id=%s, %.2fs)",
            request_id,
            elapsed,
        )

    def _handle_speak_or_chat(self) -> None:
        is_chat = self.path == "/chat"
        if is_chat:
            _metric_inc("chat_requests_total")
        else:
            _metric_inc("speak_requests_total")
        body = self._read_body(CONFIG.server.max_chat_bytes)
        payload = json.loads(body.decode("utf-8"))
        if is_chat:
            plan = self._qwen_with_retry(str(payload.get("message", "")))
        else:
            plan = parse_speech_plan(payload)
        wav, tts_meta = self._tts_with_retry(plan)
        started = time.perf_counter()
        with self.inference_lock:
            motion = self.pipeline.infer_wav(wav)
        elapsed = time.perf_counter() - started
        _metric_set("inference_seconds_sum", elapsed)
        _metric_inc("inference_seconds_count")
        motion["speech_plan"] = plan.to_dict()
        motion["tts"] = tts_meta
        motion["audio_wav_base64"] = base64.b64encode(wav).decode("ascii")
        with wave.open(io.BytesIO(wav), "rb") as audio:
            motion["audio_pcm16_base64"] = base64.b64encode(
                audio.readframes(audio.getnframes())
            ).decode("ascii")
        self.reply(200, motion)

    # -- external service calls with retries -------------------------------

    def _qwen_with_retry(self, message: str) -> Any:
        started = time.perf_counter()
        try:
            plan = _retry_with_backoff(
                lambda: qwen_chat(message),
                max_retries=CONFIG.llm.max_retries,
                backoff=CONFIG.llm.retry_backoff,
                breaker=_qwen_breaker,
            )
        except Exception:
            _metric_inc("llm_errors_total")
            raise
        elapsed = time.perf_counter() - started
        _metric_set("llm_seconds_sum", elapsed)
        _metric_inc("llm_seconds_count")
        return plan

    def _tts_with_retry(self, plan: Any) -> tuple[bytes, dict]:
        started = time.perf_counter()
        try:
            wav, meta = _retry_with_backoff(
                lambda: synthesize(plan),
                max_retries=CONFIG.tts.max_retries,
                backoff=CONFIG.tts.retry_backoff,
                breaker=_tts_breaker,
            )
        except Exception:
            _metric_inc("tts_errors_total")
            raise
        elapsed = time.perf_counter() - started
        _metric_set("tts_seconds_sum", elapsed)
        _metric_inc("tts_seconds_count")
        return wav, meta

    # -- logging -----------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("HTTP %s %s", self.command, self.path)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

_server: ThreadingHTTPServer | None = None
_shutdown_requested = threading.Event()


def _signal_handler(signum: int, _frame: object) -> None:
    logger.info("Received signal %d, shutting down...", signum)
    _shutdown_requested.set()
    if _server is not None:
        threading.Thread(target=_server.shutdown, daemon=True).start()


def main() -> None:
    global _server
    p = argparse.ArgumentParser(description="GestureLSM avatar HTTP server")
    p.add_argument("--host", default=CONFIG.server.host)
    p.add_argument("--port", type=int, default=CONFIG.server.port)
    p.add_argument("--threads", type=int, default=CONFIG.server.threads)
    p.add_argument("--timeout", type=float, default=CONFIG.server.timeout)
    args = p.parse_args()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    logger.info("Initializing GesturePipeline (threads=%s)...", args.threads)
    Handler.pipeline = GesturePipeline(args.threads)
    logger.info("Pipeline initialized in %.2fs", Handler.pipeline.timings.get("load_s", 0))

    _server = ThreadingHTTPServer((args.host, args.port), Handler)
    _server.daemon_threads = True
    _server.timeout = args.timeout

    logger.info("GestureLSM ready at http://%s:%d", args.host, args.port)
    try:
        _server.serve_forever()
    except Exception:
        logger.exception("Server crashed")
        raise
    finally:
        _server.server_close()
        logger.info("Server stopped")


if __name__ == "__main__":
    main()
