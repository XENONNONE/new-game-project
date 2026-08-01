"""Integration tests for the HTTP server with a mock pipeline."""

import json
from http.server import ThreadingHTTPServer
from threading import Thread

import pytest

from inference_runtime.config import CONFIG
from inference_runtime.server import Handler, _metrics_snapshot


@pytest.fixture
def server(mock_pipeline, mock_tts, mock_qwen):
    """Start a real HTTP server on a random port with mock dependencies."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    Handler.pipeline = mock_pipeline
    Handler.test_result = None
    Handler.inference_lock = __import__("threading").Lock()

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    yield port

    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def _post(port, path, body=None, headers=None):
    import urllib.request

    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode() if body else b""
    req = urllib.request.Request(url, data=data, method="POST")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(port, path):
    import urllib.request

    url = f"http://127.0.0.1:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class TestHealthEndpoint:
    def test_health_ok(self, server):
        status, body = _get(server, "/health")
        assert status == 200
        assert body["ok"] is True

    def test_health_has_model(self, server):
        _status, body = _get(server, "/health")
        assert body["model"] == "GestureLSM MeanFlow"

    def test_health_has_fps(self, server):
        _status, body = _get(server, "/health")
        assert body["fps"] == 30

    def test_health_has_streaming(self, server):
        _status, body = _get(server, "/health")
        assert "streaming" in body
        assert body["streaming"]["enabled"] is True
        assert body["streaming"]["ladder_step"] == 1

    def test_health_has_prompt(self, server):
        _status, body = _get(server, "/health")
        assert "qwen_system_prompt" in body

    def test_health_has_dependencies(self, server):
        _status, body = _get(server, "/health")
        assert "dependencies" in body
        deps = body["dependencies"]
        assert "llm" in deps
        assert "tts" in deps
        assert "pipeline" in deps
        assert deps["pipeline"] is True


class TestReadyEndpoint:
    def test_ready_ok(self, server):
        status, body = _get(server, "/ready")
        assert status == 200
        assert body["ready"] is True


class TestMetricsEndpoint:
    def test_metrics_ok(self, server):
        import urllib.request

        url = f"http://127.0.0.1:{server}/metrics"
        with urllib.request.urlopen(url, timeout=10) as resp:
            assert resp.status == 200
            body = resp.read().decode()
        assert "requests_total" in body
        assert "# TYPE" in body

    def test_metrics_disabled(self, server, monkeypatch):
        monkeypatch.setattr(CONFIG.server, "enable_metrics", False)
        _status, _body = _get(server, "/metrics")
        assert _status == 404


class TestInferTest:
    def test_infer_test_returns_motion(self, server):
        status, body = _post(server, "/infer_test")
        assert status == 200
        assert body["fps"] == 30
        assert "frames" in body
        assert "timings" in body

    def test_infer_test_caches(self, server):
        status1, body1 = _post(server, "/infer_test")
        status2, body2 = _post(server, "/infer_test")
        assert status1 == 200 and status2 == 200
        assert body1 == body2


class TestInferEndpoint:
    def test_infer_returns_motion(self, server, sample_wav_bytes):
        import urllib.error
        import urllib.request

        url = f"http://127.0.0.1:{server}/infer"
        req = urllib.request.Request(url, data=sample_wav_bytes, method="POST")
        req.add_header("Content-Type", "application/octet-stream")
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            body = json.loads(resp.read())
        assert status == 200
        assert body["fps"] == 30
        assert "frames" in body

    def test_infer_too_large_returns_400(self, server, monkeypatch):
        import urllib.error
        import urllib.request

        import inference_runtime.server as srv

        monkeypatch.setattr(srv.CONFIG.server, "max_request_bytes", 100)
        url = f"http://127.0.0.1:{server}/infer"
        large_data = b"\x00" * 200
        req = urllib.request.Request(url, data=large_data, method="POST")
        req.add_header("Content-Type", "application/octet-stream")
        try:
            urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as e:
            assert e.code == 400


class TestStreamEndpoint:
    def _stream_post(self, port, path, body=None):
        import urllib.error
        import urllib.request

        url = f"http://127.0.0.1:{port}{path}"
        data = body if body is not None else b""
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/octet-stream")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    def test_stream_returns_sse(self, server, sample_wav_bytes):
        status, body = self._stream_post(server, "/infer_stream", sample_wav_bytes)
        assert status == 200
        assert "data:" in body
        assert '"type":"info"' in body
        assert '"type":"frames"' in body
        assert '"type":"done"' in body

    def test_stream_has_request_id(self, server, sample_wav_bytes):
        import urllib.request

        url = f"http://127.0.0.1:{server}/infer_stream"
        req = urllib.request.Request(url, data=sample_wav_bytes, method="POST")
        req.add_header("Content-Type", "application/octet-stream")
        with urllib.request.urlopen(req, timeout=10) as resp:
            headers = dict(resp.headers)
        assert headers.get("X-Request-ID") is not None

    def test_stream_disabled_returns_503(self, server, sample_wav_bytes, monkeypatch):
        import inference_runtime.server as srv

        monkeypatch.setattr(srv.CONFIG.streaming, "enabled", False)
        try:
            status, body = self._stream_post(server, "/infer_stream", sample_wav_bytes)
            assert status == 503
            assert "error" in body
        finally:
            monkeypatch.setattr(srv.CONFIG.streaming, "enabled", True)


class TestChatEndpoint:
    def test_chat_returns_speech_plan(self, server):
        status, body = _post(server, "/chat", {"message": "Hello"})
        assert status == 200
        assert "speech_plan" in body
        assert body["speech_plan"]["reply_text"] == "Echo: Hello"

    def test_chat_returns_audio(self, server):
        status, body = _post(server, "/chat", {"message": "Hi"})
        assert status == 200
        assert "audio_wav_base64" in body
        assert "audio_pcm16_base64" in body

    def test_chat_returns_tts_info(self, server):
        status, body = _post(server, "/chat", {"message": "Hi"})
        assert status == 200
        assert "tts" in body
        assert body["tts"]["backend"] == "mock"


class TestSpeakEndpoint:
    def test_speak_returns_motion(self, server, sample_speech_plan_dict):
        status, body = _post(server, "/speak", sample_speech_plan_dict)
        assert status == 200
        assert "speech_plan" in body
        assert body["speech_plan"]["emotion"] == "happy"


class TestErrorHandling:
    def test_unknown_path_404(self, server):
        status, body = _get(server, "/unknown")
        assert status == 404
        assert "error" in body

    def test_unknown_post_path_400(self, server):
        status, body = _post(server, "/unknown", {"test": 1})
        assert status == 400
        assert "error" in body

    def test_error_no_stack_trace(self, server):
        """Error responses must not leak internal stack traces."""
        _status, body = _post(server, "/unknown", {"test": 1})
        assert "Traceback" not in json.dumps(body)
        assert "File " not in json.dumps(body)

    def test_empty_message_400(self, server):
        status, body = _post(server, "/chat", {"message": ""})
        assert status == 400
        assert "error" in body


class TestRateLimiting:
    def test_rate_limit_returns_429(self, server):
        """Server should return 429 when rate limit is exceeded."""
        import inference_runtime.server as srv

        original = srv._rate_limiter.max_requests
        srv._rate_limiter.max_requests = 1
        srv._rate_limiter._hits.clear()
        try:
            _post(server, "/infer_test")
            status, body = _post(server, "/infer_test")
            assert status == 429
            assert "error" in body
        finally:
            srv._rate_limiter.max_requests = original
            srv._rate_limiter._hits.clear()


class TestSecurityHeaders:
    def test_security_headers_present(self, server):
        import urllib.request

        url = f"http://127.0.0.1:{server}/health"
        with urllib.request.urlopen(url, timeout=10) as resp:
            headers = dict(resp.headers)
        assert headers.get("X-Content-Type-Options") == "nosniff"
        assert headers.get("X-Frame-Options") == "DENY"
        assert headers.get("Referrer-Policy") == "no-referrer"
        assert "no-store" in headers.get("Cache-Control", "")
        assert "default-src" in headers.get("Content-Security-Policy", "")

    def test_options_preflight(self, server):
        import urllib.request

        url = f"http://127.0.0.1:{server}/health"
        req = urllib.request.Request(url, method="OPTIONS")
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 204

    def test_no_cors_by_default(self, server):
        import urllib.request

        url = f"http://127.0.0.1:{server}/health"
        req = urllib.request.Request(url)
        req.add_header("Origin", "http://evil.example.com")
        with urllib.request.urlopen(req, timeout=10) as resp:
            headers = dict(resp.headers)
        assert "Access-Control-Allow-Origin" not in headers


class TestMetricsTracking:
    def test_requests_incremented(self, server):
        _get(server, "/health")
        _get(server, "/health")
        snapshot = _metrics_snapshot()
        assert snapshot["requests_total"] >= 2

    def test_infer_requests_tracked(self, server):
        _post(server, "/infer_test")
        snapshot = _metrics_snapshot()
        assert snapshot["infer_requests_total"] >= 1

    def test_uptime_increases(self, server):
        import time

        s1 = _metrics_snapshot()
        time.sleep(0.1)
        s2 = _metrics_snapshot()
        assert s2["uptime_seconds"] > s1["uptime_seconds"]
