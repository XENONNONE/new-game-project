"""Tests for the configuration module."""

from inference_runtime.config import (
    PROJECT,
    ROOT,
    Config,
    LLMConfig,
    PipelineConfig,
    ServerConfig,
    StreamingConfig,
    TTSConfig,
    load_config,
)


class TestServerConfig:
    def test_defaults(self):
        cfg = ServerConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8765
        assert cfg.threads == 6
        assert cfg.timeout == 300.0
        assert cfg.max_request_bytes == 33554432
        assert cfg.max_chat_bytes == 1048576
        assert cfg.rate_limit_per_minute == 60
        assert cfg.enable_metrics is True
        assert cfg.enable_health_details is True


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.model_name == "meanflow"
        assert cfg.seed == 42
        assert cfg.cache_infer_test is True

    def test_torch_compile_default(self):
        cfg = PipelineConfig()
        assert cfg.torch_compile is False


class TestStreamingConfig:
    def test_defaults(self):
        cfg = StreamingConfig()
        assert cfg.enabled is True
        assert cfg.window_samples == 68224
        assert cfg.overlap_frames == 16
        assert cfg.stream_fps == 30
        assert cfg.ladder_step == 1
        assert cfg.ladder_strategy == "down"
        assert cfg.max_concurrent_windows == 1
        assert cfg.chunk_duration_s == 4.264
        assert cfg.request_timeout_s == 300.0


class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig()
        assert cfg.chat_url == "http://127.0.0.1:8080/v1/chat/completions"
        assert cfg.model == "local-qwen"
        assert cfg.timeout == 120.0
        assert cfg.max_retries == 3
        assert cfg.retry_backoff == 1.5
        assert cfg.circuit_breaker_failures == 5
        assert cfg.circuit_breaker_timeout == 30.0


class TestTTSConfig:
    def test_defaults(self):
        cfg = TTSConfig()
        assert cfg.backend == "kokoro"
        assert cfg.lang == "a"
        assert cfg.timeout == 120
        assert cfg.max_retries == 2


class TestConfigLoading:
    def test_load_config_returns_config(self):
        cfg = load_config()
        assert isinstance(cfg, Config)
        assert isinstance(cfg.server, ServerConfig)
        assert isinstance(cfg.pipeline, PipelineConfig)
        assert isinstance(cfg.streaming, StreamingConfig)
        assert isinstance(cfg.llm, LLMConfig)
        assert isinstance(cfg.tts, TTSConfig)

    def test_config_has_environment(self):
        cfg = load_config()
        assert cfg.environment == "production"

    def test_config_log_level(self):
        cfg = load_config()
        assert cfg.log_level in ("INFO", "DEBUG")


class TestEnvOverrides:
    def test_env_override_port(self, monkeypatch):
        monkeypatch.setenv("AVATAR_SERVER_PORT", "9999")
        cfg = load_config()
        assert cfg.server.port == 9999

    def test_env_override_host(self, monkeypatch):
        monkeypatch.setenv("AVATAR_SERVER_HOST", "127.0.0.1")
        cfg = load_config()
        assert cfg.server.host == "127.0.0.1"

    def test_env_override_log_level(self, monkeypatch):
        monkeypatch.setenv("AVATAR_LOG_LEVEL", "DEBUG")
        cfg = load_config()
        assert cfg.log_level == "DEBUG"

    def test_env_override_llm_url(self, monkeypatch):
        monkeypatch.setenv("AVATAR_LLM_CHAT_URL", "http://localhost:9999/v1/chat")
        cfg = load_config()
        assert cfg.llm.chat_url == "http://localhost:9999/v1/chat"

    def test_env_override_tts_backend(self, monkeypatch):
        monkeypatch.setenv("AVATAR_TTS_BACKEND", "kitten")
        cfg = load_config()
        assert cfg.tts.backend == "kitten"

    def test_env_override_streaming(self, monkeypatch):
        monkeypatch.setenv("AVATAR_STREAMING_ENABLED", "0")
        monkeypatch.setenv("AVATAR_STREAMING_LADDER_STEP", "2")
        monkeypatch.setenv("AVATAR_STREAMING_OVERLAP_FRAMES", "8")
        cfg = load_config()
        assert cfg.streaming.enabled is False
        assert cfg.streaming.ladder_step == 2
        assert cfg.streaming.overlap_frames == 8

    def test_env_override_bool(self, monkeypatch):
        monkeypatch.setenv("AVATAR_SERVER_ENABLE_METRICS", "0")
        cfg = load_config()
        assert cfg.server.enable_metrics is False

    def test_env_override_int(self, monkeypatch):
        monkeypatch.setenv("AVATAR_SERVER_THREADS", "12")
        cfg = load_config()
        assert cfg.server.threads == 12

    def test_env_override_float(self, monkeypatch):
        monkeypatch.setenv("AVATAR_SERVER_TIMEOUT", "600.0")
        cfg = load_config()
        assert cfg.server.timeout == 600.0

    def test_env_override_torch_compile(self, monkeypatch):
        monkeypatch.setenv("AVATAR_PIPELINE_TORCH_COMPILE", "1")
        cfg = load_config()
        assert cfg.pipeline.torch_compile is True

    def test_invalid_env_value_ignored(self, monkeypatch):
        monkeypatch.setenv("AVATAR_SERVER_PORT", "not-a-number")
        cfg = load_config()
        assert cfg.server.port == 8765  # default preserved


class TestProjectPaths:
    def test_root_is_gesture_lsm_dir(self):
        """ROOT should be the GestureLSM directory (parent of inference_runtime)."""
        assert ROOT.name == "GestureLSM"

    def test_project_is_parent_of_root(self):
        """PROJECT should be the parent of ROOT (the repo root)."""
        assert ROOT.parent == PROJECT

    def test_config_paths_includes_root(self):
        from inference_runtime.config import DEFAULT_CONFIG_PATHS

        assert any(str(p).endswith("config.yaml") for p in DEFAULT_CONFIG_PATHS)
        assert ROOT / "config.yaml" in DEFAULT_CONFIG_PATHS
