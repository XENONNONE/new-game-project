"""YAML configuration with environment-variable overrides.

Supports a ``config.yaml`` file at the project root or any path specified
by ``AVATAR_CONFIG``.  Every value can be overridden by an environment
variable of the form ``AVATAR_<SECTION>_<KEY>`` (upper-cased, ``_``
separator).  The HTTP server, pipeline, and TTS modules all read from
the shared :data:`CONFIG` singleton.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .logging_config import get_logger

logger = get_logger("config")

PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATHS = [
    Path(os.environ.get("AVATAR_CONFIG", "")),
    ROOT / "config.yaml",
    PROJECT / "config.yaml",
    PROJECT / "GestureLSM" / "config.yaml",
    Path("/etc/avatar/config.yaml"),
]


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    threads: int = 6
    timeout: float = 300.0
    max_request_bytes: int = 33554432
    max_chat_bytes: int = 1048576
    rate_limit_per_minute: int = 60
    enable_metrics: bool = True
    enable_health_details: bool = True
    cors_origins: str = ""
    request_timeout: float = 300.0


@dataclass
class PipelineConfig:
    model_name: str = "meanflow"
    checkpoint_dir: str = ""
    threads: int | None = None
    seed: int = 42
    cache_infer_test: bool = True
    torch_compile: bool = False
    use_onnx: bool = False


@dataclass
class StreamingConfig:
    """Configuration for rolling/streaming gesture inference.

    Inspired by Rolling Diffusion (AAAI 2026) — processes audio in
    overlapping windows with seed-frame carryover and optional ladder
    acceleration for reduced latency.
    """

    enabled: bool = True
    window_samples: int = 68224
    overlap_frames: int = 16
    stream_fps: int = 30
    ladder_step: int = 1
    ladder_strategy: str = "down"
    max_concurrent_windows: int = 1
    chunk_duration_s: float = 4.264
    request_timeout_s: float = 300.0


@dataclass
class LLMConfig:
    chat_url: str = "http://127.0.0.1:8080/v1/chat/completions"
    model: str = "local-qwen"
    timeout: float = 120.0
    max_retries: int = 3
    retry_backoff: float = 1.5
    circuit_breaker_failures: int = 5
    circuit_breaker_timeout: float = 30.0


@dataclass
class TTSConfig:
    backend: str = "kitten"
    kokoro_dir: str = ""
    voice: str = ""
    lang: str = "a"
    command: str = ""
    timeout: int = 120
    max_retries: int = 2
    retry_backoff: float = 1.5


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    log_level: str = "INFO"
    log_file: str = ""
    log_json: bool = False
    environment: str = "production"


def _env_key(section: str, key: str) -> str:
    return f"AVATAR_{section.upper()}_{key.upper()}"


def _coerce(value: str, target_type: type) -> Any:
    if target_type is bool:
        return value.lower() in ("1", "true", "yes", "on")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


def _apply_env_overrides(cfg: Config) -> None:
    """Override config fields from environment variables."""
    for section_name in ("server", "pipeline", "streaming", "llm", "tts"):
        section = getattr(cfg, section_name)
        for field_name in section.__dataclass_fields__:
            env_name = _env_key(section_name, field_name)
            raw = os.environ.get(env_name)
            if raw is not None:
                field_type = type(getattr(section, field_name))
                try:
                    setattr(section, field_name, _coerce(raw, field_type))
                except (ValueError, TypeError):
                    logger.warning(
                        "Ignoring invalid env override %s=%s",
                        env_name,
                        raw,
                    )
    if level := os.environ.get("AVATAR_LOG_LEVEL"):
        cfg.log_level = level.upper()
    if log_file := os.environ.get("AVATAR_LOG_FILE"):
        cfg.log_file = log_file
    if os.environ.get("AVATAR_LOG_JSON") == "1":
        cfg.log_json = True


def _load_yaml() -> dict[str, Any]:
    for path in DEFAULT_CONFIG_PATHS:
        if path and path.is_file():
            logger.info("Loading config from %s", path)
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    logger.info("No config file found, using defaults")
    return {}


def load_config() -> Config:
    """Load YAML config and apply environment overrides."""
    raw = _load_yaml()
    cfg = Config()
    for section_name in ("server", "pipeline", "streaming", "llm", "tts"):
        section_data = raw.get(section_name, {})
        if isinstance(section_data, dict):
            section = getattr(cfg, section_name)
            for key, value in section_data.items():
                if key in section.__dataclass_fields__:
                    setattr(section, key, value)
    if "log_level" in raw:
        cfg.log_level = str(raw["log_level"]).upper()
    if "log_file" in raw:
        cfg.log_file = str(raw["log_file"])
    if "log_json" in raw:
        cfg.log_json = bool(raw["log_json"])
    if "environment" in raw:
        cfg.environment = str(raw["environment"])
    _apply_env_overrides(cfg)
    logger.info(
        "Configuration loaded: environment=%s, server=%s:%d",
        cfg.environment,
        cfg.server.host,
        cfg.server.port,
    )
    return cfg


CONFIG = load_config()
