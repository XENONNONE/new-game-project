"""Pytest fixtures and configuration for the avatar inference runtime tests."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
GESTURE_LSM = ROOT
sys.path.insert(0, str(GESTURE_LSM))

os.environ.setdefault("AVATAR_LOG_LEVEL", "DEBUG")
os.environ.setdefault("AVATAR_LOG_JSON", "0")
os.environ.setdefault("AVATAR_SERVER_RATE_LIMIT_PER_MINUTE", "10000")


@pytest.fixture
def sample_wav_bytes() -> bytes:
    """Return a minimal valid 16-bit PCM WAV (1 second of silence at 16 kHz)."""
    import io
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<h", 0) * 16000)
    return buf.getvalue()


@pytest.fixture
def sample_speech_plan_dict() -> dict:
    return {
        "reply_text": "Hello there!",
        "emotion": "happy",
        "personality": "helpful",
        "speaking_style": "warm",
        "speed": 1.0,
        "gesture_intensity": 1.2,
        "eye_contact": 0.85,
        "mood": "happy",
    }


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """Return a mock GesturePipeline that returns canned motion data."""
    pipeline = MagicMock()
    pipeline.infer_wav.return_value = {
        "fps": 30,
        "frames": [{"hips": [0.0, 0.0, 0.0, 1.0]}],
        "overlap_blend_frames": 16,
        "timings": {
            "feature_s": 0.01,
            "meanflow_s": 0.05,
            "rvq_s": 0.02,
            "retarget_s": 0.01,
            "total_s": 0.09,
        },
    }
    pipeline.timings = {"load_s": 1.5}
    pipeline.overlap_frames = 16
    pipeline.window_samples = 68224
    pipeline.ladder_step = 1
    pipeline.ladder_strategy = "down"
    pipeline.stream_fps = 30

    def _fake_stream_wav(data: bytes, reset: bool = True):
        yield {
            "type": "info",
            "wav": {"duration_s": 1.0, "output_samples": 16000},
            "windows": [],
        }
        yield {
            "type": "frames",
            "frames": [{"hips": [0.0, 0.0, 0.0, 1.0]}],
            "window_index": 0,
            "total_frames": 1,
            "timings": {"feature_s": 0.01, "meanflow_s": 0.05, "rvq_s": 0.02, "retarget_s": 0.01},
        }
        yield {
            "type": "done",
            "frames": [],
            "window_index": -1,
            "total_frames": 1,
            "expected_frames": 1,
            "timings": {
                "feature_s": 0.01,
                "meanflow_s": 0.05,
                "rvq_s": 0.02,
                "retarget_s": 0.01,
                "total_s": 0.09,
            },
        }

    pipeline.infer_stream_wav.side_effect = _fake_stream_wav
    return pipeline


@pytest.fixture
def mock_tts(monkeypatch):
    """Patch the TTS synthesize function to return canned audio."""
    import io
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<h", 0) * 32000)
    canned_wav = buf.getvalue()

    def fake_synthesize(plan):
        return canned_wav, {
            "backend": "mock",
            "emotion_voice": "Luna",
            "tts_s": 0.01,
            "note": "mock",
        }

    monkeypatch.setattr("inference_runtime.server.synthesize", fake_synthesize)
    return fake_synthesize


@pytest.fixture
def mock_qwen(monkeypatch):
    """Patch the Qwen chat function to return a canned speech plan."""
    from inference_runtime.conversation import AvatarSpeechPlan

    def fake_qwen_chat(message):
        if not message or not message.strip():
            raise ValueError("message cannot be empty")
        return AvatarSpeechPlan(
            reply_text=f"Echo: {message}",
            emotion="happy",
            speaking_style="warm",
            speed=1.0,
        )

    monkeypatch.setattr("inference_runtime.server.qwen_chat", fake_qwen_chat)
    return fake_qwen_chat
