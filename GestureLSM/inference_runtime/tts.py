"""Small local TTS adapter with emotion/style mapping and 16 kHz WAV output."""

from __future__ import annotations

import io
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .config import CONFIG
from .conversation import AvatarSpeechPlan
from .logging_config import get_logger

logger = get_logger("tts")

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
TTS_DIR = PROJECT / "models" / "tts" / "kokoro"
_KOKORO_PIPELINES: dict[str, Any] = {}
_KOKORO_VOICES: dict[str, Any] = {}
_KOKORO_LOCK = threading.Lock()
_KITTEN_MODEL: object | None = None
_KITTEN_LOCK = threading.Lock()

KOKORO_VOICES = {
    "neutral": "af_heart",
    "calm": "af_heart",
    "happy": "af_bella",
    "excited": "af_bella",
    "curious": "af_nicole",
    "thinking": "af_nicole",
    "confused": "af_nicole",
    "sad": "af_sarah",
    "angry": "am_fenrir",
    "surprised": "af_bella",
    "listening": "af_heart",
}

KITTEN_VOICES = {
    "neutral": "expr-voice-2-f",
    "calm": "expr-voice-2-f",
    "happy": "expr-voice-2-f",
    "excited": "expr-voice-4-f",
    "curious": "expr-voice-3-f",
    "thinking": "expr-voice-3-f",
    "confused": "expr-voice-3-f",
    "sad": "expr-voice-5-f",
    "angry": "expr-voice-4-m",
    "surprised": "expr-voice-2-f",
    "listening": "expr-voice-2-f",
}


def _run(command: list[str], timeout: int = 120) -> None:
    subprocess.run(
        command,
        check=True,
        timeout=timeout,
        capture_output=True,
    )


def _normalize_wav(source: Path) -> bytes:
    """Convert 24 kHz WAV to 16 kHz mono 16-bit PCM WAV."""
    import soundfile as sf

    data, rate = sf.read(str(source))
    if data.ndim > 1:
        data = data.mean(axis=1)
    count = int(len(data) * 16000 / rate)
    indices = np.linspace(0, len(data) - 1, count)
    resampled = np.interp(indices, np.arange(len(data)), data)
    int_data = (np.clip(resampled, -1.0, 1.0) * 32767).astype(np.int16)
    target = io.BytesIO()
    with wave.open(target, "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(16000)
        dst.writeframes(int_data.tobytes())
    return target.getvalue()


def _synth_kokoro(plan: AvatarSpeechPlan) -> bytes:
    """Use the official Python Kokoro package if the user installed it."""
    import soundfile as sf  # type: ignore
    import torch  # type: ignore
    from kokoro import KModel, KPipeline  # type: ignore

    lang = CONFIG.tts.lang
    with _KOKORO_LOCK:
        pipeline = _KOKORO_PIPELINES.get(lang)
        if pipeline is None:
            config = TTS_DIR / "config.json"
            checkpoint = TTS_DIR / "kokoro-v1_0.pth"
            if config.is_file() and checkpoint.is_file():
                model = KModel(config=str(config), model=str(checkpoint)).eval()
                pipeline = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", model=model)
            else:
                pipeline = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")
            _KOKORO_PIPELINES[lang] = pipeline
    voice = CONFIG.tts.voice or KOKORO_VOICES.get(plan.emotion, "af_heart")
    voice_path = TTS_DIR / "voices" / f"{voice}.pt"
    voice_obj: Any = voice
    if voice_path.is_file():
        with _KOKORO_LOCK:
            cached_voice = _KOKORO_VOICES.get(str(voice_path))
            if cached_voice is None:
                cached_voice = torch.load(voice_path, map_location="cpu", weights_only=True)
                _KOKORO_VOICES[str(voice_path)] = cached_voice
            voice_obj = cached_voice
    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "speech.wav"
        chunks = []
        with _KOKORO_LOCK:
            for _, _, audio in pipeline(
                plan.reply_text, voice=voice_obj, speed=plan.speed, split_pattern=r"\n+"
            ):
                chunks.append(audio)
        if not chunks:
            raise ValueError("Kokoro produced no audio")
        import numpy as np  # type: ignore

        sf.write(wav_path, np.concatenate(chunks), 24000)
        return _normalize_wav(wav_path)


def _load_kitten_model() -> Any:
    with _KITTEN_LOCK:
        global _KITTEN_MODEL
        if _KITTEN_MODEL is None:
            from kittentts import KittenTTS  # type: ignore

            _KITTEN_MODEL = KittenTTS()
        return _KITTEN_MODEL


def _synth_kitten(plan: AvatarSpeechPlan) -> bytes:
    """Use KittenTTS ONNX model for local CPU synthesis.

    Uses the KittenML/kitten-tts-nano-0.1 model (auto-downloaded from
    HuggingFace on first use). Voices are mapped from emotion labels
    to KittenTTS expression voices (e.g. ``expr-voice-2-f``).
    """

    model = _load_kitten_model()
    voice = CONFIG.tts.voice or KITTEN_VOICES.get(plan.emotion, "expr-voice-2-f")
    audio = model.generate(
        plan.reply_text,
        voice=voice,
        speed=plan.speed,
    )
    if audio is None or len(audio) == 0:
        raise ValueError("KittenTTS produced no audio")
    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "speech.wav"
        import soundfile as sf  # type: ignore

        sf.write(wav_path, audio, 24000)
        return _normalize_wav(wav_path)


def _synth_command(plan: AvatarSpeechPlan) -> bytes:
    """Run a user-supplied command that writes a WAV to the output path."""
    template = CONFIG.tts.command
    if not template:
        raise RuntimeError("No TTS backend configured. Install kokoro or set AVATAR_TTS_COMMAND.")
    with tempfile.TemporaryDirectory() as td:
        text_path = Path(td) / "input.txt"
        wav_path = Path(td) / "speech.wav"
        text_path.write_text(plan.reply_text, encoding="utf-8")
        command = template.format(
            text=str(text_path),
            wav=str(wav_path),
            emotion=plan.emotion,
            style=plan.speaking_style,
            speed=str(plan.speed),
        )
        _run(["sh", "-lc", command], timeout=CONFIG.tts.timeout)
        if not wav_path.is_file():
            raise FileNotFoundError(f"TTS command did not create {wav_path}")
        return _normalize_wav(wav_path)


def synthesize(plan: AvatarSpeechPlan) -> tuple[bytes, dict]:
    plan = plan.normalized()
    started = time.perf_counter()
    backend = CONFIG.tts.backend.lower()
    if backend == "command":
        wav = _synth_command(plan)
    elif backend == "kitten":
        wav = _synth_kitten(plan)
    else:
        try:
            wav = _synth_kokoro(plan)
            backend = "kokoro"
        except ModuleNotFoundError:
            logger.info("Kokoro not available, falling back to command backend")
            wav = _synth_command(plan)
            backend = "command"
    elapsed = time.perf_counter() - started
    logger.info("TTS synthesized (backend=%s, %d bytes, %.2fs)", backend, len(wav), elapsed)
    emotion_voice = (
        KITTEN_VOICES.get(plan.emotion, "expr-voice-2-f")
        if backend == "kitten"
        else KOKORO_VOICES.get(plan.emotion, "af_heart")
    )
    return wav, {
        "backend": backend,
        "emotion_voice": emotion_voice,
        "tts_s": round(elapsed, 6),
        "note": (
            "Emotion labels map to voice/style/speed; Kokoro is not a native emotion classifier."
            if backend == "kokoro"
            else "Emotion labels map to voice/style/speed via KittenTTS voice mapping."
        ),
    }
