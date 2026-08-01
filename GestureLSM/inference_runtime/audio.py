"""Dependency-light WAV preprocessing and diagnostics for GestureLSM."""

from __future__ import annotations

import io
import wave
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 68224


@dataclass(frozen=True)
class WavInfo:
    channels: int
    sample_width_bytes: int
    source_sample_rate: int
    output_sample_rate: int
    source_frames: int
    output_samples: int
    duration_s: float
    pcm_format: str
    resampled: bool

    def to_dict(self) -> dict:
        data = asdict(self)
        data["window_s"] = round(WINDOW_SAMPLES / SAMPLE_RATE, 6)
        data["expected_motion_frames"] = max(1, round(self.output_samples * 30 / SAMPLE_RATE))
        return data


def read_wav(data: bytes, *, return_info: bool = False) -> np.ndarray | tuple[np.ndarray, WavInfo]:
    with wave.open(io.BytesIO(data), "rb") as f:
        channels, width, rate = f.getnchannels(), f.getsampwidth(), f.getframerate()
        source_frames = f.getnframes()
        raw = f.readframes(f.getnframes())
    if width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        pcm_format = "pcm_s16le"
    elif width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        pcm_format = "pcm_u8"
    elif width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        pcm_format = "pcm_s32le"
    else:
        raise ValueError(f"Unsupported WAV width: {width}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(1)
    if rate != SAMPLE_RATE and audio.size:
        count = round(len(audio) * SAMPLE_RATE / rate)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, count), np.arange(len(audio)), audio
        ).astype(np.float32)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if not return_info:
        return audio  # type: ignore[no-any-return]
    info = WavInfo(
        channels=channels,
        sample_width_bytes=width,
        source_sample_rate=rate,
        output_sample_rate=SAMPLE_RATE,
        source_frames=source_frames,
        output_samples=int(audio.size),
        duration_s=round(audio.size / SAMPLE_RATE, 6),
        pcm_format=pcm_format,
        resampled=rate != SAMPLE_RATE,
    )
    return audio, info


def _rolling_max(x: np.ndarray, width: int = 1024) -> np.ndarray:
    if not len(x):
        return x
    width = min(width, len(x))
    out: np.ndarray = np.empty(len(x), np.float32)
    q: deque[int] = deque()
    for i in range(len(x) - 1, -1, -1):
        while q and q[0] >= i + width:
            q.popleft()
        while q and x[q[-1]] <= x[i]:
            q.pop()
        q.append(i)
        out[i] = x[q[0]]
    return out


def onset_amplitude(audio: np.ndarray) -> np.ndarray:
    """Sample-rate amplitude plus corrected energy-onset impulses."""
    if not len(audio):
        return np.zeros((0, 2), np.float32)
    amplitude: np.ndarray = _rolling_max(np.abs(audio))
    hop, frame = 512, 1024
    pad = np.pad(audio, (0, max(0, frame - len(audio) % hop)))
    windows = np.lib.stride_tricks.sliding_window_view(pad, frame)[::hop]
    energy = np.sqrt(np.mean(windows * windows, axis=1) + 1e-8)
    flux = np.maximum(0, np.diff(energy, prepend=energy[0]))
    median = np.median(flux)
    threshold = median + 2.5 * np.median(np.abs(flux - median))
    onset: np.ndarray = np.zeros(len(audio), np.float32)
    indices: np.ndarray = np.flatnonzero(flux > max(threshold, 1e-4)) * hop
    onset[indices[indices < len(onset)]] = 1
    result: np.ndarray = np.stack((amplitude, onset), 1)
    return result


def audio_windows(audio: np.ndarray, overlap: int = 8533) -> Any:
    """Yield padded 4.264-second windows with 16 video frames overlap."""
    step = WINDOW_SAMPLES - overlap
    if not len(audio):
        audio = np.zeros(1, np.float32)
    for start in range(0, len(audio), step):
        chunk = audio[start : start + WINDOW_SAMPLES]
        valid = len(chunk)
        if valid < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - valid))
        yield start, valid, chunk.astype(np.float32, copy=False)
        if start + WINDOW_SAMPLES >= len(audio):
            break


def window_timestamps(audio_samples: int, overlap: int = 8533) -> list[dict]:
    """Return the conditioning windows in seconds for debugging sync issues."""
    step = WINDOW_SAMPLES - overlap
    total = max(1, int(audio_samples))
    windows = []
    for index, start in enumerate(range(0, total, step)):
        valid = min(WINDOW_SAMPLES, total - start)
        windows.append(
            {
                "index": index,
                "sample_start": start,
                "sample_end": start + max(0, valid),
                "start_s": round(start / SAMPLE_RATE, 6),
                "end_s": round((start + max(0, valid)) / SAMPLE_RATE, 6),
                "valid_samples": max(0, valid),
                "padded_samples": max(0, WINDOW_SAMPLES - max(0, valid)),
                "motion_frame_start": round(start * 30 / SAMPLE_RATE),
                "motion_frame_end": round((start + max(0, valid)) * 30 / SAMPLE_RATE),
            }
        )
        if start + WINDOW_SAMPLES >= total:
            break
    return windows
