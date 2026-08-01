"""Tests for the audio preprocessing module."""

import io
import struct
import wave

import numpy as np
import pytest

from inference_runtime.audio import (
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    audio_windows,
    onset_amplitude,
    read_wav,
    window_timestamps,
)


def _make_wav(samples: int, rate: int = 16000, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{samples * channels}h", *([0] * samples * channels)))
    return buf.getvalue()


class TestReadWav:
    def test_read_mono_16bit(self):
        data = _make_wav(16000)
        audio, info = read_wav(data, return_info=True)
        assert len(audio) == 16000
        assert info.channels == 1
        assert info.sample_width_bytes == 2
        assert info.source_sample_rate == 16000
        assert info.output_sample_rate == 16000
        assert info.duration_s == 1.0
        assert info.resampled is False

    def test_read_stereo_to_mono(self):
        data = _make_wav(16000, channels=2)
        audio, info = read_wav(data, return_info=True)
        assert len(audio) == 16000
        assert info.channels == 2

    def test_read_resample(self):
        data = _make_wav(44100, rate=44100)
        audio, info = read_wav(data, return_info=True)
        assert info.resampled is True
        assert info.source_sample_rate == 44100
        assert info.output_sample_rate == 16000
        assert len(audio) == round(44100 * 16000 / 44100)

    def test_read_8bit(self):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(1)
            w.setframerate(16000)
            w.writeframes(bytes([128] * 16000))
        audio, info = read_wav(buf.getvalue(), return_info=True)
        assert info.pcm_format == "pcm_u8"
        assert len(audio) == 16000

    def test_read_32bit(self):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(4)
            w.setframerate(16000)
            w.writeframes(struct.pack(f"<{16000}i", *([0] * 16000)))
        _audio, info = read_wav(buf.getvalue(), return_info=True)
        assert info.pcm_format == "pcm_s32le"

    def test_read_unsupported_width(self):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(3)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00\x00" * 16000)
        with pytest.raises(ValueError, match="Unsupported WAV width"):
            read_wav(buf.getvalue())

    def test_info_to_dict(self):
        data = _make_wav(16000)
        _, info = read_wav(data, return_info=True)
        d = info.to_dict()
        assert d["channels"] == 1
        assert d["duration_s"] == 1.0
        assert d["window_s"] == round(WINDOW_SAMPLES / SAMPLE_RATE, 6)
        assert d["expected_motion_frames"] == max(1, round(16000 * 30 / 16000))


class TestOnsetAmplitude:
    def test_returns_correct_shape(self):
        audio = np.zeros(68224, dtype=np.float32)
        result = onset_amplitude(audio)
        assert result.shape == (68224, 2)

    def test_empty_audio(self):
        audio = np.array([], dtype=np.float32)
        result = onset_amplitude(audio)
        assert result.shape == (0, 2)

    def test_silent_audio(self):
        audio = np.zeros(68224, dtype=np.float32)
        result = onset_amplitude(audio)
        assert np.all(result == 0)

    def test_amplitude_channel(self):
        audio = np.ones(68224, dtype=np.float32) * 0.5
        result = onset_amplitude(audio)
        assert np.all(result[:, 0] == 0.5)

    def test_onset_channel_has_impulses(self):
        # Create audio with a clear onset
        audio = np.zeros(68224, dtype=np.float32)
        audio[1000:2000] = 0.8
        result = onset_amplitude(audio)
        assert np.any(result[:, 1] > 0)


class TestAudioWindows:
    def test_single_window(self):
        audio = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        windows = list(audio_windows(audio))
        assert len(windows) == 1
        _, valid, chunk = windows[0]
        assert valid == WINDOW_SAMPLES
        assert len(chunk) == WINDOW_SAMPLES

    def test_multiple_windows(self):
        audio = np.zeros(WINDOW_SAMPLES * 3, dtype=np.float32)
        windows = list(audio_windows(audio))
        assert len(windows) >= 3

    def test_padded_last_window(self):
        audio = np.zeros(WINDOW_SAMPLES + 100, dtype=np.float32)
        windows = list(audio_windows(audio))
        assert len(windows) == 2
        _, valid, chunk = windows[1]
        # Second window starts at step=WINDOW_SAMPLES-overlap, valid = total - start
        step = WINDOW_SAMPLES - 8533
        expected_valid = (WINDOW_SAMPLES + 100) - step
        assert valid == expected_valid
        assert len(chunk) == WINDOW_SAMPLES

    def test_empty_audio(self):
        audio = np.array([], dtype=np.float32)
        windows = list(audio_windows(audio))
        assert len(windows) == 1
        _, valid, chunk = windows[0]
        assert valid == 1  # empty audio is replaced with np.zeros(1)
        assert len(chunk) == WINDOW_SAMPLES

    def test_overlap(self):
        audio = np.zeros(WINDOW_SAMPLES * 2, dtype=np.float32)
        windows = list(audio_windows(audio))
        assert len(windows) == 3
        # First window starts at 0
        assert windows[0][0] == 0
        # Second window overlaps
        step = WINDOW_SAMPLES - 8533
        assert windows[1][0] == step


class TestWindowTimestamps:
    def test_returns_list_of_dicts(self):
        result = window_timestamps(68224)
        assert isinstance(result, list)
        assert len(result) >= 1
        for w in result:
            assert "index" in w
            assert "start_s" in w
            assert "end_s" in w
            assert "motion_frame_start" in w

    def test_single_window(self):
        result = window_timestamps(68224)
        assert result[0]["index"] == 0
        assert result[0]["start_s"] == 0.0
        assert result[0]["motion_frame_start"] == 0

    def test_empty_audio(self):
        result = window_timestamps(0)
        assert len(result) == 1
        assert result[0]["valid_samples"] == 1  # empty audio is replaced with np.zeros(1)
