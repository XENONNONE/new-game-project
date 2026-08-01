"""Tests for the streaming inference pipeline methods.

These tests use the real ``_blend_stream`` and ``_blend_overlap`` static
methods (which don't require model weights) plus a mock pipeline to verify
the streaming event contract.
"""

from __future__ import annotations

import numpy as np

from inference_runtime.pipeline import GesturePipeline


def _make_frame(hip_val: float = 0.0) -> dict:
    return {"hips": [hip_val, 0.0, 0.0, 1.0], "leftShoulder": [0.0, 0.0, 0.0, 1.0]}


class TestBlendStream:
    def test_empty_buffer_returns_empty(self):
        result = GesturePipeline._blend_stream([], [], 16)
        assert result == []

    def test_empty_incoming_returns_buffer(self):
        buffer = [_make_frame(1.0), _make_frame(2.0)]
        result = GesturePipeline._blend_stream(buffer, [], 16)
        assert result == buffer

    def test_partial_overlap(self):
        buffer = [_make_frame(1.0), _make_frame(2.0), _make_frame(3.0)]
        incoming = [_make_frame(4.0), _make_frame(5.0), _make_frame(6.0)]
        result = GesturePipeline._blend_stream(buffer, incoming, 2)
        assert len(result) == 2
        for frame in result:
            assert "hips" in frame

    def test_full_overlap(self):
        buffer = [_make_frame(i * 0.1) for i in range(16)]
        incoming = [_make_frame(i * 0.2) for i in range(16)]
        result = GesturePipeline._blend_stream(buffer, incoming, 16)
        assert len(result) == 16

    def test_quaternion_shortest_path(self):
        """When dot product is negative, the quaternion should be negated
        before blending to take the shortest path."""
        buffer = [{"hips": [0.0, 0.0, 0.0, 1.0]}]
        incoming = [{"hips": [0.0, 0.0, 0.0, -1.0]}]
        result = GesturePipeline._blend_stream(buffer, incoming, 1)
        assert len(result) == 1
        q = np.array(result[0]["hips"])
        assert q[3] >= 0

    def test_blend_produces_valid_quaternions(self):
        buffer = [_make_frame(0.5), _make_frame(0.3)]
        incoming = [_make_frame(0.7), _make_frame(0.2)]
        result = GesturePipeline._blend_stream(buffer, incoming, 2)
        for frame in result:
            for _bone, q in frame.items():
                q_arr = np.array(q)
                norm = np.linalg.norm(q_arr)
                assert abs(norm - 1.0) < 1e-4


class TestBlendOverlap:
    def test_blend_modifies_result_in_place(self):
        result = [_make_frame(1.0), _make_frame(2.0)]
        incoming = [_make_frame(3.0), _make_frame(4.0)]
        GesturePipeline._blend_overlap(result, incoming, 2)
        assert len(result) == 2
        for frame in result:
            assert "hips" in frame

    def test_blend_empty_result(self):
        result: list[dict] = []
        incoming = [_make_frame(1.0)]
        GesturePipeline._blend_overlap(result, incoming, 16)
        assert len(result) == 1

    def test_blend_partial_overlap_extends(self):
        result = [_make_frame(1.0)]
        incoming = [_make_frame(2.0), _make_frame(3.0)]
        GesturePipeline._blend_overlap(result, incoming, 16)
        assert len(result) == 2


class TestStreamEventContract:
    """Verify the streaming event format using a mock pipeline."""

    def test_stream_events_have_correct_types(self, mock_pipeline):
        events = list(mock_pipeline.infer_stream_wav(b""))
        types = [e["type"] for e in events]
        assert "info" in types
        assert "frames" in types
        assert "done" in types

    def test_info_event_has_wav_and_windows(self, mock_pipeline):
        events = list(mock_pipeline.infer_stream_wav(b""))
        info = next(e for e in events if e["type"] == "info")
        assert "wav" in info
        assert "windows" in info

    def test_frames_event_has_frames_and_timings(self, mock_pipeline):
        events = list(mock_pipeline.infer_stream_wav(b""))
        frames_event = next(e for e in events if e["type"] == "frames")
        assert "frames" in frames_event
        assert isinstance(frames_event["frames"], list)
        assert "timings" in frames_event
        assert "total_frames" in frames_event
        assert "window_index" in frames_event

    def test_done_event_has_total_and_expected(self, mock_pipeline):
        events = list(mock_pipeline.infer_stream_wav(b""))
        done = next(e for e in events if e["type"] == "done")
        assert done["total_frames"] == 1
        assert done["expected_frames"] == 1
        assert "timings" in done
