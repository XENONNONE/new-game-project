"""Persistent, training-free MeanFlow and RVQ decoder runtime."""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from models.MeanFlow import GestureMF
from models.vq.model import RVQVAE
from omegaconf import OmegaConf
from utils import rotation_conversions as rc
from utils.joints import hands_body_mask, lower_body_mask, upper_body_mask

import __main__

from .audio import audio_windows, onset_amplitude, read_wav, window_timestamps
from .config import CONFIG, PROJECT
from .logging_config import get_logger
from .retarget import vrm_frames

logger = get_logger("pipeline")
ROOT = Path(__file__).resolve().parents[1]

# --- Optional ONNX Runtime acceleration ---
try:
    import onnxruntime as ort
    _HAS_ONNXRUNTIME = True
except ImportError:
    _HAS_ONNXRUNTIME = False
    logger.info("onnxruntime not installed; ONNX acceleration disabled")


class ONNXDenoiser(nn.Module):
    """Drop-in replacement for GestureDenoiser using ONNX Runtime.

    Accepts the same positional arguments as the PyTorch denoiser's forward
    and returns a torch tensor so the surrounding pipeline code is unchanged.
    """

    def __init__(self, onnx_path: Path, threads: int = 4):
        super().__init__()
        if not _HAS_ONNXRUNTIME:
            raise ImportError("onnxruntime is not installed")

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.add_session_config_entry("session.disable_mem_pattern", "1")
        so.add_session_config_entry("session.disable_prepacking", "0")

        self._session = ort.InferenceSession(
            str(onnx_path),
            so,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = [m.name for m in self._session.get_inputs()]
        self._output_name = self._session.get_outputs()[0].name
        logger.info("ONNX denoiser loaded: %s (%d inputs)",
                     onnx_path.name, len(self._input_names))

    def _to_numpy(self, t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor,
                cond_time: torch.Tensor | None = None,
                seed: torch.Tensor | None = None,
                at_feat: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass matching GestureDenoiser.forward signature.

        All arguments are torch tensors (as the PyTorch model expects).
        The ONNX model was exported with inputs: x, timesteps, cond_time, seed, at_feat.
        """
        feeds = {
            "x": self._to_numpy(x),
            "timesteps": self._to_numpy(timesteps),
            "cond_time": self._to_numpy(cond_time) if cond_time is not None else np.zeros((1,), dtype=np.float32),
            "seed": self._to_numpy(seed) if seed is not None else np.zeros((1,), dtype=np.float32),
            "at_feat": self._to_numpy(at_feat) if at_feat is not None else np.zeros((1,), dtype=np.float32),
        }
        output = self._session.run([self._output_name], feeds)[0]
        return torch.from_numpy(output)

    def eval(self) -> "ONNXDenoiser":
        return self

    def parameters(self, *args, **kwargs):
        return iter([])

    def state_dict(self, *args, **kwargs):
        return {}


class SimpleONNXQuantizer(nn.Module):
    """Deterministic, ONNX-friendly reimplementation of QuantizeEMAReset.

    In eval mode the original gumbel_sample reduces to argmin(distance)
    (no noise). This version skips perplexity/commit_loss (not needed for
    inference) and clones the residual to avoid in-place mutation.
    """

    def __init__(self, quantizer):
        super().__init__()
        self.codebooks = nn.ParameterList()
        self.embedding_projs = nn.ModuleList()
        for layer in quantizer.layers:
            self.codebooks.append(
                nn.Parameter(layer.codebook.data.clone(), requires_grad=False)
            )
            self.embedding_projs.append(layer.embedding_proj)
        self.num_layers = len(self.codebooks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, T) -> quantized_out: (N, C, T)"""
        N, C, T = x.shape
        x_flat = x.permute(0, 2, 1).reshape(-1, C)  # (N*T, C)
        quantized_out = torch.zeros_like(x_flat)
        residual = x_flat.clone()

        for layer_idx in range(self.num_layers):
            projected = self.embedding_projs[layer_idx](self.codebooks[layer_idx])
            proj_sq = torch.sum(projected ** 2, dim=1, keepdim=True)
            res_sq = torch.sum(residual ** 2, dim=1, keepdim=True)
            distance = res_sq - 2 * residual @ projected.t() + proj_sq.t()
            code_idx = torch.argmin(distance, dim=-1)
            quantized = projected[code_idx]
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

        return quantized_out.view(N, T, C).permute(0, 2, 1)


class ONNXRVQDecoder(nn.Module):
    """ONNX Runtime wrapper for an RVQ quantizer + decoder pair.

    Mirrors the original ``latent2origin`` interface so the pipeline code
    is unchanged. Runs the quantizer and decoder through ONNX Runtime.
    """

    def __init__(self, quantizer_onnx: Path, decoder_onnx: Path, threads: int = 4):
        super().__init__()
        if not _HAS_ONNXRUNTIME:
            raise ImportError("onnxruntime is not installed")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self._q_session = ort.InferenceSession(
            str(quantizer_onnx), so, providers=["CPUExecutionProvider"]
        )
        self._q_output = self._q_session.get_outputs()[0].name
        self._q_input = self._q_session.get_inputs()[0].name

        self._d_session = ort.InferenceSession(
            str(decoder_onnx), so, providers=["CPUExecutionProvider"]
        )
        self._d_output = self._d_session.get_outputs()[0].name
        self._d_input = self._d_session.get_inputs()[0].name

        logger.info("ONNX RVQ decoder loaded: %s + %s",
                     quantizer_onnx.name, decoder_onnx.name)

    def _to_np(self, t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    def latent2origin(self, x: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Same interface as RVQVAE.latent2origin.

        input:  (N, T, C) — same shape the pipeline passes
        output: tuple(x_out, None, None) matching the original return
        """
        # latent2origin does x = x.permute(0, 2, 1) -> (N, C, T) for quantizer
        x_q = x.permute(0, 2, 1).contiguous()
        x_quantized = self._q_session.run(
            [self._q_output], {self._q_input: self._to_np(x_q)}
        )[0]
        x_out = self._d_session.run(
            [self._d_output], {self._d_input: x_quantized}
        )[0]
        return torch.from_numpy(x_out), None, None

    def eval(self) -> "ONNXRVQDecoder":
        return self

    def parameters(self, *args, **kwargs):
        return iter([])

    def state_dict(self, *args, **kwargs):
        return {}


# The released language-model pickle was created by a script-local Vocab class.
# Register the historical name before GestureMF opens it, regardless of entrypoint.
if not hasattr(__main__, "Vocab"):

    class Vocab:
        pass

    __main__.Vocab = Vocab


def checkpoint(name: str) -> Path:
    for path in (ROOT / "ckpt" / name, PROJECT / "ckpt" / name):
        if path.is_file() and ".part" not in path.name:
            return path
    raise FileNotFoundError(f"Missing checkpoint: {name}")


class GesturePipeline:
    """Construct once, then reuse. No trainer, dataset, SMPL mesh, or renderer."""

    def __init__(self, threads: int | None = None, seed: int = 42):
        threads = threads or CONFIG.pipeline.threads or min(6, os.cpu_count() or 4)
        torch.set_num_threads(threads)
        with contextlib.suppress(RuntimeError):
            torch.set_num_interop_threads(1)
        if CONFIG.pipeline.torch_compile:
            torch.set_float32_matmul_precision("high")
        torch.manual_seed(seed)
        self.timings: dict[str, float] = {}
        started = time.perf_counter()
        logger.info("Loading MeanFlow config and checkpoints...")
        cfg = OmegaConf.load(ROOT / "configs_new/meanflow_rvqvae_128.yaml")
        cfg.model.modality_encoder.params.data_path = (
            str(ROOT / "datasets/BEAT_SMPL/beat_v2.0.0/beat_english_v2.0.0") + "/"
        )
        self.model = GestureMF(cfg)
        state = torch.load(checkpoint("meanflow.pth"), map_location="cpu", weights_only=False)
        state = state.get("model_state_dict", state.get("model_state", state))
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.use_onnx = CONFIG.pipeline.use_onnx and _HAS_ONNXRUNTIME
        if self.use_onnx:
            onnx_candidates = [
                PROJECT / "models" / "meanflow_denoiser.int8.onnx",
                PROJECT / "meanflow_denoiser.int8.onnx",
                PROJECT / "models" / "meanflow_denoiser.onnx",
                PROJECT / "meanflow_denoiser.onnx",
            ]
            onnx_path = next((p for p in onnx_candidates if p.exists()), None)
            if onnx_path:
                logger.info("Replacing denoiser with ONNX Runtime session (%s)...", onnx_path.name)
                self.model.denoiser = ONNXDenoiser(onnx_path, threads=threads)
            else:
                logger.warning("ONNX denoiser model not found; falling back to PyTorch")
                self.use_onnx = False
        if CONFIG.pipeline.torch_compile:
            logger.info("Compiling model with torch.compile (backend=inductor)...")
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
        rvq_defs = {
            "upper": (78, "net_300000_upper.pth"),
            "hands": (180, "net_300000_hands.pth"),
            "lower": (57, "net_300000_lower.pth"),
        }
        self.decoders = {name: self._decoder(width, ckpt) for name, (width, ckpt) in rvq_defs.items()}
        if self.use_onnx:
            self._swap_decoders_onnx(rvq_defs, threads)
        self.mean = np.load(ROOT / "mean_std/beatx_2_330_mean.npy")
        self.std = np.load(ROOT / "mean_std/beatx_2_330_std.npy")
        self.seed = torch.zeros(1, 3, 128, 4)
        self.overlap_frames = CONFIG.streaming.overlap_frames
        self.window_samples = CONFIG.streaming.window_samples
        self.ladder_step = CONFIG.streaming.ladder_step
        self.ladder_strategy = CONFIG.streaming.ladder_strategy
        self.stream_fps = CONFIG.streaming.stream_fps
        self.timings["load_s"] = time.perf_counter() - started
        logger.info("Pipeline loaded in %.2fs", self.timings["load_s"])

    @staticmethod
    def _decoder(width: int, name: str) -> Any:
        args = SimpleNamespace(
            num_quantizers=6,
            shared_codebook=False,
            quantize_dropout_prob=0.2,
            quantize_dropout_cutoff_index=0,
            mu=0.99,
            beta=1.0,
        )
        model = RVQVAE(args, width, 1024, 128, 128, 2, 2, 512, 3, 3, "relu", None)
        model.load_state_dict(
            torch.load(checkpoint(name), map_location="cpu", weights_only=False)["net"],
            strict=True,
        )
        return model.eval()

    def _swap_decoders_onnx(self, rvq_defs: dict, threads: int) -> None:
        """Replace PyTorch RVQ decoders with ONNX Runtime sessions.

        Exports ONNX models on first use if they don't exist.
        """
        for name in rvq_defs:
            q_path = PROJECT / "models" / f"rvq_quantizer_{name}.onnx"
            d_path = PROJECT / "models" / f"rvq_decoder_{name}.onnx"
            if q_path.exists() and d_path.exists():
                self.decoders[name] = ONNXRVQDecoder(q_path, d_path, threads=threads)
            else:
                logger.info("ONNX RVQ models not found for %s; attempting export...", name)
                if self._export_rvq_onnx(name, rvq_defs[name][0], threads):
                    self.decoders[name] = ONNXRVQDecoder(q_path, d_path, threads=threads)
                else:
                    logger.warning("Failed to export ONNX RVQ for %s; keeping PyTorch", name)

    def _export_rvq_onnx(self, name: str, width: int, threads: int) -> bool:
        """Export quantizer + decoder for one RVQ model to ONNX."""
        try:
            rvq = self.decoders[name]
            if not isinstance(rvq, RVQVAE):
                return False
            seq_len = 32
            x_in = torch.randn(1, 128, seq_len)

            sq = SimpleONNXQuantizer(rvq.quantizer)

            torch.onnx.export(
                sq, x_in, str(PROJECT / "models" / f"rvq_quantizer_{name}.onnx"),
                input_names=["x"], output_names=["output"],
                dynamic_axes={"x": {0: "batch", 2: "seq"}, "output": {0: "batch", 2: "seq"}},
                opset_version=17, dynamo=False, do_constant_folding=True,
            )
            logger.info("Exported ONNX quantizer for %s", name)

            with torch.no_grad():
                x_quantized = sq(x_in.clone()).clone()
            torch.onnx.export(
                rvq.decoder, x_quantized, str(PROJECT / "models" / f"rvq_decoder_{name}.onnx"),
                input_names=["x_quantized"], output_names=["output"],
                dynamic_axes={"x_quantized": {0: "batch", 2: "seq"}, "output": {0: "batch", 2: "seq"}},
                opset_version=17, dynamo=False, do_constant_folding=True,
            )
            logger.info("Exported ONNX decoder for %s", name)
            return True
        except Exception as e:
            logger.error("ONNX export failed for %s: %s", name, e)
            return False

    def reset(self) -> None:
        self.seed.zero_()

    def infer_wav(self, data: bytes, reset: bool = True) -> dict:
        started = time.perf_counter()
        audio, info = read_wav(data, return_info=True)
        self.timings["wav_decode_s"] = time.perf_counter() - started
        result = self.infer_audio(audio, reset)
        result["wav"] = info.to_dict()
        overlap_samples = self.overlap_frames * (16000 // self.stream_fps)
        result["windows"] = window_timestamps(len(audio), overlap_samples)
        return result

    def infer_audio(self, audio: np.ndarray, reset: bool = True) -> dict:
        if reset:
            self.reset()
        result: list[dict] = []
        stage = dict.fromkeys(("feature_s", "meanflow_s", "rvq_s", "retarget_s"), 0.0)
        overlap_samples = round(self.overlap_frames * 16000 / self.stream_fps)
        for index, (_, _valid, chunk) in enumerate(audio_windows(audio, overlap=overlap_samples)):
            t = time.perf_counter()
            features = torch.from_numpy(onset_amplitude(chunk)).unsqueeze(0)
            stage["feature_s"] += time.perf_counter() - t
            cond = {
                "y": {
                    "audio_onset": features,
                    "word": torch.zeros(1, 128, dtype=torch.long),
                    "id": torch.zeros(1, dtype=torch.long),
                    "seed": self.seed,
                    "style_feature": torch.zeros(1, 512),
                }
            }
            t = time.perf_counter()
            with torch.inference_mode():
                raw = self.model(cond)["latents"]
            stage["meanflow_s"] += time.perf_counter() - t
            self.seed = raw.reshape(1, 3, 128, 32)[..., -4:].contiguous()
            latent = raw.squeeze(2).permute(0, 2, 1)
            t = time.perf_counter()
            matrices = self._decode(latent)
            stage["rvq_s"] += time.perf_counter() - t
            t = time.perf_counter()
            frames = vrm_frames(matrices)
            stage["retarget_s"] += time.perf_counter() - t
            if index:
                self._blend_overlap(result, frames, self.overlap_frames)
            else:
                result.extend(frames)
        result = result[: max(1, round(len(audio) * self.stream_fps / 16000))]
        self.timings.update(stage)
        self.timings["total_s"] = sum(stage.values())
        logger.debug(
            "Inference complete: %d frames, timings=%s",
            len(result),
            stage,
        )
        return {
            "fps": self.stream_fps,
            "frames": result,
            "overlap_blend_frames": self.overlap_frames,
            "timings": dict(self.timings),
        }

    def infer_stream_wav(self, data: bytes, reset: bool = True) -> Iterator[dict]:
        """Read WAV bytes and stream gesture frames incrementally.

        Wraps :meth:`infer_stream` with WAV decoding.  Each yielded event
        contains a batch of frames, the window index, and per-window timings.
        """
        started = time.perf_counter()
        audio, info = read_wav(data, return_info=True)
        self.timings["wav_decode_s"] = time.perf_counter() - started
        overlap_samples = round(self.overlap_frames * 16000 / self.stream_fps)
        yield {
            "type": "info",
            "wav": info.to_dict(),
            "windows": window_timestamps(len(audio), overlap_samples),
        }
        total_yielded = 0
        expected_frames = max(1, round(len(audio) * self.stream_fps / 16000))
        for event in self.infer_stream(audio, reset):
            total_yielded += len(event["frames"])
            yield event
        if total_yielded < expected_frames:
            logger.warning("Stream ended early: %d/%d frames", total_yielded, expected_frames)

    def infer_stream(self, audio: np.ndarray, reset: bool = True) -> Iterator[dict]:
        """Stream gesture frames incrementally as audio windows are processed.

        Inspired by Rolling Diffusion (AAAI 2026) — processes audio in
        overlapping windows with seed-frame carryover and optional ladder
        acceleration.  Frames are yielded as they become available, with a
        small buffer for overlap blending.

        Each yielded event is a dict with keys:
            - ``type``: "frames" or "done"
            - ``frames``: list of frame dicts (empty for "done")
            - ``window_index``: which audio window this batch came from
            - ``timings``: per-stage timing for this window
            - ``total_frames``: cumulative frame count
        """
        if reset:
            self.reset()
        overlap = self.overlap_frames
        overlap_samples = round(overlap * 16000 / self.stream_fps)
        buffer: list[dict] = []
        total_frames = 0
        stage = dict.fromkeys(("feature_s", "meanflow_s", "rvq_s", "retarget_s"), 0.0)
        expected_frames = max(1, round(len(audio) * self.stream_fps / 16000))

        for index, (_start, _valid, chunk) in enumerate(
            audio_windows(audio, overlap=overlap_samples)
        ):
            t = time.perf_counter()
            features = torch.from_numpy(onset_amplitude(chunk)).unsqueeze(0)
            stage["feature_s"] += time.perf_counter() - t
            cond = {
                "y": {
                    "audio_onset": features,
                    "word": torch.zeros(1, 128, dtype=torch.long),
                    "id": torch.zeros(1, dtype=torch.long),
                    "seed": self.seed,
                    "style_feature": torch.zeros(1, 512),
                }
            }
            t = time.perf_counter()
            with torch.inference_mode():
                raw = self.model(cond)["latents"]
            stage["meanflow_s"] += time.perf_counter() - t
            self.seed = raw.reshape(1, 3, 128, 32)[..., -4:].contiguous()
            latent = raw.squeeze(2).permute(0, 2, 1)
            t = time.perf_counter()
            matrices = self._decode(latent)
            stage["rvq_s"] += time.perf_counter() - t
            t = time.perf_counter()
            frames = vrm_frames(matrices)
            stage["retarget_s"] += time.perf_counter() - t

            if index:
                blended = self._blend_stream(buffer, frames, overlap)
                for frame in blended:
                    total_frames += 1
                    yield {
                        "type": "frames",
                        "frames": [frame],
                        "window_index": index,
                        "total_frames": total_frames,
                        "timings": dict(stage),
                    }
                new_frames = frames[overlap:]
            else:
                new_frames = frames[:-overlap] if len(frames) > overlap else frames

            for i in range(0, len(new_frames), self.ladder_step):
                batch = new_frames[i : i + self.ladder_step]
                total_frames += len(batch)
                yield {
                    "type": "frames",
                    "frames": batch,
                    "window_index": index,
                    "total_frames": total_frames,
                    "timings": dict(stage),
                }

            buffer = frames[-overlap:] if len(frames) >= overlap else frames

        if buffer:
            for frame in buffer:
                total_frames += 1
                yield {
                    "type": "frames",
                    "frames": [frame],
                    "window_index": -1,
                    "total_frames": total_frames,
                    "timings": dict(stage),
                }

        self.timings.update(stage)
        self.timings["total_s"] = sum(stage.values())
        yield {
            "type": "done",
            "frames": [],
            "window_index": -1,
            "total_frames": total_frames,
            "expected_frames": expected_frames,
            "timings": dict(self.timings),
        }

    @staticmethod
    def _blend_stream(buffer: list[dict], incoming: list[dict], count: int) -> list[dict]:
        """Blend buffered frames with incoming frames, returning the blended
        buffer.  The buffer is modified in-place to match the incoming frames.
        """
        overlap = min(count, len(buffer), len(incoming))
        if overlap == 0:
            return buffer
        result: list[dict] = []
        for i in range(overlap):
            alpha = (i + 1) / (overlap + 1)
            previous = buffer[i]
            current = incoming[i]
            merged: dict[str, list] = {}
            for bone in previous.keys() & current.keys():
                a = np.asarray(previous[bone], dtype=np.float64)
                b = np.asarray(current[bone], dtype=np.float64)
                if np.dot(a, b) < 0:
                    b = -b
                q = (1 - alpha) * a + alpha * b
                norm = np.linalg.norm(q)
                if norm > 1e-8:
                    q = q / norm
                merged[bone] = q.tolist()
            result.append(merged)
        return result

    @staticmethod
    def _blend_overlap(result: list[dict], incoming: list[dict], count: int) -> None:
        """Crossfade duplicate window frames with shortest-path quaternion nlerp."""
        overlap = min(count, len(result), len(incoming))
        for i in range(overlap):
            alpha = (i + 1) / (overlap + 1)
            previous = result[-overlap + i]
            current = incoming[i]
            for bone in previous.keys() & current.keys():
                a = np.asarray(previous[bone], dtype=np.float64)
                b = np.asarray(current[bone], dtype=np.float64)
                if np.dot(a, b) < 0:
                    b = -b
                q = (1 - alpha) * a + alpha * b
                norm = np.linalg.norm(q)
                if norm > 1e-8:
                    previous[bone] = (q / norm).tolist()
        result.extend(incoming[overlap:])

    def _decode(self, latent: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            u, h, lower = (x * 5 for x in latent.split(128, dim=-1))
            u = self.decoders["upper"].latent2origin(u)[0]
            h = self.decoders["hands"].latent2origin(h)[0]
            lower = self.decoders["lower"].latent2origin(lower)[0][..., :-3]
            u = u * torch.from_numpy(self.std[upper_body_mask]) + torch.from_numpy(
                self.mean[upper_body_mask]
            )
            h = h * torch.from_numpy(self.std[hands_body_mask]) + torch.from_numpy(
                self.mean[hands_body_mask]
            )
            lower = lower * torch.from_numpy(self.std[lower_body_mask]) + torch.from_numpy(
                self.mean[lower_body_mask]
            )
            pose = torch.zeros(1, u.shape[1], 55, 6, dtype=u.dtype)
            pose[:, :, [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]] = u.reshape(1, -1, 13, 6)
            pose[:, :, list(range(25, 55))] = h.reshape(1, -1, 30, 6)
            pose[:, :, [0, 1, 2, 4, 5, 7, 8, 10, 11]] = lower.reshape(1, -1, 9, 6)
            pose[:, :, 22:25, 0] = 1
            pose[:, :, 22:25, 4] = 1
            result: np.ndarray = rc.rotation_6d_to_matrix(pose).squeeze(0).cpu().numpy()
            return result
