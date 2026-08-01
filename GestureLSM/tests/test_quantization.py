"""Compare FP32 vs INT8 quantized MeanFlow + RVQ decoders.

Strategies tested:
  1. Dynamic INT8 on MeanFlow denoiser + modality_encoder (Linear layers)
  2. Dynamic INT8 on RVQ decoders (upper, hands, lower)
  3. Full: INT8 on both MeanFlow and all RVQ decoders
"""
from __future__ import annotations

import time
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GESTURE_ROOT = PROJECT_ROOT / "GestureLSM"
sys.path.insert(0, str(GESTURE_ROOT))

from inference_runtime.pipeline import GesturePipeline
from inference_runtime.audio import read_wav
from inference_runtime.config import CONFIG


def load_test_audio() -> bytes:
    wav_path = PROJECT_ROOT / "test.wav"
    if not wav_path.exists():
        raise FileNotFoundError(f"test.wav not found at {wav_path}")
    return wav_path.read_bytes()


def total_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_disk_size(model: torch.nn.Module) -> float:
    total = 0
    for name, param in model.state_dict().items():
        if not torch.is_tensor(param):
            continue
        if hasattr(param, "is_quantized") and param.is_quantized:
            total += param.numel() * param.qelement_size()
        else:
            total += param.numel() * param.element_size()
    return total / 1e6


def total_rvq_size(decoders: dict) -> float:
    return sum(model_disk_size(d) for d in decoders.values())


def quantize_meanflow(model: torch.nn.Module) -> torch.nn.Module:
    """Quantize Linear layers in denoiser + modality_encoder."""
    if hasattr(model, "denoiser"):
        model.denoiser = quantize_dynamic(model.denoiser, {nn.Linear}, dtype=torch.qint8)
    if hasattr(model, "modality_encoder"):
        model.modality_encoder = quantize_dynamic(
            model.modality_encoder, {nn.Linear}, dtype=torch.qint8
        )
    return model


def quantize_decoders(decoders: dict) -> dict:
    """Quantize Linear layers in all RVQ decoder submodules."""
    for key in decoders:
        decoders[key] = quantize_dynamic(decoders[key], {nn.Linear}, dtype=torch.qint8)
    return decoders


def run_inference(pipe: GesturePipeline, audio: bytes, label: str) -> dict:
    t0 = time.perf_counter()
    result = pipe.infer_wav(audio, reset=True)
    elapsed = time.perf_counter() - t0
    timings = result["timings"]
    print(f"[{label:20s}] Total: {elapsed:6.2f}s | MeanFlow: {timings['meanflow_s']:5.2f}s | "
          f"RVQ: {timings['rvq_s']:5.2f}s | Frames: {len(result['frames'])}")
    return {"result": result, "elapsed": elapsed, "timings": timings}


def quaternion_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    dot = np.abs(np.sum(a * b, axis=-1))
    return float(np.mean(np.clip(dot, 0, 1)))


def compare_frames(fp32_frames: list, int8_frames: list, verbose: bool = True) -> tuple[float, float]:
    if len(fp32_frames) != len(int8_frames):
        if verbose:
            print(f"  Frame mismatch: FP32={len(fp32_frames)} vs INT8={len(int8_frames)}")
        return 0.0, 0.0
    sims = []
    for f32, i8 in zip(fp32_frames, int8_frames):
        for bone in f32:
            if bone in i8:
                a = np.array(f32[bone], dtype=np.float64)
                b = np.array(i8[bone], dtype=np.float64)
                sims.append(quaternion_similarity(a, b))
    avg_sim = float(np.mean(sims)) if sims else 0
    min_sim = float(np.min(sims)) if sims else 0
    if verbose:
        print(f"  Bone comparisons: {len(sims)} | Avg cosine sim: {avg_sim:.4f} (min: {min_sim:.4f})")
        if avg_sim > 0.99:
            print("  Quality: EXCELLENT (negligible degradation)")
        elif avg_sim > 0.95:
            print("  Quality: HIGH (minimal degradation)")
        elif avg_sim > 0.85:
            print("  Quality: MODERATE (some degradation)")
        else:
            print("  Quality: LOW (significant degradation)")
    return avg_sim, min_sim


def main():
    threads = CONFIG.pipeline.threads or min(6, torch.get_num_threads())
    torch.set_num_threads(threads)
    print(f"Threads: {threads}")

    audio = load_test_audio()
    _, info = read_wav(audio, return_info=True)
    print(f"Audio: {info.duration_s:.2f}s ({info.output_samples} samples)\n")

    # ---- FP32 baseline ----
    print("=== FP32 baseline ===")
    pipe_fp32 = GesturePipeline(threads=threads)
    tp_model = total_params(pipe_fp32.model)
    tp_decoders = sum(total_params(d) for d in pipe_fp32.decoders.values())
    fp32_out = run_inference(pipe_fp32, audio, "FP32")
    fp32_size_model = model_disk_size(pipe_fp32.model)
    fp32_size_decoders = total_rvq_size(pipe_fp32.decoders)
    print(f"  Model params: {tp_model:,} | size: {fp32_size_model:.1f}MB")
    print(f"  RVQ params (3x): {tp_decoders:,} | size: {fp32_size_decoders:.1f}MB")
    print(f"  Total size: {fp32_size_model + fp32_size_decoders:.1f}MB")

    # ---- INT8 MeanFlow only ----
    print("\n=== INT8 MeanFlow (Linear only) ===")
    pipe_q1 = GesturePipeline(threads=threads)
    pipe_q1.model = quantize_meanflow(pipe_q1.model)
    q1_out = run_inference(pipe_q1, audio, "INT8-MeanFlow")
    q1_size_model = model_disk_size(pipe_q1.model)
    speedup_mf = fp32_out["timings"]["meanflow_s"] / q1_out["timings"]["meanflow_s"]
    print(f"  MeanFlow: {fp32_size_model:.1f}MB -> {q1_size_model:.1f}MB "
          f"({100*(1-q1_size_model/fp32_size_model):.0f}% smaller) | Speedup: {speedup_mf:.2f}x")
    print("  Quality:")
    sim_mf, _ = compare_frames(fp32_out["result"]["frames"], q1_out["result"]["frames"])

    # ---- INT8 RVQ only ----
    print("\n=== INT8 RVQ decoders (Linear only) ===")
    pipe_q2 = GesturePipeline(threads=threads)
    pipe_q2.decoders = quantize_decoders(pipe_q2.decoders)
    q2_out = run_inference(pipe_q2, audio, "INT8-RVQ")
    q2_size_decoders = total_rvq_size(pipe_q2.decoders)
    speedup_rvq = fp32_out["timings"]["rvq_s"] / q2_out["timings"]["rvq_s"]
    print(f"  RVQ: {fp32_size_decoders:.1f}MB -> {q2_size_decoders:.1f}MB "
          f"({100*(1-q2_size_decoders/fp32_size_decoders):.0f}% smaller) | Speedup: {speedup_rvq:.2f}x")
    print("  Quality:")
    sim_rvq, _ = compare_frames(fp32_out["result"]["frames"], q2_out["result"]["frames"])

    # ---- INT8 both ----
    print("\n=== INT8 both MeanFlow + RVQ ===")
    pipe_q3 = GesturePipeline(threads=threads)
    pipe_q3.model = quantize_meanflow(pipe_q3.model)
    pipe_q3.decoders = quantize_decoders(pipe_q3.decoders)
    q3_out = run_inference(pipe_q3, audio, "INT8-ALL")
    q3_size_model = model_disk_size(pipe_q3.model)
    q3_size_decoders = total_rvq_size(pipe_q3.decoders)
    speedup_total = fp32_out["elapsed"] / q3_out["elapsed"]
    print(f"  Total size: {fp32_size_model+fp32_size_decoders:.1f}MB -> "
          f"{q3_size_model+q3_size_decoders:.1f}MB "
          f"({100*(1-(q3_size_model+q3_size_decoders)/(fp32_size_model+fp32_size_decoders)):.0f}% smaller)")
    print(f"  Speedup (total): {speedup_total:.2f}x | RTF: {q3_out['elapsed']/info.duration_s:.2f}x")
    print("  Quality:")
    sim_all, _ = compare_frames(fp32_out["result"]["frames"], q3_out["result"]["frames"])

    # Save frames for visual inspection
    out_dir = Path("/tmp/avatar_quant_test")
    out_dir.mkdir(exist_ok=True)
    for name, frames in [("fp32", fp32_out), ("int8_mf", q1_out),
                          ("int8_rvq", q2_out), ("int8_all", q3_out)]:
        with open(out_dir / f"{name}_frames.json", "w") as f:
            json.dump(frames["result"]["frames"][:10], f, indent=2, default=str)
    print(f"\nFrames saved to {out_dir}/ (first 10 per config)")

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Config':<20s} {'Total':>8s} {'MeanFlow':>10s} {'RVQ':>8s} "
          f"{'CosSim':>8s} {'Size':>10s} {'Speedup':>8s}")
    print("-" * 90)
    configs = [
        ("FP32", fp32_out, fp32_size_model + fp32_size_decoders, 0.0, 1.0),
        ("INT8-MeanFlow", q1_out, q1_size_model + fp32_size_decoders, sim_mf, speedup_mf),
        ("INT8-RVQ", q2_out, fp32_size_model + q2_size_decoders, sim_rvq, speedup_rvq),
        ("INT8-ALL", q3_out, q3_size_model + q3_size_decoders, sim_all, speedup_total),
    ]
    for name, out, size, sim, sp in configs:
        print(f"{name:<20s} {out['elapsed']:7.2f}s {out['timings']['meanflow_s']:9.2f}s "
              f"{out['timings']['rvq_s']:7.2f}s {sim:7.4f} {size:8.1f}MB {sp:7.2f}x")
    print("=" * 90)


if __name__ == "__main__":
    main()
