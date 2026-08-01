"""Compare FP32 vs INT8 quantized MeanFlow — targets heavy submodules directly.

Tries multiple quantization strategies:
  1. Dynamic INT8 on Linear layers (torch.quantization)
  2. TorchAO Int8WeightOnly (if installed)
  3. TorchAO Int8DynamicActivationInt8Weight (if installed)
"""
from __future__ import annotations

import time
import sys
import json
import warnings
from pathlib import Path
from copy import deepcopy

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


def count_quantized_dynamic(model: torch.nn.Module) -> int:
    """Count params in torch dynamic-quantized Linear modules."""
    count = 0
    for name, module in model.named_modules():
        tname = type(module).__module__ + "." + type(module).__name__
        if "quantized.dynamic" in tname and "Linear" in type(module).__name__:
            for p in module._parameters.values():
                if p is not None and hasattr(p, "numel"):
                    count += p.numel()
    return count


def total_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def quantize_dynamic_linear(model: torch.nn.Module) -> torch.nn.Module:
    """Quantize Linear layers in the denoiser and modality_encoder."""
    if hasattr(model, "denoiser"):
        model.denoiser = quantize_dynamic(model.denoiser, {nn.Linear}, dtype=torch.qint8)
    if hasattr(model, "modality_encoder"):
        model.modality_encoder = quantize_dynamic(
            model.modality_encoder, {nn.Linear}, dtype=torch.qint8
        )
    return model


def quantize_torchao(model: torch.nn.Module, config_name: str) -> torch.nn.Module:
    """Quantize with torchao (weight-only INT8 or dynamic)."""
    from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int8DynamicActivationInt8WeightConfig

    cfg = Int8WeightOnlyConfig() if config_name == "weight_only" else Int8DynamicActivationInt8WeightConfig()
    quantize_(model.denoiser, cfg)
    quantize_(model.modality_encoder, cfg)
    return model


def run_inference(pipe: GesturePipeline, audio: bytes, label: str) -> dict:
    t0 = time.perf_counter()
    result = pipe.infer_wav(audio, reset=True)
    elapsed = time.perf_counter() - t0
    timings = result["timings"]
    print(f"[{label}] Total: {elapsed:.2f}s | MeanFlow: {timings['meanflow_s']:.2f}s | "
          f"RVQ: {timings['rvq_s']:.2f}s | Frames: {len(result['frames'])}")
    return {"result": result, "elapsed": elapsed, "timings": timings}


def quaternion_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    dot = np.abs(np.sum(a * b, axis=-1))
    return float(np.mean(np.clip(dot, 0, 1)))


def compare_frames(fp32_frames: list, int8_frames: list) -> None:
    if len(fp32_frames) != len(int8_frames):
        print(f"  Frame mismatch: FP32={len(fp32_frames)} vs INT8={len(int8_frames)}")
        return
    sims = []
    for f32, i8 in zip(fp32_frames, int8_frames):
        for bone in f32:
            if bone in i8:
                a = np.array(f32[bone], dtype=np.float64)
                b = np.array(i8[bone], dtype=np.float64)
                sims.append(quaternion_similarity(a, b))
    avg_sim = np.mean(sims) if sims else 0
    min_sim = np.min(sims) if sims else 0
    print(f"  Bone comparisons: {len(sims)} | Avg cosine sim: {avg_sim:.4f} (min: {min_sim:.4f})")
    if avg_sim > 0.99:
        print("  Quality: EXCELLENT (negligible degradation)")
    elif avg_sim > 0.95:
        print("  Quality: HIGH (minimal degradation)")
    elif avg_sim > 0.85:
        print("  Quality: MODERATE (some degradation)")
    else:
        print("  Quality: LOW (significant degradation)")


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


def main():
    threads = CONFIG.pipeline.threads or min(6, torch.get_num_threads())
    torch.set_num_threads(threads)
    print(f"Threads: {threads}")

    audio = load_test_audio()
    _, info = read_wav(audio, return_info=True)
    print(f"Audio: {info.duration_s:.2f}s ({info.output_samples} samples)")

    # FP32 baseline
    print("\n=== FP32 (original) ===")
    pipe_fp32 = GesturePipeline(threads=threads)
    tp = total_params(pipe_fp32.model)
    fp32_out = run_inference(pipe_fp32, audio, "FP32")
    fp32_size = model_disk_size(pipe_fp32.model)
    print(f"  Total params: {tp:,} | Model size: {fp32_size:.1f}MB")

    # Strategy 1: torch dynamic INT8 (Linear only)
    print("\n=== Strategy 1: Dynamic INT8 (Linear layers only) ===")
    pipe_q1 = GesturePipeline(threads=threads)
    pipe_q1.model = quantize_dynamic_linear(pipe_q1.model)
    n_quant = count_quantized_dynamic(pipe_q1.model)
    print(f"  Quantized {n_quant}/{tp} params ({n_quant/tp*100:.1f}%)")
    q1_out = run_inference(pipe_q1, audio, "INT8")
    q1_size = model_disk_size(pipe_q1.model)
    speedup = fp32_out["timings"]["meanflow_s"] / q1_out["timings"]["meanflow_s"]
    print(f"  Speedup: {speedup:.2f}x | Size: {q1_size:.1f}MB ({100*(1-q1_size/fp32_size):.0f}% smaller)")
    print("  Quality:")
    compare_frames(fp32_out["result"]["frames"], q1_out["result"]["frames"])

    # Strategy 2: torchao (if available)
    try:
        import torchao  # noqa: F401
        HAS_TORCHAO = True
    except ImportError:
        HAS_TORCHAO = False

    if HAS_TORCHAO:
        for strategy_name, config_name in [("Weight-only INT8", "weight_only"),
                                            ("Dynamic INT8", "dynamic")]:
            print(f"\n=== Strategy 2: torchao {strategy_name} ===")
            pipe_q = GesturePipeline(threads=threads)
            try:
                pipe_q.model = quantize_torchao(pipe_q.model, config_name)
                q_out = run_inference(pipe_q, audio, strategy_name)
                q_size = model_disk_size(pipe_q.model)
                speedup = fp32_out["timings"]["meanflow_s"] / q_out["timings"]["meanflow_s"]
                print(f"  Speedup: {speedup:.2f}x | Size: {q_size:.1f}MB ({100*(1-q1_size/fp32_size):.0f}% smaller)")
                print("  Quality:")
                compare_frames(fp32_out["result"]["frames"], q_out["result"]["frames"])
            except Exception as e:
                print(f"  FAILED: {e}")
    else:
        print("\n=== torchao not installed ===")
        print("Install with: pip install torchao")
        print("Then rerun for potentially better speed/quality on transformers.")

    # Save frames for visual inspection
    out_dir = Path("/tmp/avatar_quant_test")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "fp32_frames.json", "w") as f:
        json.dump(fp32_out["result"]["frames"][:10], f, indent=2, default=str)
    with open(out_dir / "int8_frames.json", "w") as f:
        json.dump(q1_out["result"]["frames"][:10], f, indent=2, default=str)
    print(f"\nFrames saved to {out_dir}/")


if __name__ == "__main__":
    main()
