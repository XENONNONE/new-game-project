"""Export MeanFlow denoiser to ONNX and benchmark vs PyTorch."""
from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore") if (warnings := __import__("warnings")) else None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GESTURE_ROOT = PROJECT_ROOT / "GestureLSM"
sys.path.insert(0, str(GESTURE_ROOT))

from inference_runtime.pipeline import GesturePipeline
from inference_runtime.audio import read_wav, onset_amplitude
from inference_runtime.config import CONFIG

ONNX_PATH = PROJECT_ROOT / "models" / "meanflow_denoiser.onnx"


def get_dummy_inputs(pipe: GesturePipeline) -> dict[str, torch.Tensor]:
    """Match the exact shapes the pipeline passes to the denoiser."""
    model = pipe.model
    d = model.denoiser
    input_dim = d.input_dim
    latent_dim = d.latent_dim
    num_joints = d.joint_num
    seq_len = d.seq_len

    # x_t shape from pipeline: (batch, input_dim*num_joints, 1, seq_len)
    x = torch.randn(1, input_dim * num_joints, 1, seq_len)

    # timesteps: scalar tensor [1.0] (inference uses 1 step)
    timesteps = torch.ones(1)

    # seed: [1, 3, 128, 4] as initialized in pipeline.__init__
    seed = torch.zeros(1, num_joints, 128, 4)

    # at_feat: output of modality_encoder
    # audio_onset [1, N, 2] -> WavEncoder -> [1, target_length, audio_dim] -> mix -> [1, T, latent_dim*joint_num]
    # After avg_pool with squeeze_scale=4: [1, target_length//4, latent_dim*joint_num]
    # config: target_length=128, latent_dim=256, joint_num=3 → [1, 32, 768]
    at_feat = torch.randn(1, seq_len, latent_dim * num_joints)

    # cond_time: zeros [1] for single-step inference
    cond_time = torch.zeros(1)

    return {
        "x": x,
        "timesteps": timesteps,
        "cond_time": cond_time,
        "seed": seed,
        "at_feat": at_feat,
    }


def export_onnx(pipe: GesturePipeline) -> None:
    model = pipe.model.denoiser
    model.eval()

    dummy = get_dummy_inputs(pipe)

    print("Exporting denoiser to ONNX...")
    input_names = ["x", "timesteps", "cond_time", "seed", "at_feat"]
    output_names = ["output"]
    dynamic_axes = {
        "x": {0: "batch"},
        "at_feat": {0: "batch"},
        "seed": {0: "batch"},
        "output": {0: "batch"},
    }

    # Try legacy TorchScript exporter first (more forgiving with dynamic shapes)
    try:
        torch.onnx.export(
            model,
            (dummy["x"], dummy["timesteps"], dummy["cond_time"], dummy["seed"], dummy["at_feat"]),
            str(ONNX_PATH),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        print(f"Exported (legacy) to {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.1f}MB)")
        return
    except Exception as e:
        print(f"Legacy export failed: {e}")
        print("Trying dynamo-based export with static shapes...")

    # Fallback: dynamo export with no dynamic axes
    try:
        torch.onnx.export(
            model,
            (dummy["x"], dummy["timesteps"], dummy["cond_time"], dummy["seed"], dummy["at_feat"]),
            str(ONNX_PATH),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,
            opset_version=18,
        )
        print(f"Exported (dynamo, static) to {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.1f}MB)")
        return
    except Exception as e2:
        print(f"Dynamo export also failed: {e2}")
        raise


def run_pytorch_baseline(pipe: GesturePipeline, inputs: dict) -> tuple[torch.Tensor, float]:
    model = pipe.model.denoiser
    model.eval()
    # Warmup
    with torch.inference_mode():
        _ = model(**inputs)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(5):
            out = model(**inputs)
    t1 = time.perf_counter()
    avg = (t1 - t0) / 5
    print(f"[PyTorch FP32] {avg:.3f}s avg (5 runs)")
    return out, avg


def run_onnx_int8_runtime(onnx_path: str, inputs: dict) -> tuple[np.ndarray, float]:
    """Run ONNX model quantized to INT8 via onnxruntime.quantization."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    int8_path = str(Path(onnx_path).with_suffix(".int8.onnx"))
    try:
        quantize_dynamic(
            onnx_path,
            int8_path,
            weight_type=QuantType.QInt8,
        )
        print(f"Quantized ONNX model saved to {int8_path}")
    except Exception as e:
        print(f"INT8 quantization failed: {e}, trying per-channel...")
        quantize_dynamic(
            onnx_path,
            int8_path,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )

    import onnxruntime as ort

    session = ort.InferenceSession(
        int8_path if Path(int8_path).exists() else onnx_path,
        providers=["CPUExecutionProvider"],
    )
    feeds = {
        "x": inputs["x"].numpy(),
        "timesteps": inputs["timesteps"].numpy(),
        "cond_time": inputs["cond_time"].numpy(),
        "seed": inputs["seed"].numpy(),
        "at_feat": inputs["at_feat"].numpy(),
    }
    _ = session.run(None, feeds)
    t0 = time.perf_counter()
    for _ in range(5):
        out = session.run(None, feeds)
    t1 = time.perf_counter()
    avg = (t1 - t0) / 5
    print(f"[ONNX INT8]      {avg:.3f}s avg (5 runs)")
    return out[0], avg


def run_onnx_runtime(onnx_path: str, inputs: dict) -> tuple[np.ndarray, float]:
    """Run ONNX model with ONNX Runtime (FP32)."""
    import onnxruntime as ort

    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )
    feeds = {
        "x": inputs["x"].numpy(),
        "timesteps": inputs["timesteps"].numpy(),
        "cond_time": inputs["cond_time"].numpy(),
        "seed": inputs["seed"].numpy(),
        "at_feat": inputs["at_feat"].numpy(),
    }
    _ = session.run(None, feeds)
    t0 = time.perf_counter()
    for _ in range(5):
        out = session.run(None, feeds)
    t1 = time.perf_counter()
    avg = (t1 - t0) / 5
    print(f"[ONNX Runtime]  {avg:.3f}s avg (5 runs)")
    return out[0], avg


def run_int8_pytorch(pipe: GesturePipeline, inputs: dict) -> tuple[torch.Tensor, float]:
    from torch.quantization import quantize_dynamic

    model = pipe.model.denoiser
    model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    model.eval()

    with torch.inference_mode():
        _ = model(**inputs)

    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(5):
            out = model(**inputs)
    t1 = time.perf_counter()
    avg = (t1 - t0) / 5
    print(f"[PyTorch INT8]   {avg:.3f}s avg (5 runs)")
    return out, avg


def main():
    threads = CONFIG.pipeline.threads or min(6, torch.get_num_threads())
    torch.set_num_threads(threads)
    print(f"Threads: {threads}")

    print("\nLoading pipeline...")
    pipe = GesturePipeline(threads=threads)
    inputs = get_dummy_inputs(pipe)
    print(f"Input shapes:")
    for k, v in inputs.items():
        print(f"  {k}: {list(v.shape)}")

    # Export
    if not ONNX_PATH.exists():
        export_onnx(pipe)
    else:
        print(f"ONNX model already exists at {ONNX_PATH}")

    print("\n=== Speed benchmark (5 runs each) ===")
    out_fp32, t_fp32 = run_pytorch_baseline(pipe, inputs)
    out_int8, t_int8 = run_int8_pytorch(pipe, inputs)
    out_onnx, t_onnx = run_onnx_runtime(str(ONNX_PATH), inputs)
    out_onnx_int8, t_onnx_int8 = run_onnx_int8_runtime(str(ONNX_PATH), inputs)

    print(f"\n=== Speedup summary ===")
    print(f"  PyTorch FP32:    {t_fp32:.3f}s")
    print(f"  PyTorch INT8:    {t_int8:.3f}s  ({t_fp32/t_int8:.2f}x vs FP32)")
    print(f"  ONNX Runtime:    {t_onnx:.3f}s  ({t_fp32/t_onnx:.2f}x vs FP32)")
    print(f"  ONNX INT8:       {t_onnx_int8:.3f}s  ({t_fp32/t_onnx_int8:.2f}x vs FP32)")

    print(f"\n=== Quality comparison ===")
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        a_flat = a.squeeze().ravel()
        b_flat = b.ravel()
        return float(np.abs(np.sum(a_flat * b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-8)))
    max_diff = float(np.max(np.abs(out_fp32.squeeze().numpy() - out_onnx)))
    cos_sim = cosine_sim(out_fp32.squeeze().numpy(), out_onnx)
    print(f"  FP32 vs ONNX FP32 max abs diff: {max_diff:.6f}")
    print(f"  FP32 vs ONNX cosine sim: {cos_sim:.6f}")

    max_diff_int8_pt = float(np.max(np.abs(out_fp32.squeeze().numpy() - out_int8.squeeze().numpy())))
    cos_sim_int8_pt = cosine_sim(out_fp32.squeeze().numpy(), out_int8.squeeze().numpy())
    print(f"  FP32 vs INT8 (PyTorch) max abs diff: {max_diff_int8_pt:.6f}")
    print(f"  FP32 vs INT8 (PyTorch) cosine sim: {cos_sim_int8_pt:.6f}")

    max_diff_int8_onnx = float(np.max(np.abs(out_fp32.squeeze().numpy() - out_onnx_int8)))
    cos_sim_int8_onnx = cosine_sim(out_fp32.squeeze().numpy(), out_onnx_int8)
    print(f"  FP32 vs ONNX INT8 max abs diff: {max_diff_int8_onnx:.6f}")
    print(f"  FP32 vs ONNX INT8 cosine sim: {cos_sim_int8_onnx:.6f}")


if __name__ == "__main__":
    main()
