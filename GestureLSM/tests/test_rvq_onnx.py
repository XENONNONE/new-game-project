"""Export RVQ quantizer + decoder to ONNX and create a full ONNX RVQ wrapper.

This script:
1. Exports the RVQ quantizer (simplified, deterministic argmin) to ONNX for each of the 3 decoders
2. Exports the RVQ decoder (Conv1d/Upsample) to ONNX for each of the 3 decoders
3. Creates an ONNXRVQDecoder class that runs quantizer+decoder through ONNX Runtime
4. Benchmarks PyTorch latent2origin vs ONNX quantizer+decoder
5. Verifies quality (cosine similarity)
"""
from __future__ import annotations

import time
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GESTURE_ROOT = PROJECT_ROOT / "GestureLSM"
sys.path.insert(0, str(GESTURE_ROOT))
ONNX_DIR = PROJECT_ROOT / "models"

RVQ_NAMES = ["upper", "hands", "lower"]
RVQ_WIDTHS = {"upper": 78, "hands": 180, "lower": 57}
RVQ_CKPTS = {
    "upper": "net_300000_upper.pth",
    "hands": "net_300000_hands.pth",
    "lower": "net_300000_lower.pth",
}


class SimpleQuantizer(nn.Module):
    """Deterministic, ONNX-friendly reimplementation of QuantizeEMAReset.

    In eval mode, gumbel_sample reduces to argmin(distance) (no noise).
    This version skips perplexity/commit_loss (not needed for inference)
    and clones the residual to avoid in-place mutation.
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
            proj_sq = torch.sum(projected ** 2, dim=1, keepdim=True)  # (nb_code, 1)
            res_sq = torch.sum(residual ** 2, dim=1, keepdim=True)    # (N*T, 1)
            distance = res_sq - 2 * residual @ projected.t() + proj_sq.t()  # (N*T, nb_code)
            code_idx = torch.argmin(distance, dim=-1)
            quantized = projected[code_idx]
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

        return quantized_out.view(N, T, C).permute(0, 2, 1)  # (N, C, T)


def load_rvq(name: str):
    """Load full RVQ model from checkpoint."""
    from models.vq.model import RVQVAE

    rvq = RVQVAE(
        SimpleNamespace(
            num_quantizers=6,
            shared_codebook=False,
            quantize_dropout_prob=0.2,
            quantize_dropout_cutoff_index=0,
            mu=0.99,
            beta=1.0,
        ),
        RVQ_WIDTHS[name],
        1024,
        128,
        128,
        2,
        2,
        512,
        3,
        3,
        "relu",
        None,
    )
    state = torch.load(
        PROJECT_ROOT / "ckpt" / RVQ_CKPTS[name], map_location="cpu", weights_only=False
    )["net"]
    rvq.load_state_dict(
        {k.removeprefix("module."): v for k, v in state.items()}, strict=True
    )
    rvq.eval()
    return rvq


def export_all():
    """Export quantizer + decoder ONNX for all 3 RVQ models."""
    import onnxruntime as ort

    print("=== Exporting RVQ ONNX models ===")

    for name in RVQ_NAMES:
        rvq = load_rvq(name)
        seq_len = 32
        x_in = torch.randn(1, 128, seq_len)  # (N, C, T) for quantizer/decoder

        # --- Quantizer ONNX ---
        sq = SimpleQuantizer(rvq.quantizer)
        quantizer_path = ONNX_DIR / f"rvq_quantizer_{name}.onnx"

        # Verify PT matches original
        with torch.inference_mode():
            x_clone = x_in.clone()
            pt_q = sq(x_clone)
            orig_out, _, _, _ = rvq.quantizer(x_in.clone(), sample_codebook_temp=0.5)
        # orig_out is (N, C, T), pt_q is (N, C, T)
        cos_q = float(
            np.sum(pt_q.numpy().ravel() * orig_out.numpy().ravel())
            / (np.linalg.norm(pt_q.numpy().ravel()) * np.linalg.norm(orig_out.numpy().ravel()) + 1e-8)
        )
        print(f"  [{name}] SimpleQuantizer vs original: cos_sim={cos_q:.6f}")

        # Export
        torch.onnx.export(
            sq,
            x_in,
            str(quantizer_path),
            input_names=["x"],
            output_names=["output"],
            dynamic_axes={
                "x": {0: "batch", 2: "seq"},
                "output": {0: "batch", 2: "seq"},
            },
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )
        print(f"  [{name}] Quantizer exported: {quantizer_path.stat().st_size / 1e6:.1f}MB")

        # --- Decoder ONNX ---
        # Get quantized input from PT quantizer for decoder export
        # Must use no_grad (not inference_mode) so tensors are usable for export tracing
        with torch.no_grad():
            x_quantized = sq(x_in.clone()).clone()
        decoder_path = ONNX_DIR / f"rvq_decoder_{name}.onnx"

        torch.onnx.export(
            rvq.decoder,
            x_quantized,
            str(decoder_path),
            input_names=["x_quantized"],
            output_names=["output"],
            dynamic_axes={
                "x_quantized": {0: "batch", 2: "seq"},
                "output": {0: "batch", 2: "seq"},
            },
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )
        print(f"  [{name}] Decoder exported: {decoder_path.stat().st_size / 1e6:.1f}MB")

        # Verify decoder ONNX matches PT
        with torch.inference_mode():
            pt_dec = rvq.decoder(x_quantized)[0]
        sess_dec = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])
        onnx_dec = sess_dec.run(None, {"x_quantized": x_quantized.numpy()})[0]
        cos_d = float(
            np.sum(pt_dec.numpy().ravel() * onnx_dec.ravel())
            / (np.linalg.norm(pt_dec.numpy().ravel()) * np.linalg.norm(onnx_dec.ravel()) + 1e-8)
        )
        print(f"  [{name}] Decoder ONNX vs PT: cos_sim={cos_d:.6f}")

        # Full pipeline: ONNX quantizer + ONNX decoder vs PT latent2origin
        with torch.inference_mode():
            pt_full = rvq.latent2origin(x_in.clone().permute(0, 2, 1))[0]
        # Note: latent2origin expects (N, T, C), permutes to (N, C, T) internally

        sess_q = ort.InferenceSession(str(quantizer_path), providers=["CPUExecutionProvider"])
        x_quantized_onnx = sess_q.run(None, {"x": x_in.numpy()})[0]
        onnx_full = sess_dec.run(None, {"x_quantized": x_quantized_onnx})[0]
        cos_full = float(
            np.sum(pt_full.numpy().ravel() * onnx_full.ravel())
            / (np.linalg.norm(pt_full.numpy().ravel()) * np.linalg.norm(onnx_full.ravel()) + 1e-8)
        )
        print(f"  [{name}] Full (quant+dec) ONNX vs PT latent2origin: cos_sim={cos_full:.6f}")

        # Benchmark
        with torch.inference_mode():
            for _ in range(3): _ = rvq.latent2origin(x_in.clone().permute(0, 2, 1))
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(10): _ = rvq.latent2origin(x_in.clone().permute(0, 2, 1))
        pt_t = (time.perf_counter() - t0) / 10

        for _ in range(3):
            _ = sess_q.run(None, {"x": x_in.numpy()})
            _ = sess_dec.run(None, {"x_quantized": x_quantized_onnx})
        t0 = time.perf_counter()
        for _ in range(10):
            q = sess_q.run(None, {"x": x_in.numpy()})
            _ = sess_dec.run(None, {"x_quantized": q[0]})
        onnx_t = (time.perf_counter() - t0) / 10

        print(f"  [{name}] latent2origin PT: {pt_t*1000:.1f}ms, quant+dec ONNX: {onnx_t*1000:.1f}ms, speedup: {pt_t/onnx_t:.1f}x")

    print("\n=== ONNX export complete ===")


def main():
    threads = 6
    torch.set_num_threads(threads)
    export_all()


if __name__ == "__main__":
    main()
