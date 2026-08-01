import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from models.MeanFlow import GestureMF


# The released vocab pickle incorrectly stores this class under __main__.
class Vocab:
    pass

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT.parent / "ckpt" / "meanflow.pth"


def main():
    torch.set_num_threads(min(8, torch.get_num_threads()))
    cfg = OmegaConf.load(ROOT / "configs_new" / "meanflow_rvqvae_128.yaml")

    started = time.perf_counter()
    model = GestureMF(cfg)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    load_seconds = time.perf_counter() - started

    # One 128-frame synthetic conditioning window. This exercises the audio/text
    # encoder and the actual one-step MeanFlow denoiser without dataset or SMPL.
    conditions = {
        "y": {
            "audio_onset": torch.zeros(1, 128, 2),
            "word": torch.zeros(1, 128, dtype=torch.long),
            "id": torch.zeros(1, dtype=torch.long),
            "seed": torch.zeros(1, 3, 128, 4),
            "style_feature": torch.zeros(1, 512),
        }
    }

    started = time.perf_counter()
    with torch.inference_mode():
        output = model(conditions)
    inference_seconds = time.perf_counter() - started
    latents = output["latents"]

    print(f"torch={torch.__version__} device=cpu threads={torch.get_num_threads()}")
    print(f"checkpoint_load_seconds={load_seconds:.3f}")
    print(f"inference_seconds={inference_seconds:.3f}")
    print(f"latents_shape={tuple(latents.shape)}")
    print(f"latents_finite={bool(torch.isfinite(latents).all())}")
    print("PASS: GestureLSM MeanFlow completed a one-step ARM64 CPU forward pass")


if __name__ == "__main__":
    main()
