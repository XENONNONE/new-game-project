"""SMPL-X to VRM humanoid transport mapping."""

from __future__ import annotations

import numpy as np

SMPLX_TO_VRM = {
    0: "hips",
    1: "leftUpperLeg",
    2: "rightUpperLeg",
    3: "spine",
    4: "leftLowerLeg",
    5: "rightLowerLeg",
    6: "chest",
    7: "leftFoot",
    8: "rightFoot",
    9: "upperChest",
    10: "leftToes",
    11: "rightToes",
    12: "neck",
    13: "leftShoulder",
    14: "rightShoulder",
    15: "head",
    16: "leftUpperArm",
    17: "rightUpperArm",
    18: "leftLowerArm",
    19: "rightLowerArm",
    20: "leftHand",
    21: "rightHand",
    23: "leftEye",
    24: "rightEye",
}
for side, base in (("left", 25), ("right", 40)):
    for fi, finger in enumerate(("Index", "Middle", "Little", "Ring", "Thumb")):
        for si, suffix in enumerate(("Proximal", "Intermediate", "Distal")):
            SMPLX_TO_VRM[base + fi * 3 + si] = side + finger + suffix


def matrix_to_quaternion(m: np.ndarray) -> np.ndarray:
    """Convert [...,3,3] matrices to normalized [x,y,z,w]."""
    q = np.empty((*m.shape[:-2], 4), np.float32)
    q[..., 3] = np.sqrt(np.maximum(0, 1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2])) / 2
    q[..., 0] = np.copysign(
        np.sqrt(np.maximum(0, 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2])) / 2,
        m[..., 2, 1] - m[..., 1, 2],
    )
    q[..., 1] = np.copysign(
        np.sqrt(np.maximum(0, 1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2])) / 2,
        m[..., 0, 2] - m[..., 2, 0],
    )
    q[..., 2] = np.copysign(
        np.sqrt(np.maximum(0, 1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2])) / 2,
        m[..., 1, 0] - m[..., 0, 1],
    )
    result: np.ndarray = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    return result


def vrm_frames(matrices: np.ndarray, ema: float = 0.35) -> list[dict]:
    quats = matrix_to_quaternion(matrices)
    previous: dict[str, np.ndarray] = {}
    frames: list[dict] = []
    for frame in quats:
        output = {}
        for joint, bone in SMPLX_TO_VRM.items():
            q, old = frame[joint], previous.get(bone)
            if old is not None:
                if np.dot(old, q) < 0:
                    q = -q
                q = old * ema + q * (1 - ema)
                q /= max(np.linalg.norm(q), 1e-8)
            previous[bone] = q
            output[bone] = [round(float(v), 6) for v in q]
        frames.append(output)
    return frames
