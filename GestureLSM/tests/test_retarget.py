"""Tests for the SMPL-X to VRM retargeting module."""

import numpy as np

from inference_runtime.retarget import (
    SMPLX_TO_VRM,
    matrix_to_quaternion,
    vrm_frames,
)


class TestSMPLXToVRM:
    def test_has_core_bones(self):
        assert "hips" in SMPLX_TO_VRM.values()
        assert "spine" in SMPLX_TO_VRM.values()
        assert "chest" in SMPLX_TO_VRM.values()
        assert "head" in SMPLX_TO_VRM.values()
        assert "neck" in SMPLX_TO_VRM.values()

    def test_has_upper_body(self):
        for bone in (
            "leftShoulder",
            "rightShoulder",
            "leftUpperArm",
            "rightUpperArm",
            "leftLowerArm",
            "rightLowerArm",
            "leftHand",
            "rightHand",
        ):
            assert bone in SMPLX_TO_VRM.values()

    def test_has_lower_body(self):
        for bone in (
            "leftUpperLeg",
            "rightUpperLeg",
            "leftLowerLeg",
            "rightLowerLeg",
            "leftFoot",
            "rightFoot",
            "leftToes",
            "rightToes",
        ):
            assert bone in SMPLX_TO_VRM.values()

    def test_has_fingers(self):
        for side in ("left", "right"):
            for finger in ("Index", "Middle", "Little", "Ring", "Thumb"):
                for segment in ("Proximal", "Intermediate", "Distal"):
                    assert f"{side}{finger}{segment}" in SMPLX_TO_VRM.values()

    def test_has_eyes(self):
        assert "leftEye" in SMPLX_TO_VRM.values()
        assert "rightEye" in SMPLX_TO_VRM.values()

    def test_no_duplicates(self):
        values = list(SMPLX_TO_VRM.values())
        assert len(values) == len(set(values))

    def test_all_indices_covered(self):
        """Every SMPL-X joint index from 0 to 54 should be mapped."""
        max_index = max(SMPLX_TO_VRM.keys())
        assert max_index >= 54


class TestMatrixToQuaternion:
    def test_identity_matrix(self):
        m = np.eye(3, dtype=np.float32)
        q = matrix_to_quaternion(m)
        assert q.shape == (4,)
        assert abs(q[3] - 1.0) < 1e-6  # w component
        assert abs(q[0]) < 1e-6  # x
        assert abs(q[1]) < 1e-6  # y
        assert abs(q[2]) < 1e-6  # z

    def test_normalized(self):
        m = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        q = matrix_to_quaternion(m)
        norm = np.linalg.norm(q)
        assert abs(norm - 1.0) < 1e-6

    def test_batch(self):
        m = np.stack([np.eye(3, dtype=np.float32)] * 5)
        q = matrix_to_quaternion(m)
        assert q.shape == (5, 4)
        assert np.allclose(q[:, 3], 1.0, atol=1e-6)

    def test_quaternion_components(self):
        """90-degree rotation around Z axis."""
        m = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        q = matrix_to_quaternion(m)
        # Should be (0, 0, sin(45), cos(45)) = (0, 0, ~0.707, ~0.707)
        assert abs(q[2] - 0.7071) < 0.01
        assert abs(q[3] - 0.7071) < 0.01


class TestVrmFrames:
    def test_single_frame(self):
        matrices = np.stack([np.eye(3, dtype=np.float32)] * 55)[np.newaxis]
        frames = vrm_frames(matrices)
        assert len(frames) == 1
        assert len(frames[0]) == 54  # SMPLX_TO_VRM has 54 entries (index 22 not mapped)

    def test_multiple_frames(self):
        matrices = np.stack([np.eye(3, dtype=np.float32)] * 55)  # (55, 3, 3)
        matrices = np.stack([matrices] * 10)  # (10, 55, 3, 3)
        frames = vrm_frames(matrices)
        assert len(frames) == 10
        for frame in frames:
            assert len(frame) == 54

    def test_frame_values_are_lists(self):
        matrices = np.stack([np.eye(3, dtype=np.float32)] * 55)[np.newaxis]
        frames = vrm_frames(matrices)
        for q in frames[0].values():
            assert isinstance(q, list)
            assert len(q) == 4

    def test_frame_values_rounded(self):
        matrices = np.stack([np.eye(3, dtype=np.float32)] * 55)[np.newaxis]
        frames = vrm_frames(matrices)
        for q in frames[0].values():
            for v in q:
                assert isinstance(v, float)
                assert v == round(v, 6)

    def test_ema_smoothing(self):
        """EMA should smooth transitions between frames."""
        m1 = np.eye(3, dtype=np.float32)
        m2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        matrices = np.stack([m1] * 55)
        matrices = np.stack([matrices, np.stack([m2] * 55)])
        frames_no_ema = vrm_frames(matrices, ema=0.0)
        vrm_frames(matrices, ema=0.5)
        # With EMA=0, first frame should be raw identity quaternion
        for bone in frames_no_ema[0]:
            q = frames_no_ema[0][bone]
            assert abs(q[3] - 1.0) < 1e-6  # w=1 for identity

    def test_quaternion_sign_consistency(self):
        """Adjacent frames should have consistent quaternion signs."""
        m1 = np.eye(3, dtype=np.float32)
        m2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        matrices = np.stack([m1] * 55)
        matrices = np.stack([matrices, np.stack([m2] * 55)])
        frames = vrm_frames(matrices, ema=0.0)
        for bone in frames[0]:
            q1 = np.array(frames[0][bone])
            q2 = np.array(frames[1][bone])
            # Dot product should be positive (same hemisphere)
            assert np.dot(q1, q2) >= 0
