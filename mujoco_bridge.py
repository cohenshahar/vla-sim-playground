"""
mujoco_bridge.py
----------------
Phase 5 / Step 3 - map an OpenVLA 7-D action onto a MuJoCo control vector.

OpenVLA's action semantics (BridgeData-v2 convention, openvla/openvla-7b):
    a[0..2] : delta end-effector position (x, y, z), in "chunks" of motion.
    a[3..5] : delta end-effector orientation (roll, pitch, yaw), axis-angle.
    a[6]    : gripper (0 = open, 1 = closed). Some ckpts emit -1 / 1.

MuJoCo environments expect very different things:
    - FetchReach-v3 (Gymnasium-Robotics): ctrl is delta-EE (dx, dy, dz, gripper)
      of length 4. So we keep a[0..2] and a[6], drop rotation. This is a clean
      fit for OpenVLA on a reach task.
    - Reacher-v5 (Gymnasium): ctrl is two joint torques. OpenVLA's 7-D action
      doesn't map to torques at all, so we do a crude projection: treat
      a[0..1] as desired (x, y) delta and map to the two torques with a gain.
      This is expected to look wiggly; it's for plumbing only.
    - custom_xml: user-provided nu from model_mj.nu. We fill the first min(nu, 6)
      entries with a[0..5] scaled by a gain, ignore gripper.

The dispatcher returns a numpy array of length `model_mj.nu`, ready for
    data.ctrl[:] = ctrl
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def openvla_action_to_ctrl(
    action: np.ndarray,
    model_mj,
    data,
    env_tag: str = "fetch_reach",
    *,
    pos_gain: float = 1.0,
    rot_gain: float = 0.25,
    torque_gain: float = 0.2,
    gripper_gain: float = 1.0,
) -> np.ndarray:
    """
    Translate OpenVLA's 7-D action into a MuJoCo ctrl vector of length nu.

    Args:
        action       : shape (7,) float32, ~[-1, 1], as returned by OpenVLA.
        model_mj     : mujoco.MjModel (for nu and actuator_ctrlrange).
        data         : mujoco.MjData  (unused for now; kept for future envs
                       that need current state).
        env_tag      : 'fetch_reach' | 'reacher' | 'custom_xml'
        *_gain       : per-component gains; conservative defaults.

    Returns:
        np.ndarray shape (model_mj.nu,), dtype float64, clipped to ctrlrange.
    """
    a = _check_action(action)
    nu = int(model_mj.nu)

    if env_tag == "fetch_reach":
        ctrl = _fetch_reach(a, nu, pos_gain=pos_gain, gripper_gain=gripper_gain)
    elif env_tag == "reacher":
        ctrl = _reacher(a, nu, torque_gain=torque_gain)
    elif env_tag == "custom_xml":
        ctrl = _custom_xml(a, nu, pos_gain=pos_gain, rot_gain=rot_gain)
    else:
        raise ValueError(f"Unknown env_tag: {env_tag!r}")

    return _clip_to_ctrlrange(ctrl, model_mj)


# ---------------------------------------------------------------------------
# Per-env projections
# ---------------------------------------------------------------------------

def _fetch_reach(a: np.ndarray, nu: int, *, pos_gain: float, gripper_gain: float) -> np.ndarray:
    """FetchReach ctrl is (dx, dy, dz, gripper). nu is normally 4."""
    ctrl = np.zeros(nu, dtype=np.float64)
    n = min(3, nu)
    ctrl[:n] = pos_gain * a[:n]
    if nu >= 4:
        ctrl[3] = gripper_gain * (2.0 * a[6] - 1.0)  # map [0,1] -> [-1,1]
    return ctrl


def _reacher(a: np.ndarray, nu: int, *, torque_gain: float) -> np.ndarray:
    """Reacher ctrl is two joint torques. Crude projection from a[0..1]."""
    ctrl = np.zeros(nu, dtype=np.float64)
    n = min(2, nu)
    ctrl[:n] = torque_gain * a[:n]
    return ctrl


def _custom_xml(a: np.ndarray, nu: int, *, pos_gain: float, rot_gain: float) -> np.ndarray:
    """Generic fallback: first min(nu,6) actuators get a[0..5] with gains."""
    ctrl = np.zeros(nu, dtype=np.float64)
    # positional part
    np_pos = min(3, nu)
    ctrl[:np_pos] = pos_gain * a[:np_pos]
    # rotational part
    np_rot_start = 3
    np_rot_end = min(6, nu)
    if np_rot_end > np_rot_start:
        ctrl[np_rot_start:np_rot_end] = rot_gain * a[np_rot_start:np_rot_end]
    return ctrl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_action(action: np.ndarray) -> np.ndarray:
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.shape[0] != 7:
        raise ValueError(f"expected 7-D action, got shape={a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"non-finite values in action: {a.tolist()}")
    return a


def _clip_to_ctrlrange(ctrl: np.ndarray, model_mj) -> np.ndarray:
    """Clip each actuator to its declared ctrlrange (if declared)."""
    cr = getattr(model_mj, "actuator_ctrlrange", None)
    if cr is None:
        return ctrl
    cr = np.asarray(cr)  # shape (nu, 2)
    # MuJoCo convention: if the actuator's ctrllimited flag is 0, cr is (0,0).
    # In that case we leave ctrl untouched for that row.
    limited = getattr(model_mj, "actuator_ctrllimited", None)
    out = ctrl.copy()
    for i in range(ctrl.shape[0]):
        if limited is not None and int(limited[i]) == 0:
            continue
        lo, hi = float(cr[i, 0]), float(cr[i, 1])
        if hi > lo:
            out[i] = float(np.clip(out[i], lo, hi))
    return out


# ---------------------------------------------------------------------------
# CLI smoke: does the dispatcher handle all three env_tags on dummy shapes?
# ---------------------------------------------------------------------------

class _DummyModel:
    def __init__(self, nu: int, ctrlrange=None, ctrllimited=None):
        self.nu = nu
        if ctrlrange is None:
            ctrlrange = np.tile(np.array([[-1.0, 1.0]]), (nu, 1))
        self.actuator_ctrlrange = np.asarray(ctrlrange, dtype=np.float64)
        if ctrllimited is None:
            ctrllimited = np.ones(nu, dtype=np.int32)
        self.actuator_ctrllimited = np.asarray(ctrllimited)


def _cli_smoke() -> int:
    print("=== phase5_openvla / mujoco_bridge CLI smoke ===")
    a = np.array([0.5, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    for tag, nu in [("fetch_reach", 4), ("reacher", 2), ("custom_xml", 6)]:
        m = _DummyModel(nu=nu)
        ctrl = openvla_action_to_ctrl(a, m, data=None, env_tag=tag)
        print(f"{tag:<12} nu={nu}  ctrl={np.round(ctrl, 3).tolist()}")
        assert ctrl.shape == (nu,), f"shape mismatch for {tag}"
        assert np.all(np.abs(ctrl) <= 1.0 + 1e-9), f"clip failed for {tag}"
    print("SMOKE_OK=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_smoke())
