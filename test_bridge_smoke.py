"""
test_bridge_smoke.py
--------------------
Phase 5 / Step 3 sanity check (no OpenVLA, no MuJoCo viewer).

Tests mujoco_bridge.openvla_action_to_ctrl on synthetic actions and dummy
models for all three env_tags. This lets you validate the bridge logic
independently of the heavy OpenVLA load.

Expected output:
    === test_bridge_smoke ===
    [fetch_reach] ctrl=[0.5, -0.2, 0.3, 1.0]                SHAPE_OK=1 RANGE_OK=1
    [reacher]     ctrl=[0.1, -0.04]                         SHAPE_OK=1 RANGE_OK=1
    [custom_xml]  ctrl=[0.5, -0.2, 0.3, 0.0, 0.0, 0.0]      SHAPE_OK=1 RANGE_OK=1
    [extremes]    ctrl clipped to [-1,1] on all envs
    SMOKE_OK=1
"""

from __future__ import annotations

import numpy as np

from mujoco_bridge import openvla_action_to_ctrl, _DummyModel


def _case(tag: str, nu: int, action: np.ndarray) -> np.ndarray:
    m = _DummyModel(nu=nu)
    ctrl = openvla_action_to_ctrl(action, m, data=None, env_tag=tag)
    return ctrl


def main() -> int:
    print("=== phase5_openvla / test_bridge_smoke ===")
    ok = True

    a_normal = np.array([0.5, -0.2, 0.3, 0.1, -0.1, 0.0, 1.0], dtype=np.float32)

    for tag, nu in [("fetch_reach", 4), ("reacher", 2), ("custom_xml", 6)]:
        ctrl = _case(tag, nu, a_normal)
        shape_ok = ctrl.shape == (nu,)
        range_ok = bool(np.all(np.abs(ctrl) <= 1.0 + 1e-9))
        finite_ok = bool(np.all(np.isfinite(ctrl)))
        print(
            f"[{tag:<11}] ctrl={np.round(ctrl, 3).tolist()}  "
            f"SHAPE_OK={int(shape_ok)} RANGE_OK={int(range_ok)} FINITE={int(finite_ok)}"
        )
        ok = ok and shape_ok and range_ok and finite_ok

    # Extremes: make sure clipping kicks in.
    a_extreme = np.array([10.0, -10.0, 5.0, 5.0, -5.0, 5.0, 1.0], dtype=np.float32)
    for tag, nu in [("fetch_reach", 4), ("reacher", 2), ("custom_xml", 6)]:
        ctrl = _case(tag, nu, a_extreme)
        clip_ok = bool(np.all(np.abs(ctrl) <= 1.0 + 1e-9))
        print(f"[extremes/{tag:<11}] ctrl={np.round(ctrl, 3).tolist()}  CLIP_OK={int(clip_ok)}")
        ok = ok and clip_ok

    print(f"SMOKE_OK={int(ok)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
