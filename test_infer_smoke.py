"""
test_infer_smoke.py
-------------------
Phase 5 / Step 2 sanity check.

Goal:
    Confirm that openvla_infer.OpenVLAInference can be instantiated and that
    .predict() returns a sane 7-D action on a black image.

This is the thing you run right after `pip install` + `python setup_hf.py`.
If this passes, you can move on to mujoco_bridge / run_openvla_mujoco.

Expected output (CPU bfloat16, first run):
    === test_infer_smoke ===
    [openvla] loading processor ...
    [openvla] processor loaded in 2.1s
    [openvla] loading model with strategy=cpu-bfloat16 dtype=torch.bfloat16 device=cpu
    [openvla] model loaded in 180.4s (total 182.5s)
    [openvla] predict_action took 64.1s -> action=[0.012, -0.004, 0.087, ...]
    SHAPE_OK=1 RANGE_OK=1 VALUE_FINITE=1
    SMOKE_OK=1

Failure triage (Windows CPU):
    - ImportError on bitsandbytes : fine, you should not be loading that path.
    - OOM on model load           : your strategy is wrong; force 'cpu-disk-offload'.
    - 403 / GatedRepoError        : re-run setup_hf.py.
    - shape != (7,)               : OpenVLA processor changed; pin transformers==4.40.1.
"""

from __future__ import annotations

import sys

import numpy as np

from config import CFG
from openvla_infer import OpenVLAInference


def main() -> int:
    print("=== phase5_openvla / test_infer_smoke ===")
    infer = OpenVLAInference()

    img = np.zeros((CFG.image_size, CFG.image_size, 3), dtype=np.uint8)
    action = infer.predict(img, "pick up the cube")

    shape_ok = action.shape == (CFG.action_dim,)
    range_ok = bool(np.all(np.abs(action) <= 1.0 + 1e-6))
    finite_ok = bool(np.all(np.isfinite(action)))

    print(f"action         : {action.tolist()}")
    print(f"SHAPE_OK={int(shape_ok)} RANGE_OK={int(range_ok)} VALUE_FINITE={int(finite_ok)}")

    ok = shape_ok and range_ok and finite_ok
    print(f"SMOKE_OK={int(ok)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
