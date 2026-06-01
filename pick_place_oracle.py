"""
pick_place_oracle.py
====================
Scripted oracle for KR6 + vacuum: pick an object off the table, place it in the bin.
Drives the arm with KR6IKBridge.servo_to_pose and toggles the vacuum via set_suction.

This is the waypoint state machine from docs/pick_place_motion_spec.md. It (a) validates
the scene/IK/suction end-to-end, (b) is the demo generator for LoRA fine-tuning, and (c)
defines the success check. OpenVLA later replaces the waypoint logic with action_to_ctrl.

STATUS: UNTESTED ON HARDWARE. Requires the vacuum scene (suction_tip_site + act_vacuum +
the bin) from the design docs. Tune the heights/thresholds in PHASES on first run.

Usage:
    python pick_place_oracle.py --xml <world.xml> [--object metal_box] [--bin bin] \
        [--ee-site suction_tip_site] [--headless] [--save-video scratch/pp.mp4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from kr6_ik_bridge import KR6IKBridge, set_suction, KR6_VACUUM_ACT


# (name, z-height or None=hold, suction, dwell_ticks) ; xy comes from object/bin per phase
PHASES = [
    ("HOVER",   "obj", 1.05, False, 0),
    ("DESCEND", "obj", 0.94, False, 0),
    ("ENGAGE",  "obj", 0.94, True,  8),
    ("LIFT",    "obj", 1.15, True,  0),
    ("TRANSIT", "bin", 1.15, True,  0),
    ("LOWER",   "bin", 1.00, True,  0),
    ("RELEASE", "bin", 1.00, False, 5),
    ("RETREAT", "bin", 1.15, False, 0),
]

POS_TOL = 0.006          # m, "waypoint reached"
MAX_TICKS_PER_PHASE = 400
SUBSTEPS = 5


def _body_xy(model, data, name):
    import mujoco
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise SystemExit(f"body {name!r} not found in model.")
    return np.array(data.xpos[bid][:2], dtype=float), bid


def run(xml_path, object_body="metal_box", bin_body="bin",
        ee_site="suction_tip_site", headless=False, save_video=None):
    import mujoco

    p = Path(xml_path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"XML not found: {p}")
    model = mujoco.MjModel.from_xml_path(str(p))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    bridge = KR6IKBridge(model, data, ee_site=ee_site)
    obj_xy, obj_bid = _body_xy(model, data, object_body)
    bin_xy, _ = _body_xy(model, data, bin_body)
    obj_rest_z = float(data.xpos[obj_bid][2])

    viewer = None
    if not headless:
        try:
            import mujoco.viewer as mjv
            viewer = mjv.launch_passive(model, data)
        except Exception as e:
            print(f"[oracle] viewer unavailable ({e}); headless.")
    renderer = mujoco.Renderer(model, 224, 224) if save_video else None
    frames = []
    warned_suction = False

    for name, src, z, suction, dwell in PHASES:
        xy = obj_xy if src == "obj" else bin_xy
        target = np.array([xy[0], xy[1], z], dtype=float)
        ok = set_suction(model, data, KR6_VACUUM_ACT, suction)
        if not ok and not warned_suction:
            print(f"[oracle] WARNING: actuator {KR6_VACUUM_ACT!r} not found; "
                  f"scene not vacuum-ready, suction disabled.")
            warned_suction = True

        ticks = 0
        while ticks < MAX_TICKS_PER_PHASE:
            if viewer is not None and not viewer.is_running():
                print("[oracle] viewer closed.")
                return 1
            diag = bridge.servo_to_pose(target)
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            if renderer is not None:
                renderer.update_scene(data)
                frames.append(np.asarray(renderer.render(), dtype=np.uint8).copy())
            ticks += 1
            if diag["pos_err"] < POS_TOL:
                break
        # dwell (e.g. let suction lock / object settle)
        for _ in range(dwell * SUBSTEPS):
            mujoco.mj_step(model, data)
        print(f"[oracle] {name:8s} reached pos_err={diag['pos_err']:.4f} "
              f"obj_z={float(data.xpos[obj_bid][2]):.3f}")

    # success check (docs/pick_place_motion_spec.md §5)
    obj = np.array(data.xpos[obj_bid], dtype=float)
    inside = (abs(obj[0] - bin_xy[0]) < 0.09 and abs(obj[1] - bin_xy[1]) < 0.09)
    below_rim = obj[2] < 0.93
    success = bool(inside and below_rim)
    print(f"[oracle] obj_final={np.round(obj,3).tolist()} inside={inside} "
          f"below_rim={below_rim} -> SUCCESS={success}")

    if viewer is not None:
        viewer.close()
    if renderer is not None and frames:
        try:
            import imageio.v2 as imageio
            outp = Path(save_video).expanduser(); outp.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(outp), frames, fps=10)
            print(f"[oracle] saved video -> {outp}")
        except Exception as e:
            print(f"[oracle] video save failed ({e}).")
    return 0 if success else 2


def _args():
    ap = argparse.ArgumentParser(description="KR6 vacuum pick-and-place oracle.")
    ap.add_argument("--xml", required=True)
    ap.add_argument("--object", default="metal_box")
    ap.add_argument("--bin", default="bin")
    ap.add_argument("--ee-site", default="suction_tip_site")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--save-video", default=None)
    return ap.parse_args()


if __name__ == "__main__":
    a = _args()
    sys.exit(run(a.xml, a.object, a.bin, a.ee_site, a.headless, a.save_video))
