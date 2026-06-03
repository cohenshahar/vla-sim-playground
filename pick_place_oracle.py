"""
pick_place_oracle.py  (Milestone-1 upgrade)
===========================================
Scripted oracle for KR6 + vacuum: pick an object off the table, place it in the bin.
Drives the arm with KR6IKBridge.servo_to_pose and toggles vacuum via set_suction.

UPGRADES over the original (added during lunch-break prep, 2026-06-03):
  1. SLOW holding phases. CPU testing in mujoco 3.1.6 showed the `adhesion` grab
     DROPS the box when the cup moves faster than ~0.002 m / physics-step, regardless
     of margin. The original single max_cart_step=0.02 over 5 substeps is in the DROP
     zone. Each phase now has its own `cart_step`; LIFT/TRANSIT/LOWER use a slow value.
  2. SEAT-GATED ENGAGE. Descend stops when the compliant cup actually seats
     (cup_compliance joint retracts past a threshold), not at a hard-coded z.
     Falls back to pos_err if the compliance joint isn't present.
  3. GRASP-CONFIRM after LIFT (object rose with the cup) -> per-trial pass/fail reason.
  4. MULTI-TRIAL runner with x/y jitter + success-rate aggregation + JSON log
     (Steps 19-20). `--trials 1` reproduces the old single-run behaviour.

Validated numbers (from suction_probe, CPU, mujoco 3.1.6): gain=40, stiffness=800,
damping=5, margin=0.004-0.01; HOLD requires lift speed <= ~0.002 m/physics-step.

STATUS: logic-checked; UNTESTED against the full KR6 scene (no scene/meshes off-box).
Run on the strong PC; tune the PHASES heights + thresholds on first run.

Usage:
    # single run, watch live
    python pick_place_oracle.py --xml scene/world.xml
    # 20 randomized trials, headless, log success rate (Step 19)
    python pick_place_oracle.py --xml scene/world.xml --trials 20 --jitter 0.05 --headless
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from kr6_ik_bridge import KR6IKBridge, set_suction, KR6_VACUUM_ACT


# (name, src, z, suction, dwell_ticks, cart_step)
#   src: "obj" -> xy from object, "bin" -> xy from bin
#   cart_step: per-phase servo speed. Holding phases are SLOW (adhesion stays engaged).
PHASES = [
    ("HOVER",   "obj", 1.05, False, 0, 0.020),
    ("DESCEND", "obj", 0.94, False, 0, 0.010),   # seat-gated (see SEAT_* below)
    ("ENGAGE",  "obj", 0.94, True,  8, 0.006),
    ("LIFT",    "obj", 1.15, True,  0, 0.006),   # SLOW: holding the box
    ("TRANSIT", "bin", 1.15, True,  0, 0.006),   # SLOW
    ("LOWER",   "bin", 1.00, True,  0, 0.006),   # SLOW
    ("RELEASE", "bin", 1.00, False, 5, 0.010),
    ("RETREAT", "bin", 1.15, False, 0, 0.020),
]

POS_TOL = 0.006              # m, "waypoint reached"
MAX_TICKS_PER_PHASE = 600
SUBSTEPS = 5

# Seat detection (docs/pick_place_motion_spec.md §4)
SEAT_COMPLIANCE_JOINT = "cup_compliance"
SEAT_RETRACT_THRESH = 0.002  # m of compliance retraction => seated
GRASP_RISE_THRESH = 0.03     # m the object must rise during LIFT to count as grasped


def _id(model, objtype, name):
    import mujoco
    return mujoco.mj_name2id(model, objtype, name)


def _body_xy(model, data, name):
    import mujoco
    bid = _id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise SystemExit(f"body {name!r} not found in model.")
    return np.array(data.xpos[bid][:2], dtype=float), bid


def _obj_free_qadr(model, obj_bid):
    """qpos address of the object's free joint (for jitter / reset). -1 if none."""
    jadr = int(model.body_jntadr[obj_bid])
    if jadr < 0:
        return -1
    return int(model.jnt_qposadr[jadr])


def _run_episode(model, data, bridge, obj_body, bin_body, viewer=None,
                 renderer=None, frames=None, verbose=True):
    import mujoco

    obj_xy, obj_bid = _body_xy(model, data, obj_body)
    bin_xy, _ = _body_xy(model, data, bin_body)
    rest_z = float(data.xpos[obj_bid][2])

    comp_jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, SEAT_COMPLIANCE_JOINT)
    comp_qadr = int(model.jnt_qposadr[comp_jid]) if comp_jid >= 0 else -1

    lifted_z = rest_z
    for name, src, z, suction, dwell, cart_step in PHASES:
        xy = obj_xy if src == "obj" else bin_xy
        target = np.array([xy[0], xy[1], z], dtype=float)
        set_suction(model, data, KR6_VACUUM_ACT, suction)

        ticks = 0
        diag = {"pos_err": 9.9}
        while ticks < MAX_TICKS_PER_PHASE:
            if viewer is not None and not viewer.is_running():
                return {"success": False, "reason": "viewer_closed"}
            diag = bridge.servo_to_pose(target, max_cart_step=cart_step)
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            if renderer is not None:
                renderer.update_scene(data)
                frames.append(np.asarray(renderer.render(), dtype=np.uint8).copy())
            ticks += 1

            reached = diag["pos_err"] < POS_TOL
            seated = False
            if name == "DESCEND" and comp_qadr >= 0:
                seated = abs(float(data.qpos[comp_qadr])) > SEAT_RETRACT_THRESH
            if reached or seated:
                break

        for _ in range(dwell * SUBSTEPS):
            mujoco.mj_step(model, data)

        if name == "LIFT":
            lifted_z = float(data.xpos[obj_bid][2])
            if lifted_z < rest_z + GRASP_RISE_THRESH:
                if verbose:
                    print(f"[oracle] GRASP FAILED at LIFT: obj_z={lifted_z:.3f} "
                          f"(rest={rest_z:.3f}); aborting episode.")
                return {"success": False, "reason": "grasp_failed",
                        "obj_z_at_lift": lifted_z}

        if verbose:
            print(f"[oracle] {name:8s} pos_err={diag['pos_err']:.4f} "
                  f"obj_z={float(data.xpos[obj_bid][2]):.3f}")

    # success check (docs/pick_place_motion_spec.md §5)
    obj = np.array(data.xpos[obj_bid], dtype=float)
    vel = float(np.linalg.norm(data.cvel[obj_bid][3:6])) if data.cvel is not None else 0.0
    inside = (abs(obj[0] - bin_xy[0]) < 0.09 and abs(obj[1] - bin_xy[1]) < 0.09)
    below_rim = obj[2] < 0.93
    at_rest = vel < 0.05
    success = bool(inside and below_rim and at_rest)
    reason = "ok" if success else (
        "not_inside" if not inside else "above_rim" if not below_rim else "moving")
    return {"success": success, "reason": reason,
            "obj_final": [round(v, 3) for v in obj.tolist()],
            "inside": inside, "below_rim": below_rim, "speed": round(vel, 3)}


def run(xml_path, object_body="metal_box", bin_body="bin",
        ee_site="suction_tip_site", headless=False, save_video=None,
        trials=1, jitter=0.0, seed=1234, log_json=None):
    import mujoco

    p = Path(xml_path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"XML not found: {p}")
    model = mujoco.MjModel.from_xml_path(str(p))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # sanity: vacuum-ready scene?
    if _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, KR6_VACUUM_ACT) < 0:
        print(f"[oracle] WARNING: actuator {KR6_VACUUM_ACT!r} not found -> scene is "
              f"NOT vacuum-ready. Run build_vacuum_scene.py first.")
    bridge = KR6IKBridge(model, data, ee_site=ee_site)
    _, obj_bid = _body_xy(model, data, object_body)
    obj_qadr = _obj_free_qadr(model, obj_bid)
    base_xy = np.array(model.qpos0[obj_qadr:obj_qadr + 2]) if obj_qadr >= 0 else None

    rng = np.random.default_rng(seed)

    viewer = None
    if not headless and trials <= 1:
        try:
            import mujoco.viewer as mjv
            viewer = mjv.launch_passive(model, data)
        except Exception as e:
            print(f"[oracle] viewer unavailable ({e}); headless.")
    renderer = mujoco.Renderer(model, 224, 224) if save_video else None

    results = []
    for t in range(trials):
        mujoco.mj_resetData(model, data)
        if jitter > 0 and obj_qadr >= 0 and base_xy is not None:
            dxy = rng.uniform(-jitter, jitter, size=2)
            data.qpos[obj_qadr:obj_qadr + 2] = base_xy + dxy
        mujoco.mj_forward(model, data)
        frames = [] if renderer else None
        r = _run_episode(model, data, bridge, object_body, bin_body,
                         viewer=viewer, renderer=renderer, frames=frames,
                         verbose=(trials <= 1))
        r["trial"] = t
        results.append(r)
        if trials > 1:
            print(f"[oracle] trial {t+1:>3}/{trials}: "
                  f"{'PASS' if r['success'] else 'FAIL':4s} ({r['reason']})")

    n_ok = sum(1 for r in results if r["success"])
    rate = n_ok / len(results) if results else 0.0
    print(f"\n[oracle] SUCCESS RATE: {n_ok}/{len(results)} = {rate*100:.1f}%  "
          f"(GATE 1 target >= 90%)")

    if renderer is not None and frames:
        try:
            import imageio.v2 as imageio
            outp = Path(save_video).expanduser(); outp.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(outp), frames, fps=10)
            print(f"[oracle] saved video -> {outp}")
        except Exception as e:
            print(f"[oracle] video save failed ({e}).")
    if log_json:
        outp = Path(log_json).expanduser(); outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(
            {"xml": str(p), "trials": trials, "jitter": jitter, "seed": seed,
             "success_rate": rate, "n_ok": n_ok, "results": results}, indent=2))
        print(f"[oracle] wrote log -> {outp}")
    if viewer is not None:
        viewer.close()
    return 0 if rate >= 0.90 else 2


def _args():
    ap = argparse.ArgumentParser(description="KR6 vacuum pick-and-place oracle.")
    ap.add_argument("--xml", required=True)
    ap.add_argument("--object", default="metal_box")
    ap.add_argument("--bin", default="bin")
    ap.add_argument("--ee-site", default="suction_tip_site")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--save-video", default=None)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--jitter", type=float, default=0.0, help="object x/y uniform jitter (m)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log-json", default=None)
    return ap.parse_args()


if __name__ == "__main__":
    a = _args()
    sys.exit(run(a.xml, a.object, a.bin, a.ee_site, a.headless, a.save_video,
                 a.trials, a.jitter, a.seed, a.log_json))
