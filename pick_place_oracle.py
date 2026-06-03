"""
pick_place_oracle.py  (Milestone-1, gravity-comp + top-down build)
==================================================================
Scripted oracle: KR6 (ceiling-mounted) + vacuum cup picks an object and places it in the bin.

Validated fixes baked in (reproduced on the real scene, mujoco 3.1.6, CPU):
  * GRAVITY COMP: the ceiling-mounted KR6 sags ~0.42 m under its own weight with the
    stock position actuators, so the EE can never reach the waypoints. We cancel the
    bias force (gravity + Coriolis) on the arm DOFs each step -> sag ~0, all waypoints
    reachable. (See _step.)
  * TOP-DOWN APPROACH: a position-only servo overshoots downward and sweeps the cup
    sideways through the object, knocking it off the table. We approach from a SAFE_HIGH
    pose directly above the object, then descend vertically. Box stays put.
  * HOLD-AT-SEAT: descent stops when the compliant cup seats; ENGAGE then HOLDS that pose
    instead of driving deeper (which over-compressed the spring and launched the box).
  * Requires: suction slide joint with limited="true" (scene fix) and the bridge's
    mju_mat2Quat fix for the orientation path.

OPEN ITEM (blocks GATE 1): pointing the cup straight DOWN. With orientation commanded the
seat is gentle, but the arm struggles to reach cup-vertical at the current object pose
(times out ~55 deg tilt), so the cup can't seal. Resolve by moving the object into the
arm's cup-down workspace OR designing an angled-approach seal. Until then the oracle runs
cleanly but won't reliably grasp.

Usage:
    python pick_place_oracle.py --xml scene/world.xml                 # single, watch live
    python pick_place_oracle.py --xml scene/world.xml --trials 20 --jitter 0.05 --headless
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from kr6_ik_bridge import KR6IKBridge, set_suction, KR6_VACUUM_ACT, KR6_ARM_JOINTS


# Top-down waypoints. src: "obj" xy from object, "bin" xy from bin. z is world height.
# (name, src, z, suction, dwell, cart_step, gate)
#   gate "reach" -> until pos_err<tol ; "seat" -> until cup compliance retracts ; "hold" -> hold seat
PHASES = [
    ("SAFE_HIGH", "obj", 1.25, False, 0, 0.020, "reach"),
    ("HOVER",     "obj", 1.10, False, 0, 0.015, "reach"),
    ("SEAT",      "obj", 0.95, False, 0, 0.004, "seat"),
    ("ENGAGE",    "obj", None, True,  8, 0.002, "hold"),   # z=None -> hold seated pose
    ("LIFT",      "obj", 1.20, True,  0, 0.006, "reach"),
    ("TRANSIT",   "bin", 1.20, True,  0, 0.006, "reach"),
    ("LOWER",     "bin", 1.00, True,  0, 0.006, "reach"),
    ("RELEASE",   "bin", 1.00, False, 5, 0.006, "reach"),
    ("RETREAT",   "bin", 1.20, False, 0, 0.015, "reach"),
]

POS_TOL = 0.008
ROT_TOL = 0.12
MAX_TICKS_PER_PHASE = 1500
SUBSTEPS = 5
SEAT_COMPLIANCE_JOINT = "cup_compliance"
SEAT_RETRACT_THRESH = 0.003
GRASP_RISE_THRESH = 0.05
TOP_DOWN = True   # command cup contact axis (site +Z) toward world -Z


def _id(model, t, n):
    import mujoco
    return mujoco.mj_name2id(model, t, n)


def _arm_dofs(model):
    return [int(model.jnt_dofadr[_id(model, __import__("mujoco").mjtObj.mjOBJ_JOINT, j)])
            for j in KR6_ARM_JOINTS]


def _step(model, data, dof, n):
    """Step physics with gravity+Coriolis compensation on the arm DOFs."""
    import mujoco
    for _ in range(n):
        mujoco.mj_step1(model, data)
        data.qfrc_applied[dof] = data.qfrc_bias[dof]
        mujoco.mj_step2(model, data)


def _cup_down_mat():
    """Rotation whose 3rd column (site +Z, the contact axis) points world -Z."""
    zt = np.array([0.0, 0.0, -1.0])
    xt = np.array([1.0, 0.0, 0.0]); xt = xt - xt.dot(zt) * zt; xt /= np.linalg.norm(xt)
    return np.column_stack([xt, np.cross(zt, xt), zt])


def _body_xy(model, data, name):
    import mujoco
    bid = _id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise SystemExit(f"body {name!r} not found.")
    return np.array(data.xpos[bid][:2], dtype=float), bid


def _obj_free_qadr(model, bid):
    jadr = int(model.body_jntadr[bid])
    return -1 if jadr < 0 else int(model.jnt_qposadr[jadr])


def _run_episode(model, data, bridge, obj_body, bin_body, viewer=None,
                 renderer=None, frames=None, verbose=True):
    import mujoco
    obj_xy, obj_bid = _body_xy(model, data, obj_body)
    bin_xy, _ = _body_xy(model, data, bin_body)
    rest_z = float(data.xpos[obj_bid][2])
    dof = _arm_dofs(model)
    comp_jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, SEAT_COMPLIANCE_JOINT)
    comp_qadr = int(model.jnt_qposadr[comp_jid]) if comp_jid >= 0 else -1
    sid = bridge._site_id
    Rt = _cup_down_mat() if TOP_DOWN else None

    seat_pos = None
    for name, src, z, suction, dwell, cart_step, gate in PHASES:
        xy = obj_xy if src == "obj" else bin_xy
        if gate == "hold" and seat_pos is not None:
            target = seat_pos
        else:
            target = np.array([xy[0], xy[1], z if z is not None else rest_z], dtype=float)
        set_suction(model, data, KR6_VACUUM_ACT, suction)

        diag = {"pos_err": 9.9, "rot_err": 9.9}
        for ticks in range(MAX_TICKS_PER_PHASE):
            if viewer is not None and not viewer.is_running():
                return {"success": False, "reason": "viewer_closed"}
            diag = bridge.servo_to_pose(target, target_mat=Rt, max_cart_step=cart_step)
            _step(model, data, dof, SUBSTEPS)
            if viewer is not None:
                viewer.sync()
            if renderer is not None:
                renderer.update_scene(data)
                frames.append(np.asarray(renderer.render(), dtype=np.uint8).copy())
            if gate == "seat" and comp_qadr >= 0 and abs(float(data.qpos[comp_qadr])) > SEAT_RETRACT_THRESH:
                seat_pos = np.array(data.site_xpos[sid], dtype=float); break
            if gate in ("reach", "hold") and diag["pos_err"] < POS_TOL and \
               (Rt is None or diag["rot_err"] < ROT_TOL):
                break
        _step(model, data, dof, dwell * SUBSTEPS)

        if name == "LIFT":
            lifted_z = float(data.xpos[obj_bid][2])
            if lifted_z < rest_z + GRASP_RISE_THRESH:
                if verbose:
                    print(f"[oracle] GRASP FAILED at LIFT: obj_z={lifted_z:.3f} (rest={rest_z:.3f})")
                return {"success": False, "reason": "grasp_failed", "obj_z_at_lift": lifted_z}
        if verbose:
            print(f"[oracle] {name:9s} pos_err={diag['pos_err']:.4f} rot_err={diag['rot_err']:.3f} "
                  f"obj_z={float(data.xpos[obj_bid][2]):.3f}")

    obj = np.array(data.xpos[obj_bid], dtype=float)
    inside = abs(obj[0] - bin_xy[0]) < 0.09 and abs(obj[1] - bin_xy[1]) < 0.09
    below_rim = obj[2] < 0.93
    success = bool(inside and below_rim)
    return {"success": success, "reason": "ok" if success else "not_in_bin",
            "obj_final": [round(v, 3) for v in obj.tolist()], "inside": inside, "below_rim": below_rim}


def run(xml_path, object_body="metal_box", bin_body="bin", ee_site="suction_tip_site",
        headless=False, save_video=None, trials=1, jitter=0.0, seed=1234, log_json=None):
    import mujoco
    p = Path(xml_path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"XML not found: {p}")
    model = mujoco.MjModel.from_xml_path(str(p))
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    if _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, KR6_VACUUM_ACT) < 0:
        print(f"[oracle] WARNING: {KR6_VACUUM_ACT!r} missing -> run build_vacuum_scene.py first.")
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
            data.qpos[obj_qadr:obj_qadr + 2] = base_xy + rng.uniform(-jitter, jitter, size=2)
        mujoco.mj_forward(model, data)
        frames = [] if renderer else None
        r = _run_episode(model, data, bridge, object_body, bin_body, viewer, renderer, frames,
                         verbose=(trials <= 1))
        r["trial"] = t; results.append(r)
        if trials > 1:
            print(f"[oracle] trial {t+1:>3}/{trials}: {'PASS' if r['success'] else 'FAIL'} ({r['reason']})")

    n_ok = sum(1 for r in results if r["success"]); rate = n_ok / max(len(results), 1)
    print(f"\n[oracle] SUCCESS RATE: {n_ok}/{len(results)} = {rate*100:.1f}%  (GATE 1 target >= 90%)")
    if renderer is not None and frames:
        try:
            import imageio.v2 as imageio
            outp = Path(save_video).expanduser(); outp.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(outp), frames, fps=10); print(f"[oracle] video -> {outp}")
        except Exception as e:
            print(f"[oracle] video save failed ({e}).")
    if log_json:
        outp = Path(log_json).expanduser(); outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps({"success_rate": rate, "n_ok": n_ok, "trials": trials,
                                    "results": results}, indent=2)); print(f"[oracle] log -> {outp}")
    if viewer is not None:
        viewer.close()
    return 0 if rate >= 0.90 else 2


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--object", default="metal_box")
    ap.add_argument("--bin", default="bin")
    ap.add_argument("--ee-site", default="suction_tip_site")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--save-video", default=None)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log-json", default=None)
    return ap.parse_args()


if __name__ == "__main__":
    a = _args()
    sys.exit(run(a.xml, a.object, a.bin, a.ee_site, a.headless, a.save_video,
                 a.trials, a.jitter, a.seed, a.log_json))
