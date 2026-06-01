"""
run_openvla_kr6.py
==================
OpenVLA -> KR6 full loop, the *correct* way:

    frame   = render(cam_wrist)                  # egocentric VLA view
    action  = OpenVLA.predict(frame, instruction)# 7-D EE action
    diag    = KR6IKBridge.action_to_ctrl(action) # DLS IK -> joint targets
    (opt)   toggle em_weld from gripper intent
    mj_step x N
    viewer.sync()

This fixes the three breaks of the generic custom_xml path:
  1. IK     : EE deltas -> joint-angle targets (kr6_ik_bridge.py)
  2. gripper: a[6] -> em_weld equality constraint (optional, --use-gripper)
  3. camera : renders from cam_wrist, not the default free camera

STATUS: UNTESTED ON HARDWARE. Validate on the strong PC. Expect to tune
pos_step / rot_step / damping on the first run. Even with correct IK, a
*pretrained* OpenVLA on KR6 is a baseline — real task success needs
fine-tuning (see docs/SESSION_PLAN). The win here is coherent, goal-directed
motion instead of jitter.

PREREQUISITE: the KR6 meshes must resolve. arm.xml hardcodes
  <compiler meshdir="/home/shahar/Desktop/phase4/VLATraining/sim/assets/urdf/meshes/">
If that path doesn't exist on the strong PC, either place the assets there or
edit the meshdir before running.

Examples:
    python run_openvla_kr6.py --xml ~/VLA_Tutorial/VLATraining/sim/scene/world.xml \
        --instruction "pick up the metal box" --max-steps 60 --infer-every 2
    python run_openvla_kr6.py --xml .../world.xml --headless --save-video scratch/kr6.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from config import CFG
from kr6_ik_bridge import KR6IKBridge, set_weld_active, KR6_EM_WELD


def run(
    xml_path: str,
    instruction: Optional[str] = None,
    max_steps: int = 60,
    infer_every: int = 2,
    substeps: int = 5,
    camera: str = "cam_wrist",
    headless: bool = False,
    use_gripper: bool = False,
    position_only: bool = False,
    save_video: Optional[str] = None,
) -> int:
    import mujoco

    from openvla_infer import OpenVLAInference

    CFG.ensure_dirs()
    instruction = instruction or CFG.instruction

    p = Path(xml_path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"XML not found: {p}")

    print(f"[kr6] loading scene: {p}")
    model = mujoco.MjModel.from_xml_path(str(p))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"[kr6] nu={model.nu} nq={model.nq} nv={model.nv}")

    # renderer that honours the named camera (the whole point vs custom_xml)
    renderer = mujoco.Renderer(model, height=CFG.image_size, width=CFG.image_size)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id < 0:
        print(f"[kr6] WARNING: camera {camera!r} not found; using default free cam.")
        camera_arg = None
    else:
        camera_arg = camera

    infer = OpenVLAInference()
    bridge = KR6IKBridge(model, data, use_orientation=not position_only)

    viewer = None
    if not headless:
        try:
            import mujoco.viewer as mjv
            viewer = mjv.launch_passive(model, data)
            print("[kr6] passive viewer up; close window or Ctrl+C to stop.")
        except Exception as e:
            print(f"[kr6] viewer unavailable ({e}); headless.")

    frames: List[np.ndarray] = []
    last_action = np.zeros(7, dtype=np.float32)
    step = 0
    t0 = time.time()
    try:
        while step < max_steps:
            if viewer is not None and not viewer.is_running():
                print("[kr6] viewer closed; stopping.")
                break

            renderer.update_scene(data, camera=camera_arg) if camera_arg \
                else renderer.update_scene(data)
            frame = np.asarray(renderer.render(), dtype=np.uint8)
            if save_video:
                frames.append(frame.copy())

            if step % infer_every == 0:
                last_action = infer.predict(frame, instruction)

            diag = bridge.action_to_ctrl(last_action)

            if use_gripper:
                ok = set_weld_active(model, data, KR6_EM_WELD, diag["gripper_closed"])
                if not ok and step == 0:
                    print(f"[kr6] WARNING: weld {KR6_EM_WELD!r} not found; gripper disabled.")

            for _ in range(substeps):
                mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()

            step += 1
            print(f"[kr6] step={step}/{max_steps} "
                  f"ee={np.round(diag['ee_pos'], 3).tolist()} "
                  f"grip={'C' if diag['gripper_closed'] else 'o'} "
                  f"dq={np.round(diag['dq'], 3).tolist()}")

    except KeyboardInterrupt:
        print("[kr6] interrupted.")
    finally:
        dt = time.time() - t0
        print(f"[kr6] done: {step} steps in {dt:.1f}s "
              f"({dt / max(step, 1):.2f} s/step)")
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
        try:
            renderer.close()
        except Exception:
            pass

    if save_video and frames:
        _write_video(frames, save_video)
    return 0


def _write_video(frames: List[np.ndarray], path: str) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        imageio.mimsave(str(out), frames, fps=10)
        print(f"[kr6] saved video -> {out} ({len(frames)} frames)")
    except Exception as e:
        # imageio is optional; fall back to a frame dump so nothing is lost.
        np.save(str(out.with_suffix(".npy")), np.asarray(frames))
        print(f"[kr6] imageio unavailable ({e}); dumped frames -> "
              f"{out.with_suffix('.npy')}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OpenVLA -> KR6 (IK bridge) loop.")
    ap.add_argument("--xml", required=True, help="Path to the KR6 world.xml")
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--infer-every", type=int, default=2)
    ap.add_argument("--substeps", type=int, default=5)
    ap.add_argument("--camera", default="cam_wrist")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--use-gripper", action="store_true",
                    help="Toggle em_weld from a[6] (naive, no proximity check).")
    ap.add_argument("--position-only", action="store_true",
                    help="Ignore a[3:6]; position-only IK (more stable first run).")
    ap.add_argument("--save-video", default=None)
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(run(
        xml_path=args.xml,
        instruction=args.instruction,
        max_steps=args.max_steps,
        infer_every=args.infer_every,
        substeps=args.substeps,
        camera=args.camera,
        headless=args.headless,
        use_gripper=args.use_gripper,
        position_only=args.position_only,
        save_video=args.save_video,
    ))
