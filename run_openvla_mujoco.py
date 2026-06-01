"""
run_openvla_mujoco.py
---------------------
Phase 5 / Step 4 - the full loop.

Per tick:
    frame  = render_rgb(model, data, renderer)         # HxWx3 uint8
    action = infer.predict(frame, instruction)         # (7,) float32
    ctrl   = openvla_action_to_ctrl(action, model, data, env_tag)
    data.ctrl[:] = ctrl
    mujoco.mj_step(model, data)
    viewer.sync()

This script supports three scenes out of the box:

    env_tag="fetch_reach"
        Uses Gymnasium-Robotics FetchReach-v3 to get the MjModel/MjData.
        This is the recommended scene because the 7-DoF reach task is
        closest to OpenVLA's training distribution.

    env_tag="reacher"
        Uses Gymnasium Reacher-v5. Tiny 2-DoF planar arm. OpenVLA's action
        is only partially meaningful here; good for plumbing only.

    env_tag="custom_xml"
        Loads an MJCF file path from env var PHASE5_XML or --xml argument.

CLI:
    python run_openvla_mujoco.py
    python run_openvla_mujoco.py --env reacher --instruction "reach the target"
    python run_openvla_mujoco.py --env custom_xml --xml path/to/arm.xml --headless
    python run_openvla_mujoco.py --max-steps 30              # cap for CPU testing

Performance note:
    On CPU-only Windows, each OpenVLA call takes 30-120 s. The viewer will
    appear frozen between ticks; this is expected. Use --max-steps to cap
    the run, or --infer-every N to only re-infer every N physics steps.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import CFG


# ---------------------------------------------------------------------------
# Scene loaders
# ---------------------------------------------------------------------------

def _load_fetch_reach():
    """Return (model, data, close_fn). Uses Gymnasium-Robotics."""
    import gymnasium as gym
    try:
        import gymnasium_robotics  # noqa: F401  (registers the envs)
    except Exception as e:
        raise RuntimeError(
            "gymnasium_robotics not installed. "
            "Run: pip install gymnasium-robotics==1.2.4"
        ) from e

    env = gym.make("FetchReach-v3", render_mode="rgb_array")
    env.reset(seed=CFG.seed)
    # Gymnasium-Robotics wraps MuJoCo; reach into the underlying simulator.
    unwrapped = env.unwrapped
    model = unwrapped.model
    data = unwrapped.data

    def close():
        env.close()

    return model, data, close, env  # env also gives us env.render()


def _load_reacher():
    import gymnasium as gym
    env = gym.make("Reacher-v5", render_mode="rgb_array")
    env.reset(seed=CFG.seed)
    unwrapped = env.unwrapped
    return unwrapped.model, unwrapped.data, env.close, env


def _load_custom_xml(xml_path: str):
    import mujoco
    p = Path(xml_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"XML not found: {p}")
    model = mujoco.MjModel.from_xml_path(str(p))
    data = mujoco.MjData(model)
    return model, data, (lambda: None), None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_frame(renderer, data, model, env, camera_name: Optional[str] = None) -> np.ndarray:
    """Return HxWx3 uint8 RGB frame.

    Prefers the Gymnasium env.render() if available (handles camera setup
    for us), else falls back to a fresh mujoco.Renderer.
    """
    if env is not None and hasattr(env, "render"):
        frame = env.render()
        if frame is None:
            raise RuntimeError("env.render() returned None (wrong render_mode?)")
        return np.asarray(frame, dtype=np.uint8)

    # Fallback: manual renderer.
    import mujoco
    renderer.update_scene(data, camera=camera_name) if camera_name else renderer.update_scene(data)
    frame = renderer.render()
    return np.asarray(frame, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    env_tag: str = "fetch_reach",
    instruction: Optional[str] = None,
    max_steps: int = 50,
    infer_every: int = 1,
    headless: bool = False,
    xml_path: Optional[str] = None,
) -> int:
    import mujoco

    from openvla_infer import OpenVLAInference
    from mujoco_bridge import openvla_action_to_ctrl

    CFG.ensure_dirs()
    instruction = instruction or CFG.instruction

    # ---- scene ----
    env = None
    if env_tag == "fetch_reach":
        model, data, close_scene, env = _load_fetch_reach()
    elif env_tag == "reacher":
        model, data, close_scene, env = _load_reacher()
    elif env_tag == "custom_xml":
        if not xml_path:
            raise SystemExit("env_tag=custom_xml requires --xml path")
        model, data, close_scene, env = _load_custom_xml(xml_path)
    else:
        raise SystemExit(f"unknown env_tag: {env_tag}")

    print(f"[run] scene loaded: env_tag={env_tag} nu={model.nu} nq={model.nq}")

    # ---- renderer (fallback if env.render is unavailable) ----
    renderer = mujoco.Renderer(model, height=CFG.image_size, width=CFG.image_size) \
        if env is None else None

    # ---- openvla ----
    infer = OpenVLAInference()

    # ---- viewer ----
    viewer = None
    if not headless:
        try:
            import mujoco.viewer as mjv
            viewer = mjv.launch_passive(model, data)
            print("[run] launched passive viewer; close the window or Ctrl+C to stop.")
        except Exception as e:
            print(f"[run] viewer unavailable ({e}); running headless.")
            viewer = None

    # ---- main loop ----
    step = 0
    last_action = np.zeros(CFG.action_dim, dtype=np.float32)
    last_action[6] = 0.0  # gripper open

    t_start = time.time()
    try:
        while step < max_steps:
            if viewer is not None and not viewer.is_running():
                print("[run] viewer closed; stopping.")
                break

            # 1) render
            frame = _render_frame(renderer, data, model, env)

            # 2) infer (maybe skip depending on infer_every)
            if step % infer_every == 0:
                last_action = infer.predict(frame, instruction)

            # 3) map to ctrl
            ctrl = openvla_action_to_ctrl(last_action, model, data, env_tag=env_tag)

            # 4) physics step (a few substeps smooth out the coarse action)
            data.ctrl[:] = ctrl
            for _ in range(5):
                mujoco.mj_step(model, data)

            # 5) viewer sync
            if viewer is not None:
                viewer.sync()

            step += 1
            print(f"[run] step={step}/{max_steps} ctrl={np.round(ctrl, 3).tolist()}")

    except KeyboardInterrupt:
        print("[run] interrupted by user")
    finally:
        dt = time.time() - t_start
        print(f"[run] done: {step} steps in {dt:.1f}s "
              f"(mean {dt / max(step, 1):.1f} s/step)")
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
        try:
            close_scene()
        except Exception:
            pass
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenVLA -> MuJoCo full loop.")
    p.add_argument("--env", default=CFG.env_tag,
                   choices=["fetch_reach", "reacher", "custom_xml"])
    p.add_argument("--instruction", default=None)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--infer-every", type=int, default=1,
                   help="Call OpenVLA every N physics steps (CPU helper).")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--xml", default=None, help="Path to MJCF for --env custom_xml")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(run(
        env_tag=args.env,
        instruction=args.instruction,
        max_steps=args.max_steps,
        infer_every=args.infer_every,
        headless=args.headless,
        xml_path=args.xml,
    ))
