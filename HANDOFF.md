# HANDOFF — vla-sim-playground

**Overwrite the "Current state" section at the end of each session.**

---

## Current state

*(2026-06-01 — repo created on Cowork, no execution yet)*

- Repo scaffolded from `phase5_openvla` (tested harness) + new KR6 IK bridge.
- **Nothing run yet.** No GPU on the Cowork machine.
- New, **untested-on-hardware** files: `kr6_ik_bridge.py`, `run_openvla_kr6.py`.
- Strong PC: Ubuntu Linux, GPU verified, VRAM-limited (expect `gpu-4bit`),
  HF token + license ready. No ROS2 / no thesis-Ubuntu today (not needed).

## Next session (strong PC) — from docs/SESSION_PLAN_2026-06-01.md §5

1. `git pull` this repo.
2. Hour 1: venv + `requirements-gpu.txt`, `check_env.py`, smoke tests, then
   `run_openvla_mujoco.py --env fetch_reach` → first MP4 (the solid win).
3. Hour 2: LIBERO/Franka — one `libero_spatial` task → real pick-and-place MP4.
4. Hour 3 (buffer): `kr6_ik_bridge.py` CLI smoke
   (`python kr6_ik_bridge.py <world.xml>`) to validate name wiring + shapes,
   then `run_openvla_kr6.py --position-only` for the first coherent KR6 motion.

## Open validation items for the KR6 bridge

- [ ] Does `world.xml` load on the strong PC? (meshdir is hardcoded to a
      `/home/shahar/Desktop/phase4/...` path — fix if absent.)
- [ ] `kr6_ik_bridge.py` CLI smoke prints `SMOKE_OK=1` (names resolve, shapes ok).
- [ ] `data.eq_active` vs `model.eq_active0` — confirm which the installed MuJoCo
      build exposes for the `em_weld` toggle.
- [ ] Tune `pos_step` / `rot_step` / `damping` / `max_joint_step` on first motion.
- [ ] Confirm `cam_wrist` renders a sane egocentric view (not inside a mesh).

## Boundary reminder

Learning infra, not thesis. Thesis sim (ROS2/KR6 on Ubuntu) untouched.
