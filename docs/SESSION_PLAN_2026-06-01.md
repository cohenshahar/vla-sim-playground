# 4-Hour Research Session Plan — OpenVLA in MuJoCo

**Date**: 2026-06-01 | **Owner**: Shahar Cohen | **Track**: Learning / practice (`phase5_openvla` + `vla_mini_pipeline`) — **NOT thesis content** (CLAUDE.md §7 #4). Sim only.

**Today's machines & state (confirmed):**
- **0:00–2:00 — Cowork (this machine)**: planning + prep. No execution, no GPU.
- **2:00–5:00 — Strong PC (Ubuntu Linux, GPU verified)**: all execution. VRAM is limited → OpenVLA must run **4-bit quantized** (`bitsandbytes load_in_4bit`). HF token ready + license approved.
- **No active Ubuntu thesis machine / no ROS2 today** — and **nothing here needs it**. `phase5_openvla` and LIBERO are pure MuJoCo.

---

## 0. Supervisor reframe — the bar for today

**Achievable and near-guaranteed:** OpenVLA-7B running 4-bit on the GPU and **visibly driving a sim arm end-to-end**, plus a **real LIBERO/Franka pick-and-place** rollout (pretrained, succeeds out of the box).

**Not today (and why):** full pick-and-place on **KR6 + vacuum**. Three independent blockers, all documented below: (a) no IK bridge, (b) gripper is a weld not a ctrl, (c) vacuum/suction model doesn't exist. KR6 success needs an IK bridge **+** demos **+** LoRA fine-tune = `vla_mini_pipeline` Phases 3–6, weeks of work. Today, KR6 is a **plumbing smoke test only**.

If you see a Franka pick-and-place on LIBERO and OpenVLA driving an arm on the GPU, **today succeeded.**

---

## 1. Why the first KR6 tests felt broken (root cause)

This is the real answer to "there's a problem with how we work the first VLA tests." It's not tuning — the KR6 path is a **category mismatch**:

1. **No inverse kinematics.** KR6 has six `position` actuators taking **target joint angles in radians** (`act_a1..a6`). `mujoco_bridge.py::_custom_xml` feeds OpenVLA's 7-D **end-effector deltas** (`a[0..5]`, ~[-1,1]) directly as joint targets. "Move EE forward" → "set A1=0.5 rad…" → nonsense pose + jitter. Not reaching.
2. **Gripper has no channel.** `_custom_xml` drops `a[6]`. KR6 grip is an **EM weld equality constraint** toggled by `em_controller.py`, not `data.ctrl`. OpenVLA cannot grasp.
3. **Wrong camera.** Custom-XML render uses the **default free camera**, not your `cam_wrist` egocentric view (no `--camera` flag).

**What a real KR6↔OpenVLA bridge needs** (scope for a later session, Phase 3):
- A Cartesian→joint mapping: take OpenVLA's EE delta, integrate onto the current EE pose of `em_contact_site`, solve to joint targets via **Jacobian damped-least-squares IK** (`mujoco.mj_jac` at the EE site) or a mocap-target weld. Write the 6 targets to `data.ctrl[0:6]`.
- Wire `a[6]` (gripper) to `em_controller.em_activate()/em_release()` with the existing proximity check.
- Render from `cam_wrist` (add a `--camera` flag to `run_openvla_mujoco.py`; `renderer.update_scene(data, camera="cam_wrist")`).

Until that exists, pointing OpenVLA at `world.xml` only proves "the loop runs on a 6-actuator model without crashing."

---

## 1.5 ROS2 vs direct MuJoCo — two separate stacks (resolves "the arm needs ROS2")

There are two distinct stacks. Mixing them is what makes the first tests feel heavy.

| | **Learning stack (today)** | **Thesis stack (later)** |
|---|---|---|
| Path | OpenVLA → `data.ctrl` → `mj_step()`, in-process Python | OpenVLA node → ROS2 topic → KR6 bridge → MuJoCo |
| ROS2? | **No** | **Yes** (real robot is commanded via ROS2; Verifier monitors topics) |
| Needs Ubuntu ROS2 box? | No | Yes |
| Purpose | Understand OpenVLA | Runtime-monitoring architecture / real robot |

Key facts:
- **MuJoCo does not need ROS2 to move an arm.** ROS2 is a messaging layer *above* the simulator. The same `world.xml` can be driven either way.
- Today's learning work uses the **direct** path — simplest, needs neither ROS2 nor the Ubuntu thesis machine.
- The **IK problem in §1 exists in both stacks**: a ROS2 node and the direct code both must convert OpenVLA's EE-deltas into joint commands. So IK is a prerequisite regardless of ROS2.
- Motion needs ROS2 **only** when integrating OpenVLA into the thesis architecture / real robot — a later milestone, worth its own RDR (`engineering:architecture`) when we get there.

## 2. What already exists — reuse, don't rebuild

`VLAResearch/phase5_openvla/` is a tested `image → OpenVLA → MuJoCo ctrl → step` loop with `check_env.py`, `test_bridge_smoke.py`, `setup_hf.py`, `test_infer_smoke.py`, `run_openvla_mujoco.py` (`--env fetch_reach|reacher|custom_xml`), and `requirements-gpu.txt`. `transformers==4.40.1` is load-bearing — do not upgrade.

The 4-week arc (LIBERO→KR6→demos→LoRA→eval) is in `vla_mini_pipeline/PLAN.md`. **Today is a slice of Phase 1 only.**

---

## 3. Biggest clock risks (in order)

1. **~14 GB OpenVLA-7B checkpoint download.** The long pole. **Start it at minute 0** of the strong-PC block. If cached already, smoke is ~20 s.
2. **4-bit / bitsandbytes.** VRAM-limited → expect `check_env.py` to pick `gpu-4bit`. Inference is slower; keep `--max-steps` modest (50–100) and you can use `--infer-every 2`. Confirm `python -c "import bitsandbytes"` works.
3. **LIBERO install.** Hour 2's only real install surface. Hard rule: if not green in ~25 min, fall back to a second FetchReach clip. Don't fight flash-attn.

---

## 4. The 2 Cowork hours (now, with me)

Decisions are **locked** (GPU=VRAM-limited→4bit, HF ready, Ubuntu strong PC, hour 3 = buffer). Remaining Cowork time, pick any — I can do these here:
- **(a)** Draft the KR6 IK-bridge skeleton (`kr6_ik_bridge.py`) + a `--camera` patch, ready to `git pull` on the strong PC, so any spare hour-3 time is productive.
- **(b)** Pre-write the exact LIBERO command sequence from `openvla/experiments/robot/libero/` so hour 2 is copy-paste.
- **(c)** Dry-run this runbook line by line and tighten it.

---

## 5. The 3 strong-PC hours (Ubuntu execution runbook)

### Hour 1 (2:00–3:00) — Env + smoke + FetchReach → the solid win

**Terminal 1 — fire the long pole immediately:**
```bash
cd ~/VLAResearch/phase5_openvla        # adjust to real path
source .venv/bin/activate || (python3 -m venv .venv && source .venv/bin/activate)
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121   # skip if present
pip install -r requirements-gpu.txt                                          # skip if present
huggingface-cli login                  # token (ready)
python setup_hf.py                     # expect READY_FOR_DOWNLOAD=1
python test_infer_smoke.py             # STARTS the 14 GB download now
```
**Terminal 2 — triage in parallel:**
```bash
nvidia-smi
python check_env.py                    # expect STRATEGY_TAG=gpu-4bit (VRAM-limited)
python -c "import bitsandbytes; print('bnb ok')"
python test_bridge_smoke.py            # expect SMOKE_OK=1, instant
export MUJOCO_GL=egl                   # only if the box is headless / over SSH
```
**Gate:** `check_env` = `gpu-*`, bnb imports, bridge = `SMOKE_OK=1`. Else stop & triage.

**Milestone 1** — `test_infer_smoke.py` prints `SMOKE_OK=1` + a 7-vector in [-1,1]: OpenVLA loads & infers 4-bit on GPU.

**Milestone 2 — the win:**
```bash
python run_openvla_mujoco.py --env fetch_reach --max-steps 100 --infer-every 2
```
Save a rollout MP4 to `scratch/`. OpenVLA visibly driving a sim arm. **If you stop here, the session already succeeded.**

### Hour 2 (3:00–4:00) — LIBERO/Franka pick-and-place → "see the task"

Use **`docs/LIBERO_RUNSHEET.md`** (exact commands, fetched from the OpenVLA README). Key point: run the **provided LIBERO-fine-tuned checkpoint** `openvla/openvla-7b-finetuned-libero-spatial` (NOT base `openvla-7b` — base is poor on LIBERO) with `--center_crop True --num_trials_per_task 3`. Check VRAM first (`nvidia-smi`): the eval loads bf16 ≈ 16 GB; under that it may OOM and need a 4-bit patch.
→ **Milestone 3:** a Franka picking a block under a language command, success/fail printed. *This is the genuine pick-and-place moment.*
**Hard rule:** not green in ~25 min → stop, fall back to another FetchReach clip with a different instruction. Don't burn the hour on deps.

### Hour 3 (4:00–5:00) — Buffer (overflow + optional KR6 smoke)

Primary purpose: absorb install/debug overrun from hours 1–2. If everything's clean and time remains, **in this order**:
1. **KR6 plumbing smoke** (expectation: non-behavior, see §1):
   ```bash
   python run_openvla_mujoco.py --env custom_xml \
     --xml ~/VLA_Tutorial/VLATraining/sim/scene/world.xml \
     --max-steps 50 --infer-every 2
   ```
   Success = "loop runs on the 6-actuator KR6 model without crashing." Nothing more. Save the clip as the documented baseline.
2. If `kr6_ik_bridge.py` was prepped in the Cowork block → wire it in and take the first real Cartesian-controlled KR6 step. (Stretch.)

### Wrap (4:50–5:00)
Save MP4s to `scratch/`. Write `phase5_openvla/HANDOFF.md`: what ran, milestones, exact errors, what's cached now. Bring errors back to Cowork for triage.

---

## 6. Paste-in starter block for Claude Code (strong PC, first message)

```
TIME-BOXED ~3-hour session on the strong GPU PC (Ubuntu Linux). Sim only,
pure MuJoCo — NO ROS2, NO thesis Ubuntu machine needed. This is learning/
practice infra (VLAResearch/phase5_openvla), NOT thesis content. The thesis
KR6 sim is READ-ONLY: load world.xml, never edit it.

Env is verified: nvidia-smi + torch.cuda.is_available() are good. VRAM is
limited, so OpenVLA must run 4-bit quantized (bitsandbytes). HF token + license
are ready. Do NOT redo driver/CUDA setup.

Goal: OpenVLA on GPU visibly driving a sim arm, then a real LIBERO/Franka
pick-and-place. KR6 will NOT do a real task today — it has no IK bridge, the
gripper is a weld not a ctrl, and there's no suction model. Don't promise KR6
behavior; treat KR6 as a plumbing smoke test only.

In order:
1. Read phase5_openvla/README.md and UBUNTU_MIGRATION.md.
2. FIRST start the ~14 GB checkpoint download (setup_hf.py then test_infer_smoke.py).
   In a 2nd terminal run check_env.py (expect gpu-4bit), import bitsandbytes,
   and test_bridge_smoke.py (SMOKE_OK=1). If headless: export MUJOCO_GL=egl.
3. Gate on those passing before proceeding; if not, stop and triage with me.
4. run_openvla_mujoco.py --env fetch_reach --max-steps 100 --infer-every 2,
   save an MP4. That's the primary win.
5. Then LIBERO: install per openvla/experiments/robot/libero/README.md, run one
   libero_spatial task, save MP4s. HARD RULE: not green in ~25 min -> fall back
   to another FetchReach clip. Do not fight flash-attn/LIBERO deps.
6. Buffer time only: KR6 plumbing smoke on ~/VLA_Tutorial/VLATraining/sim/scene/
   world.xml (expect non-behavior). Save phase5_openvla/HANDOFF.md at the end.

Rules: pin transformers==4.40.1. No invention without context — if a file/output
isn't in front of you, say so. Never commit checkpoints/datasets/tokens. Ask
before sudo/apt. Start with step 1.
```

---

## 7. Gotchas (from UBUNTU_MIGRATION.md, tuned for 4-bit)

- `bitsandbytes` import error despite working `nvidia-smi` → `pip install bitsandbytes==0.43.1 --force-reinstall`.
- `transformers` must stay `==4.40.1` (Prismatic processor breaks above it).
- Headless / SSH → `export MUJOCO_GL=egl`.
- `FetchReach-v3` not registered → `gymnasium-robotics==1.2.4` + `gymnasium[mujoco]==0.29.1`.
- HF cache on small partition → `export HF_HOME=/mnt/<big-drive>/hf_cache` **before** the download.

---

## 8. What explicitly defers (not today)

- KR6 IK bridge + gripper wiring + `cam_wrist` render (§1) — the real unblock for KR6.
- Vacuum/suction end-effector model in MuJoCo (weld/equality) — doesn't exist; KR6 today is EM.
- Demos + LoRA fine-tune — the only path to KR6 pick-and-place *success*.
- Any thesis-sim or ROS2 change. The Continuous Verifier thesis track (Phase 10 `demo_reach.py`) is untouched.

---

*VLA Research / learning track | Shahar Cohen | BGU Mechatronics | 2026-06-01*
