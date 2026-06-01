# vla-sim-playground

OpenVLA-in-simulation learning lab. Clean home for running, understanding, and
extending OpenVLA on MuJoCo arms (Franka/LIBERO and the KUKA KR6).

> **Boundary (CLAUDE.md §7 #4):** this is **learning / practice infrastructure,
> not thesis content.** The Continuous Verifier thesis direction is untouched by
> anything here. The thesis KR6 sim (ROS2 bridge, on the Ubuntu machine) is
> separate and read-only — we copy its XML, we don't edit it.

**Owner:** Shahar Cohen · BGU Mechatronics · started 2026-06-01.
Migrated from `VLAResearch/phase5_openvla` + the KR6 scene in
`VLA_Tutorial/VLATraining/sim`.

---

## What's here

| File | Purpose |
|------|---------|
| `check_env.py` | Probe Python/Torch/CUDA/MuJoCo; recommend an inference strategy. |
| `setup_hf.py` | Verify HuggingFace token + OpenVLA gated-license acceptance. |
| `config.py` | Central config (model id, paths, seed, image size, action dim). |
| `openvla_infer.py` | `OpenVLAInference.predict(image, instruction) -> action[7]`. Auto-picks gpu-float16 / gpu-4bit / cpu. |
| `mujoco_bridge.py` | Generic action→ctrl for `fetch_reach` / `reacher` / `custom_xml` (naive). |
| `run_openvla_mujoco.py` | Generic loop. **Use for the FetchReach win.** |
| `kr6_ik_bridge.py` | **NEW.** DLS inverse kinematics: OpenVLA EE-deltas → KR6 joint targets + gripper intent. |
| `run_openvla_kr6.py` | **NEW.** KR6 loop using the IK bridge + `cam_wrist` + optional `em_weld` gripper. |
| `test_bridge_smoke.py` / `test_infer_smoke.py` | Smoke tests (bridge is instant; infer downloads ~14 GB first run). |
| `docs/SESSION_PLAN_2026-06-01.md` | The 4-hour session plan + the "why KR6 first tests felt broken" analysis. |
| `UBUNTU_MIGRATION.md` | Ubuntu/GPU setup checklist + gotchas. |

## Quickstart (Ubuntu + GPU)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt
huggingface-cli login && python setup_hf.py
python check_env.py            # expect STRATEGY_TAG=gpu-* (gpu-4bit if VRAM is tight)
python test_bridge_smoke.py    # SMOKE_OK=1, instant
python test_infer_smoke.py     # downloads ~14 GB on first run, then ~1.5 s/action
```

**The reliable visible win (reach):**
```bash
python run_openvla_mujoco.py --env fetch_reach --max-steps 100 --infer-every 2
```

**Real pick-and-place (Franka):** see `docs/SESSION_PLAN_2026-06-01.md` §5 hour 2 (LIBERO).

**KR6 with the IK bridge (experimental, untested):**
```bash
python run_openvla_kr6.py --xml ~/VLA_Tutorial/VLATraining/sim/scene/world.xml \
    --instruction "pick up the metal box" --max-steps 60 --infer-every 2 --position-only
```
Pretrained OpenVLA on KR6 is a *baseline* — coherent motion is the goal, not task
success. Real success needs fine-tuning. Requires the KR6 meshes to resolve
(see the `meshdir` note in `run_openvla_kr6.py`).

## Status

`phase5_openvla` harness = tested earlier on CPU/Windows, GPU-ready. The two KR6
files (`kr6_ik_bridge.py`, `run_openvla_kr6.py`) are **UNTESTED ON HARDWARE** —
drafts to validate on the strong PC. `transformers==4.40.1` is load-bearing.
