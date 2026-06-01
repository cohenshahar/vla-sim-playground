# LIBERO run sheet — real pick-and-place with OpenVLA

Source of truth: the OpenVLA README (fetched 2026-06-01), section *"LIBERO Simulation
Benchmark Evaluations"*. Commands below are copied from there — not invented.

## What this gives you
A Franka Panda (parallel-jaw gripper) in the LIBERO sim performing language-conditioned
pick-and-place, driven by OpenVLA. Reported success ~84.7% on LIBERO-Spatial.

**Key fact:** this uses a checkpoint OpenVLA **already fine-tuned on LIBERO and
published** (`openvla/openvla-7b-finetuned-libero-spatial`). The base `openvla-7b`
does NOT do LIBERO well zero-shot. We do no fine-tuning ourselves — just download
their checkpoint and run. This is unrelated to the KR6 / vacuum arm.

## Prerequisite — check VRAM FIRST
```bash
nvidia-smi
```
The eval loads a 7B model in **bf16 ≈ 16 GB**. The script as shipped does not expose
4-bit. So:
- VRAM ≥ ~16 GB (24 GB ideal) → runs clean.
- VRAM < 16 GB → likely OOM; needs a 4-bit patch to `run_libero_eval.py` (extra work,
  not a clean today-win). Decide here before installing.

## Install (their pinned versions — use a dedicated conda env)
The OpenVLA repo pins Python 3.10, PyTorch 2.2.0, transformers 4.40.1, flash-attn 2.5.5.
Run LIBERO from the openvla repo's own env, separate from our phase5 venv.

```bash
# 1) openvla env + repo
conda create -n openvla python=3.10 -y
conda activate openvla
# install torch 2.2.0 matched to your CUDA (see pytorch.org); e.g. cu121 wheel:
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

# 2) flash-attn (can be slow/finicky; documented risk)
pip install packaging ninja
ninja --version; echo $?            # expect 0
pip install "flash-attn==2.5.5" --no-build-isolation

# 3) LIBERO itself
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e . && cd ..
pip install -r experiments/robot/libero/libero_requirements.txt
```

## Run — LIBERO-Spatial, small trial count for a quick visible result
The README default is 500 trials (10 tasks x 50). For a *today* demo, cut it down with
`--num_trials_per_task` so you see success fast.

```bash
cd openvla
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 3
```
- `--center_crop True` is REQUIRED (they fine-tuned with random crops). Omitting it tanks success.
- The checkpoint auto-downloads on first run (another ~14 GB — fire this early).
- Logs locally; add `--use_wandb True --wandb_project <p> --wandb_entity <e>` to log online.

## Known gotchas (from the README troubleshooting)
- `tensorflow-datasets` error constructing a dataset → `pip install tensorflow-datasets==4.9.3`.
- `dlimp` / `traj_map` AttributeError → `pip install --no-deps --force-reinstall git+https://github.com/moojink/dlimp_openvla`.
- flash-attn build failure → `pip cache remove flash_attn` then retry; if hopeless, this is the
  fallback-to-FetchReach trigger from the session plan (don't burn the session here).
- VRAM OOM on model load → see the prerequisite; needs a 4-bit patch.

## Where this sits in the day
Session plan Hour 2. The reliable Hour 1 win (`run_openvla_mujoco.py --env fetch_reach`,
our phase5 harness, base openvla-7b, supports 4-bit) does not depend on any of this — so
even if LIBERO install bites, Hour 1 still gives you OpenVLA driving a sim arm.

## Other suites (same pattern, swap two args)
`libero_object` → `--pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object --task_suite_name libero_object`
`libero_goal`   → `...-finetuned-libero-goal  --task_suite_name libero_goal`
`libero_10`     → `...-finetuned-libero-10    --task_suite_name libero_10`

---
*vla-sim-playground / docs | 2026-06-01 | commands quoted from openvla/openvla README*
