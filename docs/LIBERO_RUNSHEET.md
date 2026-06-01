# LIBERO run sheet — real pick-and-place with OpenVLA (4-bit, <16 GB GPU)

Source of truth: OpenVLA README + `experiments/robot/libero/run_libero_eval.py` +
`experiments/robot/openvla_utils.py` (read from source 2026-06-01). Commands and flags
below are taken from those files — not invented.

## What this gives you
A Franka Panda (parallel-jaw gripper) in LIBERO doing language-conditioned pick-and-place,
driven by OpenVLA. Reported ~84.7% success on LIBERO-Spatial. The eval **auto-saves a
replay video per episode** (`save_rollout_video`) — MP4s come for free.

**Key fact:** uses the checkpoint OpenVLA already **fine-tuned on LIBERO and published**
(`openvla/openvla-7b-finetuned-libero-spatial`). The base `openvla-7b` is poor on LIBERO.
We do NO fine-tuning ourselves. Unrelated to the KR6 / vacuum arm.

## VRAM: <16 GB confirmed → run in 4-bit
`get_vla()` passes `load_in_4bit=cfg.load_in_4bit` to the model load, and `GenerateConfig`
defines `load_in_4bit: bool = False`. So **`--load_in_4bit True` is officially supported** —
no patching. 4-bit 7B ≈ 6–8 GB. (Don't combine with `--load_in_8bit`; the script asserts
against using both.)

## Install (their pinned versions, dedicated conda env)
Python 3.10, PyTorch 2.2.0, transformers 4.40.1, flash-attn 2.5.5. Run from the openvla
repo's own env, separate from our phase5 venv.

```bash
# 1) env + repo
conda create -n openvla python=3.10 -y
conda activate openvla
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

# 2) REQUIRED for 4-bit
pip install bitsandbytes

# 3) flash-attn (mandatory unless you edit the code — see gotcha below)
pip install packaging ninja
ninja --version; echo $?            # expect 0
pip install "flash-attn==2.5.5" --no-build-isolation

# 4) LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e . && cd ..
pip install -r experiments/robot/libero/libero_requirements.txt
```

## Run — LIBERO-Spatial, 4-bit, quick trial count
```bash
cd openvla
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --load_in_4bit True \
  --num_trials_per_task 3
```
- Checkpoint auto-downloads on first run (~14 GB — fire early).
- `--center_crop True`: required (fine-tuned with random crops); it's also the default.
- Replay MP4s + a text log are written under `./experiments/logs/`.
- Add `--use_wandb True --wandb_project <p> --wandb_entity <e>` to log online (optional).

## Gotchas (confirmed from source / README)
- **flash-attn is hardcoded.** `openvla_utils.py::get_vla()` sets
  `attn_implementation="flash_attention_2"`. If flash-attn won't build on your box, edit that
  one line to `attn_implementation="sdpa"` (or `"eager"`) and skip the flash-attn install.
- `tensorflow-datasets` dataset-construct error → `pip install tensorflow-datasets==4.9.3`.
- `dlimp` / `traj_map` AttributeError → `pip install --no-deps --force-reinstall git+https://github.com/moojink/dlimp_openvla`.
- Still OOM in 4-bit? Drop other GPU users; 4-bit 7B should sit ~6–8 GB.
- Don't pass both `--load_in_4bit True` and `--load_in_8bit True` (assertion error).

## Where this sits in the day
Session plan Hour 2. Hour 1 (`run_openvla_mujoco.py --env fetch_reach`, our phase5 harness,
base openvla-7b, also 4-bit-capable) is independent — even if LIBERO install bites, Hour 1
still gives OpenVLA driving a sim arm. Hard rule: LIBERO not green in ~25 min → fall back to
a second FetchReach clip, don't burn the session on deps.

## Other suites (swap two args + the matching checkpoint)
`libero_object` → `--pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object --task_suite_name libero_object`
`libero_goal`   → `...-finetuned-libero-goal  --task_suite_name libero_goal`
`libero_10`     → `...-finetuned-libero-10    --task_suite_name libero_10`

---
*vla-sim-playground / docs | 2026-06-01 | flags verified against openvla source*
