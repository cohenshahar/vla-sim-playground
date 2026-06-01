# UBUNTU_MIGRATION.md — moving phase5_openvla from Windows to Ubuntu

**Target machine**: Ubuntu 22.04 + NVIDIA GPU (CUDA 12.1 driver).

The whole point of the Windows phase is that **no code changes are needed to
migrate**. You copy the folder, install different pip packages, and run.
If anything else changes, it is a bug — either in the Windows setup or in
this checklist. Update this doc when it happens.

---

## 0. Pre-flight checks on the Ubuntu box

```bash
nvidia-smi                       # confirms driver and CUDA runtime
python3 --version                # 3.10 or 3.11 recommended
python3 -c "import torch" 2>&1   # if torch already installed, note the CUDA version
```

If `nvidia-smi` says `CUDA Version: 12.x`, you can use the `cu121` wheel.
If it says `11.x`, switch `--index-url` to `https://download.pytorch.org/whl/cu118`
and use a torch version that matches (2.3.x supports both).

## 1. Copy the folder

The whole `phase5_openvla/` is self-contained. Two reasonable transfers:

**scp (if the Ubuntu box is reachable):**
```bash
scp -r "VLAResearch/phase5_openvla" user@ubuntu:~/VLAResearch/
```

**zip + move:**
```powershell
# Windows
Compress-Archive -Path "VLAResearch\phase5_openvla" -DestinationPath phase5_openvla.zip
```
```bash
# Ubuntu
unzip phase5_openvla.zip -d ~/VLAResearch/
```

Do **not** copy `phase5_openvla/.hf_cache/` across OSes. It is large
(>10 GB) and HF will happily re-download on Ubuntu. If you must copy it,
use rsync so partial files are not corrupted.

## 2. Create a fresh venv

```bash
cd ~/VLAResearch/phase5_openvla
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 3. Install GPU dependencies

```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `2.3.1+cu121 True <GPU name>`.

## 4. Re-authenticate with HuggingFace

The Windows token file is not portable. Log in again:

```bash
huggingface-cli login
python setup_hf.py
```

Should print `READY_FOR_DOWNLOAD=1`.

## 5. Run the smoke tests

```bash
python check_env.py              # expect STRATEGY_TAG=gpu-float16 (or -offload / -4bit)
python test_bridge_smoke.py      # instant
python test_infer_smoke.py       # ~20 s warmup + ~1.5 s/action
```

## 6. Run the full loop

```bash
python run_openvla_mujoco.py --env fetch_reach --max-steps 200
```

On GPU you can drop `--infer-every`; inference is fast enough to re-infer
every tick. The viewer should render smoothly.

---

## What changes automatically (no code edits)

| Thing | Windows | Ubuntu |
|-------|---------|--------|
| `check_env.py` strategy | `cpu-bfloat16` | `gpu-float16` (or variants) |
| `openvla_infer.py` dtype | `torch.bfloat16` | `torch.float16` |
| `openvla_infer.py` device | `cpu` | `cuda:0` |
| HF cache path | `phase5_openvla/.hf_cache/` | same |
| MuJoCo viewer | `--headless` recommended over RDP | native GLFW works |

## What may need manual tweaks

1. **Driver mismatch**. If `nvidia-smi` reports a CUDA older than what
   `torch==2.3.1+cu121` wants, either update the driver or pick a `cu118`
   wheel + downgrade bitsandbytes accordingly. Do not mix cu118 torch with
   a bitsandbytes built against cu121; it will fail at import.
2. **Headless server**. If the Ubuntu box has no display, MuJoCo needs EGL.
   Set before running:
   ```bash
   export MUJOCO_GL=egl
   ```
3. **Gymnasium-Robotics version**. If `FetchReach-v3` is not registered
   after install, try `gymnasium-robotics==1.2.4` explicitly, and
   `gymnasium[mujoco]==0.29.1`.
4. **Bitsandbytes import error**. If `nvidia-smi` works but
   `python -c "import bitsandbytes"` fails, it usually means the package
   cannot find libcuda. Fix: `pip install bitsandbytes==0.43.1 --force-reinstall`.
5. **Big HF cache on the wrong disk**. Symlink or set env var:
   ```bash
   export HF_HOME=/mnt/big-drive/hf_cache
   ```

## Post-migration validation checklist

- [ ] `check_env.py` prints `STRATEGY_TAG=gpu-*` (anything starting with `gpu-`).
- [ ] `test_bridge_smoke.py` exits 0.
- [ ] `test_infer_smoke.py` exits 0 and reports a sub-5-second `predict_action` time.
- [ ] `run_openvla_mujoco.py --env fetch_reach --max-steps 50` runs without the
      viewer freezing between ticks.
- [ ] `WINDOWS_NOTES.md` appended with "Migrated to Ubuntu on `<date>`".

## Rollback

If Ubuntu is broken and you need to keep making progress: the Windows
pipeline is unchanged and still works. The two copies are independent.

---

*VLA Research | Shahar Cohen | BGU Mechatronics | 2026-04-18*
