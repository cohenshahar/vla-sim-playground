"""
config.py
---------
Phase 5 / central config.

All knobs that multiple files share live here. No secrets. No per-OS logic
beyond path handling (Path objects are OS-agnostic).

The idea: every other script does `from config import CFG` and never
hard-codes a model id, a path, or a seed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ---- seeds ------------------------------------------------------------
# Deterministic flag group for torch + numpy + random.
# Matches CLAUDE.md §8.5: fixed seed on torch + numpy + random,
# torch.use_deterministic_algorithms(True), CUBLAS_WORKSPACE_CONFIG=:4096:8.
SEED = 1234


# ---- model -----------------------------------------------------------
MODEL_ID = "openvla/openvla-7b"
# OpenVLA's built-in processor class name (used for AutoProcessor).
# The repo registers a custom processor that transformers loads via
# trust_remote_code=True.
TRUST_REMOTE_CODE = True


# ---- paths -----------------------------------------------------------
# All paths are relative to this file so the folder is portable:
# zip phase5_openvla/, unzip on Ubuntu, everything still resolves.
PHASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PHASE_DIR.parent.parent  # VLA cowork/
HF_CACHE_DIR = PHASE_DIR / ".hf_cache"           # opt-in local cache
LOG_DIR = PHASE_DIR / "logs"
SCRATCH_DIR = PHASE_DIR / "scratch"              # screenshots, dumps


# ---- task / instruction ---------------------------------------------
# First-try instruction. Deliberately simple and in the distribution
# of BridgeData-v2 language annotations (short, imperative, English).
DEFAULT_INSTRUCTION = "move the arm upward"


# ---- MuJoCo environment ---------------------------------------------
# "fetch_reach"  - Gymnasium-Robotics FetchReach-v3 (7-DoF mobile manipulator).
# "reacher"      - Gymnasium Reacher-v5 (2-DoF planar).
# "custom_xml"   - path supplied via env var PHASE5_XML.
DEFAULT_ENV_TAG = "fetch_reach"


# ---- inference strategy ---------------------------------------------
# Possible values (matches pick_strategy in check_env.py):
#   gpu-float16, gpu-float16-offload, gpu-4bit,
#   cpu-bfloat16, cpu-disk-offload, unsupported
# 'auto' means "detect at load time from the presence of CUDA and RAM".
DEFAULT_STRATEGY = "auto"


@dataclass
class Config:
    seed: int = SEED
    model_id: str = MODEL_ID
    trust_remote_code: bool = TRUST_REMOTE_CODE

    phase_dir: Path = PHASE_DIR
    hf_cache_dir: Path = HF_CACHE_DIR
    log_dir: Path = LOG_DIR
    scratch_dir: Path = SCRATCH_DIR

    instruction: str = DEFAULT_INSTRUCTION
    env_tag: str = DEFAULT_ENV_TAG
    strategy: str = DEFAULT_STRATEGY

    # OpenVLA outputs 7D continuous actions normalized to roughly [-1, 1].
    # Semantics (BridgeData-v2 convention):
    #   a[0..2] : delta end-effector position (x, y, z)
    #   a[3..5] : delta end-effector orientation (roll, pitch, yaw), axis-angle.
    #   a[6]    : gripper (0 = open, 1 = closed) in BridgeData; some ckpts use -1/1.
    action_dim: int = 7
    action_clip: float = 1.0

    # Resolution of the image fed to OpenVLA (OpenVLA internally resizes to
    # 224x224 for DINOv2 and 384x384 for SigLIP; giving it a larger image is
    # fine, it just downsamples).
    image_size: int = 224

    # Runtime knobs
    extra_env: dict = field(default_factory=dict)

    def ensure_dirs(self) -> None:
        for p in (self.hf_cache_dir, self.log_dir, self.scratch_dir):
            p.mkdir(parents=True, exist_ok=True)


CFG = Config()


def apply_determinism(cfg: Config = CFG) -> None:
    """Set the flags required by CLAUDE.md §8.5."""
    import random
    import numpy as np
    import torch

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # These two are required by the project CLAUDE.md.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        # Some ops (e.g. certain attention kernels) have no deterministic
        # impl. warn_only=True downgrades these to warnings instead of errors.
        pass


if __name__ == "__main__":
    # Quick sanity print.
    print("=== phase5_openvla / config ===")
    print(f"model_id     : {CFG.model_id}")
    print(f"env_tag      : {CFG.env_tag}")
    print(f"instruction  : {CFG.instruction!r}")
    print(f"strategy     : {CFG.strategy}")
    print(f"phase_dir    : {CFG.phase_dir}")
    print(f"hf_cache_dir : {CFG.hf_cache_dir}")
    print(f"log_dir      : {CFG.log_dir}")
    CFG.ensure_dirs()
    print("dirs ensured.")
