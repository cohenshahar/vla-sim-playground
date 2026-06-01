"""
openvla_infer.py
----------------
Phase 5 / Step 2 - OpenVLA inference wrapper.

Goal:
    Wrap OpenVLA-7B so that the rest of the pipeline talks to it as a single
    function:  predict(image_rgb_uint8, instruction) -> np.ndarray shape (7,).

The wrapper auto-detects the device strategy:
    - 'gpu-float16'          : torch.float16 on cuda:0
    - 'gpu-float16-offload'  : torch.float16 with device_map='auto'
    - 'gpu-4bit'             : bitsandbytes 4-bit on cuda:0 (GPU only)
    - 'cpu-bfloat16'         : torch.bfloat16 on CPU (Windows-friendly)
    - 'cpu-disk-offload'     : accelerate disk offload; very slow but works
                               on low-RAM machines.

Usage:
    from openvla_infer import OpenVLAInference
    infer = OpenVLAInference()            # loads model
    action = infer.predict(frame_rgb, "move the arm upward")
    # action : np.ndarray shape (7,), dtype float32, range ~[-1, 1]

Expected first-call behavior:
    - GPU float16 : ~1.5 s / action after warmup, ~20 s warmup
    - CPU bfloat16: 30-120 s / action, ~2-5 min warmup the first time
    - Disk offload: 2-10 min / action (not interactive; use for plumbing test only)

The class loads OpenVLA lazily on __init__ so that importing this module does
not trigger any HF download until you actually instantiate it.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import CFG, Config, apply_determinism


# ---------------------------------------------------------------------------
# Strategy detection
# ---------------------------------------------------------------------------

def _detect_strategy(explicit: str = "auto") -> str:
    """Pick a strategy tag without importing heavy deps at module level."""
    if explicit and explicit != "auto":
        return explicit

    import torch  # local import keeps module import cheap

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        if vram_gb >= 12.0:
            return "gpu-float16"
        if vram_gb >= 8.0:
            return "gpu-float16-offload"
        return "gpu-4bit"

    # CPU branch.
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1e9
    except Exception:
        ram_gb = 16.0  # optimistic default
    if ram_gb >= 14.0:
        return "cpu-bfloat16"
    if ram_gb >= 7.0:
        return "cpu-disk-offload"
    return "unsupported"


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class OpenVLAInference:
    """Loads OpenVLA-7B once, exposes predict(image, instruction)."""

    def __init__(self, cfg: Optional[Config] = None, strategy: Optional[str] = None):
        self.cfg = cfg or CFG
        self.cfg.ensure_dirs()
        apply_determinism(self.cfg)

        self.strategy = _detect_strategy(self.cfg.strategy if strategy is None else strategy)
        if self.strategy == "unsupported":
            raise RuntimeError(
                "This machine has neither a CUDA GPU nor enough RAM to load OpenVLA-7B. "
                "Use a smaller VLA, or migrate to Ubuntu + GPU."
            )

        # Point HF at the local cache so weights land next to phase5_openvla.
        os.environ.setdefault("HF_HOME", str(self.cfg.hf_cache_dir))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(self.cfg.hf_cache_dir))

        self._processor = None
        self._model = None
        self._dtype = None
        self._device = None
        self._load()

    # ----- loading ----------------------------------------------------
    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq

        t0 = time.time()
        print(f"[openvla] loading processor from {self.cfg.model_id} ...")
        self._processor = AutoProcessor.from_pretrained(
            self.cfg.model_id,
            trust_remote_code=self.cfg.trust_remote_code,
        )
        print(f"[openvla] processor loaded in {time.time() - t0:.1f}s")

        kwargs, dtype, device = self._model_kwargs()
        self._dtype = dtype
        self._device = device

        t1 = time.time()
        print(f"[openvla] loading model with strategy={self.strategy} dtype={dtype} device={device}")
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.cfg.model_id,
            trust_remote_code=self.cfg.trust_remote_code,
            **kwargs,
        )
        # For 'gpu-float16' and 'cpu-bfloat16' we need to move the model
        # explicitly (device_map='auto' already placed it in the offload case).
        if "device_map" not in kwargs:
            self._model = self._model.to(device=device, dtype=dtype)
        self._model.eval()
        print(f"[openvla] model loaded in {time.time() - t1:.1f}s (total {time.time() - t0:.1f}s)")

    def _model_kwargs(self) -> tuple[dict, "object", str]:
        """Return (from_pretrained_kwargs, torch_dtype, target_device_str)."""
        import torch

        s = self.strategy
        if s == "gpu-float16":
            return ({"torch_dtype": torch.float16, "low_cpu_mem_usage": True},
                    torch.float16, "cuda:0")
        if s == "gpu-float16-offload":
            return ({"torch_dtype": torch.float16, "device_map": "auto",
                     "low_cpu_mem_usage": True},
                    torch.float16, "cuda:0")
        if s == "gpu-4bit":
            # bitsandbytes 4-bit; CUDA only. Requires bitsandbytes installed.
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            return ({"quantization_config": bnb, "device_map": "auto",
                     "low_cpu_mem_usage": True},
                    torch.float16, "cuda:0")
        if s == "cpu-bfloat16":
            return ({"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True},
                    torch.bfloat16, "cpu")
        if s == "cpu-disk-offload":
            offload = self.cfg.hf_cache_dir / "offload"
            offload.mkdir(parents=True, exist_ok=True)
            return ({"torch_dtype": torch.bfloat16, "device_map": "auto",
                     "offload_folder": str(offload), "low_cpu_mem_usage": True},
                    torch.bfloat16, "cpu")
        raise ValueError(f"Unknown strategy: {s!r}")

    # ----- inference --------------------------------------------------
    def predict(
        self,
        image_rgb_uint8: np.ndarray,
        instruction: Optional[str] = None,
        unnorm_key: str = "bridge_orig",
    ) -> np.ndarray:
        """
        Args:
            image_rgb_uint8: HxWx3 numpy array, uint8, RGB (NOT BGR).
            instruction:     natural-language command. Defaults to CFG.
            unnorm_key:      OpenVLA's per-dataset action statistics key.
                             'bridge_orig' matches the BridgeData-v2 training
                             distribution used by openvla/openvla-7b.

        Returns:
            np.ndarray of shape (action_dim,), dtype float32, in ~[-1, 1].
        """
        import torch
        from PIL import Image

        if instruction is None:
            instruction = self.cfg.instruction
        if image_rgb_uint8.dtype != np.uint8:
            image_rgb_uint8 = np.clip(image_rgb_uint8, 0, 255).astype(np.uint8)
        if image_rgb_uint8.ndim != 3 or image_rgb_uint8.shape[-1] != 3:
            raise ValueError(
                f"expected HxWx3 uint8 RGB, got shape={image_rgb_uint8.shape} "
                f"dtype={image_rgb_uint8.dtype}"
            )

        pil = Image.fromarray(image_rgb_uint8)

        # The OpenVLA processor expects a prompt in a specific format.
        prompt = f"In: What action should the robot take to {instruction.strip()}?\nOut:"

        inputs = self._processor(prompt, pil, return_tensors="pt")
        # Route tensors to the right device/dtype.
        inputs = {k: self._to_device(v) for k, v in inputs.items()}

        t0 = time.time()
        with torch.inference_mode():
            action = self._model.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False,
            )
        dt = time.time() - t0
        print(f"[openvla] predict_action took {dt:.2f}s -> action={np.round(action, 3).tolist()}")

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.cfg.action_dim:
            raise RuntimeError(
                f"OpenVLA returned action of shape {action.shape}, "
                f"expected ({self.cfg.action_dim},)"
            )
        return np.clip(action, -self.cfg.action_clip, self.cfg.action_clip)

    # ----- internals --------------------------------------------------
    def _to_device(self, t):
        import torch
        if not isinstance(t, torch.Tensor):
            return t
        # Integer tensors (tokens, attention masks) stay in their native dtype.
        if t.is_floating_point():
            return t.to(device=self._device, dtype=self._dtype)
        return t.to(device=self._device)


# ---------------------------------------------------------------------------
# CLI smoke: `python openvla_infer.py` on a black image.
# ---------------------------------------------------------------------------

def _cli_smoke() -> int:
    print("=== phase5_openvla / openvla_infer CLI smoke ===")
    infer = OpenVLAInference()
    img = np.zeros((CFG.image_size, CFG.image_size, 3), dtype=np.uint8)
    action = infer.predict(img, "pick up the cube")
    print(f"action dtype : {action.dtype}")
    print(f"action shape : {action.shape}")
    print(f"action       : {action.tolist()}")
    ok = action.shape == (CFG.action_dim,) and np.all(np.abs(action) <= 1.0 + 1e-6)
    print(f"SMOKE_OK={int(ok)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_cli_smoke())
