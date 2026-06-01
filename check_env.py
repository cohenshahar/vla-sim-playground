"""
check_env.py
------------
Phase 5 / Step 0 - Environment probe.

Goal:
    Print a compact, machine-parseable report of the local environment
    (OS, Python, Torch, CUDA, MuJoCo, RAM, free disk) and then recommend an
    OpenVLA inference strategy. This script must never hang and must never
    install anything. It is safe to run on a fresh machine.

Expected output (example, CPU-only Windows with 16 GB RAM):
    === phase5_openvla / check_env ===
    platform         : Windows-10-10.0.19045-SP0
    python           : 3.11.7
    torch            : 2.3.1            (cuda_available=False)
    cuda_device      : -
    mujoco           : 3.1.6
    transformers     : not installed
    accelerate       : not installed
    ram_total_gb     : 15.9
    ram_available_gb : 9.3
    free_disk_gb     : 220.5
    --------------------------------------------------
    RECOMMENDED STRATEGY: cpu-bfloat16
    Reason: no CUDA device detected; RAM >= 14 GB is enough to hold
            OpenVLA-7B in bfloat16 without disk offload.
    Expected inference time: 30-120 s per action.
    --------------------------------------------------

Exit codes:
    0 - probe finished (even if some imports failed; check the report).
    2 - critical import failure (python / torch) that blocks everything.
"""

from __future__ import annotations

import importlib
import platform
import shutil
import sys
from dataclasses import dataclass
from typing import Optional


# Minimum RAM (GB) needed to hold OpenVLA-7B in bfloat16 without disk offload.
RAM_FP16_MIN_GB = 14.0
# Soft floor below which even disk offload is painful.
RAM_HARD_FLOOR_GB = 7.0
# Free disk (GB) required to cache the HF weights.
DISK_MIN_GB = 30.0


@dataclass
class Probe:
    name: str
    version: Optional[str] = None
    extra: Optional[str] = None
    error: Optional[str] = None

    def line(self, width: int = 16) -> str:
        label = self.name.ljust(width)
        if self.error:
            return f"{label} : ERROR ({self.error})"
        ver = self.version or "unknown"
        if self.extra:
            return f"{label} : {ver:<16} ({self.extra})"
        return f"{label} : {ver}"


def probe_module(name: str, version_attr: str = "__version__") -> Probe:
    try:
        mod = importlib.import_module(name)
    except Exception as e:  # pragma: no cover - defensive
        return Probe(name=name, error=f"{type(e).__name__}: {e}")
    ver = getattr(mod, version_attr, None)
    return Probe(name=name, version=str(ver) if ver else "installed")


def probe_torch() -> Probe:
    try:
        import torch
    except Exception as e:
        return Probe(name="torch", error=f"{type(e).__name__}: {e}")
    cuda_ok = bool(torch.cuda.is_available())
    return Probe(
        name="torch",
        version=torch.__version__,
        extra=f"cuda_available={cuda_ok}",
    )


def probe_cuda_device() -> Probe:
    try:
        import torch
    except Exception:
        return Probe(name="cuda_device", version="-")
    if not torch.cuda.is_available():
        return Probe(name="cuda_device", version="-")
    idx = 0
    name = torch.cuda.get_device_name(idx)
    props = torch.cuda.get_device_properties(idx)
    vram_gb = props.total_memory / 1e9
    return Probe(
        name="cuda_device",
        version=name,
        extra=f"vram_gb={vram_gb:.1f}",
    )


def probe_ram() -> tuple[Optional[float], Optional[float]]:
    """Return (total_gb, available_gb). Uses psutil if present, else falls back."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total / 1e9, vm.available / 1e9
    except Exception:
        pass
    # Fallback: /proc/meminfo on Linux, else give up.
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.read().splitlines()
        kv = {}
        for ln in lines:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
        total_kb = int(kv.get("MemTotal", "0 kB").split()[0])
        avail_kb = int(kv.get("MemAvailable", "0 kB").split()[0])
        return total_kb / 1e6, avail_kb / 1e6
    except Exception:
        return None, None


def probe_disk(path: str = ".") -> Optional[float]:
    try:
        usage = shutil.disk_usage(path)
        return usage.free / 1e9
    except Exception:
        return None


def pick_strategy(
    cuda_ok: bool,
    vram_gb: Optional[float],
    ram_gb: Optional[float],
) -> tuple[str, str]:
    """Return (strategy_tag, human_reason)."""
    if cuda_ok and vram_gb is not None:
        if vram_gb >= 12.0:
            return (
                "gpu-float16",
                "CUDA GPU with >=12 GB VRAM. Load OpenVLA in float16 directly on the GPU.",
            )
        if vram_gb >= 8.0:
            return (
                "gpu-float16-offload",
                "CUDA GPU with 8-11 GB VRAM. Use float16 + device_map='auto' so LLaMA partially offloads to CPU.",
            )
        return (
            "gpu-4bit",
            "CUDA GPU with <8 GB VRAM. Use bitsandbytes 4-bit (CUDA only). Expect slow first-token.",
        )
    # No CUDA.
    if ram_gb is None:
        return (
            "cpu-bfloat16",
            "No CUDA; RAM unknown. Default to CPU bfloat16, check at load time.",
        )
    if ram_gb >= RAM_FP16_MIN_GB:
        return (
            "cpu-bfloat16",
            f"No CUDA; RAM {ram_gb:.1f} GB >= {RAM_FP16_MIN_GB:.0f} GB is enough to hold OpenVLA-7B in bfloat16 without disk offload. Expected 30-120 s per action.",
        )
    if ram_gb >= RAM_HARD_FLOOR_GB:
        return (
            "cpu-disk-offload",
            f"No CUDA; RAM {ram_gb:.1f} GB < {RAM_FP16_MIN_GB:.0f} GB. Use accelerate disk offload. Very slow (minutes per action) but functionally correct.",
        )
    return (
        "unsupported",
        f"RAM {ram_gb:.1f} GB is below the hard floor {RAM_HARD_FLOOR_GB:.0f} GB. OpenVLA-7B will not fit. Use a smaller VLA or move to Ubuntu + GPU.",
    )


def main() -> int:
    print("=== phase5_openvla / check_env ===")
    print(f"platform         : {platform.platform()}")
    print(f"python           : {sys.version.split()[0]}")

    torch_probe = probe_torch()
    print(torch_probe.line())
    if torch_probe.error:
        # torch is required downstream; print a strategy hint and exit 2.
        print("--------------------------------------------------")
        print("RECOMMENDED STRATEGY: abort")
        print("Reason: PyTorch not importable. Install it before continuing.")
        return 2

    print(probe_cuda_device().line())

    for name in ("mujoco", "transformers", "accelerate", "PIL", "numpy"):
        print(probe_module(name).line())

    ram_total, ram_avail = probe_ram()
    if ram_total is None:
        print("ram_total_gb    : unknown (install psutil)")
        print("ram_available_gb: unknown")
    else:
        print(f"ram_total_gb    : {ram_total:.1f}")
        print(f"ram_available_gb: {ram_avail:.1f}")

    free_disk = probe_disk(".")
    if free_disk is None:
        print("free_disk_gb    : unknown")
    else:
        print(f"free_disk_gb    : {free_disk:.1f}")

    # Strategy decision.
    import torch  # already imported above, re-import is cheap
    cuda_ok = bool(torch.cuda.is_available())
    vram_gb = None
    if cuda_ok:
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9

    strategy, reason = pick_strategy(cuda_ok, vram_gb, ram_total)

    print("--------------------------------------------------")
    print(f"RECOMMENDED STRATEGY: {strategy}")
    print(f"Reason: {reason}")
    if free_disk is not None and free_disk < DISK_MIN_GB:
        print(f"WARNING: free disk {free_disk:.1f} GB < {DISK_MIN_GB:.0f} GB. "
              "OpenVLA weights will not fit in the HF cache.")
    print("--------------------------------------------------")

    # Machine-parseable single line for downstream scripts.
    print(f"STRATEGY_TAG={strategy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
