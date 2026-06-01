"""
setup_hf.py
-----------
Phase 5 / pre-Step 2 - HuggingFace access bootstrap.

Goal:
    Make sure the local machine can download `openvla/openvla-7b`. This means:
      1. huggingface_hub is installed.
      2. A user token is cached locally (either HF_TOKEN env var or
         ~/.cache/huggingface/token from `huggingface-cli login`).
      3. The user has accepted the gated-model license on the website.

This script does NOT download the model. It only verifies access by calling
`HfApi.model_info` on the repo, which succeeds iff the user is authenticated
AND has accepted the license.

Expected output on a fully-configured machine:
    [hf] token source : env (HF_TOKEN) OR cache (~/.cache/huggingface/token)
    [hf] whoami       : <your-hf-username>
    [hf] model access : OK  (openvla/openvla-7b, 29 files, last_modified=...)
    READY_FOR_DOWNLOAD=1

Expected output when the license has not been accepted:
    [hf] whoami       : <your-hf-username>
    [hf] model access : FORBIDDEN
    ACTION REQUIRED:
      1. Open https://huggingface.co/openvla/openvla-7b
      2. Click "Agree and access repository".
      3. Re-run this script.
    READY_FOR_DOWNLOAD=0

Exit codes:
    0 - ready to download.
    1 - token missing; print login instructions.
    2 - license not accepted; print URL.
    3 - other error.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


MODEL_ID = "openvla/openvla-7b"
LOGIN_CMD = "huggingface-cli login"
LICENSE_URL = f"https://huggingface.co/{MODEL_ID}"


def find_token() -> tuple[str | None, str]:
    """Return (token, source) or (None, 'none')."""
    env_tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_tok:
        return env_tok, "env"
    # Default cache path used by huggingface-cli login.
    candidates = [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",  # legacy
    ]
    for p in candidates:
        if p.exists():
            try:
                tok = p.read_text().strip()
                if tok:
                    return tok, f"cache ({p})"
            except Exception:
                continue
    return None, "none"


def main() -> int:
    print("[hf] phase5_openvla / setup_hf")

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except Exception as e:
        print(f"[hf] huggingface_hub not importable: {e}")
        print("ACTION REQUIRED: pip install -r requirements-cpu.txt (or -gpu.txt)")
        return 3

    token, source = find_token()
    print(f"[hf] token source : {source}")
    if not token:
        print("ACTION REQUIRED:")
        print(f"  1. Create a read-token at https://huggingface.co/settings/tokens")
        print(f"  2. Run: {LOGIN_CMD}")
        print(f"  3. Paste the token when prompted.")
        print("READY_FOR_DOWNLOAD=0")
        return 1

    api = HfApi(token=token)

    try:
        user = api.whoami(token=token)
        print(f"[hf] whoami       : {user.get('name', '?')}")
    except Exception as e:
        print(f"[hf] whoami FAILED: {e}")
        print("ACTION REQUIRED: token is present but invalid. Re-run huggingface-cli login.")
        print("READY_FOR_DOWNLOAD=0")
        return 1

    try:
        info = api.model_info(MODEL_ID, token=token)
    except GatedRepoError:
        print(f"[hf] model access : FORBIDDEN (license not accepted)")
        print("ACTION REQUIRED:")
        print(f"  1. Open {LICENSE_URL}")
        print(f"  2. Click 'Agree and access repository'.")
        print(f"  3. Re-run this script.")
        print("READY_FOR_DOWNLOAD=0")
        return 2
    except RepositoryNotFoundError:
        print(f"[hf] model access : NOT FOUND ({MODEL_ID})")
        print("ACTION REQUIRED: check the model id in config.py.")
        print("READY_FOR_DOWNLOAD=0")
        return 3
    except Exception as e:
        print(f"[hf] model access : ERROR ({type(e).__name__}: {e})")
        print("READY_FOR_DOWNLOAD=0")
        return 3

    n_files = len(getattr(info, "siblings", []) or [])
    print(f"[hf] model access : OK  ({MODEL_ID}, {n_files} files)")
    print("READY_FOR_DOWNLOAD=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
