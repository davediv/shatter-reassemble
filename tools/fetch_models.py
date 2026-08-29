#!/usr/bin/env python3
"""Download the MediaPipe model bundles the app needs.

Kept out of the repo (see .gitignore) because they are large binaries with
their own licence; this fetches them on demand and verifies the size.
"""

from __future__ import annotations

import shutil
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"

BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "hand_landmarker.task": (
        f"{BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        True,   # required
    ),
    "selfie_segmenter.tflite": (
        f"{BASE}/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite",
        False,  # optional: only the background silhouette needs it
    ),
}


def _ssl_context() -> ssl.SSLContext | None:
    """Trust store for the download.

    Framework Python builds on macOS ship without a usable CA bundle
    unless "Install Certificates.command" has been run, so prefer certifi
    when it is importable and let the caller fall back to curl otherwise.
    """
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def _curl(url: str, dest: Path) -> bool:
    """Last resort: the system curl, which has the OS trust store."""
    exe = shutil.which("curl")
    if not exe:
        return False
    r = subprocess.run(
        [exe, "-fsSL", "--max-time", "180", "-o", str(dest), url],
        capture_output=True,
    )
    return r.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def fetch(name: str, url: str, required: bool) -> bool:
    dest = MODEL_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  {name:28} present ({dest.stat().st_size / 1024:.0f} KB)")
        return True
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60, context=_ssl_context()) as r, \
                tmp.open("wb") as f:
            while chunk := r.read(1 << 16):
                f.write(chunk)
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        if not _curl(url, tmp):
            level = "ERROR" if required else "warning"
            print(f"  {name:28} {level}: {exc}")
            return not required
    tmp.replace(dest)
    print(f"  {name:28} downloaded ({dest.stat().st_size / 1024:.0f} KB)")
    return True


def main() -> int:
    print(f"Fetching models into {MODEL_DIR}")
    ok = all(fetch(n, u, req) for n, (u, req) in MODELS.items())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
