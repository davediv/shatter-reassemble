"""Work out which MediaPipe delegate actually works on this machine.

This module exists because of a nasty property of MediaPipe on macOS: an
unsupported delegate/image-format combination does not raise, it calls
``LOG(FATAL)`` and aborts the process with SIGABRT. You cannot try/except
your way out of it, and a fallback chain written the obvious way is worse
than useless -- it walks the process into the crash.

Measured on this machine (M1, mediapipe 1.0.1), all three of these abort:

    GPU + SRGB   ImageFrame -> GPU buffer rejects 3-channel input
    CPU + SRGB   TensorsToDetectionsCalculator wants a Metal helper the
    CPU + SRGBA  CPU graph never stands up ("Service is unavailable")

leaving GPU + SRGBA as the only survivor. That is not a portable fact, so
rather than hard-coding it we probe each candidate in a *subprocess*,
where an abort costs one exit code instead of the app. The winner is
cached, so the ~3s probe happens once per machine rather than per launch.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

__all__ = ["DelegateChoice", "resolve", "CANDIDATES"]

CACHE_FILE = Path(__file__).resolve().parent.parent / "models" / ".delegate.json"

# Ordered best-first. GPU+SRGBA leads because it is both the fastest path
# and the only one that survives on macOS.
CANDIDATES: tuple[tuple[str, str], ...] = (
    ("GPU", "SRGBA"),
    ("CPU", "SRGB"),
    ("CPU", "SRGBA"),
    ("GPU", "SRGB"),
)

_PROBE_TIMEOUT = 90.0


@dataclass(frozen=True)
class DelegateChoice:
    delegate: str        # "GPU" | "CPU"
    image_format: str    # "SRGBA" | "SRGB"
    detect_ms: float     # rough per-frame cost seen during the probe

    @property
    def channels(self) -> int:
        return 4 if self.image_format == "SRGBA" else 3

    @property
    def cv_conversion(self) -> str:
        """The cv2 colour conversion this choice requires from BGR."""
        return "COLOR_BGR2RGBA" if self.channels == 4 else "COLOR_BGR2RGB"

    def describe(self) -> str:
        return f"{self.delegate}/{self.image_format} (~{self.detect_ms:.1f}ms)"


def _probe_child(model: str, delegate: str, fmt: str) -> int:
    """Run one candidate for real. Executed in a subprocess; may abort."""
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    channels = 4 if fmt == "SRGBA" else 3
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=model,
            delegate=getattr(mp_python.BaseOptions.Delegate, delegate),
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    frame = np.zeros((720, 1280, channels), np.uint8)
    frame[:] = 128
    image = mp.Image(image_format=getattr(mp.ImageFormat, fmt), data=frame)

    for i in range(3):
        landmarker.detect_for_video(image, i * 16)
    t0 = time.perf_counter()
    reps = 8
    for i in range(reps):
        landmarker.detect_for_video(image, 200 + i * 16)
    elapsed = (time.perf_counter() - t0) / reps * 1e3
    landmarker.close()
    print(f"__PROBE_OK__{elapsed:.3f}", flush=True)
    return 0


def _run_probe(model: Path, delegate: str, fmt: str) -> Optional[float]:
    env = dict(os.environ, GLOG_minloglevel="2", SHATTER_PROBE="1")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "shatter.delegate", "--probe",
             str(model), delegate, fmt],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT,
            cwd=str(Path(__file__).resolve().parent.parent), env=env,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("__PROBE_OK__"):
            try:
                return float(line[len("__PROBE_OK__"):])
            except ValueError:
                return 0.0
    return None


def _load_cache(model: Path) -> Optional[DelegateChoice]:
    try:
        data = json.loads(CACHE_FILE.read_text())
    except (OSError, ValueError):
        return None
    # Invalidate if the model or the mediapipe version changed underneath us.
    if data.get("model_mtime") != model.stat().st_mtime:
        return None
    if data.get("mediapipe") != _mediapipe_version():
        return None
    try:
        return DelegateChoice(
            data["delegate"], data["image_format"], float(data["detect_ms"])
        )
    except (KeyError, TypeError, ValueError):
        return None


def _store_cache(model: Path, choice: DelegateChoice) -> None:
    payload = asdict(choice)
    payload["model_mtime"] = model.stat().st_mtime
    payload["mediapipe"] = _mediapipe_version()
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(payload, indent=2))
    except OSError:
        pass


def _mediapipe_version() -> str:
    try:
        import mediapipe
        return str(mediapipe.__version__)
    except Exception:
        return "unknown"


def resolve(
    model: Path,
    *,
    prefer_gpu: bool = True,
    use_cache: bool = True,
    verbose: bool = True,
) -> DelegateChoice:
    """Return the fastest delegate/format pair that survives on this box."""
    if not model.exists():
        raise FileNotFoundError(
            f"{model} is missing. Run: python tools/fetch_models.py"
        )

    forced = os.environ.get("SHATTER_DELEGATE")
    if forced:
        delegate, _, fmt = forced.partition("/")
        return DelegateChoice(delegate.upper(), (fmt or "SRGBA").upper(), 0.0)

    if use_cache:
        cached = _load_cache(model)
        if cached is not None:
            if verbose:
                print(f"[delegate] cached: {cached.describe()}")
            return cached

    candidates = list(CANDIDATES)
    if not prefer_gpu:
        candidates.sort(key=lambda c: c[0] != "CPU")

    for delegate, fmt in candidates:
        if verbose:
            print(f"[delegate] probing {delegate}/{fmt} ...", end=" ", flush=True)
        ms = _run_probe(model, delegate, fmt)
        if ms is None:
            if verbose:
                print("unsupported")
            continue
        if verbose:
            print(f"ok, {ms:.1f}ms/frame")
        choice = DelegateChoice(delegate, fmt, ms)
        if use_cache:
            _store_cache(model, choice)
        return choice

    raise RuntimeError(
        "No MediaPipe delegate works on this machine. Tried: "
        + ", ".join(f"{d}/{f}" for d, f in candidates)
    )


def _main(argv: list[str]) -> int:
    if len(argv) >= 5 and argv[1] == "--probe":
        return _probe_child(argv[2], argv[3], argv[4])
    from . import config
    choice = resolve(config.HAND_MODEL, use_cache="--no-cache" not in argv)
    print(f"selected: {choice.describe()} channels={choice.channels}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
