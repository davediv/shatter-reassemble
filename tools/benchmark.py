#!/usr/bin/env python3
"""Measure the numbers the spec's acceptance criteria are written in.

Runs the real application headless against the synthetic source, drives a
shatter and a reassembly programmatically, and reports frame time
percentiles for each phase separately -- because the phases cost wildly
different amounts and a single average over all of them says nothing
useful about whether 800 shards falling holds 60fps.

Medians and p95, never means. This machine's load average reached 53
during development, and a mean over a contended run is a measurement of
the other processes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shatter import config                       # noqa: E402
from shatter.app import Phase, ShatterApp        # noqa: E402


def percentiles(samples: list) -> dict:
    if not samples:
        return {}
    array = np.array(samples)
    return {
        "frames": int(array.size),
        # min is reported first and deliberately. On a contended machine
        # it is the least contaminated estimator of the true cost: the one
        # frame that got a clean run at the CPU. p50 and above include
        # however much of the machine the rest of the system was using.
        "min": round(float(array.min()), 3),
        "p50": round(float(np.percentile(array, 50)), 3),
        "p95": round(float(np.percentile(array, 95)), 3),
        "max": round(float(array.max()), 3),
        "fps_min": round(1000.0 / max(float(array.min()), 1e-6), 1),
        "fps_p50": round(1000.0 / max(float(np.percentile(array, 50)), 1e-6), 1),
    }


def contention(app) -> float:
    """How much slower this machine is than an idle one, right now.

    A fixed GL workload whose uncontended cost was measured at 0.106ms on
    this hardware. Reporting the ratio makes every other number in the
    report interpretable instead of merely alarming -- during development
    this machine sat at load average 30-53 with headless browsers on it,
    and the same control read 3.94ms, a 37x factor.
    """
    import numpy as np
    from shatter.render.primitives import ShapeBatch

    batch = ShapeBatch(app.display)
    a = np.random.rand(60, 2).astype(np.float32) * 1000
    b = np.random.rand(60, 2).astype(np.float32) * 1000
    app.display.begin_frame()
    for _ in range(20):
        batch.begin(); batch.lines(a, b, 4.0, (1, 1, 1, 1)); batch.flush()
    app.display.ctx.finish()
    start = time.perf_counter()
    for _ in range(200):
        batch.begin(); batch.lines(a, b, 4.0, (1, 1, 1, 1)); batch.flush()
    app.display.ctx.finish()
    measured = (time.perf_counter() - start) / 200 * 1e3
    batch.release()
    return round(measured / 0.106, 2)


def phase_run(app: ShatterApp, frames: int, collect: list) -> None:
    for _ in range(frames):
        start = time.perf_counter()
        app.step()
        collect.append((time.perf_counter() - start) * 1e3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=800)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--warmup", type=int, default=90)
    parser.add_argument("--falling", type=int, default=180)
    parser.add_argument("--settled", type=int, default=180)
    parser.add_argument("--json", type=str, default="")
    parser.add_argument("--ladder", action="store_true",
                        help="leave the degradation ladder enabled")
    parser.add_argument("--no-silhouette", action="store_false",
                        dest="segmentation")
    args = parser.parse_args()

    options = config.RuntimeOptions(
        width=args.width, height=args.height, source="synthetic",
        headless=True, vsync=False, show_debug=True,
        shard_count=args.shards, ladder_enabled=args.ladder,
        segmentation=args.segmentation,
    )
    app = ShatterApp(options)

    result = {
        "shards_requested": args.shards,
        "canvas": [args.width, args.height],
        "load_average": round(os.getloadavg()[0], 2),
    }
    try:
        result["warmup_s"] = round(app.warmup(verbose=False), 2)
        result["contention_factor"] = contention(app)
        idle: list = []
        phase_run(app, args.warmup, idle)
        result["idle"] = percentiles(idle)

        app._shatter((args.width * 0.72, args.height * 0.34))
        result["shards_actual"] = app.world.count
        result["fracture_ms"] = round(app.fracture.build_ms, 3) if app.fracture else 0
        result["fracture_stages"] = {
            k: round(v, 3) for k, v in (app.fracture.stage_ms or {}).items()
        } if app.fracture else {}
        result["prewarm"] = {"hits": app.prewarmer.hits,
                             "misses": app.prewarmer.misses}

        falling: list = []
        phase_run(app, args.falling, falling)
        result["falling"] = percentiles(falling)
        result["falling_awake"] = app.world.stats.awake
        result["falling_contacts"] = app.world.stats.contacts

        settled: list = []
        phase_run(app, args.settled, settled)
        result["settled"] = percentiles(settled)
        result["settled_awake"] = app.world.stats.awake
        result["settled_contacts"] = app.world.stats.contacts

        app._reassemble()
        reassembling: list = []
        while app.phase is Phase.REASSEMBLING and len(reassembling) < 400:
            phase_run(app, 1, reassembling)
        result["reassembling"] = percentiles(reassembling)
        result["rest_error_px"] = app.reassembly.rest_error()

        result["cpu_sections"] = {k: round(v, 3)
                                  for k, v in app.profiler.sections.items()}
        result["gpu_sections"] = {k: round(v, 3)
                                  for k, v in app.gpu.results().items()}
        result["tracking"] = {
            "delegate": app.tracker.stats.delegate,
            "rate_hz": round(app.tracker.stats.rate_hz, 1),
            "detect_ms": round(app.tracker.stats.detect_ms, 3),
        }
        result["load_average_end"] = round(os.getloadavg()[0], 2)
    finally:
        app.shutdown()

    print(json.dumps(result, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
