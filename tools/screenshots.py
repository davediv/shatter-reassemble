#!/usr/bin/env python3
"""Capture the effect at each stage, so it can be looked at rather than
inferred from timings.

Numbers say the shards land in the right place. They say nothing about
whether the break radiates from the hand, whether the pile reads as glass,
or whether the silhouette is visible in the void. Those are the things
that decide whether the app is any good, and the only way to check them is
to look.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shatter import config                       # noqa: E402
from shatter.app import Phase, ShatterApp        # noqa: E402


def save(app, out: Path, name: str) -> None:
    image = app.display.read_canvas()
    cv2.imwrite(str(out / f"{name}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"  wrote {name}.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=800)
    parser.add_argument("--out", type=str, default="docs/stills")
    parser.add_argument("--source", default="synthetic")
    parser.add_argument("--debug", action="store_true",
                        help="leave the HUD and skeleton visible")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    options = config.RuntimeOptions(
        width=1920, height=1080, source=args.source, headless=True,
        vsync=False, show_debug=args.debug, shard_count=args.shards,
        ladder_enabled=False, segmentation=(args.source == "camera"),
    )
    app = ShatterApp(options)
    app.warmup(verbose=False)
    app.overlay.visible = args.debug

    try:
        for _ in range(40):
            app.step()
        save(app, out, "01-idle")

        app._shatter((1920 * 0.68, 1080 * 0.34))
        app.step()
        save(app, out, "02-snap-flash")

        for _ in range(8):
            app.step()
        save(app, out, "03-breaking")

        for _ in range(45):
            app.step()
        save(app, out, "04-falling")

        for _ in range(240):
            app.step()
        save(app, out, "05-settled-pile")

        # Reassembly is driven by wall-clock time, not by frame count. An
        # unthrottled headless loop runs at several hundred fps, so
        # stepping a fixed number of frames barely advances the animation
        # -- these wait on progress instead.
        app._reassemble()
        for label, target in (("06-reassembling-early", 0.30),
                              ("07-reassembling-late", 0.75)):
            guard = 0
            while (app.phase is Phase.REASSEMBLING
                   and app.reassembly.state.progress < target
                   and guard < 20000):
                app.step()
                guard += 1
            save(app, out, label)

        guard = 0
        while app.phase is Phase.REASSEMBLING and guard < 20000:
            app.step()
            guard += 1
        for _ in range(3):
            app.step()
        save(app, out, "08-reassembled")
        print(f"\nrest error {app.reassembly.rest_error():.6f} px, "
              f"{app.world.stats.bodies} shards")
    finally:
        app.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
