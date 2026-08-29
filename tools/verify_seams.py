#!/usr/bin/env python3
"""Prove the acceptance criterion: reassembly lands with zero visible seams.

The unit tests show that shards return to their rest transform exactly and
that neighbouring cells share identical edges. Neither of those is quite
the claim being made. The claim is about *pixels*: that the reassembled
1080p image is indistinguishable from the frozen frame it was broken out
of, with no cracks between shards.

So this renders both and compares them. It shatters, drives the reassembly
to completion, renders the final frame with every shard at rest and the
bevel closed, and diffs it against the same frozen frame drawn as a plain
fullscreen quad.

The number that matters is the count of *crack* pixels: pixels that are
much darker in the reassembled image than in the reference, which is what
a gap between two shards looks like against the black void behind them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shatter import config                       # noqa: E402
from shatter.app import Phase, ShatterApp        # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=800)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--settle", type=int, default=120)
    parser.add_argument("--save", type=str, default="",
                        help="directory to write reference/result/diff PNGs")
    args = parser.parse_args()

    options = config.RuntimeOptions(
        width=args.width, height=args.height, source="synthetic",
        headless=True, vsync=False, show_debug=False,
        shard_count=args.shards, ladder_enabled=False, segmentation=False,
    )
    app = ShatterApp(options)
    app.warmup(verbose=False)
    app.overlay.visible = False

    try:
        for _ in range(30):
            app.step()

        # The reference: the frozen frame as a plain fullscreen quad, which
        # is exactly what the shards should add up to.
        app.video.freeze()
        app.display.begin_frame()
        app.display.ctx.clear(0.0, 0.0, 0.0, 1.0)
        app.video.draw(freeze=1.0)
        reference = app.display.read_canvas().astype(np.int16)

        app._shatter((args.width * 0.66, args.height * 0.36))
        for _ in range(args.settle):
            app.step()
        # Step until every shard has landed and the bevel has closed, but
        # stop before the app tears the pile down -- finishing the phase
        # releases the shard buffer, and comparing against an empty canvas
        # proves nothing.
        app._reassemble()
        guard = 0
        while app.phase is Phase.REASSEMBLING and guard < 600:
            app.step()
            guard += 1
            state = app.reassembly.state
            if state.progress >= 1.0 and state.bevel == 0.0 and state.relief == 0.0:
                break
        if app.shards.count == 0:
            print("error: the pile was released before it could be compared")
            return 2

        # Redraw the pile one last time with every shard at rest and the
        # bevel fully closed, without the crossfade on top -- otherwise we
        # would be comparing live video against itself and proving nothing.
        app.display.begin_frame()
        app.display.ctx.clear(0.0, 0.0, 0.0, 1.0)
        app.display.canvas.clear(depth=1.0)
        app.shards.update_transforms(app.reassembly.transforms)
        app.shards.update_extras(flash=np.zeros(app.shards.count, np.float32))
        app.shards.draw(app.video.frozen, app.void.scene_color,
                        bevel=0.0, relief=0.0, refraction=0.0, bevel_shade=False)
        result = app.display.read_canvas().astype(np.int16)

        rest_error = app.reassembly.rest_error()
        diff = np.abs(result - reference).max(axis=2)
        luma_ref = reference.mean(axis=2)
        luma_out = result.mean(axis=2)
        # A crack shows as the void behind showing through: much darker
        # than the reference at that pixel.
        cracks = (luma_ref - luma_out) > 60
        total = diff.size

        print(f"shards                {app.shards.count}")
        print(f"canvas                {args.width}x{args.height}")
        print(f"rest transform error  {rest_error:.6f} px")
        print(f"max channel diff      {int(diff.max())}")
        print(f"mean channel diff     {float(diff.mean()):.4f}")
        print(f"pixels differing >8   {int((diff > 8).sum())} "
              f"({(diff > 8).sum() / total * 100:.4f}%)")
        print(f"crack pixels          {int(cracks.sum())} "
              f"({cracks.sum() / total * 100:.6f}%)")

        if args.save:
            import cv2
            out = Path(args.save)
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / "reference.png"),
                        cv2.cvtColor(reference.astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out / "reassembled.png"),
                        cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out / "diff.png"),
                        np.clip(diff * 8, 0, 255).astype(np.uint8))
            print(f"wrote reference.png, reassembled.png, diff.png to {out}")

        ok = rest_error == 0.0 and cracks.sum() == 0
        print("\nSEAMLESS" if ok else "\nSEAMS DETECTED")
        return 0 if ok else 1
    finally:
        app.shutdown()


if __name__ == "__main__":
    sys.exit(main())
