"""Command line entry point: ``python -m shatter``."""

from __future__ import annotations

import argparse
import sys

from . import config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shatter",
        description="Shatter & Reassemble -- snap to break the room, clap to "
                    "put it back.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--width", type=int, default=config.CANVAS_WIDTH,
                        help="canvas width")
    parser.add_argument("--height", type=int, default=config.CANVAS_HEIGHT,
                        help="canvas height")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                        dest="camera_index", help="camera device index")
    parser.add_argument("--source", default="camera",
                        help="'camera', 'synthetic', or a path to a video file")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--headless", action="store_true",
                        help="run with no window (benchmarks and CI)")
    parser.add_argument("--no-vsync", action="store_false", dest="vsync",
                        help="uncap the frame rate")
    parser.add_argument("--frames", type=int, default=0,
                        help="stop after N frames (0 runs until closed)")
    parser.add_argument("--tune", action="store_true", dest="tuning_mode",
                        help="start in snap tuning mode")
    parser.add_argument("--no-debug", action="store_false", dest="show_debug",
                        help="start with the debug UI hidden")
    parser.add_argument("--shards", type=int, default=config.SHARD_COUNT_TIERS[0],
                        dest="shard_count", help="shard count at full quality")
    parser.add_argument("--no-ladder", action="store_false", dest="ladder_enabled",
                        help="disable the degradation ladder")
    parser.add_argument("--no-gpu", action="store_false", dest="gpu_delegate",
                        help="prefer the CPU delegate for hand tracking")
    parser.add_argument("--no-silhouette", action="store_false", dest="segmentation",
                        help="skip person segmentation")
    parser.add_argument("--record-mode", default=config.RECORD_MODE,
                        choices=config.RECORD_MODES,
                        help="vp9 captures on the hardware encoder and "
                             "transcodes on stop; vp9-realtime encodes VP9 live "
                             "(drops frames without a hardware VP9 encoder); "
                             "h264 skips the transcode")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    options = config.RuntimeOptions(
        width=args.width,
        height=args.height,
        camera_index=args.camera_index,
        source=args.source,
        tuning_mode=args.tuning_mode,
        show_debug=args.show_debug,
        fullscreen=args.fullscreen,
        headless=args.headless,
        vsync=args.vsync,
        frames=args.frames,
        shard_count=args.shard_count,
        ladder_enabled=args.ladder_enabled,
        gpu_delegate=args.gpu_delegate,
        segmentation=args.segmentation,
        record_mode=args.record_mode,
    )

    from .app import ShatterApp

    try:
        app = ShatterApp(options)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return app.run(frames=options.frames)


if __name__ == "__main__":
    sys.exit(main())
