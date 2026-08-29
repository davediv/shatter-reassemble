"""The loop that owns everything.

Three threads and one state machine. Capture and hand tracking run on
their own threads and publish; this loop consumes whatever is newest and
never waits for either. Gesture detection runs inside the tracking thread
so it sees every tracking frame -- a snap is a 30ms transient and polling
for it from here would miss the frames it lives in.

    IDLE          live camera, nothing broken
    SHATTERED     frozen frame in shards, physics running, void behind
    REASSEMBLING  shards animating home, crossfading back to live

Render order in the broken states matters and is not obvious: the void is
drawn into an offscreen scene buffer first, blitted to the canvas as the
background, and then *sampled again* by the shard shader for refraction.
One pass, two uses -- which is what makes refracting the real background
affordable instead of faking it.
"""

from __future__ import annotations

import sys
import time
from enum import Enum
from typing import Optional

import numpy as np

from . import config, keys
from .camera import open_source
from .fracture import FracturePrewarmer, FractureResult, fracture
from .gestures import CLAP, SNAP, GestureRecognizer
from .physics import MAX_CAPSULES, PhysicsWorld
from .profiler import FrameProfiler, GpuProfiler, QualityLadder
from .reassemble import Reassembly
from .recorder import Recorder
from .render.background import VoidLayer
from .render.context import Display
from .render.debug import DebugOverlay
from .render.primitives import ShapeBatch
from .render.shards import ShardRenderer
from .render.text import TextRenderer
from .render.video import VideoLayer
from .silhouette import SilhouetteTracker
from .tracking import HandTracker
from .tuning import TuningMode
from .viewport import CoverFit

__all__ = ["ShatterApp", "Phase"]

# Bones used to build the stir capsules. The palm-closing edge is left out
# -- a capsule across the palm would sweep shards the hand never touched.
CAPSULE_CHAINS = config.FINGER_CHAINS[:5]


class Phase(Enum):
    IDLE = "idle"
    SHATTERED = "shattered"
    REASSEMBLING = "reassembling"


class ShatterApp:
    def __init__(self, options: config.RuntimeOptions) -> None:
        self.options = options
        self.tunables = options.tunables
        self.phase = Phase.IDLE
        self.running = False
        self.frame_index = 0

        # Realtime threading wants a short GIL switch interval; see config.
        sys.setswitchinterval(config.GIL_SWITCH_INTERVAL)

        self.source = open_source(options).start()
        self.fit = CoverFit(self.source.width, self.source.height,
                            options.width, options.height)

        self.display = Display(options.width, options.height,
                               headless=options.headless,
                               vsync=options.vsync,
                               fullscreen=options.fullscreen)
        self.display.on_key(self._on_key)

        self.video = VideoLayer(self.display, self.fit)
        self.void = VoidLayer(self.display, self.fit)
        capacity = max(options.shard_count, config.SHARD_COUNT_TIERS[0])
        self.shards = ShardRenderer(self.display, self.fit, capacity)
        self.shapes = ShapeBatch(self.display)
        self.text = TextRenderer(self.display)
        self.overlay = DebugOverlay(self.shapes, self.text)
        self.overlay.visible = options.show_debug
        # HUD panels change slowly compared with the skeleton and flashes.
        # Give them retained buffers of their own so ordinary frames only
        # resubmit the existing geometry instead of rebuilding it.
        self.hud_shapes = self.shapes.fork(capacity=2048)
        self.hud_text = self.text.fork(capacity=4096)
        self.hud_overlay = DebugOverlay(self.hud_shapes, self.hud_text)

        self.recognizer = GestureRecognizer(self.tunables)
        self.tracker = HandTracker(self.source, self.fit,
                                   gpu_delegate=options.gpu_delegate,
                                   on_frame=self.recognizer.update).start()
        self.silhouette = (
            SilhouetteTracker(self.source, self.tracker.choice).start()
            if options.segmentation else None
        )

        self.world = PhysicsWorld(capacity + 32, options.width, options.height)
        self.prewarmer = FracturePrewarmer().start()
        self.reassembly = Reassembly(options.width, options.height)
        self.fracture: Optional[FractureResult] = None

        self.profiler = FrameProfiler()
        self.gpu = GpuProfiler(self.display.ctx)
        self.ladder = QualityLadder(enabled=options.ladder_enabled)
        self.recorder = Recorder(self.display, mode=options.record_mode)

        self.tuning = TuningMode(self.tunables, self.recognizer, self.shapes,
                                 self.text, self.overlay)
        self.tuning.active = options.tuning_mode

        self._last_frame_index = -1
        self._interval = 1000.0 / 60.0
        self._flash = 0.0
        self._snap_time = -1e9
        self._hud_refresh_at = 0.0
        self._capsule_seg = np.zeros((MAX_CAPSULES, 4), np.float64)
        self._capsule_radius = np.zeros(MAX_CAPSULES, np.float64)
        self._capsule_vel = np.zeros((MAX_CAPSULES, 2), np.float64)
        self._status = "snap to shatter  .  clap to reassemble"

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def _on_key(self, event) -> None:
        if event.action not in (keys.PRESS, keys.REPEAT):
            return
        if self.tuning.handle_key(event.key, event.action, event.mods):
            return

        key = event.key
        if key in (keys.ESCAPE, keys.Q):
            self.running = False
        elif key == keys.H:
            self.overlay.visible = not self.overlay.visible
            self._status = f"debug UI {'shown' if self.overlay.visible else 'hidden'}"
        elif key == keys.R:
            self._toggle_recording()
        elif key == keys.T:
            self.tuning.active = not self.tuning.active
            self._status = f"tuning {'on' if self.tuning.active else 'off'}"
        elif key == keys.SPACE:
            # Manual triggers. Indispensable: they make every downstream
            # system testable without a hand, and they let you compare a
            # gesture-fired snap against a known-good one when detection
            # is being difficult.
            self._shatter((self.options.width * 0.5, self.options.height * 0.42))
        elif key == keys.C:
            self._reassemble()
        elif key == keys.F:
            self.overlay.show_raw = not self.overlay.show_raw
        elif key == keys.L:
            self.ladder.enabled = not self.ladder.enabled
            self._status = f"ladder {'on' if self.ladder.enabled else 'off'}"
        elif key == keys.LEFT_BRACKET:
            self.ladder.force(self.ladder.index - 1, time.perf_counter())
            self._status = f"quality {self.ladder.level.name}"
        elif key == keys.RIGHT_BRACKET:
            self.ladder.force(self.ladder.index + 1, time.perf_counter())
            self._status = f"quality {self.ladder.level.name}"

    def _toggle_recording(self) -> None:
        if self.recorder.recording:
            outputs = self.recorder.stop()
            names = ", ".join(p.name for p in outputs)
            self._status = (f"recorded {names}"
                            + (" (transcoding)" if self.recorder.transcoding else ""))
        elif self.recorder.start():
            self._status = f"recording ({self.recorder.mode})"
        else:
            self._status = f"record failed: {self.recorder.stats.error}"

    # ------------------------------------------------------------------
    # Gestures
    # ------------------------------------------------------------------

    def _shatter(self, origin: tuple) -> None:
        if self.phase is not Phase.IDLE:
            return
        # The ladder caps the shard count; --shards sets the ceiling. That
        # ordering matters: stepping down a rung must be able to reduce the
        # count, but it must never raise it above what the user asked for.
        count = min(self.options.shard_count, self.ladder.level.shard_count)
        self.video.freeze()

        result = self.prewarmer.take(self.options.width, self.options.height,
                                     origin, count, config.BEVEL_WIDTH)
        self.fracture = result
        self.shards.upload(result)
        self.world.load(result)
        self.world.iterations = self.ladder.level.solver_iterations
        self.world.explode(origin)

        self.phase = Phase.SHATTERED
        self._snap_time = time.perf_counter()
        self._flash = 1.0
        self._status = f"shattered into {result.count} shards"

    def _reassemble(self) -> None:
        if self.phase is not Phase.SHATTERED or self.fracture is None:
            return
        self.reassembly.begin(self.world, self.fracture, time.perf_counter())
        self.phase = Phase.REASSEMBLING
        self._status = "reassembling"

    def _handle_events(self) -> None:
        for event in self.recognizer.poll():
            if event.kind == SNAP:
                self._shatter(event.position)
            elif event.kind == CLAP:
                self._reassemble()

    def _prewarm(self, hands) -> None:
        """Build the next fracture while a hand is still armed."""
        if self.phase is not Phase.IDLE or hands is None:
            return
        signals = self.recognizer.signals()
        for slot in range(config.NUM_HANDS):
            if signals.armed[slot] and hands.present[slot]:
                point = hands.raw[slot, config.MIDDLE_TIP]
                self.prewarmer.request(
                    self.options.width, self.options.height,
                    (float(point[0]), float(point[1])),
                    min(self.options.shard_count, self.ladder.level.shard_count),
                    config.BEVEL_WIDTH,
                )
                return

    # ------------------------------------------------------------------
    # Stirring
    # ------------------------------------------------------------------

    def _update_capsules(self, hands) -> int:
        """Capsule chains along the fingers of any open palm."""
        if hands is None or self.phase is Phase.REASSEMBLING:
            self.world.clear_capsules()
            return 0
        signals = self.recognizer.signals()
        written = 0
        for slot in range(config.NUM_HANDS):
            if not hands.present[slot] or not signals.open_palm[slot]:
                continue
            span = float(hands.span[slot])
            radius = max(self.tunables.stir_capsule_radius * span, 6.0)
            points = hands.smooth[slot]
            velocity = hands.velocity[slot]
            for chain in CAPSULE_CHAINS:
                for a, b in zip(chain[:-1], chain[1:]):
                    if written >= MAX_CAPSULES:
                        break
                    self._capsule_seg[written] = (
                        points[a, 0], points[a, 1], points[b, 0], points[b, 1]
                    )
                    self._capsule_radius[written] = radius
                    self._capsule_vel[written] = (
                        (velocity[a, 0] + velocity[b, 0]) * 0.5,
                        (velocity[a, 1] + velocity[b, 1]) * 0.5,
                    )
                    written += 1
        if written:
            self.world.set_capsules(self._capsule_seg[:written],
                                    self._capsule_radius[:written],
                                    self._capsule_vel[:written])
        else:
            self.world.clear_capsules()
        return written

    # ------------------------------------------------------------------
    # Frame
    # ------------------------------------------------------------------

    def step(self) -> None:
        now = time.perf_counter()
        self.profiler.begin_frame()
        self.display.poll()

        with self.profiler.section("upload"):
            frame = self.source.latest(self._last_frame_index)
            if frame is not None:
                self.video.upload(frame)
                self._last_frame_index = frame.index
            if self.silhouette is not None:
                mask = self.silhouette.take_mask()
                if mask is not None:
                    self.void.upload_mask(mask)

        hands = self.tracker.latest()
        self._handle_events()
        self._prewarm(hands)

        level = self.ladder.level
        with self.profiler.section("physics"):
            capsules = self._update_capsules(hands)
            if self.phase is Phase.SHATTERED:
                self.world.iterations = level.solver_iterations
                # The real interval, not the smoothed average: the
                # accumulator wants elapsed time, and feeding it a
                # low-passed number lets simulated time drift away from
                # wall time, which reassembly is scheduled against.
                self.world.step(min(self._interval * 1e-3, 0.05))
                transforms = self.world.interpolated()
            elif self.phase is Phase.REASSEMBLING:
                state = self.reassembly.update(now)
                transforms = self.reassembly.transforms
                if state.finished:
                    self.phase = Phase.IDLE
                    self.shards.clear()
                    self.world.count = 0
                    self._status = "reassembled"
            else:
                transforms = None

        with self.profiler.section("render"):
            self._render(transforms, hands, now)

        with self.profiler.section("record"):
            self.recorder.capture()

        self.display.present()
        self.gpu.next_frame()
        self._interval = self.profiler.end_frame()
        self.frame_index += 1

        if self.ladder.update(self.profiler.rolling_ms, now):
            self.world.iterations = self.ladder.level.solver_iterations
            self._status = (f"quality -> {self.ladder.level.name} "
                            f"({self.ladder.last_reason})")

    def _render(self, transforms, hands, now: float) -> None:
        level = self.ladder.level
        display = self.display
        reassembly = self.reassembly.state

        if self.phase is Phase.IDLE:
            display.begin_frame()
            display.ctx.clear(0.0, 0.0, 0.0, 1.0)
            with self.gpu.section("video"):
                self.video.draw()
        else:
            # Segmentation publishes at 20Hz, so the scene texture is valid
            # for the intervening render frames. Avoid rerunning its five-tap
            # full-screen outline shader when its only input is unchanged.
            if self.void.scene_dirty:
                with self.gpu.section("void"):
                    self.void.render_scene()
            self.void.blit_to_canvas()
            display.canvas.clear(depth=1.0)

            if transforms is not None and len(transforms):
                self.shards.update_transforms(transforms)
                if self.phase is Phase.REASSEMBLING:
                    self.shards.update_extras(
                        flash=self.reassembly.flash_per_shard
                    )
                animating = self.phase is Phase.REASSEMBLING
                bevel = reassembly.bevel if animating else 1.0
                relief = reassembly.relief if animating else 1.0
                with self.gpu.section("shards"):
                    if level.shadows and relief > 0.0:
                        self.shards.draw(self.video.frozen, self.void.scene_color,
                                         bevel=bevel, relief=relief, shadow=True)
                    self.shards.draw(
                        self.video.frozen, self.void.scene_color,
                        bevel=bevel, relief=relief,
                        refraction=config.REFRACTION_STRENGTH if level.refraction else 0.0,
                        bevel_shade=level.bevel,
                    )

            # The frozen frame gives way to live video over the final 150ms.
            if self.phase is Phase.REASSEMBLING and reassembly.crossfade > 0.0:
                self.video.draw(alpha=reassembly.crossfade)

        self._draw_overlay(hands, now)

    def _draw_overlay(self, hands, now: float) -> None:
        self.shapes.begin()
        self.text.begin()

        # Snap flash, and the flash as the last shard lands.
        flash = self._flash
        if self.phase is Phase.REASSEMBLING:
            flash = max(flash, self.reassembly.state.flash)
        if flash > 0.001:
            self.shapes.rect(0, 0, self.options.width, self.options.height,
                             (1.0, 1.0, 1.0, min(flash, 1.0) * 0.85))
            self._flash *= 0.82

        if self.tuning.active:
            self.tuning.draw(self.options.width, self.options.height)
        if self.overlay.visible:
            self.overlay.draw_skeleton(hands)

        refresh_hud = (
            self.overlay.visible
            and now >= self._hud_refresh_at
        )
        if refresh_hud:
            self.hud_shapes.begin()
            self.hud_text.begin()
            self._draw_hud()
            self._hud_refresh_at = now + 1.0 / config.DEBUG_HUD_HZ

        self.shapes.flush()
        if self.overlay.visible:
            if refresh_hud:
                self.hud_shapes.flush()
            else:
                self.hud_shapes.render_cached()
        self.text.flush()
        if self.overlay.visible:
            if refresh_hud:
                self.hud_text.flush()
            else:
                self.hud_text.render_cached()

    def _draw_hud(self) -> None:
        profiler = self.profiler
        sections = profiler.sections
        tracker = self.tracker.stats
        physics = self.world.stats
        level = self.ladder.level

        def colour(value, budget):
            if value < budget * 0.75:
                return (0.45, 0.95, 0.55, 1.0)
            if value < budget:
                return (1.0, 0.85, 0.35, 1.0)
            return (1.0, 0.42, 0.38, 1.0)

        rows = [
            ("fps", f"{profiler.fps:.1f}", colour(profiler.frame_ms, 16.6)),
            ("frame", f"{profiler.frame_ms:5.2f} ms", colour(profiler.frame_ms, 16.6)),
            ("rolling 90", f"{profiler.rolling_ms:5.2f} ms",
             colour(profiler.rolling_ms, config.LADDER_STEP_DOWN_MS)),
            ("upload", f"{sections.get('upload', 0):5.2f} ms", (0.7, 0.78, 0.9, 1)),
            ("physics", f"{sections.get('physics', 0):5.2f} ms",
             colour(sections.get('physics', 0), 4.0)),
            ("render", f"{sections.get('render', 0):5.2f} ms",
             colour(sections.get('render', 0), 6.0)),
            ("record", f"{sections.get('record', 0):5.2f} ms",
             colour(sections.get('record', 0), 1.0)),
        ]
        overlay = self.hud_overlay
        end = overlay.panel(28, 28, rows, title="FRAME BUDGET")

        gpu = self.gpu.results()
        if gpu:
            rows = [(name, f"{ms:5.2f} ms", (0.7, 0.85, 1.0, 1))
                    for name, ms in sorted(gpu.items())]
            end = overlay.panel(28, end, rows, title="GPU")

        rows = [
            ("delegate", tracker.delegate, (0.8, 0.85, 0.95, 1)),
            ("rate", f"{tracker.rate_hz:.1f} Hz", (0.8, 0.85, 0.95, 1)),
            ("detect", f"{tracker.detect_ms:5.2f} ms",
             colour(tracker.detect_ms, 6.0)),
            ("camera", f"{self.source.measured_fps:.1f} fps "
                       f"({self.source.frames_dropped} dropped)", (0.7, 0.78, 0.9, 1)),
        ]
        if self.silhouette is not None and self.silhouette.stats.enabled:
            rows.append(("silhouette", f"{self.silhouette.stats.segment_ms:5.2f} ms "
                                       f"@ {self.silhouette.stats.rate_hz:.0f}Hz",
                         (0.7, 0.78, 0.9, 1)))
        end = overlay.panel(28, end, rows, title="TRACKING")

        rows = [
            ("phase", self.phase.value, (1.0, 0.9, 0.5, 1)),
            ("shards", f"{physics.bodies}", (0.8, 0.85, 0.95, 1)),
            ("awake", f"{physics.awake}", (0.8, 0.85, 0.95, 1)),
            ("contacts", f"{physics.contacts}", (0.8, 0.85, 0.95, 1)),
            ("substeps", f"{physics.substeps} @ {config.PHYSICS_HZ:.0f}Hz",
             (0.7, 0.78, 0.9, 1)),
            ("quality", f"{level.name} ({self.ladder.index})", (1.0, 0.9, 0.5, 1)),
            ("iterations", f"{level.solver_iterations}", (0.7, 0.78, 0.9, 1)),
        ]
        end = overlay.panel(28, end, rows, title="SIMULATION")

        if self.recorder.recording or self.recorder.transcoding:
            stats = self.recorder.stats
            rows = [
                ("state", "REC" if stats.recording else "transcoding",
                 (1.0, 0.35, 0.35, 1)),
                ("frames", f"{stats.frames} (+{stats.dropped} dropped)",
                 (0.9, 0.9, 0.9, 1)),
                ("elapsed", f"{stats.seconds:.1f}s", (0.9, 0.9, 0.9, 1)),
            ]
            end = overlay.panel(28, end, rows, title="RECORDING")

        overlay.hint([
            self._status,
            "",
            "space snap   C clap   R record   H hide UI   T tuning",
            "F raw/smoothed   [ ] quality   L ladder   esc quit",
        ], 28, end + 4)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def warmup(self, verbose: bool = True) -> float:
        """Pay the JIT and cache-load cost before the user can trigger it.

        numba caches compiled code to disk, but the first call in a process
        still has to load and link it -- roughly 87KB of machine code for
        the clipper alone. Left until the first snap, that lands as a
        multi-second freeze on the single most important frame in the app.

        Warmed at the real shard count and canvas size, not a token one.
        Qhull carries its own first-call cost that scales with the point
        count: warming with 64 cells left the first 800-cell fracture
        taking 32ms, of which 23ms was Voronoi construction alone.
        """
        start = time.perf_counter()
        count = min(self.options.shard_count, config.SHARD_COUNT_TIERS[0])
        prepared = fracture(
            self.options.width,
            self.options.height,
            (self.options.width * 0.5, self.options.height * 0.5),
            count,
            bevel=config.BEVEL_WIDTH,
        )
        scratch = PhysicsWorld(count + 32, self.options.width, self.options.height)
        scratch.load(prepared)
        scratch.set_capsules(
            np.array([[10.0, 10.0, 40.0, 40.0]], np.float64),
            np.array([12.0], np.float64),
            np.array([[100.0, 0.0]], np.float64),
        )
        scratch.explode((self.options.width * 0.5, self.options.height * 0.5))
        scratch.step(1.0 / 60.0)
        # The warmup fracture is production geometry, not throwaway work.
        # It is centred close enough to the keyboard snap (and most initial
        # hand snaps) to satisfy the same prediction tolerance as a normal
        # speculative result.
        self.prewarmer.prime(prepared)
        elapsed = time.perf_counter() - start
        if verbose:
            print(f"[warmup] solver and fracture ready in {elapsed:.2f}s")
        return elapsed

    def run(self, frames: int = 0) -> int:
        self.warmup(verbose=not self.options.headless or bool(frames))
        self.running = True
        try:
            while self.running:
                if self.display.should_close:
                    break
                self.step()
                if frames and self.frame_index >= frames:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            if frames:
                self.report()
            self.shutdown()
        return 0

    def report(self) -> None:
        """What the run actually cost. Printed after a bounded run."""
        profiler = self.profiler
        print(f"\n{self.frame_index} frames  "
              f"{profiler.fps:.1f} fps  "
              f"frame {profiler.rolling_ms:.2f} ms rolling, "
              f"p95 {profiler.percentile(95):.2f} ms, "
              f"worst {profiler.worst_ms:.2f} ms")
        for name, ms in sorted(profiler.sections.items()):
            print(f"    cpu  {name:12} {ms:6.2f} ms")
        for name, ms in sorted(self.gpu.results().items()):
            print(f"    gpu  {name:12} {ms:6.2f} ms")
        print(f"    tracking {self.tracker.stats.rate_hz:.1f} Hz "
              f"detect {self.tracker.stats.detect_ms:.2f} ms "
              f"[{self.tracker.stats.delegate}]")
        print(f"    physics  {self.world.stats.bodies} shards, "
              f"{self.world.stats.awake} awake, "
              f"{self.world.stats.contacts} contacts")
        print(f"    quality  {self.ladder.level.name} "
              f"(down {self.ladder.steps_down}, up {self.ladder.steps_up})")

    def shutdown(self) -> None:
        self.running = False
        if self.recorder.recording:
            self.recorder.stop()
        self.prewarmer.stop()
        if self.silhouette is not None:
            self.silhouette.stop()
        self.tracker.stop()
        self.source.stop()
        for obj in (self.shards, self.video, self.void, self.hud_shapes,
                    self.hud_text, self.shapes, self.text, self.recorder):
            try:
                obj.release()
            except Exception:
                pass
        self.display.release()
