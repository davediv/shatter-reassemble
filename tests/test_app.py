"""End-to-end smoke tests.

Every other test file checks one module against fabricated input. This one
runs the actual application -- three threads, real GL, real MediaPipe --
headless against the synthetic source, and drives it through every phase.
It is the only test that would catch a wiring mistake between two modules
that are each individually correct.

Deliberately small (120 shards) and short: it is here to catch breakage,
not to measure anything.
"""

import time
import unittest
from unittest import mock

import numpy as np

from shatter import config
from shatter.app import Phase, ShatterApp


def drive_until_idle(app, timeout=15.0):
    """Step until reassembly finishes, bounded by wall clock.

    Bounded by time rather than by a frame count on purpose: reassembly is
    scheduled against wall-clock time, and an uncapped headless loop can
    exceed 600fps, so a frame-count guard exits mid-animation on a fast
    machine and fails a test the app passed.
    """
    deadline = time.perf_counter() + timeout
    while app.phase is Phase.REASSEMBLING and time.perf_counter() < deadline:
        app.step()
    return app.phase


def make_app(shards=120, **overrides):
    options = config.RuntimeOptions(
        width=960, height=540, source="synthetic", headless=True,
        vsync=False, show_debug=True, shard_count=shards,
        ladder_enabled=False, segmentation=False, **overrides,
    )
    app = ShatterApp(options)
    app.warmup(verbose=False)
    return app


class TestLifecycle(unittest.TestCase):
    def test_runs_idle_and_stays_idle(self):
        app = make_app()
        try:
            for _ in range(20):
                app.step()
            self.assertIs(app.phase, Phase.IDLE)
            self.assertEqual(app.world.count, 0)
            self.assertGreater(app.frame_index, 0)
        finally:
            app.shutdown()

    def test_full_shatter_and_reassemble_cycle(self):
        app = make_app()
        try:
            for _ in range(10):
                app.step()
            app._shatter((600.0, 200.0))
            self.assertIs(app.phase, Phase.SHATTERED)
            self.assertEqual(app.prewarmer.hits, 1)
            self.assertEqual(app.prewarmer.misses, 0)
            self.assertGreater(app.world.count, 50)
            self.assertGreater(app.shards.vertex_count, 1000)

            for _ in range(60):
                app.step()
            # Gravity has done something.
            self.assertGreater(app.world.py[: app.world.count].mean(), 200.0)

            app._reassemble()
            self.assertIs(app.phase, Phase.REASSEMBLING)
            self.assertIs(drive_until_idle(app), Phase.IDLE)
            self.assertEqual(app.reassembly.rest_error(), 0.0)
            self.assertEqual(app.shards.vertex_count, 0)
        finally:
            app.shutdown()

    def test_a_second_shatter_works_after_reassembly(self):
        # State has to be genuinely reset, not just visually cleared.
        app = make_app()
        try:
            for cycle in range(2):
                for _ in range(5):
                    app.step()
                app._shatter((400.0 + cycle * 100, 200.0))
                self.assertIs(app.phase, Phase.SHATTERED)
                for _ in range(20):
                    app.step()
                app._reassemble()
                self.assertIs(drive_until_idle(app), Phase.IDLE, f"cycle {cycle}")
        finally:
            app.shutdown()

    def test_clap_without_a_shatter_is_ignored(self):
        app = make_app()
        try:
            app.step()
            app._reassemble()
            self.assertIs(app.phase, Phase.IDLE)
        finally:
            app.shutdown()

    def test_snap_while_shattered_is_ignored(self):
        app = make_app()
        try:
            app.step()
            app._shatter((400.0, 200.0))
            first = app.world.count
            app._shatter((100.0, 100.0))
            self.assertEqual(app.world.count, first)
        finally:
            app.shutdown()


class TestRendering(unittest.TestCase):
    def test_the_canvas_is_not_blank_in_any_phase(self):
        app = make_app()
        try:
            for _ in range(20):
                app.step()
            idle = app.display.read_canvas()
            self.assertGreater(idle.mean(), 5.0, "idle frame is black")

            app._shatter((600.0, 200.0))
            for _ in range(40):
                app.step()
            broken = app.display.read_canvas()
            self.assertGreater(broken.mean(), 2.0, "shattered frame is black")
            # The void must genuinely be darker than the intact feed.
            self.assertLess(broken.mean(), idle.mean())

            # The expensive outline pass is retained until a new mask lands.
            with mock.patch.object(app.void, "render_scene",
                                   wraps=app.void.render_scene) as render:
                app.step()
                render.assert_not_called()
                app.void.upload_mask(np.zeros((36, 64), np.uint8))
                app.step()
                render.assert_called_once()
        finally:
            app.shutdown()

    def test_hiding_the_debug_ui_changes_the_frame(self):
        app = make_app()
        try:
            for _ in range(12):
                app.step()
            with_ui = app.display.read_canvas().astype(np.int16)
            app.overlay.visible = False
            for _ in range(3):
                app.step()
            without = app.display.read_canvas().astype(np.int16)
            self.assertGreater(np.abs(with_ui - without).max(), 20)
        finally:
            app.shutdown()


class TestQualityLadder(unittest.TestCase):
    def test_stepping_down_reduces_the_next_shard_count(self):
        app = make_app(shards=800)
        try:
            app.step()
            app.ladder.force(2, 0.0)          # the 350-shard rung
            app._shatter((400.0, 200.0))
            self.assertLessEqual(app.world.count, 360)
        finally:
            app.shutdown()

    def test_shard_ceiling_is_never_raised_above_the_request(self):
        app = make_app(shards=150)
        try:
            app.step()
            app.ladder.force(0, 0.0)          # full quality: 800
            app._shatter((400.0, 200.0))
            self.assertLessEqual(app.world.count, 155)
        finally:
            app.shutdown()


if __name__ == "__main__":
    unittest.main()
