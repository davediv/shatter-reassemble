"""Frame timing and adaptive-quality regression tests."""

import unittest
from unittest import mock

from shatter.profiler import FrameProfiler, QualityLadder


class TestFrameProfiler(unittest.TestCase):
    def test_reset_starts_a_fresh_timing_window(self):
        profiler = FrameProfiler(window=8)
        profiler._filled = 8
        profiler.frame_ms = 240.0
        profiler.fps = 4.0

        with mock.patch("shatter.profiler.time.perf_counter", return_value=42.0):
            profiler.reset()

        self.assertEqual(profiler._filled, 0)
        self.assertEqual(profiler._last, 42.0)
        self.assertEqual(profiler._frame_start, 42.0)
        self.assertEqual(profiler.frame_ms, 16.6)
        self.assertEqual(profiler.fps, 60.0)


class TestVsyncAwareQuality(unittest.TestCase):
    def test_healthy_60hz_interval_does_not_step_down(self):
        ladder = QualityLadder(refresh_hz=60.0)

        self.assertGreater(ladder.step_down_ms, 1000.0 / 60.0)
        self.assertFalse(ladder.update(1000.0 / 60.0, now=2.0))
        self.assertEqual(ladder.index, 0)

    def test_missed_60hz_refresh_steps_down(self):
        ladder = QualityLadder(refresh_hz=60.0)

        self.assertTrue(ladder.update(1000.0 / 30.0, now=2.0))
        self.assertEqual(ladder.index, 1)


if __name__ == "__main__":
    unittest.main()
