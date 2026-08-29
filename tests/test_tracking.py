"""Tests for the landmark pipeline.

The spec says to verify mirroring with a skeleton overlay before building
anything else. A visual check catches a flipped image but not a slot swap
or a phantom velocity spike, and neither can be reproduced on demand in
front of a webcam -- so the meaning-bearing logic is tested here against
fabricated landmarks, and the overlay is left to confirm the wiring.
"""

import unittest

import numpy as np

from shatter import config
from shatter.tracking import HandSpanCalibrator, LandmarkPipeline
from shatter.viewport import CoverFit

CANVAS = (1920, 1080)
FIT = CoverFit(1280, 720, *CANVAS)


def make_hand(cx: float, cy: float, scale: float = 0.12) -> np.ndarray:
    """A plausible 21-landmark hand in normalised camera coordinates.

    Only the landmarks the pipeline actually reads need to be
    geometrically sensible: 0 (wrist) and 9 (middle MCP) set the span.
    """
    hand = np.zeros((config.NUM_LANDMARKS, 2), np.float32)
    hand[config.WRIST] = (cx, cy)
    hand[config.MIDDLE_MCP] = (cx, cy - scale)
    for i in range(1, config.NUM_LANDMARKS):
        if i == config.MIDDLE_MCP:
            continue
        angle = i * 0.3
        hand[i] = (cx + scale * 0.5 * np.cos(angle), cy - scale * 0.5 * np.sin(angle))
    return hand


def stack(*hands: np.ndarray) -> np.ndarray:
    return np.stack(hands).astype(np.float32) if hands else np.zeros((0, 21, 2), np.float32)


class TestMirroring(unittest.TestCase):
    def test_x_is_mirrored_exactly_once(self):
        pipe = LandmarkPipeline(FIT)
        frame = pipe.update(stack(make_hand(0.2, 0.5)), ["Left"], 0.0, 1 / 30)
        # Normalised x=0.2 in camera space must land at (1-0.2)*1920.
        self.assertAlmostEqual(float(frame.raw[0, config.WRIST, 0]), 1536.0, places=2)
        # y is never mirrored.
        self.assertAlmostEqual(float(frame.raw[0, config.WRIST, 1]), 540.0, places=2)

    def test_handedness_label_is_inverted_for_the_mirrored_view(self):
        # MediaPipe labels as if the input were mirrored; we feed it
        # camera-native frames, so the user-facing label must be flipped.
        pipe = LandmarkPipeline(FIT)
        frame = pipe.update(stack(make_hand(0.3, 0.5)), ["Left"], 0.0, 1 / 30)
        self.assertEqual(frame.handedness[0], "Right")


class TestVelocityChannel(unittest.TestCase):
    def test_no_velocity_on_the_frame_a_hand_appears(self):
        # A phantom spike here would fire a snap the instant a hand enters.
        pipe = LandmarkPipeline(FIT)
        frame = pipe.update(stack(make_hand(0.5, 0.5)), ["Left"], 0.0, 1 / 30)
        self.assertTrue(frame.present[0])
        self.assertEqual(float(np.abs(frame.velocity).max()), 0.0)

    def test_velocity_matches_displacement_over_dt(self):
        pipe = LandmarkPipeline(FIT)
        dt = 1 / 30
        pipe.update(stack(make_hand(0.5, 0.5)), ["Left"], 0.0, dt)
        frame = pipe.update(stack(make_hand(0.6, 0.5)), ["Left"], dt, dt)
        # 0.1 normalised = 192 canvas px, mirrored so it moves -x.
        expected = -0.1 * 1920 / dt
        self.assertAlmostEqual(
            float(frame.velocity[0, config.WRIST, 0]), expected, delta=1.0
        )

    def test_velocity_is_computed_from_raw_not_smoothed(self):
        # Raw must reproduce the input exactly; smoothing lags behind it.
        pipe = LandmarkPipeline(FIT)
        dt = 1 / 30
        pipe.update(stack(make_hand(0.5, 0.5)), ["Left"], 0.0, dt)
        frame = pipe.update(stack(make_hand(0.7, 0.5)), ["Left"], dt, dt)
        raw_x = float(frame.raw[0, config.WRIST, 0])
        self.assertAlmostEqual(raw_x, (1 - 0.7) * 1920, places=2)
        self.assertNotAlmostEqual(float(frame.smooth[0, config.WRIST, 0]), raw_x, places=1)


class TestSlotIdentity(unittest.TestCase):
    def test_slots_follow_position_not_a_flickering_label(self):
        pipe = LandmarkPipeline(FIT)
        dt = 1 / 30
        left, right = make_hand(0.3, 0.5), make_hand(0.7, 0.5)
        first = pipe.update(stack(left, right), ["Left", "Right"], 0.0, dt)
        slot_of_left = 0 if first.raw[0, config.WRIST, 0] > 960 else 1

        # Same positions, but MediaPipe flips both labels this frame.
        second = pipe.update(stack(left, right), ["Right", "Left"], dt, dt)
        still = 0 if second.raw[0, config.WRIST, 0] > 960 else 1
        self.assertEqual(slot_of_left, still, "a label flicker swapped the slots")
        # And no phantom velocity from an identity swap.
        self.assertLess(float(np.abs(second.velocity).max()), 1.0)

    def test_two_hands_keep_their_slots_while_approaching(self):
        pipe = LandmarkPipeline(FIT)
        dt = 1 / 30
        prev = None
        for i in range(12):
            a, b = 0.30 + i * 0.012, 0.70 - i * 0.012
            frame = pipe.update(
                stack(make_hand(a, 0.5), make_hand(b, 0.5)), ["Left", "Right"], i * dt, dt
            )
            if prev is not None:
                # Continuity: no slot may teleport between frames.
                jump = np.abs(frame.raw[:, config.WRIST] - prev).max()
                self.assertLess(jump, 60.0, f"slot jumped at step {i}")
            prev = frame.raw[:, config.WRIST].copy()

    def test_hand_leaving_and_returning_elsewhere_does_not_smear(self):
        pipe = LandmarkPipeline(FIT)
        dt = 1 / 30
        for i in range(20):
            pipe.update(stack(make_hand(0.2, 0.3)), ["Left"], i * dt, dt)
        pipe.update(np.zeros((0, 21, 2), np.float32), [], 20 * dt, dt)   # gone
        frame = pipe.update(stack(make_hand(0.8, 0.8)), ["Left"], 21 * dt, dt)
        # The smoothed pose must snap to the new hand, not lag across the
        # screen from where the old one was.
        self.assertAlmostEqual(
            float(frame.smooth[0, config.WRIST, 0]),
            float(frame.raw[0, config.WRIST, 0]),
            places=2,
        )
        self.assertEqual(float(np.abs(frame.velocity).max()), 0.0)


class TestHandSpan(unittest.TestCase):
    def test_span_is_wrist_to_middle_mcp_in_pixels(self):
        pipe = LandmarkPipeline(FIT)
        # scale 0.12 normalised in y -> 0.12 * 1080 canvas px
        frame = pipe.update(stack(make_hand(0.5, 0.5, 0.12)), ["Left"], 0.0, 1 / 30)
        self.assertAlmostEqual(float(frame.span[0]), 0.12 * 1080, delta=0.5)

    def test_median_rejects_a_single_blown_landmark(self):
        cal = HandSpanCalibrator(window=9)
        wrist = np.array([0.0, 0.0], np.float32)
        for _ in range(8):
            cal.update(0, wrist, np.array([0.0, 100.0], np.float32))
        spiked = cal.update(0, wrist, np.array([0.0, 5000.0], np.float32))
        self.assertAlmostEqual(spiked, 100.0, delta=0.01)

    def test_tiny_spans_do_not_poison_the_window(self):
        cal = HandSpanCalibrator(window=8)
        wrist = np.array([0.0, 0.0], np.float32)
        for _ in range(8):
            cal.update(0, wrist, np.array([0.0, 120.0], np.float32))
        # Below HAND_SPAN_MIN_PIXELS: too far away to be trustworthy.
        held = cal.update(0, wrist, np.array([0.0, 3.0], np.float32))
        self.assertAlmostEqual(held, 120.0, delta=0.01)

    def test_span_tracks_a_user_moving_closer(self):
        # A one-shot calibration would freeze at the first distance; the
        # rolling median has to follow the user in.
        pipe = LandmarkPipeline(FIT)
        dt = 1 / 30
        for i in range(config.HAND_SPAN_CALIBRATION_FRAMES):
            pipe.update(stack(make_hand(0.5, 0.5, 0.10)), ["Left"], i * dt, dt)
        near = 0.0
        for i in range(config.HAND_SPAN_CALIBRATION_FRAMES):
            frame = pipe.update(stack(make_hand(0.5, 0.5, 0.20)), ["Left"], (60 + i) * dt, dt)
            near = float(frame.span[0])
        self.assertAlmostEqual(near, 0.20 * 1080, delta=1.0)


if __name__ == "__main__":
    unittest.main()
