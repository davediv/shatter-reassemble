"""Gesture detection tests.

Snap detection is the finicky one, and the failures that matter are false
positives: a waving hand, a held pinch, a stray fast finger. Those are
tedious to reproduce in front of a webcam and trivial to synthesise, so
they are pinned down here. The tuning mode exists to dial the thresholds
against a real hand; these tests exist to make sure the *logic* around
those thresholds is right.

Gestures are built in canvas pixels and pushed back through CoverFit into
normalised coordinates, so every test exercises the real path: mirroring,
slot identity, raw velocity, span normalisation.
"""

import unittest

import numpy as np

from shatter import config
from shatter.gestures import CLAP, SNAP, GestureRecognizer
from shatter.tracking import LandmarkPipeline
from shatter.viewport import CoverFit

FIT = CoverFit(1280, 720, 1920, 1080)
DT = 1.0 / 60.0
SPAN = 100.0        # px, wrist -> middle MCP


def hand_px(wrist, thumb_tip, middle_tip, spread=1.0):
    """A hand specified in canvas pixels, returned in normalised coords.

    Fingertip placement follows real hand proportions -- an extended index
    tip sits about 2.0 wrist-to-MCP spans from the wrist, the pinky about
    1.7 -- because the open-palm threshold is calibrated against those
    ratios. ``spread`` curls the fingers toward the wrist.
    """
    px = np.zeros((config.NUM_LANDMARKS, 2), np.float32)
    wrist = np.asarray(wrist, np.float32)
    px[:] = wrist
    px[config.WRIST] = wrist
    px[config.MIDDLE_MCP] = wrist + (0.0, -SPAN)
    px[config.THUMB_TIP] = thumb_tip
    px[config.MIDDLE_TIP] = middle_tip
    px[config.INDEX_TIP] = wrist + (-0.45 * SPAN, -2.00 * SPAN * spread)
    px[config.RING_TIP] = wrist + (0.40 * SPAN, -1.90 * SPAN * spread)
    px[config.PINKY_TIP] = wrist + (0.75 * SPAN, -1.55 * SPAN * spread)
    return FIT.canvas_to_uv(px)


def open_hand(wrist):
    """A fully extended palm: every tip out at its anatomical reach."""
    wrist = np.asarray(wrist, np.float32)
    return hand_px(wrist, wrist + (-1.10 * SPAN, -0.90 * SPAN),
                   wrist + (0.0, -2.20 * SPAN), spread=1.0)


def closed_hand(wrist):
    """A fist: every tip curled back toward the wrist."""
    wrist = np.asarray(wrist, np.float32)
    return hand_px(wrist, wrist + (0.0, -0.40 * SPAN),
                   wrist + (0.0, -0.40 * SPAN), spread=0.35)


class GestureHarness:
    """Feeds synthetic hands through the real pipeline and recognizer."""

    def __init__(self, tunables=None):
        self.pipe = LandmarkPipeline(FIT)
        self.rec = GestureRecognizer(tunables or config.Tunables())
        self.t = 0.0
        self.fired = []

    def step(self, *hands, labels=None):
        stacked = (np.stack(hands).astype(np.float32) if hands
                   else np.zeros((0, 21, 2), np.float32))
        labels = labels or ["Left", "Right"][: len(hands)]
        frame = self.pipe.update(stacked, labels, self.t, DT)
        self.rec.update(frame)
        self.fired.extend(self.rec.poll())
        self.t += DT
        return frame

    def kinds(self):
        return [e.kind for e in self.fired]


class TestSnap(unittest.TestCase):
    def test_pinch_then_flick_fires_once(self):
        h = GestureHarness()
        wrist = (900.0, 600.0)
        pinched = (900.0, 500.0)
        for _ in range(12):                       # held pinch, hand still
            h.step(hand_px(wrist, pinched, pinched))
        self.assertEqual(h.kinds(), [])
        # Middle tip flicks 30px in one frame: 1800px/s = 18 spans/s.
        h.step(hand_px(wrist, pinched, (930.0, 520.0)))
        self.assertEqual(h.kinds(), [SNAP])
        self.assertGreater(h.fired[0].strength, config.Tunables().snap_velocity_threshold)

    def test_a_waving_hand_does_not_snap(self):
        # The whole hand translates fast with the fingers pinched. Absolute
        # fingertip velocity is far over threshold; wrist-relative velocity
        # is nil, which is the entire point of measuring it that way.
        h = GestureHarness()
        for i in range(30):
            x = 500.0 + i * 34.0                  # 2040 px/s = 20 spans/s
            h.step(hand_px((x, 600.0), (x, 500.0), (x, 500.0)))
        self.assertEqual(h.kinds(), [])

    def test_holding_a_pinch_never_fires(self):
        h = GestureHarness()
        for _ in range(120):
            h.step(hand_px((900.0, 600.0), (900.0, 500.0), (900.0, 500.0)))
        self.assertEqual(h.kinds(), [])

    def test_flick_without_a_pinch_does_not_fire(self):
        h = GestureHarness()
        wrist = (900.0, 600.0)
        far = (700.0, 500.0)                      # thumb well away: 2.0 spans
        for _ in range(10):
            h.step(hand_px(wrist, far, (900.0, 500.0)))
        h.step(hand_px(wrist, far, (930.0, 520.0)))
        self.assertEqual(h.kinds(), [])

    def test_window_runs_from_the_most_recent_pinched_frame(self):
        # People hold a pinch while they aim, then flick. Anchoring the
        # window to the start of the pinch would only fire for someone who
        # snaps the instant their fingers touch.
        h = GestureHarness()
        wrist, pinched = (900.0, 600.0), (900.0, 500.0)
        for _ in range(90):                       # 1.5s of held pinch
            h.step(hand_px(wrist, pinched, pinched))
        h.step(hand_px(wrist, pinched, (930.0, 520.0)))
        self.assertEqual(h.kinds(), [SNAP])

    def test_expired_window_rejects_a_late_flick(self):
        h = GestureHarness()
        wrist = (900.0, 600.0)
        for _ in range(10):
            h.step(hand_px(wrist, (900.0, 500.0), (900.0, 500.0)))
        # Fingers separate and stay apart well past the 120ms window.
        for _ in range(20):
            h.step(hand_px(wrist, (700.0, 500.0), (905.0, 500.0)))
        h.step(hand_px(wrist, (700.0, 500.0), (935.0, 520.0)))
        self.assertEqual(h.kinds(), [])

    def test_lockout_prevents_a_double_fire(self):
        h = GestureHarness()
        wrist, pinched = (900.0, 600.0), (900.0, 500.0)
        for _ in range(10):
            h.step(hand_px(wrist, pinched, pinched))
        for _ in range(6):                        # sustained violent flicking
            h.step(hand_px(wrist, pinched, (930.0, 520.0)))
            h.step(hand_px(wrist, pinched, pinched))
        self.assertEqual(h.kinds().count(SNAP), 1)

    def test_landmark_jitter_does_not_snap(self):
        """The failure the travel guard exists for.

        A still, pinched hand under realistic landmark noise produces
        instantaneous velocity spikes that come within ~1.3x of the flick
        threshold -- close enough to fire on an unlucky frame. Jitter is a
        zero-mean walk though, so it accumulates almost no net
        wrist-relative displacement, and the travel guard rejects it with
        roughly 6x margin. Found by watching the tuning traces.
        """
        rng = np.random.default_rng(3)
        h = GestureHarness()
        for _ in range(400):
            wrist = (900.0 + rng.normal(0, 2.0), 600.0 + rng.normal(0, 2.0))
            pinched = (wrist[0] + rng.normal(0, 3.0), wrist[1] - 100.0 + rng.normal(0, 3.0))
            h.step(hand_px(wrist, pinched, pinched))
        self.assertEqual(h.kinds(), [], "landmark jitter fired a snap")

    def test_a_real_snap_still_fires_through_jitter(self):
        # The same noise, with an actual flick on top of it.
        rng = np.random.default_rng(11)
        h = GestureHarness()
        wrist = (900.0, 600.0)
        for _ in range(40):
            pinched = (900.0 + rng.normal(0, 3.0), 500.0 + rng.normal(0, 3.0))
            h.step(hand_px(wrist, pinched, pinched))
        h.step(hand_px(wrist, (900.0, 500.0), (938.0, 528.0)))
        self.assertEqual(h.kinds(), [SNAP])

    def test_a_landmark_teleport_does_not_snap(self):
        """A tracking discontinuity is not the most emphatic snap ever.

        When MediaPipe re-associates a hand or recovers from a brief loss,
        a landmark can jump hundreds of pixels in one frame. That clears
        both the velocity and the travel gates comfortably. The upper
        sanity bound is what rejects it: no real fingertip moves at 60
        hand spans per second.
        """
        h = GestureHarness()
        wrist, pinched = (900.0, 600.0), (900.0, 500.0)
        for _ in range(15):
            h.step(hand_px(wrist, pinched, pinched))
        far = (600.0, 540.0)
        h.step(hand_px(far, far, far))          # 300px teleport
        self.assertEqual(h.kinds(), [])

    def test_snap_is_distance_invariant(self):
        # The same gesture performed at half scale -- twice as far from the
        # lens -- must fire identically, because every threshold is divided
        # by hand span.
        import shatter.tracking as tracking

        for scale in (1.0, 0.5, 2.0):
            with self.subTest(scale=scale):
                original = globals()["SPAN"]
                try:
                    globals()["SPAN"] = 100.0 * scale
                    h = GestureHarness()
                    wrist = (900.0, 600.0)
                    pinched = (900.0, 600.0 - 100.0 * scale)
                    for _ in range(15):
                        h.step(hand_px(wrist, pinched, pinched))
                    flicked = (900.0 + 30.0 * scale, pinched[1] + 20.0 * scale)
                    h.step(hand_px(wrist, pinched, flicked))
                    self.assertEqual(h.kinds(), [SNAP], f"scale {scale}")
                finally:
                    globals()["SPAN"] = original


class TestClap(unittest.TestCase):
    def _converge(self, harness, step_px=30.0, frames=23, start=700.0):
        for i in range(frames):
            gap = start - i * step_px
            left = (960.0 - gap * 0.5, 540.0)
            right = (960.0 + gap * 0.5, 540.0)
            harness.step(
                hand_px(left, left, left), hand_px(right, right, right),
                labels=["Left", "Right"],
            )

    def test_converging_palms_clap(self):
        h = GestureHarness()
        self._converge(h)
        self.assertIn(CLAP, h.kinds())

    def test_hands_held_close_but_still_do_not_clap(self):
        h = GestureHarness()
        for _ in range(60):
            left, right = (940.0, 540.0), (980.0, 540.0)
            h.step(hand_px(left, left, left), hand_px(right, right, right),
                   labels=["Left", "Right"])
        self.assertEqual(h.kinds(), [])

    def test_one_hand_cannot_clap(self):
        h = GestureHarness()
        for i in range(30):
            p = (960.0 - i * 20.0, 540.0)
            h.step(hand_px(p, p, p))
        self.assertNotIn(CLAP, h.kinds())

    def test_clap_lockout_holds(self):
        h = GestureHarness()
        self._converge(h)
        # Immediately pull apart and slam together again inside the lockout.
        self._converge(h)
        self.assertEqual(h.kinds().count(CLAP), 1)


class TestOpenPalm(unittest.TestCase):
    def test_open_and_closed_hands_are_distinguished(self):
        h = GestureHarness()
        h.step(open_hand((900.0, 600.0)))
        opened = h.rec.signals()
        self.assertTrue(opened.open_palm[0], f"extension {opened.extension[0]:.2f}")

        h2 = GestureHarness()
        h2.step(closed_hand((900.0, 600.0)))
        closed = h2.rec.signals()
        self.assertFalse(closed.open_palm[0], f"extension {closed.extension[0]:.2f}")
        # The two must be clearly separated, not marginally so, or the
        # threshold would be riding on noise.
        self.assertGreater(opened.extension[0] - closed.extension[0], 0.8)


class TestSignals(unittest.TestCase):
    def test_traces_and_signals_are_populated(self):
        h = GestureHarness()
        wrist = (900.0, 600.0)
        for _ in range(20):
            h.step(hand_px(wrist, (900.0, 500.0), (900.0, 500.0)))
        signals = h.rec.signals()
        self.assertLess(signals.pinch[0], 0.25)
        self.assertTrue(signals.armed[0])
        self.assertEqual(signals.hands, 1)
        traces = h.rec.snapshot_traces()
        self.assertEqual(len(traces["pinch"][0]), 20)
        self.assertTrue(np.all(traces["pinch"][0] < 0.25))


if __name__ == "__main__":
    unittest.main()
