"""Hand landmarks: detection, identity, mirroring, smoothing, calibration.

Runs on its own thread so that inference never sits inside the render
loop's 16.6ms budget. Consumers take the newest published HandFrame and
carry on; if tracking is running at 30Hz and rendering at 60Hz, the
renderer simply sees each tracking result twice, which is correct -- it is
better to draw a 16ms-old hand than to miss a vsync waiting for a new one.

Three things happen here that happen nowhere else:

*Mirroring.* Landmark x becomes (1-x) exactly once, inside CoverFit. The
frame handed to MediaPipe is the raw un-mirrored one, because the model
expects camera-native input.

*Identity.* Slots are assigned by proximity to the previous frame's
wrists, not by MediaPipe's handedness label. Labels flicker; a flicker
would swap two hands' filter state and snap the skeleton across the
screen.

*Calibration.* Hand span -- wrist (0) to middle MCP (9) in pixels -- is
tracked as a rolling median over 60 frames. Every gesture threshold is
divided by it, which is what makes the app behave the same whether the
user is leaning into the lens or standing back.

Two parallel channels leave this module. ``smooth`` is One Euro filtered
and is what you draw and what the stir colliders follow. ``raw`` and
``velocity`` are unfiltered and are what gesture detection reads. Never
cross them: smoothing the detection channel deletes the snap.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from . import config
from .camera import Frame, FrameSource
from .delegate import DelegateChoice, resolve
from .filters import OneEuroFilterND
from .viewport import CoverFit

__all__ = ["HandFrame", "HandTracker", "TrackerStats"]

_NL = config.NUM_LANDMARKS
_SLOTS = config.NUM_HANDS

# A detection further than this from a slot's previous wrist is treated as
# a different hand rather than a fast movement. Expressed as a fraction of
# the canvas diagonal so it scales with resolution.
_REASSOCIATE_FRACTION = 0.35


@dataclass(frozen=True)
class HandFrame:
    """One tracking result, in canvas pixel space, ready to consume."""

    index: int
    timestamp: float             # perf_counter at capture
    dt: float                    # since previous tracking frame
    present: np.ndarray          # (2,) bool
    raw: np.ndarray              # (2, 21, 2) canvas px, mirrored, unfiltered
    smooth: np.ndarray           # (2, 21, 2) canvas px, One Euro filtered
    velocity: np.ndarray         # (2, 21, 2) canvas px/s, from raw
    span: np.ndarray             # (2,) px, calibrated wrist->middle MCP
    handedness: tuple[str, ...]  # as the user sees themselves, mirrored
    latency: float               # capture -> publish, seconds

    @property
    def hand_count(self) -> int:
        return int(self.present.sum())


@dataclass
class TrackerStats:
    """Where the tracking budget actually goes. Surfaced on the HUD."""

    convert_ms: float = 0.0
    detect_ms: float = 0.0
    total_ms: float = 0.0
    rate_hz: float = 0.0
    delegate: str = "?"

    def blend(self, other: "TrackerStats", a: float = 0.1) -> None:
        """Exponential average, so the HUD reads steady instead of noisy."""
        self.convert_ms += (other.convert_ms - self.convert_ms) * a
        self.detect_ms += (other.detect_ms - self.detect_ms) * a
        self.total_ms += (other.total_ms - self.total_ms) * a


class HandSpanCalibrator:
    """Rolling-median hand span per slot.

    The spec calls for calibrating over 60 frames. A *rolling* 60-frame
    median rather than a one-shot calibration, because users walk toward
    and away from the camera mid-session; freezing the first reading would
    quietly break distance invariance the moment they moved. The median
    (not the mean) is what makes a single blown landmark harmless.
    """

    def __init__(self, window: int = config.HAND_SPAN_CALIBRATION_FRAMES) -> None:
        self._window = window
        self._ring = np.zeros((_SLOTS, window), np.float32)
        self._count = np.zeros(_SLOTS, np.int32)
        self._head = np.zeros(_SLOTS, np.int32)
        self._value = np.full(_SLOTS, config.HAND_SPAN_DEFAULT_PIXELS, np.float32)

    def reset(self, slot: int) -> None:
        self._count[slot] = 0
        self._head[slot] = 0
        self._value[slot] = config.HAND_SPAN_DEFAULT_PIXELS

    def update(self, slot: int, wrist: np.ndarray, middle_mcp: np.ndarray) -> float:
        span = float(np.hypot(*(middle_mcp - wrist)))
        if span < config.HAND_SPAN_MIN_PIXELS:
            # Too small to be a real hand at a usable distance; keep the
            # last good value rather than poisoning the window.
            return float(self._value[slot])
        h = int(self._head[slot])
        self._ring[slot, h] = span
        self._head[slot] = (h + 1) % self._window
        if self._count[slot] < self._window:
            self._count[slot] += 1
        n = int(self._count[slot])
        self._value[slot] = np.median(self._ring[slot, :n])
        return float(self._value[slot])

    @property
    def values(self) -> np.ndarray:
        return self._value

    def is_calibrated(self, slot: int) -> bool:
        return bool(self._count[slot] >= self._window)


class LandmarkPipeline:
    """Normalised landmarks in, HandFrame out. Pure, synchronous, testable.

    Everything that defines the *meaning* of a tracking frame lives here --
    mirroring, slot identity, the raw/smoothed split, span calibration --
    deliberately separated from MediaPipe and from threading so it can be
    tested against fabricated input instead of against a real hand in real
    lighting.
    """

    def __init__(self, fit: CoverFit) -> None:
        self._fit = fit
        self._filter = OneEuroFilterND(
            (_SLOTS, _NL, 2),
            config.ONE_EURO_MIN_CUTOFF,
            config.ONE_EURO_BETA,
            config.ONE_EURO_D_CUTOFF,
        )
        self._spans = HandSpanCalibrator()
        self._prev_raw = np.zeros((_SLOTS, _NL, 2), np.float32)
        self._prev_present = np.zeros(_SLOTS, np.bool_)
        self._index = 0

    @property
    def spans(self) -> HandSpanCalibrator:
        return self._spans

    def _assign_slots(self, wrists: np.ndarray, labels: list) -> tuple[list, np.ndarray]:
        """Map each detection to a stable slot.

        Proximity first, because MediaPipe's handedness label flickers and
        a flicker would swap two hands' filter state, snapping the skeleton
        across the screen. The label only seeds identity when there is no
        history to match against.

        Returns the per-detection slot assignment plus a per-slot
        continuity flag. Continuity false means a *different* hand now
        occupies that slot and its filter history must be dropped.
        """
        continuous = np.zeros(_SLOTS, np.bool_)
        n = len(labels)
        if n == 0:
            return [], continuous

        limit = _REASSOCIATE_FRACTION * float(
            np.hypot(self._fit.canvas_width, self._fit.canvas_height)
        )
        # A finite penalty rather than infinity, so a frame where only one
        # hand has history still resolves instead of discarding the match
        # we do have.
        no_match = limit * 4.0

        cost = np.full((n, _SLOTS), no_match, np.float32)
        for d in range(n):
            for slot in range(_SLOTS):
                if not self._prev_present[slot]:
                    continue
                dist = float(np.hypot(*(wrists[d] - self._prev_raw[slot, config.WRIST])))
                if dist <= limit:
                    cost[d, slot] = dist

        if self._prev_present.any():
            best, best_cost = None, np.inf
            for a in range(_SLOTS):
                candidate = [a] if n == 1 else [a, 1 - a]
                total = cost[0, a] if n == 1 else cost[0, a] + cost[1, 1 - a]
                if total < best_cost:
                    best, best_cost = candidate, total
            assignment = best
        else:
            assignment = [-1] * n
            taken: set = set()
            for d, label in enumerate(labels):
                slot = 0 if label == "Left" else 1
                if slot not in taken:
                    assignment[d] = slot
                    taken.add(slot)
            for d in range(n):
                if assignment[d] < 0:
                    free = next(s for s in range(_SLOTS) if s not in taken)
                    assignment[d] = free
                    taken.add(free)

        for d, slot in enumerate(assignment):
            continuous[slot] = cost[d, slot] < no_match
        return assignment, continuous

    def update(
        self,
        norm: np.ndarray,
        labels: list,
        timestamp: float,
        dt: float,
        latency: float = 0.0,
    ) -> HandFrame:
        """``norm`` is (n, 21, 2) normalised camera coords, n in 0..2."""
        n = int(norm.shape[0])
        dt = max(float(dt), 1e-4)

        detected = (
            self._fit.landmarks_to_canvas(norm)
            if n else np.zeros((0, _NL, 2), np.float32)
        )
        wrists = detected[:, config.WRIST] if n else np.zeros((0, 2), np.float32)
        slots, continuous = self._assign_slots(wrists, labels)

        present = np.zeros(_SLOTS, np.bool_)
        # Absent slots keep their previous pose so the filter has something
        # continuous to chew on; `present` is what says whether to use it.
        raw = self._prev_raw.copy()
        display_labels = ["", ""]
        for d, slot in enumerate(slots):
            raw[slot] = detected[d]
            present[slot] = True
            # MediaPipe assigns handedness as if the input were mirrored.
            # We feed it camera-native frames, so invert the label to match
            # what the user sees of themselves in the mirrored display.
            display_labels[slot] = "Right" if labels[d] == "Left" else "Left"

        # Velocity comes from the RAW channel only. This is the signal snap
        # and clap detection live on, and any smoothing would flatten the
        # spike they key off. A slot that was empty last frame, or that a
        # different hand just took over, has no meaningful velocity --
        # giving it one would fire a snap the moment a hand entered frame.
        velocity = np.zeros((_SLOTS, _NL, 2), np.float32)
        fresh = present & self._prev_present & continuous
        if fresh.any():
            # NB fancy indexing returns copies, so this must be a scatter
            # assignment; an `out=velocity[fresh]` would write to a
            # temporary and silently leave velocity at zero.
            velocity[fresh] = (raw[fresh] - self._prev_raw[fresh]) / np.float32(dt)

        for slot in range(_SLOTS):
            if present[slot] and not continuous[slot]:
                # A different hand just took this slot over; its span and
                # filter history belong to the previous occupant.
                self._filter.reset(slot)
                self._spans.reset(slot)
            elif not present[slot] and self._prev_present[slot]:
                # Hand left frame: drop history so its return is not
                # smeared out of the old position.
                self._filter.reset(slot)
                self._spans.reset(slot)
            if present[slot]:
                self._spans.update(
                    slot, raw[slot, config.WRIST], raw[slot, config.MIDDLE_MCP]
                )

        smooth = self._filter.filter(raw, dt).copy()

        self._prev_raw[:] = raw
        self._prev_present[:] = present
        self._index += 1

        return HandFrame(
            index=self._index,
            timestamp=timestamp,
            dt=dt,
            present=present,
            raw=raw,
            smooth=smooth,
            velocity=velocity,
            span=self._spans.values.copy(),
            handedness=tuple(display_labels),
            latency=latency,
        )


class HandTracker:
    """MediaPipe HandLandmarker on a worker thread."""

    def __init__(
        self,
        source: FrameSource,
        fit: CoverFit,
        *,
        gpu_delegate: bool = True,
        on_frame: Optional[Callable[[HandFrame], None]] = None,
    ) -> None:
        self._source = source
        self._fit = fit
        self._on_frame = on_frame
        self.stats = TrackerStats()

        self.choice = resolve(config.HAND_MODEL, prefer_gpu=gpu_delegate)
        self.stats.delegate = f"{self.choice.delegate}/{self.choice.image_format}"
        self._landmarker = self._make_landmarker(self.choice)
        self._mp_format: object = None

        self.pipeline = LandmarkPipeline(fit)

        # Persistent state across tracking frames.
        self._norm = np.zeros((_SLOTS, _NL, 2), np.float32)
        self._rgb: Optional[np.ndarray] = None
        self._small: Optional[np.ndarray] = None

        self._latest: Optional[HandFrame] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_index = 0
        self._last_ts = 0.0
        self._mp_ts = 0
        self._t0 = 0.0
        self._rate_t0 = 0.0
        self._rate_n = 0
        self.error: Optional[BaseException] = None

    # -- construction -----------------------------------------------------

    def _make_landmarker(self, choice: DelegateChoice):
        """Build the landmarker on the delegate the probe already proved.

        No fallback chain here on purpose: an unsupported delegate aborts
        the process rather than raising, so the decision has to have been
        made out-of-process. See shatter/delegate.py.
        """
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(config.HAND_MODEL),
                delegate=getattr(mp_python.BaseOptions.Delegate, choice.delegate),
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config.NUM_HANDS,
            min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        return vision.HandLandmarker.create_from_options(options)

    # -- lifecycle --------------------------------------------------------

    def start(self) -> "HandTracker":
        if self._thread is None:
            self._t0 = time.perf_counter()
            self._rate_t0 = self._t0
            self._thread = threading.Thread(
                target=self._run, name="HandTracker", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self._landmarker.close()
        except Exception:
            pass

    def __enter__(self) -> "HandTracker":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- consumer API -----------------------------------------------------

    def latest(self) -> Optional[HandFrame]:
        with self._lock:
            return self._latest

    # -- slot identity ----------------------------------------------------

    # -- main loop --------------------------------------------------------

    def _run(self) -> None:
        import mediapipe as mp

        last_index = -1
        try:
            while not self._stop.is_set():
                frame = self._source.wait_for_frame(last_index, 0.25)
                if frame is None:
                    continue
                last_index = frame.index
                self._process(frame, mp)
        except BaseException as exc:
            self.error = exc

    def _process(self, frame: Frame, mp) -> None:
        t_start = time.perf_counter()

        # Downscale first, then convert. MediaPipe returns normalised
        # coordinates, so an aspect-preserving downscale is free in mapping
        # terms, and it is the single cheapest way to buy back detection
        # time. Convert into the layout the probed delegate demands --
        # MediaPipe gets the raw, un-mirrored image, because the model is
        # trained on camera-native input and the mirror belongs in CoverFit.
        source = frame.data
        tw = config.TRACKING_INPUT_WIDTH
        if tw and source.shape[1] > tw:
            th = int(round(source.shape[0] * tw / source.shape[1]))
            if self._small is None or self._small.shape[:2] != (th, tw):
                self._small = np.empty((th, tw, 3), np.uint8)
            cv2.resize(source, (tw, th), dst=self._small,
                       interpolation=cv2.INTER_AREA)
            source = self._small

        channels = self.choice.channels
        if self._rgb is None or self._rgb.shape[:2] != source.shape[:2] \
                or self._rgb.shape[2] != channels:
            self._rgb = np.empty(
                (source.shape[0], source.shape[1], channels), np.uint8
            )
            self._mp_format = getattr(mp.ImageFormat, self.choice.image_format)
        cv2.cvtColor(source, getattr(cv2, self.choice.cv_conversion), dst=self._rgb)
        t_convert = time.perf_counter()

        image = mp.Image(image_format=self._mp_format, data=self._rgb)
        # MediaPipe requires strictly increasing millisecond stamps.
        ts = int((frame.timestamp - self._t0) * 1000.0)
        self._mp_ts = max(ts, self._mp_ts + 1)
        result = self._landmarker.detect_for_video(image, self._mp_ts)
        t_detect = time.perf_counter()

        self._publish(frame, result, t_start, t_convert, t_detect)

    def _publish(self, frame, result, t_start, t_convert, t_detect) -> None:
        hands = getattr(result, "hand_landmarks", None) or []
        handedness = getattr(result, "handedness", None) or []

        n = min(len(hands), _SLOTS)
        labels: list[str] = []
        norm = self._norm
        for d in range(n):
            lm = hands[d]
            for i in range(_NL):
                point = lm[i]
                norm[d, i, 0] = point.x
                norm[d, i, 1] = point.y
            categories = handedness[d] if d < len(handedness) else None
            labels.append(categories[0].category_name if categories else "Left")

        now = frame.timestamp
        dt = now - self._last_ts if self._last_ts else 1.0 / max(self._source.fps, 1.0)
        self._last_ts = now

        hand_frame = self.pipeline.update(
            norm[:n], labels, now, dt, latency=time.perf_counter() - now
        )
        with self._lock:
            self._latest = hand_frame

        published = time.perf_counter()
        self.stats.blend(TrackerStats(
            convert_ms=(t_convert - t_start) * 1e3,
            detect_ms=(t_detect - t_convert) * 1e3,
            total_ms=(published - t_start) * 1e3,
        ))
        self._rate_n += 1
        elapsed = published - self._rate_t0
        if elapsed >= 1.0:
            self.stats.rate_hz = self._rate_n / elapsed
            self._rate_t0, self._rate_n = published, 0

        if self._on_frame is not None:
            self._on_frame(hand_frame)
