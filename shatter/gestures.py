"""Snap, clap and open-palm detection off the unsmoothed velocity channel.

Runs inside the tracking thread, on every tracking frame. That placement
is deliberate: a snap is a ~30ms transient, and polling for it from a
render loop that may be running faster or slower than tracking would drop
exactly the frames the spike lives in.

Everything here is normalised by hand span, so a snap 30cm from the lens
and a snap across the room read identically.

Snap
----
The spec's rule is: thumb tip and middle tip come within 0.25 hand spans,
followed by a middle-tip velocity spike within 120ms. Two refinements on
top of that, both of which earn their place:

*The window runs from the most recent pinched frame, not from the frame
the pinch began.* People hold the pinch while they aim, then flick. A
window anchored to the start of the pinch would only ever fire for someone
who snaps the instant their fingers touch.

*The spike is measured relative to the wrist.* Absolute fingertip velocity
cannot tell a snap from a hand being waved -- during a wave every landmark
including the middle tip exceeds any threshold a real snap would. Working
in the hand's own frame makes the detector immune to whole-hand
translation, which in testing is the single largest source of false
positives.

Clap
----
Palm centroids converging, closer than 0.5 spans, with approach speed
above threshold. Approach is projected from the velocity channel rather
than differenced from distance -- differencing a distance costs a frame of
latency and squares the noise.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np

from . import config
from .filters import TraceBuffer
from .tracking import HandFrame

__all__ = ["GestureEvent", "GestureSignals", "GestureRecognizer",
           "SNAP", "CLAP", "TRACE_CAPACITY"]

SNAP = "snap"
CLAP = "clap"

TRACE_CAPACITY = 420        # ~14s at 30Hz, ~7s at 60Hz

# Hand span can only be trusted once there is a hand to measure.
_MIN_SPAN = config.HAND_SPAN_MIN_PIXELS


@dataclass(frozen=True)
class GestureEvent:
    kind: str
    timestamp: float
    position: tuple[float, float]    # canvas px -- where the break radiates from
    slot: int = -1
    strength: float = 0.0            # spans/s at the moment it fired


@dataclass
class GestureSignals:
    """A snapshot of what the detectors are seeing, for HUD and tuning."""

    pinch: tuple[float, float] = (9.9, 9.9)        # spans, per slot
    flick: tuple[float, float] = (0.0, 0.0)        # spans/s, wrist-relative
    armed: tuple[bool, bool] = (False, False)
    lockout: tuple[float, float] = (0.0, 0.0)      # seconds remaining
    palm_distance: float = 9.9                     # spans
    palm_approach: float = 0.0                     # spans/s, + is closing
    clap_lockout: float = 0.0
    open_palm: tuple[bool, bool] = (False, False)
    extension: tuple[float, float] = (0.0, 0.0)
    hands: int = 0
    span: tuple[float, float] = (0.0, 0.0)


class GestureRecognizer:
    """Owns the three detectors and the traces the tuning mode plots."""

    def __init__(self, tunables: config.Tunables) -> None:
        self.tunables = tunables
        self._events: Deque[GestureEvent] = deque(maxlen=64)
        self._lock = threading.Lock()
        self._signals = GestureSignals()

        slots = config.NUM_HANDS
        self._last_pinch_time = np.full(slots, -1e9, np.float64)
        self._last_snap_time = np.full(slots, -1e9, np.float64)
        self._last_clap_time = -1e9
        self._pinch_latch = np.zeros(slots, np.bool_)

        # Traces are written here and read by the renderer, so they are
        # snapshotted under the same lock as the signals.
        self.trace_pinch = [TraceBuffer(TRACE_CAPACITY) for _ in range(slots)]
        self.trace_flick = [TraceBuffer(TRACE_CAPACITY) for _ in range(slots)]
        self.trace_palm = TraceBuffer(TRACE_CAPACITY)
        self.trace_approach = TraceBuffer(TRACE_CAPACITY)
        # Fire markers are pushed on the same schedule as the signals, so a
        # marker always lines up with the exact sample that triggered it.
        # Recording wall-clock timestamps instead would leave the markers
        # drifting against the trace whenever the tracking rate wobbled.
        self.trace_snap_fired = [TraceBuffer(TRACE_CAPACITY) for _ in range(slots)]
        self.trace_clap_fired = TraceBuffer(TRACE_CAPACITY)
        self.trace_events: Deque[tuple[float, str]] = deque(maxlen=64)

        self.snap_count = 0
        self.clap_count = 0

    # -- consumer API -----------------------------------------------------

    def poll(self) -> list[GestureEvent]:
        """Drain pending events. Called from the render thread."""
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events

    def signals(self) -> GestureSignals:
        with self._lock:
            return self._signals

    def snapshot_traces(self) -> dict:
        """Copies of the traces, safe to read while tracking keeps writing."""
        with self._lock:
            return {
                "pinch": [t.snapshot().copy() for t in self.trace_pinch],
                "flick": [t.snapshot().copy() for t in self.trace_flick],
                "palm": self.trace_palm.snapshot().copy(),
                "approach": self.trace_approach.snapshot().copy(),
                "snap_fired": [t.snapshot().copy() for t in self.trace_snap_fired],
                "clap_fired": self.trace_clap_fired.snapshot().copy(),
                "events": list(self.trace_events),
            }

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._last_pinch_time[:] = -1e9
            self._last_snap_time[:] = -1e9
            self._last_clap_time = -1e9

    # -- detection --------------------------------------------------------

    def update(self, frame: HandFrame) -> None:
        """Called on the tracking thread, once per tracking frame."""
        tun = self.tunables
        now = frame.timestamp

        pinch = [9.9, 9.9]
        flick = [0.0, 0.0]
        armed = [False, False]
        extension = [0.0, 0.0]
        open_palm = [False, False]
        lockout = [0.0, 0.0]
        events: list[GestureEvent] = []

        for slot in range(config.NUM_HANDS):
            if not frame.present[slot]:
                self._pinch_latch[slot] = False
                continue
            span = float(frame.span[slot])
            if span < _MIN_SPAN:
                continue

            points = frame.raw[slot]
            velocity = frame.velocity[slot]

            # -- snap ------------------------------------------------------
            thumb = points[config.THUMB_TIP]
            middle = points[config.MIDDLE_TIP]
            pinch_distance = float(np.hypot(*(middle - thumb))) / span
            pinch[slot] = pinch_distance

            # Wrist-relative, so waving the whole hand does not read as a
            # flick. This is the single biggest false-positive rejection.
            relative = velocity[config.MIDDLE_TIP] - velocity[config.WRIST]
            flick_speed = float(np.hypot(*relative)) / span
            flick[slot] = flick_speed

            if pinch_distance < tun.snap_pinch_distance:
                self._last_pinch_time[slot] = now
                self._pinch_latch[slot] = True

            in_window = (now - self._last_pinch_time[slot]) <= tun.snap_window
            armed[slot] = bool(in_window and self._pinch_latch[slot])
            since_snap = now - self._last_snap_time[slot]
            lockout[slot] = max(0.0, tun.snap_lockout - since_snap)

            if (armed[slot]
                    and flick_speed > tun.snap_velocity_threshold
                    and since_snap > tun.snap_lockout):
                self._last_snap_time[slot] = now
                self._pinch_latch[slot] = False
                self.snap_count += 1
                events.append(GestureEvent(
                    SNAP, now, (float(middle[0]), float(middle[1])),
                    slot, flick_speed,
                ))

            # -- open palm -------------------------------------------------
            tips = points[[config.THUMB_TIP, config.INDEX_TIP, config.MIDDLE_TIP,
                           config.RING_TIP, config.PINKY_TIP]]
            spread = float(np.mean(np.hypot(
                tips[:, 0] - points[config.WRIST, 0],
                tips[:, 1] - points[config.WRIST, 1],
            ))) / span
            extension[slot] = spread
            open_palm[slot] = spread > tun.open_palm_extension

        # -- clap ----------------------------------------------------------
        palm_distance = 9.9
        approach = 0.0
        both = bool(frame.present.all())
        if both:
            span = float(frame.span.mean())
            if span >= _MIN_SPAN:
                a = frame.raw[0, config.MIDDLE_MCP]
                b = frame.raw[1, config.MIDDLE_MCP]
                delta = b - a
                gap = float(np.hypot(*delta))
                palm_distance = gap / span
                if gap > 1e-3:
                    direction = delta / gap
                    relative = frame.velocity[0, config.MIDDLE_MCP] - \
                        frame.velocity[1, config.MIDDLE_MCP]
                    # Positive when the palms are closing on each other.
                    approach = float(np.dot(relative, direction)) / span

                since_clap = now - self._last_clap_time
                if (palm_distance < tun.clap_distance
                        and approach > tun.clap_velocity_threshold
                        and since_clap > tun.clap_lockout):
                    self._last_clap_time = now
                    self.clap_count += 1
                    midpoint = (a + b) * 0.5
                    events.append(GestureEvent(
                        CLAP, now, (float(midpoint[0]), float(midpoint[1])),
                        -1, approach,
                    ))

        snapshot = GestureSignals(
            pinch=(pinch[0], pinch[1]),
            flick=(flick[0], flick[1]),
            armed=(armed[0], armed[1]),
            lockout=(lockout[0], lockout[1]),
            palm_distance=palm_distance,
            palm_approach=approach,
            clap_lockout=max(0.0, tun.clap_lockout - (now - self._last_clap_time)),
            open_palm=(open_palm[0], open_palm[1]),
            extension=(extension[0], extension[1]),
            hands=frame.hand_count,
            span=(float(frame.span[0]), float(frame.span[1])),
        )

        snapped = [False] * config.NUM_HANDS
        clapped = False
        for event in events:
            if event.kind == SNAP and 0 <= event.slot < config.NUM_HANDS:
                snapped[event.slot] = True
            elif event.kind == CLAP:
                clapped = True

        with self._lock:
            self._signals = snapshot
            for slot in range(config.NUM_HANDS):
                self.trace_pinch[slot].push(min(pinch[slot], 2.0))
                self.trace_flick[slot].push(flick[slot])
                self.trace_snap_fired[slot].push(1.0 if snapped[slot] else 0.0)
            self.trace_palm.push(min(palm_distance, 4.0))
            self.trace_approach.push(approach)
            self.trace_clap_fired.push(1.0 if clapped else 0.0)
            for event in events:
                self._events.append(event)
                self.trace_events.append((event.timestamp, event.kind))
