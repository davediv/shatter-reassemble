"""Frame instrumentation, and the ladder that acts on it.

Two measurements, because they answer different questions. CPU section
timers say where the frame's Python and numba time went. GPU timer queries
(ARB_timer_query -- the desktop equivalent of the spec's
EXT_disjoint_timer_query_webgl2, and confirmed working on Apple silicon)
say what the GPU actually spent, which no amount of CPU timing can tell
you because draw calls return long before the work is done.

GPU queries are read from several frames back rather than immediately.
Asking for a result the GPU has not reached yet blocks until it has, which
would turn the profiler into the very stall it is trying to find. Six
frames of slack rather than two or three, because an unthrottled loop can
run well ahead of the GPU and a shallow ring starts blocking exactly when
the frame is busiest.

The ladder runs off wall-clock frame time rather than off either timer.
That is deliberate: the thing the user experiences is the frame interval,
and it captures costs -- driver stalls, vsync misses, other processes --
that no section timer will ever attribute to anything.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from . import config

__all__ = ["FrameProfiler", "QualityLadder", "GpuProfiler"]


class GpuProfiler:
    """GL_TIME_ELAPSED queries, read from ``depth`` frames ago."""

    def __init__(self, ctx, depth: int = 6) -> None:
        self.ctx = ctx
        self.depth = depth
        self.available = True
        self._queries: Dict[str, list] = {}
        self._slot = 0
        self._results: Dict[str, float] = {}
        self._pending: Dict[str, list] = {}
        self._active: Optional[str] = None
        self._zero_streak = 0

    def _ensure(self, name: str) -> None:
        if name not in self._queries:
            try:
                self._queries[name] = [
                    self.ctx.query(time=True) for _ in range(self.depth)
                ]
            except Exception:
                self.available = False
                self._queries[name] = []
                return
            self._pending[name] = [False] * self.depth
            self._results[name] = 0.0

    @contextmanager
    def section(self, name: str):
        """Time one GL section. Sections may not nest -- GL allows only one
        TIME_ELAPSED query in flight at a time."""
        if not self.available or self._active is not None:
            yield
            return
        self._ensure(name)
        queries = self._queries.get(name)
        if not queries:
            yield
            return
        slot = self._slot % self.depth
        query = queries[slot]

        # Harvest the result this slot is still holding before reusing it;
        # by now it is `depth` frames old and certainly complete.
        if self._pending[name][slot]:
            try:
                elapsed = query.elapsed
                self._results[name] += (elapsed * 1e-6 - self._results[name]) * 0.2
                if elapsed == 0:
                    self._zero_streak += 1
                    if self._zero_streak > 240:
                        # The driver is accepting queries and reporting
                        # nothing. Stop paying for them.
                        self.available = False
                else:
                    self._zero_streak = 0
            except Exception:
                self.available = False
        self._active = name
        try:
            with query:
                yield
            self._pending[name][slot] = True
        finally:
            self._active = None

    def next_frame(self) -> None:
        self._slot += 1

    def results(self) -> Dict[str, float]:
        return dict(self._results)


@dataclass
class FrameProfiler:
    """Rolling CPU section timings plus the frame interval itself."""

    window: int = config.LADDER_WINDOW_FRAMES
    sections: Dict[str, float] = field(default_factory=dict)
    frame_ms: float = 16.6
    fps: float = 60.0

    def __post_init__(self) -> None:
        self._history = np.full(self.window, 16.6, np.float64)
        self._cursor = 0
        self._filled = 0
        self._last = time.perf_counter()
        self._frame_start = self._last

    @contextmanager
    def section(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1e3
            previous = self.sections.get(name, elapsed)
            # Exponential average: a HUD of raw per-frame numbers is
            # unreadable, and the ladder wants a trend, not a sample.
            self.sections[name] = previous + (elapsed - previous) * 0.15

    def begin_frame(self) -> None:
        self._frame_start = time.perf_counter()

    def end_frame(self) -> float:
        now = time.perf_counter()
        interval = (now - self._last) * 1e3
        self._last = now
        self._history[self._cursor] = interval
        self._cursor = (self._cursor + 1) % self.window
        self._filled = min(self._filled + 1, self.window)
        self.frame_ms += (interval - self.frame_ms) * 0.15
        if interval > 1e-6:
            self.fps += (1000.0 / interval - self.fps) * 0.15
        return interval

    @property
    def rolling_ms(self) -> float:
        if self._filled == 0:
            return 16.6
        return float(self._history[: self._filled].mean())

    @property
    def worst_ms(self) -> float:
        if self._filled == 0:
            return 16.6
        return float(self._history[: self._filled].max())

    def percentile(self, q: float) -> float:
        if self._filled == 0:
            return 16.6
        return float(np.percentile(self._history[: self._filled], q))

    def reset(self) -> None:
        self._history[:] = 16.6
        self._filled = 0
        self._cursor = 0


class QualityLadder:
    """Steps quality down when the rolling average slips, and back up later.

    Hysteresis is the whole design. Stepping down at 15ms and back up at
    the same 15ms would oscillate on the boundary, and an app that
    flickers between refraction on and off looks far worse than one that
    simply left it off. The dwell time stops it thrashing through several
    rungs on a single hitch -- like the one every snap causes when the
    fracture lands.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.index = 0
        self.changed_at = 0.0
        self.last_reason = ""
        self.steps_down = 0
        self.steps_up = 0

    @property
    def level(self) -> config.QualityLevel:
        return config.QUALITY_LADDER[self.index]

    def force(self, index: int, now: float) -> None:
        self.index = max(0, min(index, len(config.QUALITY_LADDER) - 1))
        self.changed_at = now

    def update(self, rolling_ms: float, now: float) -> bool:
        """Returns True if the rung changed this frame."""
        if not self.enabled:
            return False
        if now - self.changed_at < config.LADDER_DWELL_SECONDS:
            return False

        if rolling_ms > config.LADDER_STEP_DOWN_MS:
            if self.index < len(config.QUALITY_LADDER) - 1:
                self.index += 1
                self.changed_at = now
                self.steps_down += 1
                self.last_reason = f"{rolling_ms:.1f}ms > {config.LADDER_STEP_DOWN_MS:.0f}"
                return True
        elif rolling_ms < config.LADDER_STEP_UP_MS and self.index > 0:
            self.index -= 1
            self.changed_at = now
            self.steps_up += 1
            self.last_reason = f"{rolling_ms:.1f}ms < {config.LADDER_STEP_UP_MS:.0f}"
            return True
        return False
