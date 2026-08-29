"""The snap tuning mode: two live traces, adjustable thresholds, on-screen.

The spec calls snap detection finicky and says to build this before any
visuals, which is exactly right. Snap thresholds cannot be reasoned out on
paper -- they depend on how a particular person snaps, how bright the room
is and how fast the camera runs. What you need is to watch the two signals
scroll past while you snap, see where the spike actually lands relative to
the line, and drag the line to the gap.

So this shows precisely the two signals the detector reads:

    pinch distance   thumb tip to middle tip, in hand spans
    flick speed      middle tip velocity relative to the wrist, spans/sec

with the live threshold drawn across each, and a marker on every frame
that fired. A well-tuned threshold sits in clear air: above every peak the
idle hand produces, below every peak a real snap produces. If those two
overlap on the trace, no threshold will work and the gesture needs to
change, not the number -- which is the kind of thing you only ever learn
by looking at the trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import config, keys
from .gestures import TRACE_CAPACITY, GestureRecognizer
from .render.debug import DebugOverlay
from .render.primitives import ShapeBatch
from .render.text import TextRenderer

__all__ = ["TuningMode", "TunableSpec"]

PANEL_BG = (0.04, 0.05, 0.07, 0.86)
PANEL_EDGE = (0.30, 0.36, 0.48, 0.6)
GRID = (1.0, 1.0, 1.0, 0.06)
THRESHOLD_COLOR = (1.0, 0.42, 0.32, 0.95)
FIRE_COLOR = (1.0, 0.85, 0.25, 0.85)
SLOT_COLORS = ((0.25, 0.95, 1.00, 0.95), (1.00, 0.35, 0.85, 0.95))
SELECTED = (1.0, 0.92, 0.4, 1.0)
DIM = (0.60, 0.66, 0.76, 1.0)


@dataclass(frozen=True)
class TunableSpec:
    label: str
    attr: str
    step: float
    minimum: float
    maximum: float
    unit: str = ""
    digits: int = 3

    def format(self, value: float) -> str:
        return f"{value:.{self.digits}f}{self.unit}"


SPECS: tuple[TunableSpec, ...] = (
    TunableSpec("snap pinch distance", "snap_pinch_distance", 0.01, 0.05, 1.00, " span", 2),
    TunableSpec("snap flick speed", "snap_velocity_threshold", 0.5, 1.0, 60.0, " sp/s", 1),
    TunableSpec("snap min travel", "snap_min_travel", 0.02, 0.02, 1.50, " span", 2),
    TunableSpec("snap max velocity", "snap_max_velocity", 1.0, 20.0, 200.0, " sp/s", 1),
    TunableSpec("snap window", "snap_window", 0.010, 0.020, 0.500, " s", 3),
    TunableSpec("snap lockout", "snap_lockout", 0.05, 0.05, 2.00, " s", 2),
    TunableSpec("clap distance", "clap_distance", 0.02, 0.10, 2.00, " span", 2),
    TunableSpec("clap approach", "clap_velocity_threshold", 0.10, 0.20, 20.0, " sp/s", 2),
    TunableSpec("open palm extension", "open_palm_extension", 0.05, 0.50, 3.00, " span", 2),
    TunableSpec("stir capsule radius", "stir_capsule_radius", 0.01, 0.02, 0.60, " span", 2),
)


class TuningMode:
    """Draws the traces and owns the keyboard editing of thresholds."""

    def __init__(
        self,
        tunables: config.Tunables,
        recognizer: GestureRecognizer,
        shapes: ShapeBatch,
        text: TextRenderer,
        overlay: DebugOverlay,
    ) -> None:
        self.tunables = tunables
        self.recognizer = recognizer
        self.shapes = shapes
        self.text = text
        self.overlay = overlay
        self.active = False
        self.selected = 0
        self.status = "arrow keys adjust  .  S saves  .  D restores defaults"
        # Auto-ranged axes, smoothed so the plot does not breathe.
        self._flick_top = 20.0
        self._approach_top = 6.0

    # -- input ------------------------------------------------------------

    def handle_key(self, key: int, action: int, mods: int) -> bool:
        """Returns True if the key was consumed by the tuning UI."""
        if not self.active or action not in (keys.PRESS, keys.REPEAT):
            return False
        if key == keys.UP:
            self.selected = (self.selected - 1) % len(SPECS)
        elif key == keys.DOWN:
            self.selected = (self.selected + 1) % len(SPECS)
        elif key in (keys.LEFT, keys.RIGHT):
            self._adjust(-1 if key == keys.LEFT else 1, mods)
        elif key == keys.S:
            path = self.tunables.save()
            self.status = f"saved to {path.name}"
        elif key == keys.D:
            defaults = config.Tunables()
            for spec in SPECS:
                setattr(self.tunables, spec.attr, getattr(defaults, spec.attr))
            self.status = "restored defaults (unsaved)"
        else:
            return False
        return True

    def _adjust(self, direction: int, mods: int) -> None:
        spec = SPECS[self.selected]
        # Shift for a coarse pass, then fine once you are close.
        step = spec.step * (10.0 if mods & keys.MOD_SHIFT else 1.0)
        value = getattr(self.tunables, spec.attr) + direction * step
        value = min(max(value, spec.minimum), spec.maximum)
        setattr(self.tunables, spec.attr, round(value, 6))
        self.status = f"{spec.label} = {spec.format(value)}   (unsaved)"

    # -- drawing ----------------------------------------------------------

    def draw(self, canvas_w: int, canvas_h: int) -> None:
        if not self.active:
            return
        traces = self.recognizer.snapshot_traces()
        signals = self.recognizer.signals()

        margin = 46.0
        plot_w = canvas_w - margin * 2 - 470.0
        # Four stacked plots have to fit 1080 with room for titles:
        # top + 3 * (plot_h + gap) + plot_h must clear the bottom edge.
        plot_h = 168.0
        gap = 50.0
        top = 148.0

        self.text.draw("SNAP TUNING", margin, 44.0, 38.0, (1.0, 1.0, 1.0, 1.0))
        self.text.draw(
            "a real snap must clear every line; an idle hand must clear none",
            margin, 92.0, 20.0, (0.55, 0.62, 0.74, 1.0),
        )

        # -- signal 1: pinch distance -------------------------------------
        self._plot(
            margin, top, plot_w, plot_h,
            series=[(traces["pinch"][s], SLOT_COLORS[s]) for s in range(config.NUM_HANDS)],
            lo=0.0, hi=1.2,
            threshold=self.tunables.snap_pinch_distance,
            threshold_below=True,
            markers=[(traces["snap_fired"][s], FIRE_COLOR) for s in range(config.NUM_HANDS)],
            title="1  pinch distance   thumb tip -> middle tip",
            readout=f"{min(signals.pinch):.2f} span",
            unit="span",
        )

        # -- signal 2: flick speed ----------------------------------------
        observed = max((float(t.max()) if t.size else 0.0) for t in traces["flick"])
        target = max(observed * 1.15, self.tunables.snap_velocity_threshold * 1.6, 8.0)
        self._flick_top += (target - self._flick_top) * 0.08
        self._plot(
            margin, top + plot_h + gap, plot_w, plot_h,
            series=[(traces["flick"][s], SLOT_COLORS[s]) for s in range(config.NUM_HANDS)],
            lo=0.0, hi=self._flick_top,
            threshold=self.tunables.snap_velocity_threshold,
            threshold_below=False,
            markers=[(traces["snap_fired"][s], FIRE_COLOR) for s in range(config.NUM_HANDS)],
            title="2  flick speed   middle tip velocity relative to wrist",
            readout=f"{max(signals.flick):.1f} sp/s",
            unit="sp/s",
        )

        # -- signal 3: travel ---------------------------------------------
        self._plot(
            margin, top + (plot_h + gap) * 2, plot_w, plot_h,
            series=[(traces["travel"][s], SLOT_COLORS[s]) for s in range(config.NUM_HANDS)],
            lo=0.0, hi=max(self.tunables.snap_min_travel * 3.0, 0.6),
            threshold=self.tunables.snap_min_travel,
            threshold_below=False,
            markers=[(traces["snap_fired"][s], FIRE_COLOR) for s in range(config.NUM_HANDS)],
            title="3  travel   net wrist-relative displacement over the window",
            readout=f"{max(signals.travel):.2f} span",
            unit="span",
        )

        # -- clap ----------------------------------------------------------
        approach_target = max(
            float(np.abs(traces["approach"]).max()) * 1.2 if traces["approach"].size else 4.0,
            self.tunables.clap_velocity_threshold * 2.0, 4.0,
        )
        self._approach_top += (approach_target - self._approach_top) * 0.08
        self._plot(
            margin, top + (plot_h + gap) * 3, plot_w, plot_h,
            series=[(traces["palm"], (0.45, 1.0, 0.6, 0.95)),
                    (traces["approach"] / max(self._approach_top, 1e-3) * self.tunables.clap_distance * 2.0,
                     (1.0, 0.75, 0.35, 0.8))],
            lo=0.0, hi=max(self.tunables.clap_distance * 3.0, 1.5),
            threshold=self.tunables.clap_distance,
            threshold_below=True,
            markers=[(traces["clap_fired"], FIRE_COLOR)],
            title="4  clap   palm gap (green) and approach speed (amber, scaled)",
            readout=f"{signals.palm_distance:.2f} span  {signals.palm_approach:+.1f} sp/s",
            unit="span",
        )

        panel_x = canvas_w - 470.0 + 24.0
        end = self._draw_parameters(panel_x, top, 424.0)
        self._draw_state(panel_x, end + 16.0, 424.0, signals)

    def _plot(
        self, x: float, y: float, w: float, h: float, *,
        series, lo: float, hi: float, threshold: Optional[float],
        threshold_below: bool, markers, title: str, readout: str, unit: str,
    ) -> None:
        shapes = self.shapes
        shapes.rect(x, y, w, h, PANEL_BG)
        shapes.rect_outline(x, y, w, h, 1.5, PANEL_EDGE)
        self.text.draw(title, x + 12.0, y - 26.0, 21.0, (0.82, 0.87, 0.95, 1.0))
        self.text.draw(readout, x + w - 220.0, y - 26.0, 21.0, (1.0, 1.0, 1.0, 1.0))

        span = max(hi - lo, 1e-6)

        def to_y(values: np.ndarray) -> np.ndarray:
            norm = (values - lo) / span
            return y + h - np.clip(norm, 0.0, 1.0) * h

        for level in (0.25, 0.5, 0.75):
            gy = y + h - level * h
            shapes.line((x, gy), (x + w, gy), 1.0, GRID)
        self.text.draw(f"{hi:.2f}", x + w + 8.0, y - 4.0, 17.0, (0.5, 0.55, 0.65, 1.0))
        self.text.draw(f"{lo:.2f}", x + w + 8.0, y + h - 16.0, 17.0, (0.5, 0.55, 0.65, 1.0))

        step = w / max(TRACE_CAPACITY - 1, 1)

        # Fire markers first, so the traces draw over them.
        for fired, colour in markers:
            if fired.size == 0:
                continue
            hits = np.flatnonzero(fired > 0.5)
            if hits.size:
                xs = x + w - (fired.size - 1 - hits) * step
                top = np.full(hits.size, y, np.float32)
                bottom = np.full(hits.size, y + h, np.float32)
                shapes.lines(np.stack([xs, top], axis=1),
                             np.stack([xs, bottom], axis=1), 2.0, colour)

        if threshold is not None and lo <= threshold <= hi:
            ty = float(to_y(np.array([threshold], np.float32))[0])
            shapes.line((x, ty), (x + w, ty), 2.0, THRESHOLD_COLOR)
            side = "fire below" if threshold_below else "fire above"
            self.text.draw(f"{threshold:.2f} {unit}  {side}", x + 12.0, ty - 24.0,
                           18.0, THRESHOLD_COLOR)

        for values, colour in series:
            if values.size < 2:
                continue
            xs = x + w - (values.size - 1 - np.arange(values.size)) * step
            points = np.stack([xs, to_y(values)], axis=1).astype(np.float32)
            shapes.polyline(points, 2.0, colour)

    def _draw_parameters(self, x: float, y: float, w: float) -> float:
        row_h = 34.0
        height = 44.0 + len(SPECS) * row_h + 12.0
        self.shapes.rect(x, y, w, height, PANEL_BG)
        self.shapes.rect_outline(x, y, w, height, 1.5, PANEL_EDGE)
        self.text.draw("THRESHOLDS", x + 16.0, y + 10.0, 24.0, (1.0, 1.0, 1.0, 1.0))

        cursor = y + 48.0
        for i, spec in enumerate(SPECS):
            chosen = i == self.selected
            colour = SELECTED if chosen else DIM
            if chosen:
                self.shapes.rect(x + 6.0, cursor - 4.0, w - 12.0, row_h - 4.0,
                                 (1.0, 0.92, 0.4, 0.10))
                self.text.draw(">", x + 12.0, cursor, 20.0, SELECTED)
            self.text.draw(spec.label, x + 32.0, cursor, 20.0, colour)
            value = getattr(self.tunables, spec.attr)
            self.text.draw(spec.format(value), x + w - 148.0, cursor, 20.0,
                           (1.0, 1.0, 1.0, 1.0) if chosen else DIM)
            cursor += row_h
        return y + height

    def _draw_state(self, x: float, y: float, w: float, signals) -> None:
        rows = []
        for slot in range(config.NUM_HANDS):
            state = "ARMED" if signals.armed[slot] else "-"
            if signals.lockout[slot] > 0:
                state = f"lockout {signals.lockout[slot]:.2f}s"
            rows.append((
                f"hand {slot}", state,
                SLOT_COLORS[slot] if signals.armed[slot] else DIM,
            ))
        rows.append(("hands seen", str(signals.hands), DIM))
        rows.append(("span px", f"{signals.span[0]:.0f} / {signals.span[1]:.0f}", DIM))
        rows.append(("travel span", f"{max(signals.travel):.2f}", DIM))
        rows.append(("snaps fired", str(self.recognizer.snap_count), (0.5, 1.0, 0.6, 1.0)))
        rows.append(("claps fired", str(self.recognizer.clap_count), (0.5, 1.0, 0.6, 1.0)))
        end = self.overlay.panel(x, y, rows, title="LIVE STATE", width=w)
        self.overlay.hint([
            "up / down    select threshold",
            "left / right adjust   (shift = x10)",
            "S            save to tuning.json",
            "D            restore defaults",
            "T            leave tuning mode",
        ], x + 8.0, end + 6.0, 19.0)
        # Wrapped, because the status can run longer than the panel.
        status = self.status
        line_y = end + 6.0 + 5 * 25.0 + 12.0
        while status:
            self.text.draw(status[:34], x + 8.0, line_y, 19.0, (1.0, 0.85, 0.4, 1.0))
            status = status[34:]
            line_y += 24.0
