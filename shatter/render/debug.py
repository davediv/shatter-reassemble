"""Debug overlays: the skeleton, and the HUD that reports the budget.

The spec is explicit that the skeleton comes before any effect code, and
the reason is worth stating: mirroring bugs are invisible in the numbers.
Landmarks that are flipped, swapped between hands, or lagging behind the
image all produce perfectly plausible-looking arrays. You only catch them
by watching a hand move and seeing the drawn skeleton move the same way.

So this draws the smoothed skeleton over the mirrored feed, and offers a
raw-versus-smoothed mode that makes the One Euro filter's lag visible
directly -- which is how you confirm the filter is light enough to leave
the velocity spikes that snap detection needs.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from .. import config
from ..tracking import HandFrame
from .primitives import ShapeBatch
from .text import TextRenderer

__all__ = ["DebugOverlay", "SKELETON_BONES"]


def _bones() -> np.ndarray:
    pairs: list[tuple[int, int]] = []
    for chain in config.FINGER_CHAINS:
        pairs.extend(zip(chain[:-1], chain[1:]))
    return np.array(pairs, np.int32)


SKELETON_BONES = _bones()

# Slot 0 and slot 1 get distinct hues so a slot swap is instantly obvious:
# if the colours trade places while the hands hold still, identity broke.
SLOT_COLORS = (
    (0.25, 0.95, 1.00, 0.95),   # cyan
    (1.00, 0.35, 0.85, 0.95),   # magenta
)
TIP_COLOR = (1.0, 0.92, 0.35, 1.0)
RAW_COLOR = (1.0, 1.0, 1.0, 0.28)
PANEL_BG = (0.03, 0.04, 0.06, 0.78)
PANEL_EDGE = (0.35, 0.42, 0.55, 0.55)

FINGERTIPS = (config.THUMB_TIP, config.INDEX_TIP, config.MIDDLE_TIP,
              config.RING_TIP, config.PINKY_TIP)


class DebugOverlay:
    def __init__(self, shapes: ShapeBatch, text: TextRenderer) -> None:
        self.shapes = shapes
        self.text = text
        self.visible = True
        self.show_raw = False

    # -- skeleton ---------------------------------------------------------

    def draw_skeleton(self, frame: Optional[HandFrame]) -> None:
        if not self.visible or frame is None:
            return
        shapes = self.shapes
        for slot in range(config.NUM_HANDS):
            if not frame.present[slot]:
                continue
            colour = SLOT_COLORS[slot]
            points = frame.smooth[slot]

            if self.show_raw:
                # The gap between these two is the filter's lag, made
                # visible. It should be small and should vanish the moment
                # the hand moves fast -- that is the One Euro filter doing
                # its job rather than flattening the signal.
                raw = frame.raw[slot]
                shapes.lines(raw[SKELETON_BONES[:, 0]], raw[SKELETON_BONES[:, 1]],
                             2.0, RAW_COLOR)

            shapes.lines(points[SKELETON_BONES[:, 0]], points[SKELETON_BONES[:, 1]],
                         5.0, colour)
            shapes.discs(points, np.full(config.NUM_LANDMARKS, 6.0, np.float32),
                         colour, segments=10)
            shapes.discs(points[list(FINGERTIPS)], np.full(len(FINGERTIPS), 10.0, np.float32),
                         TIP_COLOR, segments=12)

            # The calibrated span, drawn as the thing it actually is: the
            # wrist-to-middle-MCP measurement every threshold divides by.
            span = float(frame.span[slot])
            wrist = points[config.WRIST]
            shapes.line(wrist, points[config.MIDDLE_MCP], 3.0, (1.0, 1.0, 1.0, 0.6))
            self.text.draw(
                f"{frame.handedness[slot] or '?'}  span {span:.0f}px",
                float(wrist[0]) - 60.0, float(wrist[1]) + 22.0, 24.0, colour,
            )

    def draw_capsules(self, centers_a: np.ndarray, centers_b: np.ndarray,
                      radius: float, colour=(1.0, 0.55, 0.2, 0.30)) -> None:
        """The stir colliders, drawn as the capsules the solver actually uses."""
        if not self.visible or centers_a.shape[0] == 0:
            return
        self.shapes.lines(centers_a, centers_b, radius * 2.0, colour)
        self.shapes.discs(centers_a, np.full(centers_a.shape[0], radius, np.float32),
                          colour, segments=10)
        self.shapes.discs(centers_b, np.full(centers_b.shape[0], radius, np.float32),
                          colour, segments=10)

    # -- HUD --------------------------------------------------------------

    def panel(
        self,
        x: float,
        y: float,
        rows: Sequence[tuple[str, str, tuple]],
        title: Optional[str] = None,
        width: float = 430.0,
        row_height: float = 27.0,
        text_size: float = 21.0,
    ) -> float:
        """A key/value panel. Returns the y just past its bottom edge."""
        if not self.visible:
            return y
        header = 34.0 if title else 8.0
        height = header + len(rows) * row_height + 10.0
        self.shapes.rect(x, y, width, height, PANEL_BG)
        self.shapes.rect_outline(x, y, width, height, 1.5, PANEL_EDGE)

        cursor = y + 8.0
        if title:
            self.text.draw(title, x + 14.0, cursor, 24.0, (0.98, 0.98, 1.0, 1.0))
            cursor += 30.0
        for label, value, colour in rows:
            self.text.draw(label, x + 14.0, cursor, text_size, (0.62, 0.68, 0.78, 1.0))
            self.text.draw(value, x + width * 0.52, cursor, text_size, colour)
            cursor += row_height
        return y + height + 10.0

    def bar(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        fraction: float,
        colour,
        background=(1.0, 1.0, 1.0, 0.12),
    ) -> None:
        """A budget bar. Reads faster than a number when it is the number
        going over budget that matters."""
        if not self.visible:
            return
        self.shapes.rect(x, y, width, height, background)
        clamped = max(0.0, min(1.0, float(fraction)))
        if clamped > 0.0:
            self.shapes.rect(x, y, width * clamped, height, colour)

    def hint(self, lines: Iterable[str], x: float, y: float,
             size: float = 20.0, colour=(0.75, 0.80, 0.88, 0.85)) -> None:
        if not self.visible:
            return
        for i, line in enumerate(lines):
            self.text.draw(line, x, y + i * (size + 6.0), size, colour)
