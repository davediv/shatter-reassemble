"""Batched coloured geometry for overlays.

The debug skeleton, the tuning traces and the HUD panels are all built out
of thick lines, quads and discs. Drawing each one individually would be
dozens of draw calls per frame for something that is supposed to be a
diagnostic, so everything accumulates into one vertex buffer and goes out
in a single call.

Thick lines are expanded to quads on the CPU because core-profile OpenGL
caps glLineWidth at 1.0 on macOS -- a one-pixel skeleton over a 1080p
camera feed is illegible.
"""

from __future__ import annotations

import moderngl
import numpy as np

from .context import Display, make_program

__all__ = ["ShapeBatch"]

_FLOATS_PER_VERTEX = 6      # x, y, r, g, b, a


class ShapeBatch:
    """Accumulate triangles in canvas pixel space; flush in one draw call."""

    def __init__(self, display: Display, capacity: int = 8192,
                 *, shared: "ShapeBatch | None" = None) -> None:
        self.display = display
        self.ctx = display.ctx
        self._data = np.zeros((capacity, _FLOATS_PER_VERTEX), np.float32)
        self._count = 0
        self._cached_count = 0
        self._program = (
            shared._program
            if shared is not None
            else make_program(self.ctx, "shapes.vert", "shapes.frag")
        )
        self._buffer = self.ctx.buffer(reserve=self._data.nbytes, dynamic=True)
        self._vao = self.ctx.vertex_array(
            self._program, [(self._buffer, "2f 4f", "in_pos", "in_color")]
        )
        # Unit circle, resolved once and scaled per disc.
        self._unit_circle = {}

    def fork(self, capacity: int = 8192) -> "ShapeBatch":
        """Create an independent batch that shares the compiled shader."""
        return ShapeBatch(self.display, capacity, shared=self)

    # -- capacity ---------------------------------------------------------

    def _reserve(self, extra: int) -> int:
        need = self._count + extra
        if need > self._data.shape[0]:
            new_capacity = max(need, self._data.shape[0] * 2)
            grown = np.zeros((new_capacity, _FLOATS_PER_VERTEX), np.float32)
            grown[: self._count] = self._data[: self._count]
            self._data = grown
            self._buffer.orphan(self._data.nbytes)
            self._vao.release()
            self._vao = self.ctx.vertex_array(
                self._program, [(self._buffer, "2f 4f", "in_pos", "in_color")]
            )
        start = self._count
        self._count = need
        return start

    def begin(self) -> None:
        self._count = 0

    @property
    def vertex_count(self) -> int:
        return self._count

    # -- shapes -----------------------------------------------------------

    def lines(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        width: float,
        color,
    ) -> None:
        """N thick line segments at once. ``p0``/``p1`` are (N, 2)."""
        p0 = np.asarray(p0, np.float32).reshape(-1, 2)
        p1 = np.asarray(p1, np.float32).reshape(-1, 2)
        if p0.shape[0] == 0:
            return

        direction = p1 - p0
        length = np.hypot(direction[:, 0], direction[:, 1])
        keep = length > 1e-5
        if not keep.any():
            return
        p0, p1, direction, length = p0[keep], p1[keep], direction[keep], length[keep]

        normal = np.empty_like(direction)
        half = np.float32(width * 0.5)
        normal[:, 0] = -direction[:, 1] / length * half
        normal[:, 1] = direction[:, 0] / length * half

        n = p0.shape[0]
        corners = np.empty((n, 4, 2), np.float32)
        corners[:, 0] = p0 - normal
        corners[:, 1] = p0 + normal
        corners[:, 2] = p1 + normal
        corners[:, 3] = p1 - normal

        start = self._reserve(n * 6)
        out = self._data[start: start + n * 6].reshape(n, 6, _FLOATS_PER_VERTEX)
        out[:, 0, :2] = corners[:, 0]
        out[:, 1, :2] = corners[:, 1]
        out[:, 2, :2] = corners[:, 2]
        out[:, 3, :2] = corners[:, 0]
        out[:, 4, :2] = corners[:, 2]
        out[:, 5, :2] = corners[:, 3]
        out[:, :, 2:] = self._colors(color, n)[:, None, :]

    def line(self, p0, p1, width: float, color) -> None:
        self.lines(np.asarray(p0, np.float32)[None], np.asarray(p1, np.float32)[None],
                   width, color)

    def polyline(self, points: np.ndarray, width: float, color) -> None:
        points = np.asarray(points, np.float32).reshape(-1, 2)
        if points.shape[0] < 2:
            return
        self.lines(points[:-1], points[1:], width, color)

    def rect(self, x: float, y: float, w: float, h: float, color) -> None:
        start = self._reserve(6)
        out = self._data[start: start + 6]
        out[0, :2] = (x, y)
        out[1, :2] = (x + w, y)
        out[2, :2] = (x + w, y + h)
        out[3, :2] = (x, y)
        out[4, :2] = (x + w, y + h)
        out[5, :2] = (x, y + h)
        out[:, 2:] = self._colors(color, 1)[0]

    def rect_outline(self, x: float, y: float, w: float, h: float,
                     width: float, color) -> None:
        corners = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32
        )
        self.lines(corners, np.roll(corners, -1, axis=0), width, color)

    def discs(self, centers: np.ndarray, radii, color, segments: int = 14) -> None:
        centers = np.asarray(centers, np.float32).reshape(-1, 2)
        n = centers.shape[0]
        if n == 0:
            return
        radii = np.broadcast_to(np.asarray(radii, np.float32).reshape(-1), (n,))

        if segments not in self._unit_circle:
            angles = np.linspace(0.0, 2.0 * np.pi, segments + 1, dtype=np.float32)
            self._unit_circle[segments] = np.stack(
                [np.cos(angles), np.sin(angles)], axis=1
            )
        ring = self._unit_circle[segments]

        start = self._reserve(n * segments * 3)
        out = self._data[start: start + n * segments * 3].reshape(
            n, segments, 3, _FLOATS_PER_VERTEX
        )
        scaled = ring[None, :, :] * radii[:, None, None] + centers[:, None, :]
        out[:, :, 0, :2] = centers[:, None, :]
        out[:, :, 1, :2] = scaled[:, :-1]
        out[:, :, 2, :2] = scaled[:, 1:]
        out[:, :, :, 2:] = self._colors(color, n)[:, None, None, :]

    def disc(self, center, radius: float, color, segments: int = 14) -> None:
        self.discs(np.asarray(center, np.float32)[None], np.float32([radius]),
                   color, segments)

    # -- output -----------------------------------------------------------

    @staticmethod
    def _colors(color, n: int) -> np.ndarray:
        arr = np.asarray(color, np.float32)
        if arr.ndim == 1:
            return np.broadcast_to(arr, (n, 4))
        return arr.reshape(-1, 4)

    def flush(self, depth_test: bool = False) -> int:
        """Draw everything accumulated. Returns the vertex count drawn."""
        if self._count == 0:
            self._cached_count = 0
            return 0
        # A contiguous row slice satisfies the buffer protocol, so this
        # uploads straight out of the accumulation array with no copy.
        self._buffer.write(self._data[: self._count])
        self._program["u_canvas_size"].value = (
            self.display.canvas_width, self.display.canvas_height
        )
        if depth_test:
            self.ctx.enable(moderngl.DEPTH_TEST)
        else:
            self.ctx.disable(moderngl.DEPTH_TEST)
        self._vao.render(moderngl.TRIANGLES, vertices=self._count)
        drawn = self._count
        self._cached_count = drawn
        self._count = 0
        return drawn

    def render_cached(self, depth_test: bool = False) -> int:
        """Redraw the last flushed buffer without rebuilding or uploading it."""
        if self._cached_count == 0:
            return 0
        self._program["u_canvas_size"].value = (
            self.display.canvas_width, self.display.canvas_height
        )
        if depth_test:
            self.ctx.enable(moderngl.DEPTH_TEST)
        else:
            self.ctx.disable(moderngl.DEPTH_TEST)
        self._vao.render(moderngl.TRIANGLES, vertices=self._cached_count)
        return self._cached_count

    def release(self) -> None:
        for obj in (self._vao, self._buffer):
            try:
                obj.release()
            except Exception:
                pass
