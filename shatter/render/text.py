"""A bitmap font atlas, baked at startup, drawn as one batched call.

The HUD needs to report frame budgets, ladder state and live gesture
signals -- numbers that change every frame. Rasterising text on the CPU
and uploading it per frame would cost more than the effects it is
measuring, so glyphs are baked once into an atlas texture and drawn as
textured quads.

Laid out on a fixed grid, which makes the font effectively monospaced.
That matters more than it sounds: a HUD of jittering proportional digits
is unreadable at 60fps, while monospaced columns hold still.
"""

from __future__ import annotations

import cv2
import moderngl
import numpy as np

from .context import Display, make_program

__all__ = ["TextRenderer"]

FIRST_CHAR = 32
LAST_CHAR = 126
COLUMNS = 16

# Baked oversized and minified at draw time, so the HUD stays crisp at any
# scale without a second atlas.
CELL_W = 40
CELL_H = 56
_FLOATS_PER_VERTEX = 8      # x, y, u, v, r, g, b, a


class TextRenderer:
    def __init__(self, display: Display, capacity: int = 4096) -> None:
        self.display = display
        self.ctx = display.ctx
        self.atlas, self.advance = self._bake()
        self._texture = self.ctx.texture(
            (self.atlas.shape[1], self.atlas.shape[0]), 1, self.atlas.tobytes()
        )
        self._texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._texture.repeat_x = self._texture.repeat_y = False

        self._program = make_program(self.ctx, "text.vert", "text.frag")
        self._data = np.zeros((capacity, _FLOATS_PER_VERTEX), np.float32)
        self._count = 0
        self._buffer = self.ctx.buffer(reserve=self._data.nbytes, dynamic=True)
        self._vao = self._make_vao()

    def _make_vao(self) -> moderngl.VertexArray:
        return self.ctx.vertex_array(
            self._program,
            [(self._buffer, "2f 2f 4f", "in_pos", "in_uv", "in_color")],
        )

    # -- atlas ------------------------------------------------------------

    @staticmethod
    def _bake() -> tuple[np.ndarray, float]:
        count = LAST_CHAR - FIRST_CHAR + 1
        rows = (count + COLUMNS - 1) // COLUMNS
        atlas = np.zeros((rows * CELL_H, COLUMNS * CELL_W), np.uint8)

        # Pick the scale that makes the widest glyph fit its cell, so no
        # glyph is ever clipped by the grid.
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 1.0
        widest = max(
            cv2.getTextSize(chr(c), font, scale, 2)[0][0]
            for c in range(FIRST_CHAR, LAST_CHAR + 1)
        )
        scale *= (CELL_W - 8) / max(widest, 1)

        baseline = int(CELL_H * 0.72)
        for i in range(count):
            char = chr(FIRST_CHAR + i)
            cx = (i % COLUMNS) * CELL_W
            cy = (i // COLUMNS) * CELL_H
            (w, _), _ = cv2.getTextSize(char, font, scale, 2)
            cv2.putText(
                atlas, char, (cx + (CELL_W - w) // 2, cy + baseline),
                font, scale, 255, 2, cv2.LINE_AA,
            )
        # Monospace advance: a little tighter than the cell so columns of
        # digits read as a block rather than as scattered characters.
        return atlas, 0.62

    # -- drawing ----------------------------------------------------------

    def begin(self) -> None:
        self._count = 0

    def measure(self, text: str, size: float) -> tuple[float, float]:
        return len(text) * size * self.advance, size

    def draw(
        self,
        text: str,
        x: float,
        y: float,
        size: float = 22.0,
        color=(1.0, 1.0, 1.0, 1.0),
    ) -> float:
        """Queue a string with its top-left at (x, y). Returns the end x."""
        if not text:
            return x
        printable = [c for c in text if FIRST_CHAR <= ord(c) <= LAST_CHAR]
        needed = self._count + len(printable) * 6
        if needed > self._data.shape[0]:
            grown = np.zeros((max(needed, self._data.shape[0] * 2),
                              _FLOATS_PER_VERTEX), np.float32)
            grown[: self._count] = self._data[: self._count]
            self._data = grown
            self._buffer.orphan(self._data.nbytes)
            self._vao.release()
            self._vao = self._make_vao()

        step = size * self.advance
        glyph_w = size * (CELL_W / CELL_H)
        du, dv = 1.0 / COLUMNS, CELL_H / self.atlas.shape[0]
        rgba = np.asarray(color, np.float32)

        pen = x
        for char in printable:
            index = ord(char) - FIRST_CHAR
            if char != " ":
                u = (index % COLUMNS) * du
                v = (index // COLUMNS) * dv
                out = self._data[self._count: self._count + 6]
                corners = (
                    (pen, y, u, v),
                    (pen + glyph_w, y, u + du, v),
                    (pen + glyph_w, y + size, u + du, v + dv),
                    (pen, y, u, v),
                    (pen + glyph_w, y + size, u + du, v + dv),
                    (pen, y + size, u, v + dv),
                )
                for i, (px, py, pu, pv) in enumerate(corners):
                    out[i, 0] = px
                    out[i, 1] = py
                    out[i, 2] = pu
                    out[i, 3] = pv
                out[:, 4:] = rgba
                self._count += 6
            pen += step
        return pen

    def flush(self) -> int:
        if self._count == 0:
            return 0
        self._buffer.write(self._data[: self._count])
        self._texture.use(0)
        self._program["u_atlas"].value = 0
        self._program["u_canvas_size"].value = (
            self.display.canvas_width, self.display.canvas_height
        )
        self.ctx.disable(moderngl.DEPTH_TEST)
        self._vao.render(moderngl.TRIANGLES, vertices=self._count)
        drawn = self._count
        self._count = 0
        return drawn

    def release(self) -> None:
        for obj in (self._vao, self._buffer, self._texture):
            try:
                obj.release()
            except Exception:
                pass
