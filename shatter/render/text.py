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
        self._queued: list = []
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
        self._queued.clear()

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
        """Queue a string with its top-left at (x, y). Returns the end x.

        Queued, not built. The HUD draws dozens of short strings a frame,
        and numpy's per-call overhead on a twelve-character array dwarfs
        the work itself -- so every queued string is expanded together in
        one pass at flush time, which makes the cost depend on the total
        glyph count rather than on how many strings it arrived in.
        """
        step = size * self.advance
        if text:
            self._queued.append((text, float(x), float(y), float(size),
                                 tuple(color)))
        return x + len(text) * step

    def _build(self) -> int:
        if not self._queued:
            return 0

        codes = []
        origin_x = []
        origin_y = []
        sizes = []
        colors = []
        for text, x, y, size, color in self._queued:
            raw = np.frombuffer(text.encode("latin-1", "replace"), np.uint8)
            codes.append(raw)
            step = size * self.advance
            origin_x.append(x + np.arange(raw.size, dtype=np.float32) * step)
            origin_y.append(np.full(raw.size, y, np.float32))
            sizes.append(np.full(raw.size, size, np.float32))
            colors.append(np.tile(np.asarray(color, np.float32), (raw.size, 1)))

        codes = np.concatenate(codes).astype(np.int32) - FIRST_CHAR
        pen = np.concatenate(origin_x)
        top = np.concatenate(origin_y)
        size = np.concatenate(sizes)
        color = np.concatenate(colors)

        visible = (codes >= 0) & (codes <= LAST_CHAR - FIRST_CHAR) & (codes != 32 - FIRST_CHAR)
        if not visible.any():
            return 0
        codes = codes[visible]
        pen = pen[visible]
        top = top[visible]
        size = size[visible]
        color = color[visible]
        count = codes.size

        needed = count * 6
        if needed > self._data.shape[0]:
            self._data = np.zeros((max(needed, self._data.shape[0] * 2),
                                   _FLOATS_PER_VERTEX), np.float32)
            self._buffer.orphan(self._data.nbytes)
            self._vao.release()
            self._vao = self._make_vao()

        du, dv = 1.0 / COLUMNS, CELL_H / self.atlas.shape[0]
        u0 = (codes % COLUMNS).astype(np.float32) * du
        v0 = (codes // COLUMNS).astype(np.float32) * dv
        u1, v1 = u0 + du, v0 + dv
        x0 = pen
        x1 = pen + size * (CELL_W / CELL_H)
        y0 = top
        y1 = top + size

        out = self._data[: count * 6].reshape(count, 6, _FLOATS_PER_VERTEX)
        for slot, (px, py, pu, pv) in enumerate((
            (x0, y0, u0, v0), (x1, y0, u1, v0), (x1, y1, u1, v1),
            (x0, y0, u0, v0), (x1, y1, u1, v1), (x0, y1, u0, v1),
        )):
            out[:, slot, 0] = px
            out[:, slot, 1] = py
            out[:, slot, 2] = pu
            out[:, slot, 3] = pv
        out[:, :, 4:] = color[:, None, :]
        self._count = count * 6
        return self._count

    def flush(self) -> int:
        if self._build() == 0:
            self._queued.clear()
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
        self._queued.clear()
        return drawn

    def release(self) -> None:
        for obj in (self._vao, self._buffer, self._texture):
            try:
                obj.release()
            except Exception:
                pass
