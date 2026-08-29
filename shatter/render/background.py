"""The void behind the shards, and the scene buffer the glass refracts.

Two jobs in one place because they are the same pixels. The void -- black
with the person's silhouette faintly outlined -- is drawn into an
offscreen scene buffer, that buffer is blitted to the canvas as the
background, and the shard shader then samples it for refraction. Drawing
it once and reading it twice is what makes refraction of the *actual*
background affordable rather than a guess at what might be behind.
"""

from __future__ import annotations

import moderngl
import numpy as np

from ..viewport import CoverFit
from .context import Display, make_program

__all__ = ["VoidLayer"]

# Neutral cold white; the silhouette should read as absence, not as a
# coloured light source.
DEFAULT_TINT = (0.62, 0.78, 0.95)


class VoidLayer:
    def __init__(self, display: Display, fit: CoverFit) -> None:
        self.display = display
        self.ctx = display.ctx
        self.fit = fit

        self.scene_color = self.ctx.texture(
            (display.canvas_width, display.canvas_height), 3, dtype="f1"
        )
        self.scene_color.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scene_color.repeat_x = self.scene_color.repeat_y = False
        self.scene = self.ctx.framebuffer(color_attachments=[self.scene_color])

        # A single dark texel stands in until segmentation produces
        # anything, so the shaders never branch on whether it exists.
        self.mask = self.ctx.texture((1, 1), 1, data=b"\x00")
        self.mask.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.mask.repeat_x = self.mask.repeat_y = False
        self._mask_size = (1, 1)
        self.has_mask = False
        self.scene_dirty = True

        self._program = make_program(self.ctx, "fullscreen.vert", "void.frag")
        self._vao = self.ctx.vertex_array(self._program, [])
        self._blit = make_program(self.ctx, "fullscreen.vert", "blit.frag")
        self._blit_vao = self.ctx.vertex_array(self._blit, [])

    def upload_mask(self, mask: np.ndarray) -> None:
        height, width = mask.shape[:2]
        if (width, height) != self._mask_size:
            self.mask.release()
            self.mask = self.ctx.texture((width, height), 1,
                                         np.ascontiguousarray(mask).tobytes())
            self.mask.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.mask.repeat_x = self.mask.repeat_y = False
            self._mask_size = (width, height)
        else:
            self.mask.write(np.ascontiguousarray(mask))
        self.has_mask = True
        self.scene_dirty = True

    def render_scene(self, outline: float = 1.25, fill: float = 0.055,
                     tint=DEFAULT_TINT) -> None:
        """Draw the void into the offscreen scene buffer."""
        self.scene.use()
        self.ctx.viewport = (0, 0, self.display.canvas_width,
                             self.display.canvas_height)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.disable(moderngl.DEPTH_TEST)

        self.mask.use(0)
        self._program["u_mask"].value = 0
        self._program["u_canvas_size"].value = (
            self.display.canvas_width, self.display.canvas_height
        )
        self._program["u_uv"].value = self.fit.uv_transform()
        self._program["u_mask_texel"].value = (
            1.0 / max(self._mask_size[0], 1), 1.0 / max(self._mask_size[1], 1)
        )
        self._program["u_outline"].value = float(outline) if self.has_mask else 0.0
        self._program["u_fill"].value = float(fill) if self.has_mask else 0.0
        self._program["u_tint"].value = tint
        self._vao.render(moderngl.TRIANGLES, vertices=3)
        self.scene_dirty = False

    def blit_to_canvas(self) -> None:
        """Copy the scene buffer into the canvas as the background."""
        self.display.begin_frame()
        self.scene_color.use(0)
        self._blit["u_canvas"].value = 0
        self._blit["u_canvas_size"].value = (
            self.display.canvas_width, self.display.canvas_height
        )
        self._blit["u_rect"].value = (
            0.0, 0.0,
            1.0 / self.display.canvas_width, 1.0 / self.display.canvas_height,
        )
        self.ctx.disable(moderngl.DEPTH_TEST)
        self._blit_vao.render(moderngl.TRIANGLES, vertices=3)

    def release(self) -> None:
        for obj in (self.scene, self.scene_color, self.mask):
            try:
                obj.release()
            except Exception:
                pass
