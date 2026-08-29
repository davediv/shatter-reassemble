"""The camera feed as a canvas-filling layer, plus the frozen frame.

Two textures live here for the whole run: the live camera upload, and the
snapshot taken at the instant of the snap. The frozen one is what every
shard carries a slice of -- the reason a pile of glass on the floor is
still recognisably *your room* -- and it is captured with a GPU-side copy
so the snap frame pays nothing for it.

Both are RGBA8 even though the camera hands us three channels. RGB8 is
not a native texture format on Apple silicon and the driver expands it
during upload: measured 1.152ms per 1280x720 frame against 0.477ms for
RGBA8. Padding to four channels on the CPU costs 0.08ms, so the swap buys
back roughly 0.6ms of every frame. The alpha byte is never read -- the
shader still swizzles .bgr, because the bytes are OpenCV's BGR order.
"""

from __future__ import annotations

import cv2
import moderngl
import numpy as np

from ..camera import Frame
from ..viewport import CoverFit
from .context import Display, make_program

__all__ = ["VideoLayer"]


class VideoLayer:
    def __init__(self, display: Display, fit: CoverFit) -> None:
        self.display = display
        self.ctx = display.ctx
        self.fit = fit

        size = (fit.camera_width, fit.camera_height)
        self.live = self._make_texture(size)
        self.frozen = self._make_texture(size)
        # Staging buffer for the BGR -> BGRA pad. Allocated once; the
        # conversion writes into it in place every frame.
        self._staged = np.empty((fit.camera_height, fit.camera_width, 4), np.uint8)

        self._freeze_fbo = self.ctx.framebuffer(color_attachments=[self.frozen])
        self._program = make_program(self.ctx, "fullscreen.vert", "video.frag")
        self._vao = self.ctx.vertex_array(self._program, [])
        self._uploaded = -1
        self.has_frozen = False

    def _make_texture(self, size: tuple[int, int]) -> moderngl.Texture:
        texture = self.ctx.texture(size, 4, dtype="f1")
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = texture.repeat_y = False
        return texture

    def upload(self, frame: Frame) -> bool:
        """Push a camera frame to the GPU. Returns False if already current."""
        if frame.index == self._uploaded:
            return False
        data = frame.data
        if data.shape[2] == 3:
            cv2.cvtColor(data, cv2.COLOR_BGR2BGRA, dst=self._staged)
            data = self._staged
        self.live.write(data)
        self._uploaded = frame.index
        return True

    def freeze(self) -> None:
        """Snapshot the live texture. GPU-to-GPU, no readback."""
        self.ctx.copy_framebuffer(self._freeze_fbo, self._live_fbo)
        self.has_frozen = True

    @property
    def _live_fbo(self) -> moderngl.Framebuffer:
        fbo = getattr(self, "_live_fbo_cache", None)
        if fbo is None:
            fbo = self.ctx.framebuffer(color_attachments=[self.live])
            self._live_fbo_cache = fbo
        return fbo

    def draw(self, freeze: float = 0.0, exposure: float = 1.0) -> None:
        self.live.use(0)
        self.frozen.use(1)
        self._program["u_video"].value = 0
        self._program["u_frozen"].value = 1
        self._program["u_canvas_size"].value = (
            self.display.canvas_width, self.display.canvas_height
        )
        self._program["u_uv"].value = self.fit.uv_transform()
        self._program["u_freeze"].value = float(freeze)
        self._program["u_exposure"].value = float(exposure)
        self._vao.render(moderngl.TRIANGLES, vertices=3)

    def release(self) -> None:
        for obj in (self.live, self.frozen, self._freeze_fbo):
            try:
                obj.release()
            except Exception:
                pass
