"""The one place that knows how camera space maps onto canvas space.

Tracking converts normalised landmarks into canvas pixels; the shaders
convert canvas pixels back into texture coordinates. If those two
disagree by even a little, the fracture does not radiate from the hand and
shard UVs slide against the frozen frame. So both sides derive their
numbers from a single CoverFit.

The mirror lives here too. The spec is explicit that the flip happens
*once*: the display is mirrored and landmark x becomes (1 - x). Folding it
into this transform means there is exactly one minus sign in the codebase
and no chance of double-flipping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CoverFit"]


@dataclass(frozen=True)
class CoverFit:
    """Aspect-preserving 'cover' fit of a camera image onto the canvas.

    The camera image is scaled up until it covers the canvas entirely and
    centred, so the overflowing axis is cropped equally at both ends.
    Nothing is ever letterboxed -- a black bar down the side of a shattered
    room would look like a bug.
    """

    camera_width: int
    camera_height: int
    canvas_width: int
    canvas_height: int
    mirror: bool = True

    @property
    def scale(self) -> float:
        return max(
            self.canvas_width / self.camera_width,
            self.canvas_height / self.camera_height,
        )

    @property
    def offset(self) -> tuple[float, float]:
        """Canvas-space top-left of the scaled camera image."""
        s = self.scale
        return (
            (self.canvas_width - self.camera_width * s) * 0.5,
            (self.canvas_height - self.camera_height * s) * 0.5,
        )

    # -- normalised landmarks -> canvas pixels ---------------------------

    def landmarks_to_canvas(self, norm: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Map normalised (x, y) in [0,1] camera space to canvas pixels.

        ``norm`` is any array whose last axis is 2. Mirroring is applied
        here and nowhere else.
        """
        s = self.scale
        ox, oy = self.offset
        if out is None:
            out = np.empty_like(norm, dtype=np.float32)
        w = np.float32(self.camera_width * s)
        h = np.float32(self.camera_height * s)
        if self.mirror:
            np.multiply(norm[..., 0], np.float32(-w), out=out[..., 0])
            np.add(out[..., 0], np.float32(ox + w), out=out[..., 0])
        else:
            np.multiply(norm[..., 0], w, out=out[..., 0])
            np.add(out[..., 0], np.float32(ox), out=out[..., 0])
        np.multiply(norm[..., 1], h, out=out[..., 1])
        np.add(out[..., 1], np.float32(oy), out=out[..., 1])
        return out

    # -- canvas pixels -> texture coordinates ----------------------------

    def uv_transform(self) -> tuple[float, float, float, float]:
        """``(ax, bx, ay, by)`` such that ``uv = (a*px + b)``.

        Handed straight to the shaders as a vec4 so sampling the camera
        texture is two multiply-adds with the mirror already baked in.
        """
        s = self.scale
        ox, oy = self.offset
        w = self.camera_width * s
        h = self.camera_height * s
        if self.mirror:
            ax, bx = -1.0 / w, 1.0 + ox / w
        else:
            ax, bx = 1.0 / w, -ox / w
        return (ax, bx, 1.0 / h, -oy / h)

    def canvas_to_uv(self, px: np.ndarray) -> np.ndarray:
        ax, bx, ay, by = self.uv_transform()
        out = np.empty_like(px, dtype=np.float32)
        out[..., 0] = px[..., 0] * ax + bx
        out[..., 1] = px[..., 1] * ay + by
        return out

    # -- diagnostics ------------------------------------------------------

    @property
    def visible_camera_rect(self) -> tuple[float, float, float, float]:
        """Portion of the camera image that survives the crop, in [0,1]."""
        s = self.scale
        vis_w = self.canvas_width / (self.camera_width * s)
        vis_h = self.canvas_height / (self.camera_height * s)
        return ((1.0 - vis_w) * 0.5, (1.0 - vis_h) * 0.5, vis_w, vis_h)
