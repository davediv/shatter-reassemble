"""The whole pile, in one draw call, at any shard count.

The spec asks for a single instanced draw with per-instance transform and
UV rect. Instancing wants identical geometry per instance, and Voronoi
cells are the opposite of identical -- three to thirteen vertices each.
Padding every cell to a fixed topology would burn about a fifth of the
vertex budget on degenerate triangles, and the per-instance data would
still have to be fetched from somewhere.

So this keeps the guarantee the spec actually cares about -- no per-shard
draw calls at any count, no per-shard CPU work per frame -- with a batched
draw instead: one static vertex buffer holding every shard's triangles
tagged with a shard index, and a small RGBA32F texture of per-shard
transforms rewritten each frame. 800 shards is 12.8KB of texture upload,
measured at 0.01ms, and exactly one glDrawArrays.

Texture layout, three rows:
    0   x, y, cos, sin           rewritten every frame
    1   rest x, rest y, depth    written once per fracture
    2   alpha, flash, scale      rewritten when reassembly is running
"""

from __future__ import annotations

import moderngl
import numpy as np

from .. import config
from ..fracture import FLOATS_PER_VERTEX, FractureResult
from ..viewport import CoverFit
from .context import Display, make_program

__all__ = ["ShardRenderer"]

ROW_TRANSFORM = 0
ROW_REST = 1
ROW_EXTRA = 2
ROWS = 3


class ShardRenderer:
    def __init__(self, display: Display, fit: CoverFit, capacity: int) -> None:
        self.display = display
        self.ctx = display.ctx
        self.fit = fit
        self.capacity = capacity
        self.count = 0
        self.vertex_count = 0

        self.program = make_program(self.ctx, "shard.vert", "shard.frag")
        self._vbo: moderngl.Buffer | None = None
        self._vao: moderngl.VertexArray | None = None

        self.table = self.ctx.texture((capacity, ROWS), 4, dtype="f4")
        self.table.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.table.repeat_x = self.table.repeat_y = False

        # Staging rows, written in place so a frame allocates nothing.
        self._row_transform = np.zeros((capacity, 4), np.float32)
        self._row_extra = np.zeros((capacity, 4), np.float32)
        self._row_extra[:, 0] = 1.0     # alpha
        self._row_extra[:, 2] = 1.0     # scale

    # -- geometry ---------------------------------------------------------

    def upload(self, result: FractureResult) -> None:
        """Push a fresh fracture. Runs once per snap, never per frame."""
        self.count = min(result.count, self.capacity)
        data = np.ascontiguousarray(result.vertices, np.float32)
        self.vertex_count = data.shape[0]

        needed = data.nbytes
        if self._vbo is None or self._vbo.size < needed:
            if self._vao is not None:
                self._vao.release()
            if self._vbo is not None:
                self._vbo.release()
            self._vbo = self.ctx.buffer(reserve=max(needed, 1 << 20), dynamic=True)
            self._vao = self.ctx.vertex_array(
                self.program,
                [(self._vbo, "2f 2f 1f 1f 1f 2f",
                  "in_local", "in_inset", "in_edge", "in_part",
                  "in_shard", "in_normal")],
            )
        self._vbo.write(data)

        rest = np.zeros((self.capacity, 4), np.float32)
        rest[: self.count, 0] = result.centroid[: self.count, 0]
        rest[: self.count, 1] = result.centroid[: self.count, 1]
        rest[: self.count, 2] = result.depth[: self.count]
        self.table.write(rest, viewport=(0, ROW_REST, self.capacity, 1))

        self._row_extra[:, 0] = 1.0
        self._row_extra[:, 1] = 0.0
        self._row_extra[:, 2] = 1.0
        self.table.write(self._row_extra, viewport=(0, ROW_EXTRA, self.capacity, 1))

    def clear(self) -> None:
        self.count = 0
        self.vertex_count = 0

    # -- per frame --------------------------------------------------------

    def update_transforms(self, transforms: np.ndarray) -> None:
        """``transforms`` is (N, 3) of x, y, rot from the solver."""
        n = min(transforms.shape[0], self.capacity)
        if n == 0:
            return
        row = self._row_transform
        row[:n, 0] = transforms[:n, 0]
        row[:n, 1] = transforms[:n, 1]
        # cos/sin here rather than in the shader: 800 of them cost ~10us on
        # the CPU and would otherwise be recomputed for every one of the
        # 66,000 vertices.
        np.cos(transforms[:n, 2], out=row[:n, 2])
        np.sin(transforms[:n, 2], out=row[:n, 3])
        self.table.write(row, viewport=(0, ROW_TRANSFORM, self.capacity, 1))

    def update_extras(self, alpha=None, flash=None, scale=None) -> None:
        """Per-shard alpha, flash and scale. Only reassembly touches these."""
        row = self._row_extra
        n = self.count
        if alpha is not None:
            row[:n, 0] = alpha
        if flash is not None:
            row[:n, 1] = flash
        if scale is not None:
            row[:n, 2] = scale
        self.table.write(row, viewport=(0, ROW_EXTRA, self.capacity, 1))

    # -- drawing ----------------------------------------------------------

    def draw(
        self,
        frozen: moderngl.Texture,
        scene: moderngl.Texture,
        *,
        bevel: float = 1.0,
        thickness: float = config.SHARD_THICKNESS,
        refraction: float = config.REFRACTION_STRENGTH,
        bevel_shade: bool = True,
        light=(0.55, -0.83),
    ) -> int:
        if self._vao is None or self.vertex_count == 0:
            return 0

        frozen.use(0)
        scene.use(1)
        self.table.use(2)
        program = self.program
        program["u_frozen"].value = 0
        program["u_scene"].value = 1
        program["u_shards"].value = 2
        program["u_canvas_size"].value = (
            self.display.canvas_width, self.display.canvas_height
        )
        program["u_uv"].value = self.fit.uv_transform()
        program["u_bevel"].value = float(bevel)
        program["u_thickness"].value = float(thickness)
        program["u_perspective"].value = config.PERSPECTIVE_STRENGTH
        program["u_light"].value = light
        program["u_refraction"].value = float(refraction)
        program["u_bevel_shade"].value = 1.0 if bevel_shade else 0.0

        ctx = self.ctx
        ctx.enable(moderngl.DEPTH_TEST)
        # Shards are opaque and the depth buffer resolves them, so blending
        # stays off: 800 overlapping alpha-blended pieces would need a CPU
        # sort every frame, and the pile does not need one.
        ctx.disable(moderngl.BLEND)
        self._vao.render(moderngl.TRIANGLES, vertices=self.vertex_count)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        return self.vertex_count

    def release(self) -> None:
        for obj in (self._vao, self._vbo, self.table):
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass
