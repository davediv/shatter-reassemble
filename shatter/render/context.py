"""GL context, offscreen canvas, and the window that shows it.

Everything is drawn into an offscreen 1920x1080 canvas framebuffer and
only then blitted to the window. Three things fall out of that:

* Canvas coordinates are fixed, so physics, fracture and shard geometry
  all work in one stable pixel space no matter how the window is sized or
  which display it lands on.
* The recorder has a clean, correctly-sized source that never includes
  window chrome or a resize-in-progress.
* Headless mode is the same code path minus the window, which is what
  makes the benchmarks in tools/ measure the real renderer instead of an
  approximation of it.

Canvas space is y-down, origin top-left. Every module in this codebase
uses that convention, so y is flipped exactly once -- in fullscreen.vert.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import moderngl
import numpy as np

SHADER_DIR = Path(__file__).resolve().parent.parent / "shaders"

__all__ = ["Display", "load_shader", "make_program"]

_shader_cache: dict[str, str] = {}


def load_shader(name: str) -> str:
    """Read a shader source, with an ``#include``-free one-level cache."""
    if name not in _shader_cache:
        _shader_cache[name] = (SHADER_DIR / name).read_text()
    return _shader_cache[name]


def make_program(ctx: moderngl.Context, vert: str, frag: str) -> moderngl.Program:
    try:
        return ctx.program(
            vertex_shader=load_shader(vert), fragment_shader=load_shader(frag)
        )
    except Exception as exc:
        raise RuntimeError(f"failed to build {vert} + {frag}:\n{exc}") from exc


@dataclass
class KeyEvent:
    key: int
    scancode: int
    action: int
    mods: int


class Display:
    """Owns the GL context, the canvas framebuffer and (optionally) a window."""

    def __init__(
        self,
        canvas_width: int,
        canvas_height: int,
        *,
        headless: bool = False,
        vsync: bool = True,
        fullscreen: bool = False,
        title: str = "Shatter & Reassemble",
        window_scale: float = 0.75,
    ) -> None:
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.headless = headless
        self.vsync = vsync
        self._window = None
        self._glfw = None
        self.key_events: list[KeyEvent] = []
        self._on_key: Optional[Callable[[KeyEvent], None]] = None

        if headless:
            self.ctx = moderngl.create_standalone_context(require=410)
            self.window_width = canvas_width
            self.window_height = canvas_height
        else:
            self.ctx = self._create_window(
                canvas_width, canvas_height, fullscreen, title, window_scale, vsync
            )

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # The canvas: colour + depth. Depth is what lets 800 overlapping
        # shards be drawn in one call without sorting them on the CPU.
        self.canvas_color = self.ctx.texture((canvas_width, canvas_height), 4, dtype="f1")
        self.canvas_color.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.canvas_color.repeat_x = False
        self.canvas_color.repeat_y = False
        self.canvas_depth = self.ctx.depth_texture((canvas_width, canvas_height))
        self.canvas = self.ctx.framebuffer(
            color_attachments=[self.canvas_color], depth_attachment=self.canvas_depth
        )

        self._blit = make_program(self.ctx, "fullscreen.vert", "blit.frag")
        self._blit_vao = self.ctx.vertex_array(self._blit, [])
        self.frame_index = 0

    # -- window -----------------------------------------------------------

    def _create_window(
        self, cw: int, ch: int, fullscreen: bool, title: str,
        window_scale: float, vsync: bool,
    ) -> moderngl.Context:
        import glfw

        self._glfw = glfw
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")

        # macOS gives OpenGL 4.1 core and nothing newer, and only with
        # forward compatibility requested.
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)

        monitor = glfw.get_primary_monitor() if fullscreen else None
        if fullscreen:
            mode = glfw.get_video_mode(monitor)
            width, height = mode.size.width, mode.size.height
        else:
            width = int(cw * window_scale)
            height = int(ch * window_scale)

        window = glfw.create_window(width, height, title, monitor, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("could not create a GL 4.1 window")
        self._window = window
        glfw.make_context_current(window)
        glfw.swap_interval(1 if vsync else 0)
        glfw.set_key_callback(window, self._key_callback)

        ctx = moderngl.create_context(require=410)
        self.window_width, self.window_height = glfw.get_framebuffer_size(window)
        return ctx

    def _key_callback(self, window, key, scancode, action, mods) -> None:
        event = KeyEvent(key, scancode, action, mods)
        self.key_events.append(event)
        if self._on_key is not None:
            self._on_key(event)

    def on_key(self, callback: Callable[[KeyEvent], None]) -> None:
        self._on_key = callback

    # -- frame ------------------------------------------------------------

    def begin_frame(self) -> None:
        self.canvas.use()
        self.ctx.viewport = (0, 0, self.canvas_width, self.canvas_height)

    def present(self) -> None:
        """Blit the canvas into the window and swap."""
        self.frame_index += 1
        if self.headless:
            return

        glfw = self._glfw
        self.window_width, self.window_height = glfw.get_framebuffer_size(self._window)
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, self.window_width, self.window_height)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        # Contain the canvas in the window, preserving aspect.
        scale = min(
            self.window_width / self.canvas_width,
            self.window_height / self.canvas_height,
        )
        draw_w = self.canvas_width * scale
        draw_h = self.canvas_height * scale
        off_x = (self.window_width - draw_w) * 0.5
        off_y = (self.window_height - draw_h) * 0.5

        self.canvas_color.use(0)
        self._blit["u_canvas"].value = 0
        self._blit["u_canvas_size"].value = (self.window_width, self.window_height)
        self._blit["u_rect"].value = (off_x, off_y, 1.0 / draw_w, 1.0 / draw_h)
        # No save/restore of depth state here: moderngl's depth_func is
        # write-only and reading it raises. Every pass sets up the state it
        # needs anyway, so there is nothing to preserve.
        self.ctx.disable(moderngl.DEPTH_TEST)
        self._blit_vao.render(moderngl.TRIANGLES, vertices=3)
        glfw.swap_buffers(self._window)

    def poll(self) -> None:
        if not self.headless:
            self._glfw.poll_events()

    def drain_keys(self) -> list[KeyEvent]:
        events, self.key_events = self.key_events, []
        return events

    @property
    def should_close(self) -> bool:
        if self.headless:
            return False
        return bool(self._glfw.window_should_close(self._window))

    def request_close(self) -> None:
        if not self.headless:
            self._glfw.set_window_should_close(self._window, True)

    def read_canvas(self) -> np.ndarray:
        """Canvas contents as an (H, W, 3) uint8 array, top row first."""
        raw = self.canvas.read(components=3, alignment=1)
        image = np.frombuffer(raw, np.uint8).reshape(
            self.canvas_height, self.canvas_width, 3
        )
        # GL hands back bottom-up; flip so row 0 is the top of the image.
        return image[::-1]

    def release(self) -> None:
        try:
            self.canvas.release()
            self.canvas_color.release()
            self.canvas_depth.release()
        except Exception:
            pass
        if not self.headless and self._window is not None:
            self._glfw.destroy_window(self._window)
            self._glfw.terminate()
            self._window = None
