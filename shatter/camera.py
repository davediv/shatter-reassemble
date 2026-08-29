"""Frame acquisition, decoupled from everything downstream.

Capture runs on its own thread with latest-wins semantics: the render loop
never blocks waiting on the camera, and a slow consumer never builds a
backlog of stale frames. What you get from ``latest()`` is always the
newest frame that exists, which is the only thing a real-time effect can
use anyway.

Buffer lifetime
---------------
The capture thread cycles through a small ring of preallocated frame
buffers so steady-state capture allocates nothing. That means a published
frame is only valid until the ring wraps around to it again -- RING_SIZE
capture periods, 66ms at 60fps. Every consumer here finishes with a frame
in well under that (GL upload <1ms, hand tracking ~10-15ms), so the frame
is stable while in use. Anything that needs to hold a frame longer must
copy it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import config

__all__ = ["Frame", "FrameSource", "CameraSource", "SyntheticSource",
           "VideoFileSource", "open_source"]

# Enough slack that a consumer can hold a frame for several capture periods.
RING_SIZE = 4


@dataclass(frozen=True)
class Frame:
    """One captured frame. ``data`` is BGR uint8, HxWx3.

    Colour conversion is deliberately *not* done here. The GL upload
    swizzles BGR in the shader for free, and only the frames the tracker
    actually consumes need converting to RGB -- which is strictly less
    work than converting every frame at capture.
    """

    data: np.ndarray
    index: int
    timestamp: float          # perf_counter seconds at capture


class FrameSource:
    """Common machinery for anything that produces frames on a thread."""

    def __init__(self, width: int, height: int, fps: float) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.frames_captured = 0
        self.frames_dropped = 0

        self._ring: list[np.ndarray] = []
        self._slot = 0
        self._latest: Optional[Frame] = None
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None
        self._measured_fps = 0.0
        self._fps_t0 = 0.0
        self._fps_n0 = 0
        # Set on publish, cleared as soon as any consumer takes the frame.
        # A publish that lands while it is still set means nobody kept up.
        self._unconsumed = False

    # -- lifecycle --------------------------------------------------------

    def start(self) -> "FrameSource":
        if self._thread is not None:
            return self
        self._ring = [
            np.empty((self.height, self.width, 3), np.uint8)
            for _ in range(RING_SIZE)
        ]
        self._fps_t0 = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run, name=type(self).__name__, daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._release()

    def __enter__(self) -> "FrameSource":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- consumer API -----------------------------------------------------

    def latest(self, after_index: int = -1) -> Optional[Frame]:
        """Newest frame, or None if nothing newer than ``after_index``."""
        with self._lock:
            f = self._latest
            if f is None or f.index <= after_index:
                return None
            self._unconsumed = False
            return f

    def wait_for_frame(self, after_index: int, timeout: float) -> Optional[Frame]:
        """Block until a frame newer than ``after_index`` exists."""
        deadline = time.perf_counter() + timeout
        with self._new_frame:
            while True:
                f = self._latest
                if f is not None and f.index > after_index:
                    self._unconsumed = False
                    return f
                remaining = deadline - time.perf_counter()
                if remaining <= 0 or self._stop.is_set():
                    return None
                self._new_frame.wait(remaining)

    @property
    def measured_fps(self) -> float:
        return self._measured_fps

    @property
    def error(self) -> Optional[BaseException]:
        return self._error

    # -- subclass hooks ---------------------------------------------------

    def _acquire(self, into: np.ndarray) -> bool:
        raise NotImplementedError

    def _release(self) -> None:
        pass

    # -- internals --------------------------------------------------------

    def _publish(self, buf: np.ndarray) -> None:
        now = time.perf_counter()
        self.frames_captured += 1
        frame = Frame(buf, self.frames_captured, now)
        with self._new_frame:
            if self._unconsumed:
                # We are overwriting a frame no consumer ever looked at.
                # Not an error -- it is exactly the latest-wins behaviour
                # we want -- but a high rate means a stalled consumer.
                self.frames_dropped += 1
            self._latest = frame
            self._unconsumed = True
            self._new_frame.notify_all()

        elapsed = now - self._fps_t0
        if elapsed >= 1.0:
            self._measured_fps = (self.frames_captured - self._fps_n0) / elapsed
            self._fps_t0, self._fps_n0 = now, self.frames_captured

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                buf = self._ring[self._slot]
                if not self._acquire(buf):
                    if self._stop.is_set():
                        break
                    time.sleep(0.002)
                    continue
                self._slot = (self._slot + 1) % RING_SIZE
                self._publish(buf)
        except BaseException as exc:            # surfaced to the main thread
            self._error = exc
        finally:
            with self._new_frame:
                self._new_frame.notify_all()


class CameraSource(FrameSource):
    """Live webcam via AVFoundation (macOS) or the platform default.

    Negotiates 1280x720 at 60fps and falls back to 30 if the device will
    not deliver it. Cameras lie about what they accept, so the requested
    rate is verified against what the driver reports *and* against the
    rate actually measured once frames are flowing.
    """

    def __init__(
        self,
        index: int = config.CAMERA_INDEX,
        width: int = config.CAMERA_WIDTH,
        height: int = config.CAMERA_HEIGHT,
        fps: int = config.CAMERA_FPS,
    ) -> None:
        self._requested_fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._index = index
        self._open(index, width, height, fps)
        super().__init__(self._actual_w, self._actual_h, self._actual_fps)

    def _open(self, index: int, width: int, height: int, fps: int) -> None:
        backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(
                f"could not open camera {index}. On macOS the terminal or IDE "
                f"needs Camera permission in System Settings > Privacy."
            )

        # MJPG lets most webcams actually hit 60fps at 720p; without it they
        # fall back to raw YUY2 which is bandwidth-capped to 30.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        # Shortest queue the backend will accept: we want the freshest
        # frame, never a buffered one.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        reported = cap.get(cv2.CAP_PROP_FPS)
        if fps == config.CAMERA_FPS and 0 < reported < fps * 0.9:
            cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS_FALLBACK)
            reported = cap.get(cv2.CAP_PROP_FPS)

        self._cap = cap
        self._actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        self._actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
        self._actual_fps = float(reported) if reported > 0 else float(fps)

    def _acquire(self, into: np.ndarray) -> bool:
        cap = self._cap
        if cap is None or not cap.grab():
            return False
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            return False
        if frame.shape == into.shape:
            # retrieve() allocates; copy into the ring so downstream sees a
            # stable buffer and the allocation is freed immediately.
            np.copyto(into, frame)
        else:
            cv2.resize(frame, (into.shape[1], into.shape[0]), dst=into,
                       interpolation=cv2.INTER_AREA)
        return True

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class SyntheticSource(FrameSource):
    """A deterministic moving scene, for benchmarks and camera-less tests.

    Deliberately busy -- text, edges, colour blocks -- so that when it is
    shattered you can tell at a glance whether the shards really carry
    their slice of the frozen frame, and whether reassembly is seamless.
    """

    def __init__(
        self,
        width: int = config.CAMERA_WIDTH,
        height: int = config.CAMERA_HEIGHT,
        fps: int = config.CAMERA_FPS,
        realtime: bool = True,
    ) -> None:
        super().__init__(width, height, fps)
        self._realtime = realtime
        self._t = 0.0
        self._next_due = 0.0
        self._base = self._make_base(width, height)

    @staticmethod
    def _make_base(w: int, h: int) -> np.ndarray:
        img = np.empty((h, w, 3), np.uint8)
        # Vertical gradient background.
        ramp = np.linspace(28, 96, h, dtype=np.float32)[:, None]
        img[:, :, 0] = np.clip(ramp * 1.15, 0, 255).astype(np.uint8)
        img[:, :, 1] = np.clip(ramp * 0.90, 0, 255).astype(np.uint8)
        img[:, :, 2] = np.clip(ramp * 0.70, 0, 255).astype(np.uint8)
        # A grid, so misalignment after reassembly is obvious to the eye.
        for x in range(0, w, 64):
            img[:, x] = (70, 70, 78)
        for y in range(0, h, 64):
            img[y, :] = (70, 70, 78)
        cv2.putText(img, "SHATTER", (int(w * 0.06), int(h * 0.30)),
                    cv2.FONT_HERSHEY_DUPLEX, w / 480.0, (240, 235, 225), 3, cv2.LINE_AA)
        cv2.putText(img, "REASSEMBLE", (int(w * 0.06), int(h * 0.52)),
                    cv2.FONT_HERSHEY_DUPLEX, w / 620.0, (120, 190, 255), 2, cv2.LINE_AA)
        return img

    def _acquire(self, into: np.ndarray) -> bool:
        if self._realtime:
            now = time.perf_counter()
            if self._next_due == 0.0:
                self._next_due = now
            if now < self._next_due:
                time.sleep(min(self._next_due - now, 0.004))
                return False
            self._next_due += 1.0 / self.fps

        np.copyto(into, self._base)
        h, w = into.shape[:2]
        t = self._t
        self._t += 1.0 / self.fps
        # A couple of moving objects to make the frozen frame obviously
        # frozen the moment a snap lands.
        cx = int(w * (0.5 + 0.34 * np.cos(t * 1.7)))
        cy = int(h * (0.62 + 0.18 * np.sin(t * 2.3)))
        cv2.circle(into, (cx, cy), int(h * 0.11), (60, 120, 240), -1, cv2.LINE_AA)
        cv2.circle(into, (cx, cy), int(h * 0.11), (250, 250, 250), 2, cv2.LINE_AA)
        bx = int(w * (0.5 + 0.30 * np.sin(t * 1.1)))
        cv2.rectangle(into, (bx - 60, int(h * 0.72)), (bx + 60, int(h * 0.92)),
                      (90, 220, 130), -1)
        return True


class VideoFileSource(FrameSource):
    """Loop a video file. Useful for reproducing a specific gesture take."""

    def __init__(self, path: str, fps: Optional[float] = None) -> None:
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video file: {path}")
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = fps or self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._path = path
        self._next_due = 0.0
        super().__init__(w, h, float(f))

    def _acquire(self, into: np.ndarray) -> bool:
        now = time.perf_counter()
        if self._next_due == 0.0:
            self._next_due = now
        if now < self._next_due:
            time.sleep(min(self._next_due - now, 0.004))
            return False
        self._next_due += 1.0 / self.fps

        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if not ok:
                return False
        np.copyto(into, frame)
        return True

    def _release(self) -> None:
        self._cap.release()


def open_source(options: config.RuntimeOptions) -> FrameSource:
    """Build the frame source named by ``options.source``."""
    if options.source == "synthetic":
        return SyntheticSource()
    if options.source == "camera":
        return CameraSource(index=options.camera_index)
    return VideoFileSource(options.source)
