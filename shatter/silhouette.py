"""Person segmentation for the silhouette that survives in the void.

Behind the shards there is nothing but black -- except the faint outline
of the person still moving in it. That outline is the emotional point of
the shattered state: the user keeps living while their world lies broken
on the floor. It is also, budget-wise, a background detail.

So it runs here, on its own thread, at its own rate, on a deliberately
small input. Measured on an M1 Air: 8.9ms/frame at 1280x720, 2.8ms at
256x144. Stacking either onto the hand-tracking thread would have blown
the tracking budget for something the user perceives as a soft glow.

The model reports a single 'selfie' confidence mask in [0,1]. The category
mask is available but awkward -- it uses 255 as the background sentinel,
so 'no person' reads as 255 rather than 0 -- and a soft confidence value
makes a much better outline than a hard label anyway.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import config
from .camera import FrameSource
from .delegate import DelegateChoice

__all__ = ["SilhouetteTracker", "SilhouetteStats"]


@dataclass
class SilhouetteStats:
    enabled: bool = False
    segment_ms: float = 0.0
    rate_hz: float = 0.0
    width: int = 0
    height: int = 0


class SilhouetteTracker:
    """Selfie segmentation on a worker thread, rate-limited and downscaled."""

    def __init__(
        self,
        source: FrameSource,
        choice: DelegateChoice,
        *,
        max_hz: float = config.SEGMENT_MAX_HZ,
        width: int = config.SEGMENT_INPUT_WIDTH,
    ) -> None:
        self._source = source
        self._choice = choice
        self._max_hz = max_hz
        self._width = width
        self.stats = SilhouetteStats()

        self._segmenter = self._make_segmenter()
        self.stats.enabled = self._segmenter is not None

        self._small: Optional[np.ndarray] = None
        self._rgb: Optional[np.ndarray] = None
        self._mp_format = None
        # Two publish buffers, alternating: the consumer can hold one while
        # the next is being written.
        self._buffers: list[np.ndarray] = []
        self._buf_index = 0
        self._mask: Optional[np.ndarray] = None
        self._mask_dirty = False

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0
        self._mp_ts = 0
        self._rate_t0 = 0.0
        self._rate_n = 0
        self._failures = 0

    def _make_segmenter(self):
        if not config.SEGMENT_MODEL.exists():
            return None
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.ImageSegmenterOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(config.SEGMENT_MODEL),
                    delegate=getattr(
                        mp_python.BaseOptions.Delegate, self._choice.delegate
                    ),
                ),
                running_mode=vision.RunningMode.VIDEO,
                output_category_mask=False,
                output_confidence_masks=True,
            )
            return vision.ImageSegmenter.create_from_options(options)
        except Exception:
            return None

    # -- lifecycle --------------------------------------------------------

    def start(self) -> "SilhouetteTracker":
        if self._segmenter is not None and self._thread is None:
            self._t0 = time.perf_counter()
            self._rate_t0 = self._t0
            self._thread = threading.Thread(
                target=self._run, name="Silhouette", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._segmenter is not None:
            try:
                self._segmenter.close()
            except Exception:
                pass
            self._segmenter = None

    def __enter__(self) -> "SilhouetteTracker":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- consumer API -----------------------------------------------------

    def take_mask(self) -> Optional[np.ndarray]:
        """Newest mask if it changed since the last call, else None.

        Returning None on an unchanged mask lets the renderer skip a
        redundant texture upload on the ~2 out of 3 frames where
        segmentation has not produced anything new.
        """
        with self._lock:
            if not self._mask_dirty:
                return None
            self._mask_dirty = False
            return self._mask

    # -- worker -----------------------------------------------------------

    def _run(self) -> None:
        import mediapipe as mp

        last_index = -1
        min_period = 1.0 / self._max_hz if self._max_hz > 0 else 0.0
        next_due = 0.0
        while not self._stop.is_set():
            frame = self._source.wait_for_frame(last_index, 0.25)
            if frame is None:
                continue
            last_index = frame.index
            now = time.perf_counter()
            if now < next_due:
                continue
            next_due = max(now + min_period, next_due + min_period)
            try:
                self._segment(frame, mp, now)
            except Exception:
                self._failures += 1
                if self._failures >= 5:
                    self.stats.enabled = False
                    return

    def _segment(self, frame, mp, now: float) -> None:
        src = frame.data
        w = self._width
        h = int(round(src.shape[0] * w / src.shape[1]))
        if self._small is None or self._small.shape[:2] != (h, w):
            self._small = np.empty((h, w, 3), np.uint8)
            self._rgb = np.empty((h, w, self._choice.channels), np.uint8)
            self._buffers = [np.empty((h, w), np.uint8) for _ in range(2)]
            self._mp_format = getattr(mp.ImageFormat, self._choice.image_format)
            self.stats.width, self.stats.height = w, h

        t0 = time.perf_counter()
        cv2.resize(src, (w, h), dst=self._small, interpolation=cv2.INTER_AREA)
        cv2.cvtColor(self._small, getattr(cv2, self._choice.cv_conversion),
                     dst=self._rgb)
        image = mp.Image(image_format=self._mp_format, data=self._rgb)
        ts = int((frame.timestamp - self._t0) * 1000.0)
        self._mp_ts = max(ts, self._mp_ts + 1)
        result = self._segmenter.segment_for_video(image, self._mp_ts)

        masks = getattr(result, "confidence_masks", None)
        if not masks:
            return
        conf = masks[0].numpy_view()
        if conf.ndim == 3:
            conf = conf[:, :, 0]

        buf = self._buffers[self._buf_index]
        self._buf_index ^= 1
        # float32 [0,1] -> uint8. 256 levels is far more than a faint
        # outline needs, and it is a quarter of the upload.
        cv2.convertScaleAbs(conf, dst=buf, alpha=255.0)

        elapsed = time.perf_counter() - t0
        self.stats.segment_ms += (elapsed * 1e3 - self.stats.segment_ms) * 0.15
        with self._lock:
            self._mask = buf
            self._mask_dirty = True

        self._rate_n += 1
        span = time.perf_counter() - self._rate_t0
        if span >= 1.0:
            self.stats.rate_hz = self._rate_n / span
            self._rate_t0, self._rate_n = time.perf_counter(), 0
