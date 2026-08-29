"""Canvas capture to VP9, in 16:9 and 9:16, without stalling the frame.

The browser version of this is one line: MediaRecorder on
canvas.captureStream(60). Off the browser it splits into three problems,
and each has a wrong answer that costs frames.

*Getting pixels off the GPU.* A plain glReadPixels blocks until the GPU
has finished the frame, which hands back every millisecond the pipeline
was buying. Reads go into a ring of pixel buffer objects instead and are
mapped two frames later, by which time the copy is long done.

*Encoding.* Apple silicon has no VP9 encoder -- VP9 is decode-only -- and
software libvpx managed 2.7fps for both outputs at 1080p. Realtime VP9 at
60fps is unreachable here at any setting, so capture runs on the hardware
HEVC encoder and the VP9 the spec asks for is transcoded once recording
stops. The output is still VP9 12Mbps in both aspect ratios; it just
arrives shortly after you stop rather than during.

*Not letting the encoder become the app's problem.* The queue is bounded.
If the encoder falls behind, frames are dropped and counted rather than
allowed to back-pressure into the render loop -- a recording with a few
dropped frames is a recording; a recording that drags the app to 40fps
ruins the thing being recorded.

Both aspect ratios come out of one ffmpeg invocation: the raw pipe is
split, one branch encoded as-is and the other centre-cropped to 9:16, so
the 6MB/frame of raw video crosses the process boundary once.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

import moderngl

from . import config

__all__ = ["Recorder", "RecorderStats"]

PBO_DEPTH = 3
# Frames in flight between the render thread and the encoder. Each is a
# full 1080p RGB frame, so this is the memory ceiling for recording:
# 6 x 6.2MB = 37MB.
BUFFER_POOL = 6


@dataclass
class RecorderStats:
    recording: bool = False
    transcoding: bool = False
    frames: int = 0
    dropped: int = 0
    seconds: float = 0.0
    capture_ms: float = 0.0
    queue_depth: int = 0
    outputs: tuple = ()
    mode: str = config.RECORD_MODE
    error: str = ""


class Recorder:
    def __init__(
        self,
        display,
        *,
        fps: int = config.RECORD_FPS,
        bitrate: str = config.RECORD_BITRATE,
        directory: Path = config.RECORDING_DIR,
        mode: str = config.RECORD_MODE,
    ) -> None:
        self.display = display
        self.ctx = display.ctx
        self.fps = fps
        self.bitrate = bitrate
        self.directory = Path(directory)
        self.mode = mode if mode in config.RECORD_MODES else config.RECORD_MODE
        self.stats = RecorderStats(mode=self.mode)
        self._transcoder: Optional[threading.Thread] = None

        self.width = display.canvas_width
        self.height = display.canvas_height
        self._frame_bytes = self.width * self.height * 3

        self._pbos = [
            self.ctx.buffer(reserve=self._frame_bytes) for _ in range(PBO_DEPTH)
        ]
        self._pending = [False] * PBO_DEPTH
        self._slot = 0

        # A free list rather than fresh allocations. moderngl's Buffer.read()
        # returns bytes, which means allocating and freeing 6.2MB per frame
        # -- 373MB/s of churn at 60fps, measured at ~9.5ms per call against
        # 2.0ms for read_into on a buffer that already exists.
        self._pool = [bytearray(self._frame_bytes) for _ in range(BUFFER_POOL)]
        self._free: Queue = Queue()
        self._queue: Queue = Queue()
        self._writer: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None
        self._stop = threading.Event()
        self._started_at = 0.0
        self._outputs: tuple = ()
        self._capture_paths: tuple = ()

    # -- lifecycle --------------------------------------------------------

    @property
    def recording(self) -> bool:
        return self._process is not None

    def toggle(self) -> bool:
        if self.recording:
            self.stop()
        else:
            self.start()
        return self.recording

    def start(self) -> bool:
        if self.recording:
            return True
        if shutil.which("ffmpeg") is None:
            self.stats.error = "ffmpeg not found on PATH"
            return False

        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        if self.mode == "h264":
            suffix = config.RECORD_INTERMEDIATE_SUFFIX
        elif self.mode == "vp9":
            suffix = config.RECORD_INTERMEDIATE_SUFFIX
        else:
            suffix = ".webm"
        landscape = self.directory / f"shatter-{stamp}-16x9{suffix}"
        portrait = self.directory / f"shatter-{stamp}-9x16{suffix}"
        self._capture_paths = (landscape, portrait)
        self._outputs = (
            (landscape.with_suffix(".webm"), portrait.with_suffix(".webm"))
            if self.mode == "vp9" else (landscape, portrait)
        )

        try:
            self._process = subprocess.Popen(
                self._command(landscape, portrait),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            self._process = None
            self.stats.error = str(exc)
            return False

        self._stop.clear()
        for q in (self._queue, self._free):
            while not q.empty():
                try:
                    q.get_nowait()
                except Empty:
                    break
        for buffer in self._pool:
            self._free.put_nowait(buffer)
        self._pending = [False] * PBO_DEPTH
        self._slot = 0
        self._started_at = time.perf_counter()
        self.stats = RecorderStats(recording=True, outputs=self._outputs,
                                   mode=self.mode)

        self._writer = threading.Thread(target=self._write_loop,
                                        name="Recorder", daemon=True)
        self._writer.start()
        return True

    def stop(self) -> tuple:
        if not self.recording:
            return ()
        # Drain what the GPU still owes us before shutting the pipe.
        self._flush_pending()
        self._stop.set()
        if self._writer is not None:
            self._writer.join(timeout=5.0)
            self._writer = None
        process = self._process
        self._process = None
        if process is not None:
            try:
                if process.stdin:
                    process.stdin.close()
                process.wait(timeout=30)
            except Exception:
                process.kill()
        self.stats.recording = False
        if self.mode == "vp9":
            # The VP9 the spec asks for, produced off the render thread now
            # that nothing is depending on realtime.
            self.stats.transcoding = True
            self._transcoder = threading.Thread(
                target=self._transcode, name="Transcode", daemon=True
            )
            self._transcoder.start()
        return self._outputs

    def _command(self, landscape: Path, portrait: Path) -> list:
        # 9:16 out of 1920x1080 is a 608-wide centre crop, scaled up.
        crop_w = (self.height * 9 // 16) // 2 * 2
        crop_x = (self.width - crop_w) // 2
        pw, ph = config.RECORD_PORTRAIT

        if self.mode == "vp9-realtime":
            # Spec-literal: VP9 straight off the canvas. Measured at 2.7fps
            # for both outputs on this machine, so it will drop most of
            # them. Kept because on hardware with a VP9 encoder it is the
            # right answer, and because it is worth being able to show the
            # difference.
            encode = [
                "-c:v", config.RECORD_CODEC, "-b:v", self.bitrate,
                "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1",
                "-pix_fmt", "yuv420p",
            ]
        else:
            encode = [
                "-c:v", config.RECORD_INTERMEDIATE_CODEC,
                "-b:v", config.RECORD_INTERMEDIATE_BITRATE,
                "-tag:v", "hvc1", "-pix_fmt", "yuv420p",
            ]

        return [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}", "-r", str(self.fps),
            "-i", "-",
            "-filter_complex",
            f"[0:v]split=2[a][b];[b]crop={crop_w}:{self.height}:{crop_x}:0,"
            f"scale={pw}:{ph}[c]",
            "-map", "[a]", *encode, str(landscape),
            "-map", "[c]", *encode, str(portrait),
        ]

    def _transcode(self) -> None:
        """Intermediate -> VP9 12Mbps, after recording has stopped."""
        try:
            for source, target in zip(self._capture_paths, self._outputs):
                if not source.exists():
                    continue
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                     "-i", str(source),
                     "-c:v", config.RECORD_CODEC, "-b:v", self.bitrate,
                     "-row-mt", "1", "-deadline", "good", "-cpu-used", "4",
                     "-pix_fmt", "yuv420p", str(target)],
                    check=False,
                )
                if target.exists() and target.stat().st_size > 0:
                    source.unlink(missing_ok=True)
        except Exception as exc:
            self.stats.error = f"transcode failed: {exc}"
        finally:
            self.stats.transcoding = False

    @property
    def transcoding(self) -> bool:
        return bool(self.stats.transcoding)

    def wait_for_transcode(self, timeout: float = 600.0) -> None:
        if self._transcoder is not None:
            self._transcoder.join(timeout=timeout)

    # -- per frame --------------------------------------------------------

    def capture(self) -> None:
        """Queue this frame. Costs a readback request and, two frames
        later, one memcpy."""
        if not self.recording:
            return
        start = time.perf_counter()
        slot = self._slot
        pbo = self._pbos[slot]

        # Harvest the frame this slot is holding from PBO_DEPTH frames ago
        # before overwriting it. By now the copy has certainly landed, so
        # mapping it does not stall.
        if self._pending[slot]:
            self._harvest(pbo)
            self._pending[slot] = False

        self.display.canvas.read_into(pbo, components=3, alignment=1)
        self._pending[slot] = True
        self._slot = (slot + 1) % PBO_DEPTH

        elapsed = (time.perf_counter() - start) * 1e3
        self.stats.capture_ms += (elapsed - self.stats.capture_ms) * 0.1
        self.stats.seconds = time.perf_counter() - self._started_at
        self.stats.queue_depth = self._queue.qsize()

    def _flush_pending(self) -> None:
        for offset in range(PBO_DEPTH):
            slot = (self._slot + offset) % PBO_DEPTH
            if self._pending[slot]:
                self._harvest(self._pbos[slot])
                self._pending[slot] = False

    def _harvest(self, pbo: moderngl.Buffer) -> None:
        """Copy one completed readback out of the PBO and hand it over."""
        try:
            destination = self._free.get_nowait()
        except Empty:
            # Every buffer is still queued or being written: the encoder is
            # behind. Drop this frame, count it, and keep rendering. Letting
            # it back-pressure into the render loop would ruin the very
            # thing being recorded.
            self.stats.dropped += 1
            return
        pbo.read_into(destination)
        self._queue.put_nowait(destination)
        self.stats.frames += 1

    def _write_loop(self) -> None:
        process = self._process
        while True:
            try:
                payload = self._queue.get(timeout=0.2)
            except Empty:
                if self._stop.is_set():
                    return
                continue
            if process is None or process.stdin is None:
                return
            try:
                process.stdin.write(payload)
            except (BrokenPipeError, ValueError):
                self.stats.error = "encoder closed the pipe"
                return
            finally:
                # Back on the free list whatever happened, so a stalled
                # encoder starves the pool instead of leaking it.
                self._free.put_nowait(payload)

    def release(self) -> None:
        if self.recording:
            self.stop()
        for pbo in self._pbos:
            try:
                pbo.release()
            except Exception:
                pass
