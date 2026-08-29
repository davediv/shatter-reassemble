"""Putting it back. The part people watch to the end for.

Every shard knows its rest transform because the fracture produced it, so
reassembly is an animation problem rather than a physics one. Three things
make it read as a reversal rather than a reset:

*Stagger.* Delay is proportional to distance from frame centre, so the
image reforms outward from the middle instead of everything snapping home
at once.

*Overshoot.* Each shard eases past its rest pose and settles back, which
gives the landing weight.

*An exact landing.* At the end of its flight a shard is assigned its rest
transform verbatim -- not the last value of an interpolation. Combined
with the bevel animating to zero, which returns every vertex to the true
cell boundary, adjacent shards close up on identical edges and there is
nothing left to see a seam in.

Shards that have not started yet keep falling ballistically rather than
freezing in mid-air. If the user claps while the pile is still settling, a
frozen shard hanging in space for 400ms reads as a bug; a falling one
reads as a shard that has not been called home yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import config

__all__ = ["Reassembly", "ReassemblyState"]


@dataclass
class ReassemblyState:
    active: bool = False
    finished: bool = False
    progress: float = 0.0        # 0..1 across the whole animation
    bevel: float = 1.0           # shader uniform: 1 normal, 0 fully closed
    flash: float = 0.0           # screen flash on the final landing
    crossfade: float = 0.0       # 0 frozen shards, 1 live video
    landed: int = 0


def _ease_out_back(t: np.ndarray, tension: float) -> np.ndarray:
    """easeOutBack: overshoots the target, then settles onto it."""
    u = t - 1.0
    return 1.0 + (tension + 1.0) * u * u * u + tension * u * u


class Reassembly:
    def __init__(self, canvas_width: int, canvas_height: int) -> None:
        self.width = canvas_width
        self.height = canvas_height
        self.state = ReassemblyState()

        self._count = 0
        self._start_time = 0.0
        self._delay: Optional[np.ndarray] = None
        self._rest = np.zeros((0, 3), np.float64)
        self._launch = np.zeros((0, 3), np.float64)
        self._velocity = np.zeros((0, 3), np.float64)
        self._transforms = np.zeros((0, 3), np.float32)
        self._flash = np.zeros(0, np.float32)
        self._total = 1.0

    # -- lifecycle --------------------------------------------------------

    def begin(self, world, result, now: float) -> None:
        """Capture the pose to fly home from. Called on the clap."""
        n = world.count
        self._count = n
        self._start_time = now

        self._launch = np.empty((n, 3), np.float64)
        self._launch[:, 0] = world.px[:n]
        self._launch[:, 1] = world.py[:n]
        self._launch[:, 2] = world.rot[:n]

        self._velocity = np.empty((n, 3), np.float64)
        self._velocity[:, 0] = world.vx[:n]
        self._velocity[:, 1] = world.vy[:n]
        self._velocity[:, 2] = world.w[:n]

        self._rest = np.empty((n, 3), np.float64)
        self._rest[:, 0] = result.centroid[:n, 0]
        self._rest[:, 1] = result.centroid[:n, 1]
        # Unwind to the nearest whole turn rather than to zero, so a shard
        # that spun three times on the way down does not spin three times
        # back. The pose is identical either way; the journey is not.
        self._rest[:, 2] = np.round(self._launch[:, 2] / (2.0 * np.pi)) * (2.0 * np.pi)

        centre = np.array([self.width * 0.5, self.height * 0.5])
        distance = np.hypot(self._rest[:, 0] - centre[0], self._rest[:, 1] - centre[1])
        reach = float(np.hypot(centre[0], centre[1]))
        self._delay = (distance / max(reach, 1.0)) * config.REASSEMBLE_MAX_DELAY

        self._transforms = np.zeros((n, 3), np.float32)
        self._flash = np.zeros(n, np.float32)
        self._total = float(self._delay.max() + config.REASSEMBLE_DURATION)

        self.state = ReassemblyState(active=True, progress=0.0, bevel=1.0)

    def cancel(self) -> None:
        self.state = ReassemblyState()
        self._count = 0

    @property
    def active(self) -> bool:
        return self.state.active

    # -- per frame --------------------------------------------------------

    def update(self, now: float) -> ReassemblyState:
        if not self.state.active or self._count == 0:
            return self.state

        elapsed = now - self._start_time
        duration = config.REASSEMBLE_DURATION
        local = np.clip((elapsed - self._delay) / duration, -10.0, 1.0)

        flying = local > 0.0
        waiting = ~flying
        out = self._transforms

        if waiting.any():
            # Still falling: carry on ballistically from the clap pose.
            # Collisions are ignored for at most REASSEMBLE_MAX_DELAY, and
            # a shard about to be yanked home does not need them.
            t = np.maximum(elapsed, 0.0)
            idx = np.flatnonzero(waiting)
            out[idx, 0] = self._launch[idx, 0] + self._velocity[idx, 0] * t
            out[idx, 1] = (self._launch[idx, 1] + self._velocity[idx, 1] * t
                           + 0.5 * config.GRAVITY * t * t)
            out[idx, 2] = self._launch[idx, 2] + self._velocity[idx, 2] * t

        if flying.any():
            idx = np.flatnonzero(flying)
            # Ease from wherever the shard had fallen to when its turn
            # came, computed analytically so the handover is seamless.
            hold = np.maximum(self._delay[idx], 0.0)
            from_x = self._launch[idx, 0] + self._velocity[idx, 0] * hold
            from_y = (self._launch[idx, 1] + self._velocity[idx, 1] * hold
                      + 0.5 * config.GRAVITY * hold * hold)
            from_r = self._launch[idx, 2] + self._velocity[idx, 2] * hold

            eased = _ease_out_back(local[idx], config.REASSEMBLE_OVERSHOOT)
            out[idx, 0] = from_x + (self._rest[idx, 0] - from_x) * eased
            out[idx, 1] = from_y + (self._rest[idx, 1] - from_y) * eased
            out[idx, 2] = from_r + (self._rest[idx, 2] - from_r) * eased

            # The landing is assigned, never interpolated. An eased value
            # at t=1 is only equal to the target up to rounding, and this
            # app's whole payoff is that the final pose is exact.
            landed = idx[local[idx] >= 1.0]
            if landed.size:
                out[landed, 0] = self._rest[landed, 0]
                out[landed, 1] = self._rest[landed, 1]
                out[landed, 2] = self._rest[landed, 2]

        progress = float(np.clip(elapsed / self._total, 0.0, 1.0))
        landed_count = int(np.count_nonzero(local >= 1.0))

        # Per-shard sparkle as each one lands, and a screen flash when the
        # last one does.
        self._flash[:] = 0.0
        just = np.flatnonzero((local > 0.86) & (local < 1.0))
        if just.size:
            self._flash[just] = ((local[just] - 0.86) / 0.14).astype(np.float32) * 0.42

        overrun = elapsed - self._total
        flash = 0.0
        if overrun >= 0.0:
            flash = float(max(0.0, 1.0 - overrun / config.REASSEMBLE_FLASH_DURATION))

        # The bevel closes over the last quarter, so every vertex is back
        # on the true cell boundary by the time the last shard lands.
        bevel = float(np.clip((0.94 - progress) / 0.19, 0.0, 1.0))

        # Cross-fade to live video over the final 150ms of the flash tail.
        fade_start = self._total
        crossfade = float(np.clip(
            (elapsed - fade_start) / config.REASSEMBLE_CROSSFADE, 0.0, 1.0
        ))

        finished = elapsed >= self._total + max(
            config.REASSEMBLE_CROSSFADE, config.REASSEMBLE_FLASH_DURATION
        )
        self.state = ReassemblyState(
            active=not finished,
            finished=finished,
            progress=progress,
            bevel=bevel,
            flash=flash,
            crossfade=crossfade,
            landed=landed_count,
        )
        return self.state

    @property
    def transforms(self) -> np.ndarray:
        return self._transforms[: self._count]

    @property
    def flash_per_shard(self) -> np.ndarray:
        return self._flash[: self._count]

    def rest_error(self) -> float:
        """Max distance from any shard to its rest pose, in pixels.

        The acceptance number: after the animation this must be zero, not
        merely small.
        """
        if self._count == 0:
            return 0.0
        delta = self._transforms[: self._count, :2] - self._rest[: self._count, :2]
        return float(np.hypot(delta[:, 0], delta[:, 1]).max())
