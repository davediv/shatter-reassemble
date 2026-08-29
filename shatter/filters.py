"""Signal conditioning for the tracking pipeline.

The One Euro filter (Casiez, Roussel & Vogel, CHI 2012) is a first-order
low-pass whose cutoff frequency rises with the speed of the signal. Slow
movement gets heavy smoothing (jitter dies); fast movement gets almost
none (lag dies). That adaptive behaviour is exactly what a hand needs.

A warning that shapes the whole design of this module: this app is driven
by *velocity spikes*. A snap is a 30ms transient and a clap is a
deceleration edge. Over-smooth and you delete the very events you are
trying to detect. So smoothing here is strictly for what the user *sees* --
the skeleton overlay and the capsule colliders. Gesture detection reads a
parallel unsmoothed channel, and never touches these filters.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["OneEuroFilterND", "TraceBuffer", "alpha_for_cutoff"]

# Guard rails on dt. A stalled camera or a debugger breakpoint can hand us
# an absurd delta; clamping keeps the filter from either freezing (dt->0
# makes alpha->0) or teleporting (dt huge makes alpha->1).
_MIN_DT = 1e-4
_MAX_DT = 0.2

_TWO_PI = 2.0 * math.pi


def alpha_for_cutoff(cutoff: float, dt: float) -> float:
    """Smoothing factor of a first-order low-pass at ``cutoff`` Hz."""
    tau = 1.0 / (_TWO_PI * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilterND:
    """One Euro filter applied element-wise across a fixed-shape array.

    Shaped for the tracking pipeline: ``(slots, landmarks, axes)``, i.e.
    one independent filter per landmark per axis, as specified. The whole
    set updates in a handful of vectorised numpy ops rather than 84 Python
    objects, and every intermediate is preallocated so a steady-state frame
    allocates nothing.

    Slots carry tracking identity (left hand / right hand). When a hand
    disappears its slot must be reset, otherwise the filter would smear the
    old hand's position into the new one's first frame.
    """

    __slots__ = (
        "shape", "min_cutoff", "beta", "d_cutoff", "_alpha_d",
        "_x_prev", "_dx_prev", "_ready", "_dx", "_cutoff", "_alpha", "_out",
    )

    def __init__(
        self,
        shape: tuple[int, ...],
        min_cutoff: float,
        beta: float,
        d_cutoff: float,
    ) -> None:
        self.shape = shape
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._alpha_d = 0.0

        self._x_prev = np.zeros(shape, np.float32)
        self._dx_prev = np.zeros(shape, np.float32)
        # One readiness flag per slot, not per element: a hand's landmarks
        # all appear and vanish together.
        self._ready = np.zeros(shape[0], np.bool_)

        # Preallocated scratch.
        self._dx = np.zeros(shape, np.float32)
        self._cutoff = np.zeros(shape, np.float32)
        self._alpha = np.zeros(shape, np.float32)
        self._out = np.zeros(shape, np.float32)

    def reset(self, slot: int | None = None) -> None:
        """Forget history for one slot, or all of them."""
        if slot is None:
            self._ready[:] = False
        else:
            self._ready[slot] = False

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        return self.filter(x, dt)

    def filter(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Smooth ``x`` and return an internal buffer (do not retain it).

        The returned array is reused on the next call. Callers that need to
        keep the values across frames must copy.
        """
        dt = min(max(float(dt), _MIN_DT), _MAX_DT)

        dx, dx_prev = self._dx, self._dx_prev
        x_prev, out = self._x_prev, self._out

        # Derivative, low-passed at a fixed cutoff.
        np.subtract(x, x_prev, out=dx)
        np.multiply(dx, np.float32(1.0 / dt), out=dx)
        a_d = np.float32(alpha_for_cutoff(self.d_cutoff, dt))
        self._alpha_d = float(a_d)
        # dx_prev += a_d * (dx - dx_prev)
        np.subtract(dx, dx_prev, out=dx)
        np.multiply(dx, a_d, out=dx)
        np.add(dx_prev, dx, out=dx_prev)

        # Speed-adaptive cutoff, then the position low-pass.
        np.abs(dx_prev, out=self._cutoff)
        np.multiply(self._cutoff, np.float32(self.beta), out=self._cutoff)
        np.add(self._cutoff, np.float32(self.min_cutoff), out=self._cutoff)

        # alpha = 1 / (1 + tau/dt), tau = 1/(2*pi*cutoff)
        #       = 1 / (1 + 1/(2*pi*cutoff*dt))
        np.multiply(self._cutoff, np.float32(_TWO_PI * dt), out=self._alpha)
        np.reciprocal(self._alpha, out=self._alpha)
        np.add(self._alpha, np.float32(1.0), out=self._alpha)
        np.reciprocal(self._alpha, out=self._alpha)

        # out = x_prev + alpha * (x - x_prev)
        np.subtract(x, x_prev, out=out)
        np.multiply(out, self._alpha, out=out)
        np.add(x_prev, out, out=out)

        # Slots seeing their first frame pass straight through, and seed
        # their history from it.
        if not self._ready.all():
            fresh = ~self._ready
            out[fresh] = x[fresh]
            dx_prev[fresh] = 0.0
            self._ready[:] = True

        x_prev[...] = out
        return out


class TraceBuffer:
    """Fixed-capacity ring of scalars for live signal plots.

    The tuning mode draws snap's two signals as scrolling traces; this is
    what they scroll through. Writes are O(1) with no allocation, and
    ``snapshot`` hands back a contiguous oldest-to-newest view.
    """

    __slots__ = ("_data", "_head", "_count", "_scratch")

    def __init__(self, capacity: int) -> None:
        self._data = np.zeros(capacity, np.float32)
        self._scratch = np.zeros(capacity, np.float32)
        self._head = 0
        self._count = 0

    def __len__(self) -> int:
        return self._count

    @property
    def capacity(self) -> int:
        return self._data.shape[0]

    def push(self, value: float) -> None:
        self._data[self._head] = value
        self._head = (self._head + 1) % self._data.shape[0]
        if self._count < self._data.shape[0]:
            self._count += 1

    def clear(self) -> None:
        self._head = 0
        self._count = 0
        self._data[:] = 0.0

    def snapshot(self) -> np.ndarray:
        """Oldest-to-newest view of the buffer contents."""
        cap = self._data.shape[0]
        if self._count < cap:
            return self._data[: self._count]
        n = cap - self._head
        self._scratch[:n] = self._data[self._head:]
        self._scratch[n:] = self._data[: self._head]
        return self._scratch

    def latest(self, default: float = 0.0) -> float:
        if self._count == 0:
            return default
        return float(self._data[(self._head - 1) % self._data.shape[0]])
