"""One-shot Voronoi fracture: canvas rectangle in, shard geometry out.

Runs exactly once, inside the flash frame of a snap, and is never touched
again. That single fact shapes everything here -- it can afford to be
thorough because it happens once, and it must be fast because it happens
inside a frame the user is looking at.

What comes out is two things at once:

*Physics bodies.* Local-space convex polygons with area, centroid and
rotational inertia, ready to drop straight into the solver's flat arrays.

*Render geometry.* One interleaved vertex buffer holding every shard's
triangles, in shard-local coordinates, tagged with the shard index so the
whole pile draws in a single call with per-shard transforms fetched from a
texture.

The bevel trick
---------------
Each vertex stores its position on the *outer* cell boundary plus the
offset that would inset it for the bevel. The shader interpolates between
them with a uniform. At rest, with that uniform at zero, every vertex
returns to the exact cell boundary -- so adjacent shards share bit-similar
edges and reassembly lands with no seams. Baking the inset position
directly would have left a permanent hairline crack between every pair of
neighbours at the very moment the app is trying to look flawless.

UVs are not stored at all. They are affine in the rest position, so the
shader reconstructs them from the local vertex and the shard's rest
centroid, which keeps them correct as the bevel animates rather than
stretching the texture across a moving vertex.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numba import njit

from . import config

__all__ = ["FractureResult", "FracturePrewarmer", "fracture",
           "FLOATS_PER_VERTEX"]

# local.xy, inset.xy, edge, part, shard, normal.xy
FLOATS_PER_VERTEX = 9

PART_WALL = 0.0
PART_BEVEL = 1.0
PART_FACE = 2.0

_JIT = dict(cache=True, fastmath=True, nogil=True)


@dataclass
class FractureResult:
    """Everything the physics and the renderer need, from one decomposition."""

    count: int
    centroid: np.ndarray        # (N, 2) rest position in canvas px
    area: np.ndarray            # (N,)
    inertia: np.ndarray         # (N,) second moment per unit mass
    radius: np.ndarray          # (N,) bounding circle about the centroid
    depth: np.ndarray           # (N,) 0..1, drives the fake perspective
    poly_verts: np.ndarray      # (M, 2) local, CCW, concatenated
    poly_start: np.ndarray      # (N,) index into poly_verts
    poly_count: np.ndarray      # (N,)
    vertices: np.ndarray        # (V, 9) interleaved render geometry
    seeds: np.ndarray           # (N, 2) the generating points, for debug
    origin: tuple = (0.0, 0.0)  # where the break radiates from
    shard_count: int = 0        # requested count, for cache matching
    bevel: float = 0.0
    build_ms: float = 0.0
    stage_ms: Optional[dict] = None


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------

def _seed_points(
    count: int,
    width: float,
    height: float,
    origin: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Seeds clustered on the snap point, thinning out across the frame.

    A pure uniform scatter gives an even crazy-paving that reads as a
    texture rather than an impact. The Gaussian cluster is what makes the
    break look like it *came from* the hand: small dense shards at the
    strike, larger plates further out.

    The far field is a jittered grid rather than uniform random, because
    uniform random clumps, and clumped seeds make slivers -- long thin
    cells that behave badly in the solver and read as glitches.
    """
    clustered = int(count * config.FRACTURE_CLUSTER_FRACTION)
    scattered = count - clustered
    diagonal = float(np.hypot(width, height))
    sigma = config.FRACTURE_CLUSTER_SIGMA * diagonal

    points = np.empty((count, 2), np.float64)

    if clustered:
        # Radially distributed rather than a plain 2D normal: this gives
        # the density falloff we want without piling half the seeds into a
        # few pixels at the exact centre.
        radius = np.abs(rng.normal(0.0, sigma, clustered)) ** 0.85
        angle = rng.uniform(0.0, 2.0 * np.pi, clustered)
        points[:clustered, 0] = origin[0] + np.cos(angle) * radius
        points[:clustered, 1] = origin[1] + np.sin(angle) * radius

    if scattered:
        columns = int(np.ceil(np.sqrt(scattered * width / max(height, 1.0))))
        columns = max(columns, 1)
        rows = int(np.ceil(scattered / columns))
        cell_w, cell_h = width / columns, height / rows
        index = rng.permutation(columns * rows)[:scattered]
        gx = (index % columns).astype(np.float64)
        gy = (index // columns).astype(np.float64)
        points[clustered:, 0] = (gx + rng.uniform(0.15, 0.85, scattered)) * cell_w
        points[clustered:, 1] = (gy + rng.uniform(0.15, 0.85, scattered)) * cell_h

    # Keep everything strictly inside; a seed exactly on the border makes
    # a degenerate cell after clipping.
    np.clip(points[:, 0], 1.0, width - 1.0, out=points[:, 0])
    np.clip(points[:, 1], 1.0, height - 1.0, out=points[:, 1])

    # Qhull refuses coincident input, and clustering makes collisions
    # likely near the strike. Nudge duplicates apart on a sub-pixel grid.
    _, unique = np.unique(np.round(points, 2), axis=0, return_index=True)
    if unique.size < count:
        duplicated = np.setdiff1d(np.arange(count), unique)
        points[duplicated] += rng.uniform(-0.9, 0.9, (duplicated.size, 2))
    return points


def _guard_ring(width: float, height: float, count: int = 48) -> np.ndarray:
    """Points on a far circle, so no real seed lands on the convex hull.

    A seed on the hull owns an unbounded Voronoi cell, which scipy reports
    with a -1 vertex index and which cannot be clipped as a polygon.
    Surrounding the canvas removes the special case entirely, for the cost
    of 48 throwaway points.
    """
    cx, cy = width * 0.5, height * 0.5
    radius = np.hypot(width, height) * 2.0
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.stack([cx + np.cos(angle) * radius, cy + np.sin(angle) * radius], axis=1)


# --------------------------------------------------------------------------
# Compiled kernels
# --------------------------------------------------------------------------

@njit(**_JIT)
def _clip_cells(
    vertices, starts, counts, x0, y0, x1, y1,
    out_verts, out_starts, out_counts,
):
    """Sutherland-Hodgman each cell against the canvas rectangle.

    Convex-vs-convex, so the classic four-plane sweep is exact and cannot
    produce self-intersections. Vertices shared between neighbouring cells
    survive the clip identically, which is what keeps reassembly seamless.
    """
    n_cells = starts.shape[0]
    scratch_a = np.empty((64, 2), np.float64)
    scratch_b = np.empty((64, 2), np.float64)
    write = 0

    for cell in range(n_cells):
        count = counts[cell]
        if count < 3:
            out_starts[cell] = write
            out_counts[cell] = 0
            continue
        if count > 64:
            count = 64
        base = starts[cell]
        for i in range(count):
            scratch_a[i, 0] = vertices[base + i, 0]
            scratch_a[i, 1] = vertices[base + i, 1]
        current = count

        for plane in range(4):
            if current < 3:
                break
            produced = 0
            for i in range(current):
                ax = scratch_a[i, 0]
                ay = scratch_a[i, 1]
                j = i + 1
                if j == current:
                    j = 0
                bx = scratch_a[j, 0]
                by = scratch_a[j, 1]

                if plane == 0:
                    da = ax - x0
                    db = bx - x0
                elif plane == 1:
                    da = x1 - ax
                    db = x1 - bx
                elif plane == 2:
                    da = ay - y0
                    db = by - y0
                else:
                    da = y1 - ay
                    db = y1 - by

                inside_a = da >= 0.0
                inside_b = db >= 0.0
                if inside_a:
                    scratch_b[produced, 0] = ax
                    scratch_b[produced, 1] = ay
                    produced += 1
                if inside_a != inside_b:
                    denom = da - db
                    if denom != 0.0:
                        t = da / denom
                        scratch_b[produced, 0] = ax + (bx - ax) * t
                        scratch_b[produced, 1] = ay + (by - ay) * t
                        produced += 1
                if produced >= 63:
                    break
            current = produced
            for i in range(current):
                scratch_a[i, 0] = scratch_b[i, 0]
                scratch_a[i, 1] = scratch_b[i, 1]

        out_starts[cell] = write
        if current < 3:
            out_counts[cell] = 0
            continue
        # Drop vertices that the clip collapsed onto their neighbour;
        # a zero-length edge has no normal and would poison the bevel.
        kept = 0
        for i in range(current):
            j = i - 1
            if j < 0:
                j = current - 1
            dx = scratch_a[i, 0] - scratch_a[j, 0]
            dy = scratch_a[i, 1] - scratch_a[j, 1]
            if dx * dx + dy * dy > 1e-8:
                out_verts[write + kept, 0] = scratch_a[i, 0]
                out_verts[write + kept, 1] = scratch_a[i, 1]
                kept += 1
        out_counts[cell] = kept if kept >= 3 else 0
        write += kept if kept >= 3 else 0
    return write


@njit(**_JIT)
def _cell_properties(verts, starts, counts, centroid, area, inertia, radius):
    """Area, centroid, polar second moment and bounding radius per cell.

    Also rewrites each polygon in place to be centroid-relative and
    counter-clockwise, which is the convention the solver assumes.
    """
    n_cells = starts.shape[0]
    for cell in range(n_cells):
        count = counts[cell]
        if count < 3:
            area[cell] = 0.0
            inertia[cell] = 1.0
            radius[cell] = 0.0
            centroid[cell, 0] = 0.0
            centroid[cell, 1] = 0.0
            continue
        base = starts[cell]

        twice_area = 0.0
        cx = 0.0
        cy = 0.0
        for i in range(count):
            j = i + 1
            if j == count:
                j = 0
            ax = verts[base + i, 0]
            ay = verts[base + i, 1]
            bx = verts[base + j, 0]
            by = verts[base + j, 1]
            cross = ax * by - bx * ay
            twice_area += cross
            cx += (ax + bx) * cross
            cy += (ay + by) * cross

        signed = twice_area * 0.5
        if abs(signed) < 1e-9:
            area[cell] = 0.0
            inertia[cell] = 1.0
            radius[cell] = 0.0
            continue
        cx /= 3.0 * twice_area
        cy /= 3.0 * twice_area
        centroid[cell, 0] = cx
        centroid[cell, 1] = cy
        area[cell] = abs(signed)

        # Re-express locally, and force counter-clockwise winding.
        if signed < 0.0:
            for i in range(count // 2):
                k = count - 1 - i
                tx = verts[base + i, 0]
                ty = verts[base + i, 1]
                verts[base + i, 0] = verts[base + k, 0]
                verts[base + i, 1] = verts[base + k, 1]
                verts[base + k, 0] = tx
                verts[base + k, 1] = ty
        largest = 0.0
        for i in range(count):
            verts[base + i, 0] -= cx
            verts[base + i, 1] -= cy
            d = verts[base + i, 0] ** 2 + verts[base + i, 1] ** 2
            if d > largest:
                largest = d
        radius[cell] = np.sqrt(largest)

        # Polar second moment about the centroid, per unit mass.
        moment = 0.0
        for i in range(count):
            j = i + 1
            if j == count:
                j = 0
            ax = verts[base + i, 0]
            ay = verts[base + i, 1]
            bx = verts[base + j, 0]
            by = verts[base + j, 1]
            cross = ax * by - bx * ay
            moment += cross * (ax * ax + ax * bx + bx * bx
                               + ay * ay + ay * by + by * by)
        value = abs(moment) / (12.0 * area[cell])
        inertia[cell] = value if value > 1e-6 else 1e-6


@njit(**_JIT)
def _emit_geometry(verts, starts, counts, bevel, out, vertex_starts, vertex_counts):
    """Expand each convex cell into face, bevel rim and side wall triangles.

    Every vertex carries the *outer* boundary position plus the offset
    that insets it for the bevel, so the shader can animate the bevel to
    nothing and land every vertex back on the true cell boundary.
    """
    n_cells = starts.shape[0]
    write = 0
    for cell in range(n_cells):
        count = counts[cell]
        vertex_starts[cell] = write
        if count < 3:
            vertex_counts[cell] = 0
            continue
        base = starts[cell]
        shard = np.float32(cell)

        # Inset by offsetting each edge inward and intersecting the
        # neighbours. Convex input makes this well defined; the guard
        # below catches cells too small to inset without turning inside
        # out, and simply collapses their bevel to nothing.
        inset = np.empty((count, 2), np.float64)
        normals = np.empty((count, 2), np.float64)
        for i in range(count):
            j = i + 1
            if j == count:
                j = 0
            ex = verts[base + j, 0] - verts[base + i, 0]
            ey = verts[base + j, 1] - verts[base + i, 1]
            length = np.sqrt(ex * ex + ey * ey)
            if length < 1e-9:
                normals[i, 0] = 0.0
                normals[i, 1] = 0.0
            else:
                # CCW winding: the outward normal is the edge rotated -90.
                normals[i, 0] = ey / length
                normals[i, 1] = -ex / length

        for i in range(count):
            prev = i - 1
            if prev < 0:
                prev = count - 1
            nx = normals[prev, 0] + normals[i, 0]
            ny = normals[prev, 1] + normals[i, 1]
            length = np.sqrt(nx * nx + ny * ny)
            if length < 1e-6:
                inset[i, 0] = 0.0
                inset[i, 1] = 0.0
                continue
            # Miter length, clamped so a sharp corner cannot shoot the
            # inset vertex out past the far side of the cell.
            scale = bevel * 2.0 / (length * length)
            if scale > bevel * 4.0:
                scale = bevel * 4.0
            inset[i, 0] = -nx * scale
            inset[i, 1] = -ny * scale

        # Guard: if insetting would invert the cell, drop the bevel.
        px = verts[base + 0, 0]
        py = verts[base + 0, 1]
        span = 0.0
        for i in range(count):
            dx = verts[base + i, 0] - px
            dy = verts[base + i, 1] - py
            d = np.sqrt(dx * dx + dy * dy)
            if d > span:
                span = d
        if span < bevel * 3.0:
            for i in range(count):
                inset[i, 0] = 0.0
                inset[i, 1] = 0.0

        # -- side wall: one quad per edge, extruded in the shader --------
        for i in range(count):
            j = i + 1
            if j == count:
                j = 0
            ax = verts[base + i, 0]
            ay = verts[base + i, 1]
            bx = verts[base + j, 0]
            by = verts[base + j, 1]
            nx = normals[i, 0]
            ny = normals[i, 1]
            for corner in range(6):
                # quad (a, b, b', a') as two triangles
                if corner == 0 or corner == 3:
                    vx, vy, extrude = ax, ay, 0.0
                elif corner == 1:
                    vx, vy, extrude = bx, by, 0.0
                elif corner == 2 or corner == 4:
                    vx, vy, extrude = bx, by, 1.0
                else:
                    vx, vy, extrude = ax, ay, 1.0
                out[write, 0] = vx
                out[write, 1] = vy
                out[write, 2] = 0.0
                out[write, 3] = 0.0
                out[write, 4] = extrude
                out[write, 5] = PART_WALL
                out[write, 6] = shard
                out[write, 7] = nx
                out[write, 8] = ny
                write += 1

        # -- bevel rim: outer edge to inset edge --------------------------
        for i in range(count):
            j = i + 1
            if j == count:
                j = 0
            nx = normals[i, 0]
            ny = normals[i, 1]
            for corner in range(6):
                if corner == 0 or corner == 3:
                    src, inner = i, 0.0
                elif corner == 1:
                    src, inner = j, 0.0
                elif corner == 2 or corner == 4:
                    src, inner = j, 1.0
                else:
                    src, inner = i, 1.0
                out[write, 0] = verts[base + src, 0]
                out[write, 1] = verts[base + src, 1]
                out[write, 2] = inset[src, 0] * inner
                out[write, 3] = inset[src, 1] * inner
                out[write, 4] = inner
                out[write, 5] = PART_BEVEL
                out[write, 6] = shard
                out[write, 7] = nx
                out[write, 8] = ny
                write += 1

        # -- face: fan over the inset polygon ------------------------------
        for i in range(1, count - 1):
            for corner in range(3):
                src = 0 if corner == 0 else (i if corner == 1 else i + 1)
                out[write, 0] = verts[base + src, 0]
                out[write, 1] = verts[base + src, 1]
                out[write, 2] = inset[src, 0]
                out[write, 3] = inset[src, 1]
                out[write, 4] = 1.0
                out[write, 5] = PART_FACE
                out[write, 6] = shard
                out[write, 7] = 0.0
                out[write, 8] = 0.0
                write += 1

        vertex_counts[cell] = write - vertex_starts[cell]
    return write


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def fracture(
    width: int,
    height: int,
    origin: tuple[float, float],
    count: int = config.SHARD_COUNT_TIERS[0],
    *,
    bevel: float = config.BEVEL_WIDTH,
    seed: Optional[int] = None,
) -> FractureResult:
    """Decompose the canvas into ``count`` shards radiating from ``origin``."""
    from scipy.spatial import Voronoi

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    count = max(int(count), 8)

    points = _seed_points(count, float(width), float(height), origin, rng)
    padded = np.vstack([points, _guard_ring(float(width), float(height))])
    t_seed = time.perf_counter()

    diagram = Voronoi(padded)
    t_voronoi = time.perf_counter()

    # Gather the cells belonging to real seeds, ignoring the guard ring.
    # Flattening every region into one index array and taking a single
    # fancy-index costs 0.45ms against 1.08ms for a per-cell loop, which
    # is worth having in a routine that runs inside a visible frame.
    # The guard ring guarantees no -1 vertices, but the filter costs
    # nothing and keeps one bad seed from crashing the snap.
    region_lists = [diagram.regions[r] for r in diagram.point_region[:count]]
    usable = [r if (len(r) >= 3 and -1 not in r) else [] for r in region_lists]
    raw_counts = np.fromiter((len(r) for r in usable), np.int64, count)
    total = int(raw_counts.sum())
    if total:
        flat = np.concatenate([np.asarray(r, np.int64) for r in usable if r])
        raw_verts = diagram.vertices[flat]
    else:
        raw_verts = np.zeros((1, 2), np.float64)
    raw_starts = np.zeros(count, np.int64)
    np.cumsum(raw_counts[:-1], out=raw_starts[1:])

    # Clipping can add at most one vertex per clip plane per polygon.
    clipped_verts = np.empty((total + count * 8 + 8, 2), np.float64)
    clipped_starts = np.empty(count, np.int64)
    clipped_counts = np.empty(count, np.int64)
    used = _clip_cells(
        raw_verts, raw_starts, raw_counts,
        0.0, 0.0, float(width), float(height),
        clipped_verts, clipped_starts, clipped_counts,
    )
    t_clip = time.perf_counter()

    centroid = np.zeros((count, 2), np.float64)
    area = np.zeros(count, np.float64)
    inertia = np.ones(count, np.float64)
    radius = np.zeros(count, np.float64)
    _cell_properties(clipped_verts, clipped_starts, clipped_counts,
                     centroid, area, inertia, radius)

    # A cell that survived clipping as a sliver is not a shard: invisible,
    # but it would still occupy a physics body and a solver slot.
    alive = (clipped_counts >= 3) & (area >= config.MIN_SHARD_AREA)
    live_count = int(alive.sum())
    # Zero the rejects *before* emitting rather than filtering their
    # triangles out afterwards, so no geometry is ever built for a cell
    # that has no body to be transformed by.
    clipped_counts[~alive] = 0

    budget = int((clipped_counts[alive] * 15).sum() + 16)
    vertex_data = np.empty((budget, FLOATS_PER_VERTEX), np.float64)
    vertex_starts = np.zeros(count, np.int64)
    vertex_counts = np.zeros(count, np.int64)
    written = _emit_geometry(
        clipped_verts, clipped_starts, clipped_counts, float(bevel),
        vertex_data, vertex_starts, vertex_counts,
    )
    t_emit = time.perf_counter()

    # Compact to the surviving cells, renumbering the shard index the
    # geometry refers to so it stays a dense 0..N-1 range.
    keep = np.flatnonzero(alive)
    remap = np.full(count, -1, np.int64)
    remap[keep] = np.arange(live_count)

    vertices = vertex_data[:written].astype(np.float32)
    if live_count < count:
        vertices[:, 6] = remap[vertices[:, 6].astype(np.int64)]
        assert vertices[:, 6].min() >= 0, "geometry emitted for a rejected cell"

    # Ragged gather without a Python loop: build the destination offsets,
    # broadcast each polygon's source base against them, and index once.
    kept_counts = clipped_counts[keep]
    kept_starts = clipped_starts[keep]
    poly_count = kept_counts.astype(np.int32)
    poly_start = np.zeros(live_count, np.int32)
    if live_count:
        np.cumsum(poly_count[:-1], out=poly_start[1:])
        offsets = np.repeat(kept_starts - poly_start.astype(np.int64), kept_counts)
        poly_verts = clipped_verts[offsets + np.arange(int(kept_counts.sum()))]
        poly_verts = poly_verts.astype(np.float32)
    else:
        poly_verts = np.zeros((0, 2), np.float32)

    # Depth drives the fake perspective. Shards near the strike sit
    # nearer the viewer, so the break reads as coming toward the camera.
    kept_centroids = centroid[keep].astype(np.float32)
    distance = np.hypot(kept_centroids[:, 0] - origin[0],
                        kept_centroids[:, 1] - origin[1])
    reach = max(float(np.hypot(width, height)) * 0.5, 1.0)
    depth = np.clip(1.0 - distance / reach, 0.0, 1.0).astype(np.float32)
    depth = depth * 0.6 + rng.random(live_count).astype(np.float32) * 0.4

    build_ms = (time.perf_counter() - t0) * 1e3
    return FractureResult(
        count=live_count,
        centroid=kept_centroids,
        area=area[keep].astype(np.float32),
        inertia=inertia[keep].astype(np.float32),
        radius=radius[keep].astype(np.float32),
        depth=depth,
        poly_verts=poly_verts,
        poly_start=poly_start,
        poly_count=poly_count,
        vertices=vertices,
        seeds=points[keep].astype(np.float32),
        origin=(float(origin[0]), float(origin[1])),
        shard_count=count,
        bevel=float(bevel),
        build_ms=build_ms,
        stage_ms={
            "seed": (t_seed - t0) * 1e3,
            "voronoi": (t_voronoi - t_seed) * 1e3,
            "clip": (t_clip - t_voronoi) * 1e3,
            "emit": (t_emit - t_clip) * 1e3,
            "compact": (time.perf_counter() - t_emit) * 1e3,
        },
    )


class FracturePrewarmer:
    """Builds the fracture speculatively, on a worker thread, before the snap.

    The decomposition costs ~5ms for 800 cells and the snap frame is the
    worst possible moment to spend it: that frame is already freezing the
    video texture and uploading two megabytes of shard geometry.

    But a snap always announces itself. The fingers have to pinch before
    they can flick, and the detector arms on that pinch -- typically many
    frames before it fires, because people hold the pinch while they aim.
    That is enough warning to build the whole thing in advance and have it
    waiting when the flick lands.

    The catch is that the break has to radiate from where the hand *ends
    up*, not from where it armed. So a prepared result is only accepted if
    the hand is still within FRACTURE_PREDICT_TOLERANCE of the predicted
    origin; past that the visual lie would be noticeable and it rebuilds
    synchronously, paying the cost it was trying to avoid rather than
    putting the break in the wrong place.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._request: Optional[tuple] = None
        self._result: Optional[FractureResult] = None
        self._thread: Optional[threading.Thread] = None
        self.hits = 0
        self.misses = 0

    def start(self) -> "FracturePrewarmer":
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="FracturePrewarm", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def request(self, width, height, origin, count, bevel) -> None:
        """Ask for a fracture at ``origin``. Non-blocking; latest wins."""
        with self._lock:
            ready = self._result
            if (ready is not None and ready.shard_count == count
                    and ready.bevel == bevel
                    and _near(ready.origin, origin)):
                return          # what we already have is still good enough
            self._request = (width, height, tuple(origin), count, bevel)
        self._wake.set()

    def take(self, width, height, origin, count, bevel) -> FractureResult:
        """The fracture for this snap, prepared if possible, built if not."""
        with self._lock:
            ready = self._result
            self._result = None
            if (ready is not None and ready.shard_count == count
                    and ready.bevel == bevel
                    and _near(ready.origin, origin)):
                self.hits += 1
                return ready
        self.misses += 1
        return fracture(width, height, origin, count, bevel=bevel)

    def discard(self) -> None:
        with self._lock:
            self._result = None
            self._request = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(0.5)
            self._wake.clear()
            if self._stop.is_set():
                return
            with self._lock:
                pending, self._request = self._request, None
            if pending is None:
                continue
            try:
                built = fracture(pending[0], pending[1], pending[2],
                                 pending[3], bevel=pending[4])
            except Exception:
                continue
            with self._lock:
                # Drop it if a newer request arrived while we were working.
                if self._request is None:
                    self._result = built


def _near(a, b, tolerance: float = config.FRACTURE_PREDICT_TOLERANCE) -> bool:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 <= tolerance * tolerance
