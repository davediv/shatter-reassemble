"""A purpose-built 2.5D rigid body solver for a pile of glass.

Not a general physics engine. It does exactly what a pile of shards
needs -- convex planar bodies, sequential impulses, a sorted-axis
broadphase and aggressive sleeping -- and nothing else, which is what
makes 800 bodies affordable inside a 4ms budget.

Layout
------
Every quantity lives in a flat, preallocated float32 array in
structure-of-arrays order. Nothing is a Python object, nothing is
allocated per frame, and the hot loops are numba-compiled with the GIL
released so the solver genuinely runs beside the render thread rather
than merely appearing to.

Sleeping
--------
This is the single thing that makes the target reachable. A pile that has
settled costs almost nothing because settled bodies leave the solver
entirely. Sleeping is approximated without full island detection: a body
only accumulates sleep time while every body it touches is also slow, so
a stack cannot sleep out from under the shard resting on it.

The 2.5D part
-------------
Bodies are strictly planar. Depth is a per-shard scalar that the renderer
turns into scale, parallax and shading, and that the solver uses only to
bias contact response slightly, so shards at different depths slide past
each other instead of grinding. It is a lie that reads as perspective and
costs one multiply.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from . import config

__all__ = ["PhysicsWorld", "PhysicsStats", "MAX_CAPSULES"]

_JIT = dict(cache=True, fastmath=True, nogil=True)

MAX_CONTACTS = 16384
MAX_PAIRS = 16384
MAX_CAPSULES = 48
# Open-addressed warm-start cache. Power of two so the probe is a mask.
CACHE_SLOTS = 32768
CACHE_EMPTY = np.int64(-1)

# Boundary plane ids, encoded in the contact's second body index.
STATIC_BODY = np.int32(-1)

# Warm-start keys carry a class tag in the top bits. Without one, a
# boundary contact for body i keyed as (i<<22)|(plane<<2)|which is
# indistinguishable from a body-body contact between i and body `plane`,
# so a shard touching body 1 or 2 warm-starts from the floor's impulse and
# vice versa -- injecting energy into a pile that is trying to settle.
KEY_BODY = np.int64(0) << np.int64(60)
KEY_BOUND = np.int64(1) << np.int64(60)
KEY_CAPSULE = np.int64(2) << np.int64(60)


@dataclass
class PhysicsStats:
    substeps: int = 0
    bodies: int = 0
    awake: int = 0
    pairs: int = 0
    contacts: int = 0
    step_ms: float = 0.0
    broad_ms: float = 0.0
    narrow_ms: float = 0.0
    solve_ms: float = 0.0


# --------------------------------------------------------------------------
# Compiled kernels
# --------------------------------------------------------------------------

@njit(**_JIT)
def _effective_mass(inv_mass, inv_inertia, awake, solver_inv_mass,
                    solver_inv_inertia, count):
    """A sleeping body is an immovable obstacle, not a free body.

    Without this the solver keeps applying impulses to sleeping bodies:
    their velocities accumulate while their positions never integrate, and
    the moment something wakes them they leap. Zeroing their inverse mass
    for the duration is also what makes sleeping *pay* -- a settled shard
    becomes ground for everything resting on it, so the stack above it
    converges in far fewer iterations.
    """
    for i in range(count):
        if awake[i] == 0:
            solver_inv_mass[i] = 0.0
            solver_inv_inertia[i] = 0.0
        else:
            solver_inv_mass[i] = inv_mass[i]
            solver_inv_inertia[i] = inv_inertia[i]


@njit(**_JIT)
def _integrate_velocities(vx, vy, w, awake, inv_mass, gravity, dt,
                          linear_damping, angular_damping, count):
    lin = 1.0 / (1.0 + dt * linear_damping)
    ang = 1.0 / (1.0 + dt * angular_damping)
    for i in range(count):
        if awake[i] == 0 or inv_mass[i] == 0.0:
            continue
        vy[i] = (vy[i] + gravity * dt) * lin
        vx[i] = vx[i] * lin
        w[i] = w[i] * ang


@njit(**_JIT)
def _update_world_vertices(verts, vstart, vcount, px, py, rot, awake,
                           wverts, wnormals, count):
    """Local polygons to world space, once per substep, awake bodies only."""
    for i in range(count):
        if awake[i] == 0:
            continue
        c = np.cos(rot[i])
        s = np.sin(rot[i])
        base = vstart[i]
        n = vcount[i]
        for k in range(n):
            lx = verts[base + k, 0]
            ly = verts[base + k, 1]
            wverts[base + k, 0] = px[i] + c * lx - s * ly
            wverts[base + k, 1] = py[i] + s * lx + c * ly
        for k in range(n):
            j = k + 1
            if j == n:
                j = 0
            ex = wverts[base + j, 0] - wverts[base + k, 0]
            ey = wverts[base + j, 1] - wverts[base + k, 1]
            length = np.sqrt(ex * ex + ey * ey)
            if length < 1e-9:
                wnormals[base + k, 0] = 0.0
                wnormals[base + k, 1] = 0.0
            else:
                wnormals[base + k, 0] = ey / length
                wnormals[base + k, 1] = -ex / length


@njit(**_JIT)
def _sort_axis(order, px, radius, count):
    """Insertion sort on the sweep axis.

    Bodies barely move between 120Hz substeps, so the order is almost
    always already sorted and this is O(n) in practice -- far cheaper
    than a full sort every substep, which is the whole point of keeping
    the ordering persistent.
    """
    for i in range(1, count):
        item = order[i]
        key = px[item] - radius[item]
        j = i - 1
        while j >= 0 and (px[order[j]] - radius[order[j]]) > key:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = item


@njit(**_JIT)
def _sweep_pairs(order, px, py, radius, awake, layer, pairs, count):
    """Sweep-and-prune along the sorted axis, within a depth layer."""
    found = 0
    for a in range(count):
        ia = order[a]
        ax_max = px[ia] + radius[ia]
        ay = py[ia]
        ar = radius[ia]
        for b in range(a + 1, count):
            ib = order[b]
            if px[ib] - radius[ib] > ax_max:
                break                       # sorted: nothing further can touch
            if awake[ia] == 0 and awake[ib] == 0:
                continue                    # two sleepers cannot collide
            if layer[ia] != layer[ib]:
                continue                    # different depth planes
            reach = ar + radius[ib]
            dy = py[ib] - ay
            if dy > reach or dy < -reach:
                continue
            dx = px[ib] - px[ia]
            if dx * dx + dy * dy > reach * reach:
                continue
            if found < MAX_PAIRS:
                pairs[found, 0] = ia
                pairs[found, 1] = ib
                found += 1
    return found


@njit(inline="always")
def _cache_slot(key, cache_key):
    slot = np.int64((key * np.int64(2654435761)) & np.int64(CACHE_SLOTS - 1))
    for _ in range(16):
        existing = cache_key[slot]
        if existing == key or existing == CACHE_EMPTY:
            return slot
        slot = (slot + 1) & np.int64(CACHE_SLOTS - 1)
    return np.int64(-1)


@njit(**_JIT)
def _collide_polygons(
    ia, ib, wverts, wnormals, vstart, vcount, px, py,
    contacts_a, contacts_b, contact_normal, contact_point,
    contact_depth, contact_key, written,
):
    """SAT with reference-face clipping. Up to two contact points."""
    base_a, count_a = vstart[ia], vcount[ia]
    base_b, count_b = vstart[ib], vcount[ib]

    best_depth = 1e30
    best_face = -1
    best_ref = 0
    # Least-penetrating axis over A's faces, then B's.
    for k in range(count_a):
        nx = wnormals[base_a + k, 0]
        ny = wnormals[base_a + k, 1]
        if nx == 0.0 and ny == 0.0:
            continue
        ox = wverts[base_a + k, 0]
        oy = wverts[base_a + k, 1]
        lowest = 1e30
        for m in range(count_b):
            d = ((wverts[base_b + m, 0] - ox) * nx
                 + (wverts[base_b + m, 1] - oy) * ny)
            if d < lowest:
                lowest = d
        if lowest > 0.0:
            return written                  # separating axis: no contact
        if -lowest < best_depth:
            best_depth = -lowest
            best_face = k
            best_ref = 0

    for k in range(count_b):
        nx = wnormals[base_b + k, 0]
        ny = wnormals[base_b + k, 1]
        if nx == 0.0 and ny == 0.0:
            continue
        ox = wverts[base_b + k, 0]
        oy = wverts[base_b + k, 1]
        lowest = 1e30
        for m in range(count_a):
            d = ((wverts[base_a + m, 0] - ox) * nx
                 + (wverts[base_a + m, 1] - oy) * ny)
            if d < lowest:
                lowest = d
        if lowest > 0.0:
            return written
        if -lowest < best_depth:
            best_depth = -lowest
            best_face = k
            best_ref = 1

    if best_face < 0:
        return written

    if best_ref == 0:
        ref_base, ref_count = base_a, count_a
        inc_base, inc_count = base_b, count_b
    else:
        ref_base, ref_count = base_b, count_b
        inc_base, inc_count = base_a, count_a

    nx = wnormals[ref_base + best_face, 0]
    ny = wnormals[ref_base + best_face, 1]

    # Incident face: the one on the other body most anti-parallel to it.
    incident = 0
    smallest = 1e30
    for k in range(inc_count):
        d = wnormals[inc_base + k, 0] * nx + wnormals[inc_base + k, 1] * ny
        if d < smallest:
            smallest = d
            incident = k

    j = incident + 1
    if j == inc_count:
        j = 0
    p0x = wverts[inc_base + incident, 0]
    p0y = wverts[inc_base + incident, 1]
    p1x = wverts[inc_base + j, 0]
    p1y = wverts[inc_base + j, 1]

    # Clip the incident edge to the reference face's side planes.
    rj = best_face + 1
    if rj == ref_count:
        rj = 0
    rax = wverts[ref_base + best_face, 0]
    ray = wverts[ref_base + best_face, 1]
    rbx = wverts[ref_base + rj, 0]
    rby = wverts[ref_base + rj, 1]
    tx = rbx - rax
    ty = rby - ray
    length = np.sqrt(tx * tx + ty * ty)
    if length < 1e-9:
        return written
    tx /= length
    ty /= length

    lo = (p0x - rax) * tx + (p0y - ray) * ty
    hi = (p1x - rax) * tx + (p1y - ray) * ty
    # Parametric clip of [lo, hi] against [0, length].
    denom = hi - lo
    t0, t1 = 0.0, 1.0
    if abs(denom) > 1e-9:
        a = (0.0 - lo) / denom
        b = (length - lo) / denom
        if a > b:
            a, b = b, a
        if a > t0:
            t0 = a
        if b < t1:
            t1 = b
    else:
        if lo < 0.0 or lo > length:
            return written
    if t1 < t0:
        return written

    for which in range(2):
        t = t0 if which == 0 else t1
        cx = p0x + (p1x - p0x) * t
        cy = p0y + (p1y - p0y) * t
        depth = -((cx - rax) * nx + (cy - ray) * ny)
        if depth < 0.0:
            continue
        if which == 1 and t1 - t0 < 1e-6:
            continue
        if written >= MAX_CONTACTS:
            return written
        # Normal always points from A toward B, whichever body was
        # reference, so the solver never has to branch on it.
        sign = 1.0 if best_ref == 0 else -1.0
        contacts_a[written] = ia
        contacts_b[written] = ib
        contact_normal[written, 0] = nx * sign
        contact_normal[written, 1] = ny * sign
        contact_point[written, 0] = cx
        contact_point[written, 1] = cy
        contact_depth[written] = depth
        contact_key[written] = KEY_BODY | (np.int64(ia) << np.int64(32)) \
            | (np.int64(ib) << np.int64(2)) | np.int64(which)
        written += 1
    return written


@njit(**_JIT)
def _polygon_vs_circle(
    ia, cx, cy, cradius, svx, svy, key_tag,
    wverts, wnormals, vstart, vcount,
    contacts_a, contacts_b, contact_normal, contact_point,
    contact_depth, contact_key, static_vx, static_vy, written,
):
    """One contact between a shard and a swept circle (a capsule slice)."""
    base, count = vstart[ia], vcount[ia]
    best = -1e30
    face = 0
    for k in range(count):
        nx = wnormals[base + k, 0]
        ny = wnormals[base + k, 1]
        if nx == 0.0 and ny == 0.0:
            continue
        d = (cx - wverts[base + k, 0]) * nx + (cy - wverts[base + k, 1]) * ny
        if d > best:
            best = d
            face = k
    if best > cradius:
        return written
    if written >= MAX_CONTACTS:
        return written

    j = face + 1
    if j == count:
        j = 0
    ax = wverts[base + face, 0]
    ay = wverts[base + face, 1]
    bx = wverts[base + j, 0]
    by = wverts[base + j, 1]

    if best < 0.0:
        # Centre inside the polygon: push out along the closest face.
        nx = wnormals[base + face, 0]
        ny = wnormals[base + face, 1]
        depth = cradius - best
        contact_x = cx - nx * best
        contact_y = cy - ny * best
    else:
        ex = bx - ax
        ey = by - ay
        length2 = ex * ex + ey * ey
        t = 0.0
        if length2 > 1e-12:
            t = ((cx - ax) * ex + (cy - ay) * ey) / length2
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
        qx = ax + ex * t
        qy = ay + ey * t
        dx = cx - qx
        dy = cy - qy
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > cradius:
            return written
        if dist < 1e-6:
            nx = wnormals[base + face, 0]
            ny = wnormals[base + face, 1]
        else:
            nx = dx / dist
            ny = dy / dist
        depth = cradius - dist
        contact_x = qx
        contact_y = qy

    # Normal points from the shard toward the capsule.
    contacts_a[written] = ia
    contacts_b[written] = STATIC_BODY
    contact_normal[written, 0] = nx
    contact_normal[written, 1] = ny
    contact_point[written, 0] = contact_x
    contact_point[written, 1] = contact_y
    contact_depth[written] = depth
    contact_key[written] = KEY_CAPSULE | (np.int64(ia) << np.int64(32)) \
        | np.int64(key_tag)
    static_vx[written] = svx
    static_vy[written] = svy
    return written + 1


@njit(**_JIT)
def _boundary_contacts(
    wverts, vstart, vcount, awake, px, py, width, height, floor_y,
    contacts_a, contacts_b, contact_normal, contact_point,
    contact_depth, contact_key, static_vx, static_vy, written, count,
):
    """Walls and floor as half-planes, two deepest vertices each."""
    for i in range(count):
        if awake[i] == 0:
            continue
        base, n = vstart[i], vcount[i]
        for plane in range(3):
            # plane 0: floor (normal up), 1: left wall, 2: right wall.
            if plane == 0:
                nx, ny = 0.0, -1.0
            elif plane == 1:
                nx, ny = 1.0, 0.0
            else:
                nx, ny = -1.0, 0.0

            first, second = -1, -1
            d_first, d_second = 0.0, 0.0
            for k in range(n):
                if plane == 0:
                    depth = wverts[base + k, 1] - floor_y
                elif plane == 1:
                    depth = -wverts[base + k, 0]
                else:
                    depth = wverts[base + k, 0] - width
                if depth <= 0.0:
                    continue
                if depth > d_first:
                    second, d_second = first, d_first
                    first, d_first = k, depth
                elif depth > d_second:
                    second, d_second = k, depth
            if first < 0:
                continue
            for which in range(2):
                k = first if which == 0 else second
                if k < 0:
                    continue
                if written >= MAX_CONTACTS:
                    return written
                depth = d_first if which == 0 else d_second
                contacts_a[written] = i
                contacts_b[written] = STATIC_BODY
                # Normal points from the shard out toward the wall.
                contact_normal[written, 0] = -nx
                contact_normal[written, 1] = -ny
                contact_point[written, 0] = wverts[base + k, 0]
                contact_point[written, 1] = wverts[base + k, 1]
                contact_depth[written] = depth
                contact_key[written] = KEY_BOUND | (np.int64(i) << np.int64(32)) \
                    | (np.int64(plane) << np.int64(2)) | np.int64(which)
                static_vx[written] = 0.0
                static_vy[written] = 0.0
                written += 1
    return written


@njit(**_JIT)
def _prepare_contacts(
    contacts_a, contacts_b, contact_normal, contact_point, contact_depth,
    contact_key, px, py, vx, vy, w, inv_mass, inv_inertia,
    normal_mass, tangent_mass, bias, pos_bias, normal_impulse, tangent_impulse,
    pseudo_impulse, ra, rb, cache_key, cache_normal, cache_tangent,
    restitution, slop, baumgarte, inv_dt, static_vx, static_vy, n_contacts,
):
    for c in range(n_contacts):
        ia = contacts_a[c]
        ib = contacts_b[c]
        nx = contact_normal[c, 0]
        ny = contact_normal[c, 1]

        rax = contact_point[c, 0] - px[ia]
        ray = contact_point[c, 1] - py[ia]
        ra[c, 0] = rax
        ra[c, 1] = ray
        rn_a = rax * ny - ray * nx
        rt_a = rax * nx + ray * ny

        k_normal = inv_mass[ia] + inv_inertia[ia] * rn_a * rn_a
        k_tangent = inv_mass[ia] + inv_inertia[ia] * rt_a * rt_a

        if ib >= 0:
            rbx = contact_point[c, 0] - px[ib]
            rby = contact_point[c, 1] - py[ib]
            rb[c, 0] = rbx
            rb[c, 1] = rby
            rn_b = rbx * ny - rby * nx
            rt_b = rbx * nx + rby * ny
            k_normal += inv_mass[ib] + inv_inertia[ib] * rn_b * rn_b
            k_tangent += inv_mass[ib] + inv_inertia[ib] * rt_b * rt_b
            rel_vx = (vx[ib] - w[ib] * rby) - (vx[ia] - w[ia] * ray)
            rel_vy = (vy[ib] + w[ib] * rbx) - (vy[ia] + w[ia] * rax)
        else:
            rb[c, 0] = 0.0
            rb[c, 1] = 0.0
            rel_vx = static_vx[c] - (vx[ia] - w[ia] * ray)
            rel_vy = static_vy[c] - (vy[ia] + w[ia] * rax)

        normal_mass[c] = 1.0 / k_normal if k_normal > 1e-12 else 0.0
        tangent_mass[c] = 1.0 / k_tangent if k_tangent > 1e-12 else 0.0

        approach = rel_vx * nx + rel_vy * ny
        penetration = contact_depth[c] - slop
        if penetration < 0.0:
            penetration = 0.0
        # Penetration recovery is a *pseudo* velocity, solved separately
        # and applied to position only. Folding it into the velocity solve
        # the obvious way (plain Baumgarte) injects real kinetic energy
        # into every resting contact, and a pile of 800 shards then
        # vibrates forever and never sleeps -- measured: zero bodies
        # asleep after six seconds.
        pos_bias[c] = baumgarte * inv_dt * penetration
        # Restitution only for genuine impacts; applying it to resting
        # contacts is the other classic way to make a pile jitter.
        b = 0.0
        if approach < -180.0:
            b = -restitution * approach
        bias[c] = b
        pseudo_impulse[c] = 0.0

        # Warm start from the previous substep's accumulated impulse.
        # Sequential impulses converge far faster from last frame's answer
        # than from zero, and a pile of 800 shards is exactly the case
        # where 8 iterations from cold is not enough.
        slot = _cache_slot(contact_key[c], cache_key)
        if slot >= 0 and cache_key[slot] == contact_key[c]:
            normal_impulse[c] = cache_normal[slot]
            tangent_impulse[c] = cache_tangent[slot]
        else:
            normal_impulse[c] = 0.0
            tangent_impulse[c] = 0.0


@njit(**_JIT)
def _apply_warm_start(
    contacts_a, contacts_b, contact_normal, ra, rb,
    normal_impulse, tangent_impulse, vx, vy, w, inv_mass, inv_inertia,
    n_contacts,
):
    for c in range(n_contacts):
        ia = contacts_a[c]
        ib = contacts_b[c]
        nx = contact_normal[c, 0]
        ny = contact_normal[c, 1]
        tx, ty = -ny, nx
        ix = nx * normal_impulse[c] + tx * tangent_impulse[c]
        iy = ny * normal_impulse[c] + ty * tangent_impulse[c]
        vx[ia] -= ix * inv_mass[ia]
        vy[ia] -= iy * inv_mass[ia]
        w[ia] -= inv_inertia[ia] * (ra[c, 0] * iy - ra[c, 1] * ix)
        if ib >= 0:
            vx[ib] += ix * inv_mass[ib]
            vy[ib] += iy * inv_mass[ib]
            w[ib] += inv_inertia[ib] * (rb[c, 0] * iy - rb[c, 1] * ix)


@njit(**_JIT)
def _solve_velocity(
    contacts_a, contacts_b, contact_normal, ra, rb,
    normal_mass, tangent_mass, bias, normal_impulse, tangent_impulse,
    vx, vy, w, inv_mass, inv_inertia, static_vx, static_vy,
    friction, n_contacts,
):
    for c in range(n_contacts):
        ia = contacts_a[c]
        ib = contacts_b[c]
        nx = contact_normal[c, 0]
        ny = contact_normal[c, 1]
        rax, ray = ra[c, 0], ra[c, 1]

        if ib >= 0:
            rbx, rby = rb[c, 0], rb[c, 1]
            rel_vx = (vx[ib] - w[ib] * rby) - (vx[ia] - w[ia] * ray)
            rel_vy = (vy[ib] + w[ib] * rbx) - (vy[ia] + w[ia] * rax)
        else:
            rbx = rby = 0.0
            rel_vx = static_vx[c] - (vx[ia] - w[ia] * ray)
            rel_vy = static_vy[c] - (vy[ia] + w[ia] * rax)

        vn = rel_vx * nx + rel_vy * ny
        lam = (bias[c] - vn) * normal_mass[c]
        # Clamp the *accumulated* impulse, not the increment: this is what
        # lets a contact pull back an over-correction from an earlier
        # iteration without ever applying a sticky negative impulse.
        total = normal_impulse[c] + lam
        if total < 0.0:
            total = 0.0
        lam = total - normal_impulse[c]
        normal_impulse[c] = total

        ix = nx * lam
        iy = ny * lam
        vx[ia] -= ix * inv_mass[ia]
        vy[ia] -= iy * inv_mass[ia]
        w[ia] -= inv_inertia[ia] * (rax * iy - ray * ix)
        if ib >= 0:
            vx[ib] += ix * inv_mass[ib]
            vy[ib] += iy * inv_mass[ib]
            w[ib] += inv_inertia[ib] * (rbx * iy - rby * ix)

        # Coulomb friction against the accumulated normal impulse.
        tx, ty = -ny, nx
        if ib >= 0:
            rel_vx = (vx[ib] - w[ib] * rby) - (vx[ia] - w[ia] * ray)
            rel_vy = (vy[ib] + w[ib] * rbx) - (vy[ia] + w[ia] * rax)
        else:
            rel_vx = static_vx[c] - (vx[ia] - w[ia] * ray)
            rel_vy = static_vy[c] - (vy[ia] + w[ia] * rax)
        vt = rel_vx * tx + rel_vy * ty
        lam_t = -vt * tangent_mass[c]
        limit = friction * normal_impulse[c]
        total_t = tangent_impulse[c] + lam_t
        if total_t > limit:
            total_t = limit
        elif total_t < -limit:
            total_t = -limit
        lam_t = total_t - tangent_impulse[c]
        tangent_impulse[c] = total_t

        ix = tx * lam_t
        iy = ty * lam_t
        vx[ia] -= ix * inv_mass[ia]
        vy[ia] -= iy * inv_mass[ia]
        w[ia] -= inv_inertia[ia] * (rax * iy - ray * ix)
        if ib >= 0:
            vx[ib] += ix * inv_mass[ib]
            vy[ib] += iy * inv_mass[ib]
            w[ib] += inv_inertia[ib] * (rbx * iy - rby * ix)


@njit(**_JIT)
def _solve_position(
    contacts_a, contacts_b, contact_normal, ra, rb,
    normal_mass, pos_bias, pseudo_impulse,
    pvx, pvy, pw, inv_mass, inv_inertia, n_contacts,
):
    """Split impulse: resolve overlap in a velocity channel that only ever
    reaches position, never the real velocities. The pile pushes itself
    apart without gaining energy, which is what lets it go to sleep."""
    for c in range(n_contacts):
        ia = contacts_a[c]
        ib = contacts_b[c]
        nx = contact_normal[c, 0]
        ny = contact_normal[c, 1]
        rax, ray = ra[c, 0], ra[c, 1]

        if ib >= 0:
            rbx, rby = rb[c, 0], rb[c, 1]
            rel_vx = (pvx[ib] - pw[ib] * rby) - (pvx[ia] - pw[ia] * ray)
            rel_vy = (pvy[ib] + pw[ib] * rbx) - (pvy[ia] + pw[ia] * rax)
        else:
            rbx = rby = 0.0
            rel_vx = -(pvx[ia] - pw[ia] * ray)
            rel_vy = -(pvy[ia] + pw[ia] * rax)

        vn = rel_vx * nx + rel_vy * ny
        lam = (pos_bias[c] - vn) * normal_mass[c]
        total = pseudo_impulse[c] + lam
        if total < 0.0:
            total = 0.0
        lam = total - pseudo_impulse[c]
        pseudo_impulse[c] = total

        ix = nx * lam
        iy = ny * lam
        pvx[ia] -= ix * inv_mass[ia]
        pvy[ia] -= iy * inv_mass[ia]
        pw[ia] -= inv_inertia[ia] * (rax * iy - ray * ix)
        if ib >= 0:
            pvx[ib] += ix * inv_mass[ib]
            pvy[ib] += iy * inv_mass[ib]
            pw[ib] += inv_inertia[ib] * (rbx * iy - rby * ix)


@njit(**_JIT)
def _store_impulses(contact_key, normal_impulse, tangent_impulse,
                    cache_key, cache_normal, cache_tangent, n_contacts):
    """Replace the cache wholesale rather than adding to it.

    Warm starting only ever wants the *previous* substep's impulses. An
    accumulate-only table looks fine for a few seconds and then quietly
    stops working: shards keep forming new pairs, so the set of distinct
    contact keys grows without bound until every slot is occupied, every
    lookup walks its full 16-slot probe and misses, and the solver is
    suddenly running cold with no warm start at all. That showed up as
    settled-frame spikes of 40-70ms.
    """
    cache_key[:] = CACHE_EMPTY
    for c in range(n_contacts):
        slot = _cache_slot(contact_key[c], cache_key)
        if slot >= 0:
            cache_key[slot] = contact_key[c]
            cache_normal[slot] = normal_impulse[c]
            cache_tangent[slot] = tangent_impulse[c]


@njit(**_JIT)
def _integrate_positions(px, py, rot, vx, vy, w, pvx, pvy, pw, awake, dt, count):
    for i in range(count):
        if awake[i] == 0:
            continue
        px[i] += (vx[i] + pvx[i]) * dt
        py[i] += (vy[i] + pvy[i]) * dt
        rot[i] += (w[i] + pw[i]) * dt
        # Pseudo velocity is consumed entirely by this one step; it must
        # never survive into the next or it becomes real motion.
        pvx[i] = 0.0
        pvy[i] = 0.0
        pw[i] = 0.0


@njit(**_JIT)
def _update_sleep(vx, vy, w, awake, sleep_time, disturbed, dt,
                  linear_threshold, angular_threshold, sleep_after, count):
    still_awake = 0
    for i in range(count):
        if awake[i] == 0:
            continue
        slow = (vx[i] * vx[i] + vy[i] * vy[i]) < linear_threshold * linear_threshold
        if slow and abs(w[i]) < angular_threshold and disturbed[i] == 0:
            sleep_time[i] += dt
            if sleep_time[i] > sleep_after:
                awake[i] = 0
                vx[i] = 0.0
                vy[i] = 0.0
                w[i] = 0.0
                continue
        else:
            sleep_time[i] = 0.0
        still_awake += 1
    return still_awake


@njit(**_JIT)
def _mark_disturbed(contacts_a, contacts_b, vx, vy, w, awake, disturbed,
                    sleep_time, linear_threshold, angular_threshold,
                    disturb_scale, n_contacts):
    """Approximate islands: a body cannot sleep while a neighbour is moving.

    Without this a shard resting on a still-sliding one falls asleep and
    the pile locks into a pose that is visibly wrong. Full island
    detection would be the rigorous answer; propagating "someone next to
    me is moving" through the contact graph costs one pass and is
    indistinguishable at this scale.

    The disturb threshold is deliberately higher than the sleep threshold.
    With both at the same value, any body sitting just over the line keeps
    every neighbour awake, that cascades through a dense pile, and nothing
    ever sleeps -- measured at 796 of 798 bodies still awake after eight
    seconds. The gap gives the pile somewhere to converge to.
    """
    linear_threshold = linear_threshold * disturb_scale
    angular_threshold = angular_threshold * disturb_scale
    for c in range(n_contacts):
        ia = contacts_a[c]
        ib = contacts_b[c]
        if ib < 0:
            continue
        fast_a = ((vx[ia] * vx[ia] + vy[ia] * vy[ia]) >
                  linear_threshold * linear_threshold) or abs(w[ia]) > angular_threshold
        fast_b = ((vx[ib] * vx[ib] + vy[ib] * vy[ib]) >
                  linear_threshold * linear_threshold) or abs(w[ib]) > angular_threshold
        if fast_a:
            disturbed[ib] = 1
            if awake[ib] == 0:
                awake[ib] = 1
                sleep_time[ib] = 0.0
        if fast_b:
            disturbed[ia] = 1
            if awake[ia] == 0:
                awake[ia] = 1
                sleep_time[ia] = 0.0


@njit(**_JIT)
def _capsule_contacts(
    capsule_seg, capsule_radius, capsule_vel, n_capsules,
    px, py, radius, awake, sleep_time,
    wverts, wnormals, vstart, vcount,
    contacts_a, contacts_b, contact_normal, contact_point,
    contact_depth, contact_key, static_vx, static_vy, written, count,
):
    """Shards against the capsule chain following the user's fingers.

    Each capsule is tested as a swept circle: the closest point on the
    segment to the shard becomes a circle, and the shard is collided
    against that. It is an approximation only in that a shard spanning a
    whole capsule sees one contact rather than two, which at finger scale
    is invisible.
    """
    for c in range(n_capsules):
        ax = capsule_seg[c, 0]
        ay = capsule_seg[c, 1]
        bx = capsule_seg[c, 2]
        by = capsule_seg[c, 3]
        cr = capsule_radius[c]
        ex = bx - ax
        ey = by - ay
        length2 = ex * ex + ey * ey

        lo_x = min(ax, bx) - cr
        hi_x = max(ax, bx) + cr
        lo_y = min(ay, by) - cr
        hi_y = max(ay, by) + cr

        for i in range(count):
            if px[i] + radius[i] < lo_x or px[i] - radius[i] > hi_x:
                continue
            if py[i] + radius[i] < lo_y or py[i] - radius[i] > hi_y:
                continue

            t = 0.0
            if length2 > 1e-12:
                t = ((px[i] - ax) * ex + (py[i] - ay) * ey) / length2
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
            qx = ax + ex * t
            qy = ay + ey * t
            dx = px[i] - qx
            dy = py[i] - qy
            reach = cr + radius[i]
            if dx * dx + dy * dy > reach * reach:
                continue

            # A hand sweeping through the pile has to wake it, or the
            # shards would simply ignore being touched.
            if awake[i] == 0:
                # World vertices are stale for a body that was asleep --
                # they are only refreshed for awake bodies at the top of a
                # substep. Wake it and let the next substep collide it
                # properly, rather than resolving against a stale pose.
                awake[i] = 1
                sleep_time[i] = 0.0
                continue

            before = written
            written = _polygon_vs_circle(
                i, qx, qy, cr, capsule_vel[c, 0], capsule_vel[c, 1],
                np.int64(c),
                wverts, wnormals, vstart, vcount,
                contacts_a, contacts_b, contact_normal, contact_point,
                contact_depth, contact_key, static_vx, static_vy, written,
            )
            if written > before:
                sleep_time[i] = 0.0
    return written


@njit(**_JIT)
def _simulate(
    substeps, dt, count,
    px, py, rot, vx, vy, w, prev_x, prev_y, prev_rot,
    inv_mass, inv_inertia, solver_inv_mass, solver_inv_inertia,
    radius, awake, sleep_time, disturbed, layer,
    verts, vstart, vcount, wverts, wnormals,
    order, pairs,
    contacts_a, contacts_b, contact_normal, contact_point, contact_depth,
    contact_key, static_vx, static_vy,
    normal_mass, tangent_mass, bias, pos_bias, normal_impulse, tangent_impulse,
    pseudo_impulse, ra, rb, pvx, pvy, pw,
    cache_key, cache_normal, cache_tangent,
    capsule_seg, capsule_radius, capsule_vel, n_capsules,
    gravity, linear_damping, angular_damping, restitution, friction,
    slop, baumgarte, iterations, position_iterations, width, height, floor_y,
    sleep_linear, sleep_angular, sleep_after, sleep_disturb_scale, stats,
):
    """Run ``substeps`` fixed steps. One call per frame, nothing allocated."""
    total_pairs = 0
    total_contacts = 0
    awake_count = 0
    inv_dt = 1.0 / dt

    for _ in range(substeps):
        for i in range(count):
            prev_x[i] = px[i]
            prev_y[i] = py[i]
            prev_rot[i] = rot[i]
            disturbed[i] = 0

        _effective_mass(inv_mass, inv_inertia, awake,
                        solver_inv_mass, solver_inv_inertia, count)
        _integrate_velocities(vx, vy, w, awake, inv_mass, gravity, dt,
                              linear_damping, angular_damping, count)
        _update_world_vertices(verts, vstart, vcount, px, py, rot, awake,
                               wverts, wnormals, count)

        _sort_axis(order, px, radius, count)
        n_pairs = _sweep_pairs(order, px, py, radius, awake, layer, pairs, count)
        total_pairs += n_pairs

        written = 0
        for p in range(n_pairs):
            ia = pairs[p, 0]
            ib = pairs[p, 1]
            # Keep the lower index first so the warm-start key is stable
            # regardless of which order the sweep produced them in.
            if ia > ib:
                ia, ib = ib, ia
            written = _collide_polygons(
                ia, ib, wverts, wnormals, vstart, vcount, px, py,
                contacts_a, contacts_b, contact_normal, contact_point,
                contact_depth, contact_key, written,
            )
        # Body-body contacts leave the static velocity slots untouched, so
        # clear them before the static passes append their own.
        for c in range(written):
            static_vx[c] = 0.0
            static_vy[c] = 0.0

        written = _boundary_contacts(
            wverts, vstart, vcount, awake, px, py, width, height, floor_y,
            contacts_a, contacts_b, contact_normal, contact_point,
            contact_depth, contact_key, static_vx, static_vy, written, count,
        )
        if n_capsules > 0:
            written = _capsule_contacts(
                capsule_seg, capsule_radius, capsule_vel, n_capsules,
                px, py, radius, awake, sleep_time,
                wverts, wnormals, vstart, vcount,
                contacts_a, contacts_b, contact_normal, contact_point,
                contact_depth, contact_key, static_vx, static_vy, written, count,
            )
        total_contacts += written

        _prepare_contacts(
            contacts_a, contacts_b, contact_normal, contact_point, contact_depth,
            contact_key, px, py, vx, vy, w, solver_inv_mass, solver_inv_inertia,
            normal_mass, tangent_mass, bias, pos_bias, normal_impulse,
            tangent_impulse, pseudo_impulse, ra, rb,
            cache_key, cache_normal, cache_tangent,
            restitution, slop, baumgarte, inv_dt, static_vx, static_vy, written,
        )
        _apply_warm_start(
            contacts_a, contacts_b, contact_normal, ra, rb,
            normal_impulse, tangent_impulse, vx, vy, w,
            solver_inv_mass, solver_inv_inertia, written,
        )
        for _ in range(iterations):
            _solve_velocity(
                contacts_a, contacts_b, contact_normal, ra, rb,
                normal_mass, tangent_mass, bias, normal_impulse, tangent_impulse,
                vx, vy, w, solver_inv_mass, solver_inv_inertia,
                static_vx, static_vy, friction, written,
            )
        # Disturbance is measured *after* the velocity solve, never
        # before. Before the solve, every resting body still carries the
        # gravity impulse just applied to it -- about 22px/s at 120Hz --
        # so the whole pile reads as moving, every body marks its
        # neighbours awake through the contact graph, and nothing ever
        # sleeps. Measured: 743 of 798 bodies awake after nine seconds.
        _mark_disturbed(contacts_a, contacts_b, vx, vy, w, awake, disturbed,
                        sleep_time, sleep_linear, sleep_angular,
                        sleep_disturb_scale, written)

        for _ in range(position_iterations):
            _solve_position(
                contacts_a, contacts_b, contact_normal, ra, rb,
                normal_mass, pos_bias, pseudo_impulse,
                pvx, pvy, pw, solver_inv_mass, solver_inv_inertia, written,
            )
        _store_impulses(contact_key, normal_impulse, tangent_impulse,
                        cache_key, cache_normal, cache_tangent, written)

        _integrate_positions(px, py, rot, vx, vy, w, pvx, pvy, pw,
                             awake, dt, count)
        awake_count = _update_sleep(
            vx, vy, w, awake, sleep_time, disturbed, dt,
            sleep_linear, sleep_angular, sleep_after, count,
        )

    stats[0] = total_pairs
    stats[1] = total_contacts
    stats[2] = awake_count


@njit(**_JIT)
def _interpolate(prev_x, prev_y, prev_rot, px, py, rot, alpha, out, count):
    beta = 1.0 - alpha
    for i in range(count):
        out[i, 0] = prev_x[i] * beta + px[i] * alpha
        out[i, 1] = prev_y[i] * beta + py[i] * alpha
        out[i, 2] = prev_rot[i] * beta + rot[i] * alpha


class PhysicsWorld:
    """Flat arrays, fixed substeps, one compiled call per frame."""

    def __init__(self, capacity: int, width: float, height: float) -> None:
        self.capacity = capacity
        self.width = float(width)
        self.height = float(height)
        self.floor_y = float(height)
        self.count = 0
        self.stats = PhysicsStats()
        self.iterations = config.SOLVER_ITERATIONS

        f32 = np.float32
        zeros = lambda n=capacity, dtype=f32: np.zeros(n, dtype)   # noqa: E731

        # State. Kept in float64 for position and rotation: reassembly
        # demands sub-pixel accuracy after thousands of integration steps,
        # and float32 accumulates visible drift over that many additions.
        self.px = np.zeros(capacity, np.float64)
        self.py = np.zeros(capacity, np.float64)
        self.rot = np.zeros(capacity, np.float64)
        self.vx = np.zeros(capacity, np.float64)
        self.vy = np.zeros(capacity, np.float64)
        self.w = np.zeros(capacity, np.float64)
        self.prev_x = np.zeros(capacity, np.float64)
        self.prev_y = np.zeros(capacity, np.float64)
        self.prev_rot = np.zeros(capacity, np.float64)

        self.inv_mass = np.zeros(capacity, np.float64)
        self.inv_inertia = np.zeros(capacity, np.float64)
        self.solver_inv_mass = np.zeros(capacity, np.float64)
        self.solver_inv_inertia = np.zeros(capacity, np.float64)
        self.radius = np.zeros(capacity, np.float64)
        self.depth = zeros()
        self.awake = np.zeros(capacity, np.uint8)
        self.sleep_time = np.zeros(capacity, np.float64)
        self.disturbed = np.zeros(capacity, np.uint8)
        self.layer = np.zeros(capacity, np.int32)
        self.order = np.arange(capacity, dtype=np.int32)

        # Polygon geometry, sized on load.
        self.verts = np.zeros((1, 2), np.float64)
        self.wverts = np.zeros((1, 2), np.float64)
        self.wnormals = np.zeros((1, 2), np.float64)
        self.vstart = np.zeros(capacity, np.int32)
        self.vcount = np.zeros(capacity, np.int32)

        self.pairs = np.zeros((MAX_PAIRS, 2), np.int32)
        self.contacts_a = np.zeros(MAX_CONTACTS, np.int32)
        self.contacts_b = np.zeros(MAX_CONTACTS, np.int32)
        self.contact_normal = np.zeros((MAX_CONTACTS, 2), np.float64)
        self.contact_point = np.zeros((MAX_CONTACTS, 2), np.float64)
        self.contact_depth = np.zeros(MAX_CONTACTS, np.float64)
        self.contact_key = np.zeros(MAX_CONTACTS, np.int64)
        self.static_vx = np.zeros(MAX_CONTACTS, np.float64)
        self.static_vy = np.zeros(MAX_CONTACTS, np.float64)
        self.normal_mass = np.zeros(MAX_CONTACTS, np.float64)
        self.tangent_mass = np.zeros(MAX_CONTACTS, np.float64)
        self.bias = np.zeros(MAX_CONTACTS, np.float64)
        self.pos_bias = np.zeros(MAX_CONTACTS, np.float64)
        self.normal_impulse = np.zeros(MAX_CONTACTS, np.float64)
        self.tangent_impulse = np.zeros(MAX_CONTACTS, np.float64)
        self.pseudo_impulse = np.zeros(MAX_CONTACTS, np.float64)
        self.pvx = np.zeros(capacity, np.float64)
        self.pvy = np.zeros(capacity, np.float64)
        self.pw = np.zeros(capacity, np.float64)
        self.ra = np.zeros((MAX_CONTACTS, 2), np.float64)
        self.rb = np.zeros((MAX_CONTACTS, 2), np.float64)

        self.cache_key = np.full(CACHE_SLOTS, CACHE_EMPTY, np.int64)
        self.cache_normal = np.zeros(CACHE_SLOTS, np.float64)
        self.cache_tangent = np.zeros(CACHE_SLOTS, np.float64)

        self.capsule_seg = np.zeros((MAX_CAPSULES, 4), np.float64)
        self.capsule_radius = np.zeros(MAX_CAPSULES, np.float64)
        self.capsule_vel = np.zeros((MAX_CAPSULES, 2), np.float64)
        self.n_capsules = 0

        self.transforms = np.zeros((capacity, 3), np.float32)
        self._stats_buffer = np.zeros(4, np.float64)
        self._accumulator = 0.0

    # -- setup ------------------------------------------------------------

    def load(self, result, density: float = 1.0) -> None:
        """Populate from a FractureResult. Rest pose becomes the start pose."""
        n = min(result.count, self.capacity)
        self.count = n

        self.px[:n] = result.centroid[:n, 0]
        self.py[:n] = result.centroid[:n, 1]
        self.rot[:n] = 0.0
        self.vx[:n] = 0.0
        self.vy[:n] = 0.0
        self.w[:n] = 0.0
        self.prev_x[:n] = self.px[:n]
        self.prev_y[:n] = self.py[:n]
        self.prev_rot[:n] = 0.0

        mass = np.maximum(result.area[:n].astype(np.float64) * density, 1e-3)
        self.inv_mass[:n] = 1.0 / mass
        self.inv_inertia[:n] = 1.0 / np.maximum(
            result.inertia[:n].astype(np.float64) * mass, 1e-3
        )
        self.radius[:n] = result.radius[:n]
        self.depth[:n] = result.depth[:n]
        self.awake[:n] = 1
        self.sleep_time[:n] = 0.0
        self.layer[:n] = np.clip(
            (result.depth[:n] * config.DEPTH_LAYERS).astype(np.int32),
            0, config.DEPTH_LAYERS - 1,
        )
        self.order[:n] = np.arange(n, dtype=np.int32)

        total_verts = int(result.poly_count[:n].sum())
        self.verts = result.poly_verts[:total_verts].astype(np.float64).copy()
        self.wverts = np.zeros_like(self.verts)
        self.wnormals = np.zeros_like(self.verts)
        self.vstart[:n] = result.poly_start[:n]
        self.vcount[:n] = result.poly_count[:n]

        self.cache_key[:] = CACHE_EMPTY
        self._accumulator = 0.0

    def explode(self, origin, strength: float = 900.0, spin: float = 5.0,
                seed: int | None = None) -> None:
        """Radial kick from the strike, so the break bursts outward."""
        n = self.count
        if n == 0:
            return
        rng = np.random.default_rng(seed)
        dx = self.px[:n] - origin[0]
        dy = self.py[:n] - origin[1]
        distance = np.hypot(dx, dy)
        distance[distance < 1e-3] = 1e-3
        # Falls off with distance, so the strike bursts and the far edges
        # merely sag -- a uniform kick reads as an explosion, not a break.
        falloff = 1.0 / (1.0 + distance / (self.width * 0.16))
        push = strength * falloff
        self.vx[:n] += dx / distance * push + rng.normal(0, 60, n)
        self.vy[:n] += dy / distance * push * 0.55 + rng.normal(0, 60, n)
        self.w[:n] += rng.normal(0, spin, n) * falloff
        self.awake[:n] = 1
        self.sleep_time[:n] = 0.0

    def set_capsules(self, segments: np.ndarray, radii: np.ndarray,
                     velocities: np.ndarray) -> None:
        n = min(len(radii), MAX_CAPSULES)
        self.n_capsules = n
        if n:
            self.capsule_seg[:n] = segments[:n]
            self.capsule_radius[:n] = radii[:n]
            self.capsule_vel[:n] = velocities[:n]

    def clear_capsules(self) -> None:
        self.n_capsules = 0

    def wake_all(self) -> None:
        self.awake[: self.count] = 1
        self.sleep_time[: self.count] = 0.0

    # -- stepping ---------------------------------------------------------

    def step(self, frame_dt: float) -> int:
        """Advance by ``frame_dt`` in fixed 120Hz substeps.

        The accumulator is what keeps reassembly timing from drifting: the
        solver only ever sees one dt, so identical inputs give identical
        motion regardless of what the frame rate did.
        """
        if self.count == 0:
            return 0
        dt = config.PHYSICS_DT
        self._accumulator += frame_dt
        substeps = int(self._accumulator / dt)
        if substeps <= 0:
            return 0
        if substeps > config.MAX_SUBSTEPS_PER_FRAME:
            # Drop the backlog rather than spiralling: trying to catch up
            # after a stall makes the next frame slower still.
            substeps = config.MAX_SUBSTEPS_PER_FRAME
            self._accumulator = 0.0
        else:
            self._accumulator -= substeps * dt

        t0 = time.perf_counter()
        _simulate(
            substeps, dt, self.count,
            self.px, self.py, self.rot, self.vx, self.vy, self.w,
            self.prev_x, self.prev_y, self.prev_rot,
            self.inv_mass, self.inv_inertia,
            self.solver_inv_mass, self.solver_inv_inertia,
            self.radius, self.awake, self.sleep_time, self.disturbed, self.layer,
            self.verts, self.vstart, self.vcount, self.wverts, self.wnormals,
            self.order, self.pairs,
            self.contacts_a, self.contacts_b, self.contact_normal,
            self.contact_point, self.contact_depth, self.contact_key,
            self.static_vx, self.static_vy,
            self.normal_mass, self.tangent_mass, self.bias, self.pos_bias,
            self.normal_impulse, self.tangent_impulse, self.pseudo_impulse,
            self.ra, self.rb, self.pvx, self.pvy, self.pw,
            self.cache_key, self.cache_normal, self.cache_tangent,
            self.capsule_seg, self.capsule_radius, self.capsule_vel,
            self.n_capsules,
            config.GRAVITY, config.LINEAR_DAMPING, config.ANGULAR_DAMPING,
            config.RESTITUTION, config.FRICTION, config.PENETRATION_SLOP,
            config.BAUMGARTE, self.iterations, config.POSITION_ITERATIONS,
            self.width, self.height, self.floor_y,
            config.SLEEP_LINEAR_VELOCITY, config.SLEEP_ANGULAR_VELOCITY,
            config.SLEEP_TIME, config.SLEEP_DISTURB_SCALE, self._stats_buffer,
        )
        elapsed = (time.perf_counter() - t0) * 1e3

        self.stats.substeps = substeps
        self.stats.bodies = self.count
        self.stats.pairs = int(self._stats_buffer[0] / max(substeps, 1))
        self.stats.contacts = int(self._stats_buffer[1] / max(substeps, 1))
        self.stats.awake = int(self._stats_buffer[2])
        self.stats.step_ms += (elapsed - self.stats.step_ms) * 0.2
        return substeps

    @property
    def alpha(self) -> float:
        """Fraction of a substep elapsed, for render interpolation."""
        return min(max(self._accumulator / config.PHYSICS_DT, 0.0), 1.0)

    def interpolated(self) -> np.ndarray:
        """(N, 3) of x, y, rot at the render instant."""
        _interpolate(self.prev_x, self.prev_y, self.prev_rot,
                     self.px, self.py, self.rot, self.alpha,
                     self.transforms, self.count)
        return self.transforms[: self.count]
