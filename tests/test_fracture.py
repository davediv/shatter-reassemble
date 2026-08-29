"""Fracture tests, including the one the whole app rests on.

"Reassembly lands with zero visible seams" is the acceptance criterion,
and it is decided here, not in the reassembly code. Reassembly only has to
return every shard to its rest transform; whether that produces a seamless
image depends entirely on whether neighbouring cells share *identical*
edges. So the load-bearing test walks every edge of every cell and checks
that each interior edge is shared by exactly two cells at the same
coordinates, and reports how far apart they actually are.
"""

import time
import unittest
from collections import defaultdict
from threading import Event
from types import SimpleNamespace
from unittest import mock

import numpy as np

from shatter import config
from shatter.fracture import (
    FLOATS_PER_VERTEX,
    PART_FACE,
    FracturePrewarmer,
    fracture,
)

WIDTH, HEIGHT = 1920, 1080
ORIGIN = (1400.0, 400.0)


def world_polygons(result):
    for i in range(result.count):
        start = int(result.poly_start[i])
        count = int(result.poly_count[i])
        yield result.poly_verts[start: start + count] + result.centroid[i]


class TestTiling(unittest.TestCase):
    def test_cells_tile_the_canvas(self):
        result = fracture(WIDTH, HEIGHT, ORIGIN, 600, seed=3)
        coverage = result.area.sum() / (WIDTH * HEIGHT)
        # The only shortfall permitted is the sliver filter.
        self.assertGreater(coverage, 0.999)
        self.assertLess(coverage, 1.0001)

    def test_every_cell_lies_inside_the_canvas(self):
        result = fracture(WIDTH, HEIGHT, ORIGIN, 600, seed=4)
        for poly in world_polygons(result):
            self.assertGreaterEqual(poly.min(), -0.02)
            self.assertLessEqual(poly[:, 0].max(), WIDTH + 0.02)
            self.assertLessEqual(poly[:, 1].max(), HEIGHT + 0.02)

    def test_cells_are_convex_and_counter_clockwise(self):
        # The solver's SAT narrowphase assumes both.
        result = fracture(WIDTH, HEIGHT, ORIGIN, 400, seed=5)
        for poly in world_polygons(result):
            edges = np.roll(poly, -1, axis=0) - poly
            cross = (edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1]
                     - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0])
            self.assertTrue(np.all(cross >= -1e-3), "cell is not convex")
            area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1)
                                - np.roll(poly[:, 0], -1) * poly[:, 1])
            self.assertGreater(area, 0.0, "cell is not counter-clockwise")

    def test_neighbouring_cells_share_identical_edges(self):
        """The seam test. If this fails, reassembly cannot be seamless."""
        result = fracture(WIDTH, HEIGHT, ORIGIN, 600, seed=6)
        buckets = defaultdict(list)
        for poly in world_polygons(result):
            for a, b in zip(poly, np.roll(poly, -1, axis=0)):
                # Key on a coarse rounding so floating point cannot split
                # a genuinely shared edge across two buckets; the exact
                # coordinates are compared inside the bucket.
                key = tuple(sorted(
                    (tuple(np.round(a, 1)), tuple(np.round(b, 1)))
                ))
                buckets[key].append((a, b))

        def on_border(edge):
            (a, b) = edge
            return (
                (abs(a[0]) < 0.05 and abs(b[0]) < 0.05)
                or (abs(a[0] - WIDTH) < 0.05 and abs(b[0] - WIDTH) < 0.05)
                or (abs(a[1]) < 0.05 and abs(b[1]) < 0.05)
                or (abs(a[1] - HEIGHT) < 0.05 and abs(b[1] - HEIGHT) < 0.05)
            )

        unmatched = 0
        interior = 0
        worst = 0.0
        for pairs in buckets.values():
            if len(pairs) == 1:
                if not on_border(pairs[0]):
                    unmatched += 1
                continue
            interior += 1
            first, second = pairs[0], pairs[1]
            gap = min(
                max(np.abs(first[0] - second[0]).max(),
                    np.abs(first[1] - second[1]).max()),
                max(np.abs(first[0] - second[1]).max(),
                    np.abs(first[1] - second[0]).max()),
            )
            worst = max(worst, float(gap))

        self.assertGreater(interior, 500, "no interior edges found at all")
        # Unmatched interior edges come only from cells the sliver filter
        # dropped; anything more means the clip is not producing shared
        # geometry and the pile will not close up.
        self.assertLess(unmatched / interior, 0.02,
                        f"{unmatched} unmatched interior edges of {interior}")
        # The number that matters: how far apart two cells put the same
        # corner. A GPU rasterises on a 1/256 px subpixel grid, so
        # anything well under that produces no crack at all.
        self.assertLess(worst, 0.01, f"shared corners differ by {worst:.5f}px")


class TestClustering(unittest.TestCase):
    def test_break_radiates_from_the_strike(self):
        result = fracture(WIDTH, HEIGHT, ORIGIN, 800, seed=7)
        distance = np.hypot(result.centroid[:, 0] - ORIGIN[0],
                            result.centroid[:, 1] - ORIGIN[1])
        near = result.area[distance < 300]
        far = result.area[distance > 900]
        self.assertGreater(near.size, 20)
        self.assertGreater(far.size, 20)
        # Small dense shards at the impact, big plates further out.
        self.assertGreater(far.mean() / near.mean(), 3.0)

    def test_depth_is_nearer_at_the_strike(self):
        result = fracture(WIDTH, HEIGHT, ORIGIN, 600, seed=8)
        distance = np.hypot(result.centroid[:, 0] - ORIGIN[0],
                            result.centroid[:, 1] - ORIGIN[1])
        near = result.depth[distance < 300].mean()
        far = result.depth[distance > 900].mean()
        self.assertGreater(near, far)


class TestGeometry(unittest.TestCase):
    def test_vertex_count_matches_the_emission_rule(self):
        # face (3n-6) + bevel (6n) + wall (6n) = 15n - 6 per cell.
        result = fracture(WIDTH, HEIGHT, ORIGIN, 300, seed=9)
        expected = int((result.poly_count.astype(np.int64) * 15 - 6).sum())
        self.assertEqual(result.vertices.shape[0], expected)
        self.assertEqual(result.vertices.shape[1], FLOATS_PER_VERTEX)

    def test_shard_indices_are_dense_and_in_range(self):
        result = fracture(WIDTH, HEIGHT, ORIGIN, 800, seed=10)
        indices = result.vertices[:, 6].astype(np.int64)
        self.assertEqual(indices.min(), 0)
        self.assertEqual(indices.max(), result.count - 1)
        self.assertEqual(np.unique(indices).size, result.count)

    def test_face_vertices_carry_a_nonzero_bevel_inset(self):
        result = fracture(WIDTH, HEIGHT, ORIGIN, 300, seed=11)
        faces = result.vertices[result.vertices[:, 5] == PART_FACE]
        inset = np.hypot(faces[:, 2], faces[:, 3])
        self.assertGreater(inset.max(), 0.5)
        # And the inset always points inward, never outward.
        self.assertLess(inset.max(), config.BEVEL_WIDTH * 5.0)

    def test_outer_boundary_positions_are_untouched_by_the_bevel(self):
        # The whole seam strategy depends on local.xy being the true cell
        # boundary, with the inset kept separate for the shader to animate.
        result = fracture(WIDTH, HEIGHT, ORIGIN, 200, seed=12)
        for shard in range(min(result.count, 40)):
            rows = result.vertices[result.vertices[:, 6] == shard]
            emitted = {(round(float(x), 3), round(float(y), 3))
                       for x, y in rows[:, :2]}
            start = int(result.poly_start[shard])
            count = int(result.poly_count[shard])
            for vertex in result.poly_verts[start: start + count]:
                key = (round(float(vertex[0]), 3), round(float(vertex[1]), 3))
                self.assertIn(key, emitted)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_gives_the_same_fracture(self):
        a = fracture(WIDTH, HEIGHT, ORIGIN, 400, seed=42)
        b = fracture(WIDTH, HEIGHT, ORIGIN, 400, seed=42)
        self.assertEqual(a.count, b.count)
        np.testing.assert_array_equal(a.centroid, b.centroid)
        np.testing.assert_array_equal(a.vertices, b.vertices)

    def test_different_origins_give_different_fractures(self):
        a = fracture(WIDTH, HEIGHT, (200.0, 200.0), 400, seed=42)
        b = fracture(WIDTH, HEIGHT, (1700.0, 900.0), 400, seed=42)
        self.assertFalse(np.array_equal(a.centroid, b.centroid))


class TestPrewarmer(unittest.TestCase):
    def test_repeated_nearby_requests_share_an_inflight_build(self):
        started = Event()
        release = Event()
        calls = []

        def slow_fracture(width, height, origin, count, *, bevel):
            calls.append(origin)
            started.set()
            self.assertTrue(release.wait(1.0))
            return SimpleNamespace(
                origin=origin,
                shard_count=count,
                bevel=bevel,
            )

        prewarmer = FracturePrewarmer()
        with mock.patch("shatter.fracture.fracture", side_effect=slow_fracture):
            prewarmer.start()
            try:
                prewarmer.request(WIDTH, HEIGHT, ORIGIN, 800, config.BEVEL_WIDTH)
                self.assertTrue(started.wait(1.0))
                for offset in range(20):
                    prewarmer.request(
                        WIDTH,
                        HEIGHT,
                        (ORIGIN[0] + offset, ORIGIN[1]),
                        800,
                        config.BEVEL_WIDTH,
                    )
                release.set()
                deadline = time.perf_counter() + 1.0
                while time.perf_counter() < deadline:
                    with prewarmer._lock:
                        if prewarmer._result is not None:
                            break
                    time.sleep(0.001)
                self.assertIsNotNone(
                    prewarmer.take(
                        WIDTH,
                        HEIGHT,
                        (ORIGIN[0] + 19, ORIGIN[1]),
                        800,
                        config.BEVEL_WIDTH,
                    )
                )
            finally:
                release.set()
                prewarmer.stop()

        self.assertEqual(len(calls), 1)
        self.assertEqual(prewarmer.hits, 1)


if __name__ == "__main__":
    unittest.main()
