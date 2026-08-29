"""Reassembly tests.

"Sub-pixel accuracy on the final pose is mandatory -- any misalignment
kills the whole app." So the central assertion here is not that the error
is small, it is that the error is exactly zero: the landing is assigned,
never interpolated.
"""

import unittest

import numpy as np

from shatter import config
from shatter.fracture import fracture
from shatter.physics import PhysicsWorld
from shatter.reassemble import Reassembly

WIDTH, HEIGHT = 1920, 1080


def scattered_world(count=200, seconds=2.0, seed=4):
    result = fracture(WIDTH, HEIGHT, (1400.0, 400.0), count, seed=seed)
    world = PhysicsWorld(count + 16, WIDTH, HEIGHT)
    world.load(result)
    world.explode((1400.0, 400.0), seed=seed)
    for _ in range(int(seconds * 60)):
        world.step(1 / 60)
    return world, result


class TestLanding(unittest.TestCase):
    def test_every_shard_lands_exactly_on_its_rest_pose(self):
        world, result = scattered_world()
        anim = Reassembly(WIDTH, HEIGHT)
        anim.begin(world, result, 0.0)
        t = 0.0
        while anim.active:
            t += 1 / 60
            anim.update(t)
        self.assertEqual(anim.rest_error(), 0.0)

    def test_rotation_lands_on_a_whole_turn(self):
        world, result = scattered_world()
        anim = Reassembly(WIDTH, HEIGHT)
        anim.begin(world, result, 0.0)
        t = 0.0
        while anim.active:
            t += 1 / 60
            anim.update(t)
        # Rest rotation is the nearest whole turn, so cos/sin are exactly
        # the identity rotation and the shard lands square.
        turns = anim.transforms[:, 2] / (2.0 * np.pi)
        np.testing.assert_allclose(turns, np.round(turns), atol=1e-9)

    def test_shards_do_not_spin_the_long_way_home(self):
        # A shard that spun three times on the way down must not spin
        # three times back; it unwinds to the nearest whole turn.
        world, result = scattered_world()
        n = world.count
        world.rot[:n] = np.linspace(-8 * np.pi, 8 * np.pi, n)
        anim = Reassembly(WIDTH, HEIGHT)
        anim.begin(world, result, 0.0)
        travel = np.abs(anim._rest[:, 2] - world.rot[:n])
        self.assertLess(travel.max(), np.pi + 1e-6)


class TestTiming(unittest.TestCase):
    def test_progress_is_monotonic_and_completes(self):
        world, result = scattered_world(count=120)
        anim = Reassembly(WIDTH, HEIGHT)
        anim.begin(world, result, 0.0)
        previous = -1.0
        t = 0.0
        for _ in range(400):
            t += 1 / 60
            state = anim.update(t)
            self.assertGreaterEqual(state.progress, previous - 1e-9)
            previous = state.progress
            if state.finished:
                break
        self.assertTrue(state.finished)
        self.assertAlmostEqual(state.progress, 1.0, places=6)

    def test_bevel_closes_before_the_end(self):
        # The bevel must reach zero, or every neighbouring pair keeps a
        # hairline gap exactly when the app is trying to look flawless.
        world, result = scattered_world(count=120)
        anim = Reassembly(WIDTH, HEIGHT)
        anim.begin(world, result, 0.0)
        t = 0.0
        state = anim.update(t)
        self.assertAlmostEqual(state.bevel, 1.0, places=3)
        while anim.active:
            t += 1 / 60
            state = anim.update(t)
        self.assertEqual(state.bevel, 0.0)

    def test_crossfade_runs_over_the_final_window(self):
        world, result = scattered_world(count=120)
        anim = Reassembly(WIDTH, HEIGHT)
        anim.begin(world, result, 0.0)
        t = 0.0
        seen = []
        while anim.active and t < 10.0:
            t += 1 / 120
            seen.append(anim.update(t).crossfade)
        self.assertEqual(seen[0], 0.0)
        self.assertAlmostEqual(seen[-1], 1.0, places=3)
        # It must be a fade, not a cut.
        partial = [c for c in seen if 0.05 < c < 0.95]
        self.assertGreater(len(partial), 3)

    def test_stagger_makes_the_centre_land_first(self):
        world, result = scattered_world(count=300)
        anim = Reassembly(WIDTH, HEIGHT)
        anim.begin(world, result, 0.0)
        centre = np.hypot(result.centroid[: world.count, 0] - WIDTH / 2,
                          result.centroid[: world.count, 1] - HEIGHT / 2)
        near = anim._delay[centre < np.percentile(centre, 25)].mean()
        far = anim._delay[centre > np.percentile(centre, 75)].mean()
        self.assertLess(near, far)
        self.assertLessEqual(anim._delay.max(), config.REASSEMBLE_MAX_DELAY + 1e-6)

    def test_overshoot_actually_overshoots(self):
        world, result = scattered_world(count=120)
        anim = Reassembly(WIDTH, HEIGHT)
        anim.begin(world, result, 0.0)
        rest = anim._rest[:, :2].copy()
        start = anim._launch[:, :2].copy()
        overshot = False
        t = 0.0
        while anim.active:
            t += 1 / 120
            anim.update(t)
            # Past the rest pose, measured along the direction of travel.
            direction = rest - start
            travel = anim.transforms[:, :2] - start
            norm = np.einsum("ij,ij->i", direction, direction)
            good = norm > 1.0
            if good.any():
                frac = np.einsum("ij,ij->i", travel[good], direction[good]) / norm[good]
                if frac.max() > 1.005:
                    overshot = True
                    break
        self.assertTrue(overshot, "no shard overshot its rest pose")


if __name__ == "__main__":
    unittest.main()
