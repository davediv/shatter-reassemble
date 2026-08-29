"""Solver tests.

Deliberately assertions about *behaviour*, not timings: this machine's
load varies enormously and a wall-clock assertion would be a flaky test.
Behaviour is deterministic given the fixed timestep, so these hold
regardless of how contended the box is.

Several of these are regressions for bugs that were invisible in the
visual output and only showed up as "the pile never sleeps".
"""

import unittest

import numpy as np

from shatter import config
from shatter.fracture import fracture
from shatter.physics import PhysicsWorld

WIDTH, HEIGHT = 1920, 1080


def build(count=150, seed=3, explode=True):
    result = fracture(WIDTH, HEIGHT, (1400.0, 400.0), count, seed=seed)
    world = PhysicsWorld(count + 16, WIDTH, HEIGHT)
    world.load(result)
    if explode:
        world.explode((1400.0, 400.0), seed=seed)
    return world, result


def run(world, seconds, dt=1 / 60):
    for _ in range(int(seconds / dt)):
        world.step(dt)


class TestContainment(unittest.TestCase):
    def test_nothing_escapes_the_canvas(self):
        world, _ = build()
        run(world, 6.0)
        n = world.count
        # A little overlap at the walls is the solver's slop; a shard
        # hundreds of pixels out has tunnelled.
        self.assertGreater(world.px[:n].min(), -60.0)
        self.assertLess(world.px[:n].max(), WIDTH + 60.0)
        self.assertLess(world.py[:n].max(), HEIGHT + 60.0)

    def test_shards_fall(self):
        world, result = build(explode=False)
        before = world.py[: world.count].mean()
        run(world, 1.0)
        self.assertGreater(world.py[: world.count].mean(), before + 20.0)

    def test_the_pile_empties_the_top_of_the_frame(self):
        # Depth layering is what makes this possible: a perfect tiling has
        # nowhere to compact to, so without layers the top never clears.
        world, _ = build(count=400)
        run(world, 6.0)
        top_tenth = (world.py[: world.count] < HEIGHT * 0.25).mean()
        self.assertLess(top_tenth, 0.22)


class TestSleeping(unittest.TestCase):
    def test_the_pile_settles_and_sleeps(self):
        world, _ = build(count=400)
        run(world, 8.0)
        asleep = world.count - world.stats.awake
        self.assertGreater(asleep / world.count, 0.85,
                           f"only {asleep}/{world.count} asleep")

    def test_sleeping_collapses_the_contact_count(self):
        # The entire performance argument for sleeping.
        world, _ = build(count=400)
        run(world, 1.0)
        busy = world.stats.contacts
        run(world, 7.0)
        settled = world.stats.contacts
        self.assertGreater(busy, 200)
        self.assertLess(settled, busy * 0.2,
                        f"contacts only fell {busy} -> {settled}")

    def test_sleeping_bodies_hold_still(self):
        """Regression: the solver used to keep applying impulses to
        sleeping bodies. Their velocities accumulated while their
        positions never integrated, so they leapt when woken."""
        world, _ = build(count=300)
        run(world, 8.0)
        n = world.count
        asleep = world.awake[:n] == 0
        self.assertTrue(asleep.any())
        self.assertEqual(float(np.abs(world.vx[:n][asleep]).max()), 0.0)
        self.assertEqual(float(np.abs(world.vy[:n][asleep]).max()), 0.0)
        self.assertEqual(float(np.abs(world.w[:n][asleep]).max()), 0.0)

    def test_a_settled_pile_does_not_regain_energy(self):
        """Regression: boundary and body-body contacts shared warm-start
        keys, so shards warm-started from the floor's impulse and the pile
        pumped itself back up instead of settling."""
        world, _ = build(count=300)
        run(world, 8.0)
        n = world.count
        speed = np.hypot(world.vx[:n], world.vy[:n])
        energy_before = float((speed ** 2).sum())
        run(world, 4.0)
        speed = np.hypot(world.vx[:n], world.vy[:n])
        energy_after = float((speed ** 2).sum())
        self.assertLess(energy_after, max(energy_before, 1.0) * 2.0 + 1.0)

    def test_wake_all_restarts_the_pile(self):
        world, _ = build(count=200)
        run(world, 8.0)
        self.assertLess(world.stats.awake, world.count)
        world.wake_all()
        world.step(1 / 60)
        self.assertEqual(world.stats.awake, world.count)


class TestFixedTimestep(unittest.TestCase):
    def test_substep_count_is_independent_of_frame_rate(self):
        """The reason for the accumulator: reassembly timing must not
        drift with frame rate."""
        a, _ = build(count=120, seed=5)
        b, _ = build(count=120, seed=5)
        for _ in range(20):
            a.step(1 / 60)
            a.step(1 / 60)
        for _ in range(20):
            b.step(1 / 30)
        np.testing.assert_allclose(a.px[: a.count], b.px[: b.count], atol=1e-9)
        np.testing.assert_allclose(a.py[: a.count], b.py[: b.count], atol=1e-9)

    def test_a_stall_does_not_spiral(self):
        world, _ = build(count=120)
        world.step(2.0)          # a two-second hitch
        self.assertLessEqual(world.stats.substeps, config.MAX_SUBSTEPS_PER_FRAME)

    def test_interpolation_stays_between_the_substeps(self):
        world, _ = build(count=120)
        run(world, 0.5)
        world.step(1 / 90)       # leave the accumulator part-full
        out = world.interpolated()
        n = world.count
        # The render transforms are float32 on purpose -- they are uploaded
        # straight into a float32 texture -- so the tolerance is one ulp at
        # canvas scale, not zero. That is still ~30x finer than the GPU's
        # 1/256 subpixel rasterisation grid.
        tolerance = 2e-4
        low = np.minimum(world.prev_x[:n], world.px[:n]) - tolerance
        high = np.maximum(world.prev_x[:n], world.px[:n]) + tolerance
        self.assertTrue(np.all(out[:, 0] >= low) and np.all(out[:, 0] <= high))


class TestDeterminism(unittest.TestCase):
    def test_identical_inputs_give_identical_output(self):
        a, _ = build(count=200, seed=9)
        b, _ = build(count=200, seed=9)
        run(a, 3.0)
        run(b, 3.0)
        np.testing.assert_array_equal(a.px[: a.count], b.px[: b.count])
        np.testing.assert_array_equal(a.rot[: a.count], b.rot[: b.count])


class TestCapsules(unittest.TestCase):
    def test_a_moving_hand_wakes_and_pushes_the_pile(self):
        world, _ = build(count=300)
        run(world, 8.0)
        self.assertLess(world.stats.awake, world.count * 0.3)

        n = world.count
        before = world.px[:n].copy()
        # A capsule swept through the settled pile, moving fast.
        target_y = float(np.percentile(world.py[:n], 80))
        segments = np.array([[200.0, target_y, 600.0, target_y]], np.float64)
        world.set_capsules(segments, np.array([90.0]), np.array([[2600.0, 0.0]]))
        run(world, 0.6)
        self.assertGreater(world.stats.awake, 5, "the hand woke nothing")
        moved = np.abs(world.px[:n] - before)
        self.assertGreater(moved.max(), 4.0, "the hand moved nothing")

    def test_capsules_can_be_cleared(self):
        world, _ = build(count=120)
        world.set_capsules(np.array([[0.0, 0.0, 10.0, 10.0]], np.float64),
                           np.array([20.0]), np.array([[0.0, 0.0]]))
        self.assertEqual(world.n_capsules, 1)
        world.clear_capsules()
        self.assertEqual(world.n_capsules, 0)
        world.step(1 / 60)


if __name__ == "__main__":
    unittest.main()
