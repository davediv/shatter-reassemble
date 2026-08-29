"""The camera->canvas mapping is load-bearing for alignment, so it gets a
round-trip test at several aspect ratios. If this breaks, the fracture
stops radiating from the hand and shard UVs slide off the frozen frame.
"""

import unittest

import numpy as np

from shatter.viewport import CoverFit

CASES = [
    (1280, 720, 1920, 1080),    # matching 16:9
    (640, 480, 1920, 1080),     # 4:3 source, cropped top and bottom
    (1920, 1080, 1280, 720),    # downscale
    (1280, 720, 1080, 1920),    # portrait canvas, cropped left and right
]

PROBES = np.array(
    [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.25, 0.75], [0.9, 0.1]], np.float32
)


class TestCoverFit(unittest.TestCase):
    def test_uv_is_exact_inverse_of_landmark_mapping(self):
        for cw, ch, ww, wh in CASES:
            with self.subTest(camera=(cw, ch), canvas=(ww, wh)):
                fit = CoverFit(cw, ch, ww, wh)
                back = fit.canvas_to_uv(fit.landmarks_to_canvas(PROBES))
                np.testing.assert_allclose(back, PROBES, atol=1e-6)

    def test_mirror_flips_horizontally_and_only_once(self):
        fit = CoverFit(1280, 720, 1920, 1080)
        px = fit.landmarks_to_canvas(np.array([[0.0, 0.5], [1.0, 0.5]], np.float32))
        self.assertAlmostEqual(float(px[0, 0]), 1920.0, places=3)
        self.assertAlmostEqual(float(px[1, 0]), 0.0, places=3)
        # y is never mirrored
        self.assertAlmostEqual(float(px[0, 1]), 540.0, places=3)

    def test_mirror_can_be_disabled(self):
        fit = CoverFit(1280, 720, 1920, 1080, mirror=False)
        px = fit.landmarks_to_canvas(np.array([[0.0, 0.5]], np.float32))
        self.assertAlmostEqual(float(px[0, 0]), 0.0, places=3)

    def test_cover_never_letterboxes(self):
        # Every canvas corner must land inside the scaled camera image.
        for cw, ch, ww, wh in CASES:
            with self.subTest(camera=(cw, ch), canvas=(ww, wh)):
                fit = CoverFit(cw, ch, ww, wh)
                corners = np.array(
                    [[0, 0], [ww, 0], [0, wh], [ww, wh]], np.float32
                )
                uv = fit.canvas_to_uv(corners)
                self.assertTrue(np.all(uv >= -1e-6) and np.all(uv <= 1 + 1e-6))

    def test_out_parameter_is_filled_in_place(self):
        fit = CoverFit(1280, 720, 1920, 1080)
        out = np.zeros_like(PROBES)
        result = fit.landmarks_to_canvas(PROBES, out=out)
        self.assertIs(result, out)
        self.assertTrue(np.any(out != 0))


if __name__ == "__main__":
    unittest.main()
