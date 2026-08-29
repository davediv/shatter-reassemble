"""Frame capture keeps the latest-wins ring allocation-free."""

import unittest

import numpy as np

from shatter.camera import _retrieve_into


class FakeCapture:
    def __init__(self, returned=None):
        self.returned = returned
        self.output = None

    def retrieve(self, output):
        self.output = output
        if self.returned is None:
            output[:] = 37
            return True, output
        return True, self.returned


class TestBufferedDecode(unittest.TestCase):
    def test_matching_frame_decodes_into_the_ring_buffer(self):
        destination = np.zeros((8, 12, 3), np.uint8)
        capture = FakeCapture()

        self.assertTrue(_retrieve_into(capture, destination))
        self.assertIs(capture.output, destination)
        self.assertTrue(np.all(destination == 37))

    def test_backend_that_ignores_output_still_copies_safely(self):
        returned = np.full((8, 12, 3), 91, np.uint8)
        destination = np.zeros_like(returned)

        self.assertTrue(_retrieve_into(FakeCapture(returned), destination))
        np.testing.assert_array_equal(destination, returned)


if __name__ == "__main__":
    unittest.main()
