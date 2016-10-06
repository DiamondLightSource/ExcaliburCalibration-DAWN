import unittest

import numpy as np

from excaliburcalibrationdawn import arrayutil


class FunctionsTest(unittest.TestCase):

    def setUp(self):
        self.array = np.array(range(25)).reshape([5, 5])

    def test_grab_slice(self):
        expected_array = np.array([[6, 7, 8],
                                   [11, 12, 13],
                                   [16, 17, 18]])

        subarray = arrayutil.grab_slice(self.array, [1, 1], [3, 3])

        np.testing.assert_array_equal(expected_array, subarray)

    def test_set_slice(self):
        expected_array = np.array([[0, 1, 2, 3, 4],
                                   [5, 0, 0, 0, 9],
                                   [10, 0, 0, 0, 14],
                                   [15, 0, 0, 0, 19],
                                   [20, 21, 22, 23, 24]])
        subarray = np.zeros([3, 3])

        arrayutil.set_slice(self.array, [1, 1], [3, 3], subarray)

        np.testing.assert_array_equal(expected_array, self.array)
