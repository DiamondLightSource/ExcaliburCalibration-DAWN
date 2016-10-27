import unittest

from pkg_resources import require
require("mock")
from mock import patch, MagicMock, ANY

util_patch_path = "excaliburcalibrationdawn.util"

import numpy as np

from excaliburcalibrationdawn import util


class FunctionsTest(unittest.TestCase):

    def setUp(self):
        self.array = np.array(range(25)).reshape([5, 5])

    def test_grab_slice(self):
        expected_array = np.array([[6, 7, 8],
                                   [11, 12, 13],
                                   [16, 17, 18]])

        subarray = util.grab_slice(self.array, [1, 1], [3, 3])

        np.testing.assert_array_equal(expected_array, subarray)

    def test_set_slice(self):
        expected_array = np.array([[0, 1, 2, 3, 4],
                                   [5, 0, 0, 0, 9],
                                   [10, 0, 0, 0, 14],
                                   [15, 0, 0, 0, 19],
                                   [20, 21, 22, 23, 24]])
        subarray = np.zeros([3, 3])

        util.set_slice(self.array, [1, 1], [3, 3], subarray)

        np.testing.assert_array_equal(expected_array, self.array)

    @patch('numpy.rot90')
    @patch('numpy.savetxt')
    @patch('numpy.loadtxt')
    def test_rotate_config(self, load_mock, save_mock, rotate_mock):
        test_path = 'path/to/config'

        util.rotate_config(test_path)

        load_mock.assert_called_once_with(test_path)
        rotate_mock.assert_called_once_with(load_mock.return_value, 2)
        save_mock.assert_called_once_with(test_path, rotate_mock.return_value,
                                          fmt='%.18g', delimiter=' ')

    datetime_mock = MagicMock()
    datetime_mock.now.return_value.isoformat.return_value = "20161020~154548.834130"

    @patch(util_patch_path + '.datetime',
           new=datetime_mock)
    def test_get_time_stamp(self):
        expected_time_stamp = "20161020~154548"

        time_stamp = util.get_time_stamp()

        self.assertEqual(expected_time_stamp, time_stamp)

    def test_to_list_given_value_then_return_list(self):
        response = util.to_list(1)

        self.assertEqual([1], response)

    def test_to_list_given_list_then_return(self):
        response = util.to_list([1])

        self.assertEqual([1], response)

