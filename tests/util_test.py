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

    @patch(util_patch_path + '.grab_slice')
    @patch(util_patch_path + '.generate_chip_range')
    def test_grab_chip_slice(self, generate_mock, grab_mock):
        array = MagicMock()
        generate_mock.return_value = MagicMock(), MagicMock

        value = util.grab_chip_slice(array, 1)

        generate_mock.assert_called_once_with(1)
        grab_mock.assert_called_once_with(array, generate_mock.return_value[0],
                                          generate_mock.return_value[1])
        self.assertEqual(grab_mock.return_value, value)

    @patch(util_patch_path + '.set_slice')
    @patch(util_patch_path + '.generate_chip_range')
    def test_set_chip_slice(self, generate_mock, set_mock):
        array = np.array([[1, 2, 3, 4, 5],
                          [10, 20, 30, 40, 50],
                          [100, 200, 300, 400, 500]])
        generate_mock.return_value = MagicMock(), MagicMock

        util.set_chip_slice(array, 1, 0)

        generate_mock.assert_called_once_with(1)
        set_mock.assert_called_once_with(array, generate_mock.return_value[0], generate_mock.return_value[1], 0)

    def test_generate_chip_range(self):
        expected_start = [0, 256]
        expected_stop = [255, 511]

        start, stop = util.generate_chip_range(1)

        np.testing.assert_array_equal(expected_start, start)
        np.testing.assert_array_equal(expected_stop, stop)

    @patch('numpy.rot90')
    @patch(util_patch_path + '.save_array')
    @patch('numpy.loadtxt')
    def test_rotate_config(self, load_mock, save_mock, rotate_mock):
        test_path = 'path/to/config'

        util.rotate_array(test_path)

        load_mock.assert_called_once_with(test_path)
        rotate_mock.assert_called_once_with(load_mock.return_value, 2)
        save_mock.assert_called_once_with(test_path, rotate_mock.return_value)

    @patch('numpy.savetxt')
    def test_save_array(self, save_mock):
        test_path = "path/to/config"
        mock_array = MagicMock()

        util.save_array(test_path, mock_array)

        save_mock.assert_called_once_with(test_path, mock_array,
                                          fmt="%.18g", delimiter=" ")

    datetime_mock = MagicMock()
    datetime_mock.now.return_value.isoformat.return_value = "20161020~154548.834130"

    @patch(util_patch_path + '.datetime',
           new=datetime_mock)
    def test_get_time_stamp(self):
        expected_time_stamp = "20161020~154548"

        time_stamp = util.get_time_stamp()

        self.assertEqual(expected_time_stamp, time_stamp)

    @patch(util_patch_path + '.get_time_stamp', return_value="20161020~154548")
    def test_tag_plot_name(self, get_mock):
        plot_name = util.tag_plot_name("TestImage", "Node 1")

        get_mock.assert_called_once_with()
        self.assertEqual("Node 1 - TestImage - 20161020~154548", plot_name)

    def test_to_list_given_value_then_return_list(self):
        response = util.to_list(1)

        self.assertEqual([1], response)

    def test_to_list_given_list_then_return(self):
        response = util.to_list([1])

        self.assertEqual([1], response)

    @patch(util_patch_path + '.time.sleep')
    @patch(util_patch_path + '.os.path.isfile',
           side_effect=[False, False, False, False, False, False, False, True])
    def test_wait_for_file_appears(self, isfile_mock, sleep_mock):

        response = util.wait_for_file("/path/to/file", 5)

        self.assertTrue(response)
        self.assertEqual([0.1] * 8,
                         [call[0][0] for call in sleep_mock.call_args_list])
        self.assertEqual(8, sleep_mock.call_count)
        self.assertEqual(8, isfile_mock.call_count)

    @patch(util_patch_path + '.time.sleep')
    @patch(util_patch_path + '.os.path.isfile', return_value=False)
    def test_wait_for_file_timeout(self, isfile_mock, sleep_mock):

        response = util.wait_for_file("/path/to/file", 5)

        self.assertFalse(response)
        self.assertEqual([0.1] * 50,
                         [call[0][0] for call in sleep_mock.call_args_list])
        self.assertEqual(50, sleep_mock.call_count)
        self.assertEqual(50, isfile_mock.call_count)

    @patch('filecmp.cmp')
    def test_file_match(self, cmp_mock):

        response = util.files_match("/path/to/file", "/path/to/file2")

        cmp_mock.assert_called_once_with("/path/to/file", "/path/to/file2")
        self.assertEqual(cmp_mock.return_value, response)

    @patch(util_patch_path + '._ReturnThread')
    def test_spawn_thread(self, thread_init_mock):
        thread_mock = MagicMock()
        thread_init_mock.return_value = thread_mock
        function_mock = MagicMock()

        response = util.spawn_thread(function_mock, "arg1", arg2="arg2")

        thread_init_mock.assert_called_once_with(target=function_mock,
                                                 args=("arg1",),
                                                 kwargs=dict(arg2="arg2"))
        thread_mock.start.assert_called_once_with()
        self.assertEqual(thread_mock, response)

    def test_wait_for_threads(self):
        mocks = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]

        response = util.wait_for_threads(mocks)

        for idx, mock in enumerate(mocks):
            mock.join.assert_called_once_with()
            self.assertEqual(mock.join.return_value, response[idx])


class ReturnThreadTest(unittest.TestCase):

    @patch('threading.Thread.__init__')
    def test_init(self, thread_mock):
        thread = util._ReturnThread(group="group", target="target",
                                    name="name", args="args", kwargs="kwargs",
                                    verbose="verbose")

        thread_mock.assert_called_once_with("group", "target", "name", "args",
                                            "kwargs", "verbose")
        self.assertIsNone(thread._return)

    def test_run(self):
        function_mock = MagicMock()
        thread = util._ReturnThread(target=function_mock,
                                    args="args", kwargs=dict(arg2="arg2"))
        thread._Thread__target = function_mock
        thread._Thread__args = ["args"]
        thread._Thread__kwargs = dict(arg2="arg2")

        thread.run()

        function_mock.assert_called_once_with("args", arg2="arg2")
        self.assertEqual(thread._return, function_mock.return_value)

    @patch('threading.Thread.join')
    def test_join(self, join_mock):
        thread = util._ReturnThread()
        return_mock = MagicMock()
        thread._return = return_mock

        response = thread.join("timeout")

        join_mock.assert_called_once_with("timeout")
        self.assertEqual(response, return_mock)
