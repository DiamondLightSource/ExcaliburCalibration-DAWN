import unittest

from pkg_resources import require
require("mock")
from mock import patch, MagicMock, ANY

from excaliburcalibrationdawn.excalibur1M import Excalibur1M
from excaliburcalibrationdawn.excaliburnode import ExcaliburNode, np
E1M_patch_path = "excaliburcalibrationdawn.excalibur1M.Excalibur1M"
Node_patch_path = "excaliburcalibrationdawn.excalibur1M.ExcaliburNode"
DAWN_patch_path = "excaliburcalibrationdawn.excalibur1M.ExcaliburDAWN"
util_patch_path = "excaliburcalibrationdawn.arrayutil"


class InitTest(unittest.TestCase):

    def setUp(self):
        self.e = Excalibur1M("test-server", 1, 2)

    def test_class_attributes_set(self):
        self.assertIsInstance(self.e.nodes[0], ExcaliburNode)
        self.assertIsInstance(self.e.nodes[1], ExcaliburNode)

        self.assertEqual(1, self.e.nodes[0].fem)
        self.assertEqual(2, self.e.nodes[1].fem)


class FunctionsTest(unittest.TestCase):

    def setUp(self):
        self.e = Excalibur1M("test-server", 1, 2)
        self.node1_mock = MagicMock()
        self.node2_mock = MagicMock()
        self.e.nodes = [self.node1_mock, self.node2_mock]

    def test_read_chip_ids(self):
        self.e.read_chip_ids()

        self.node1_mock.read_chip_ids.assert_called_once_with()
        self.node2_mock.read_chip_ids.assert_called_once_with()

    def test_threshold_equalization(self):
        self.e.threshold_equalization([[0], [0]])

        self.node1_mock.threshold_equalization.assert_called_once_with([0])
        self.node2_mock.threshold_equalization.assert_called_once_with([0])

    @patch(E1M_patch_path + '._grab_node_slice')
    def test_optimize_dac_disc(self, grab_mock):
        roi_mock = MagicMock()
        self.e.optimize_dac_disc([[0], [0]], roi_mock)

        self.assertEqual((roi_mock, 0), grab_mock.call_args_list[0][0])
        self.assertEqual((roi_mock, 1), grab_mock.call_args_list[1][0])
        self.assertEqual(([0], grab_mock.return_value),
                         self.node1_mock.optimize_disc_l.call_args_list[0][0])
        self.assertEqual(([0], grab_mock.return_value),
                         self.node1_mock.optimize_disc_h.call_args_list[0][0])
        self.assertEqual(([0], grab_mock.return_value),
                         self.node2_mock.optimize_disc_l.call_args_list[0][0])
        self.assertEqual(([0], grab_mock.return_value),
                         self.node2_mock.optimize_disc_h.call_args_list[0][0])

    @patch(DAWN_patch_path + '.plot_image')
    def test_expose(self, plot_mock):
        mock_array = np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                               [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]])
        self.node1_mock.expose.return_value = mock_array
        self.node2_mock.expose.return_value = mock_array
        expected_array = np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                                   [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                                   [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                                   [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]])

        self.e.expose(100)

        self.node1_mock.expose.assert_called_once_with()
        self.node2_mock.expose.assert_called_once_with()
        plot_mock.assert_called_once_with(ANY, "Excalibur1M Image")
        np.testing.assert_array_equal(expected_array, plot_mock.call_args[0][0])


class UtilTest(unittest.TestCase):

    def setUp(self):
        self.e = Excalibur1M("test-server", 1, 2)

    @patch(util_patch_path + '.grab_slice')
    @patch(E1M_patch_path + '._generate_node_range', return_value=["start", "stop"])
    def test_grab_node_slice(self, generate_mock, grab_mock):
        mock_array = MagicMock()
        self.e._grab_node_slice(mock_array, 1)

        generate_mock.assert_called_once_with(1)
        grab_mock.assert_called_once_with(mock_array, "start", "stop")

    def test_generate_node_range(self):
        expected_start = [256, 0]
        expected_stop = [511, 2047]

        start, stop = self.e._generate_node_range(1)

        self.assertEqual(expected_start, start)
        self.assertEqual(expected_stop, stop)
