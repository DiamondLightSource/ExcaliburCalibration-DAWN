import unittest

from pkg_resources import require
require("mock")
from mock import patch, MagicMock, ANY

import numpy as np

from excaliburcalibrationdawn import ExcaliburDetector, ExcaliburNode
Detector_patch_path = "excaliburcalibrationdawn.excaliburdetector" \
                      ".ExcaliburDetector"
Node_patch_path = "excaliburcalibrationdawn.excalibur1M.ExcaliburNode"
DAWN_patch_path = "excaliburcalibrationdawn.excaliburdetector.ExcaliburDAWN"
util_patch_path = "excaliburcalibrationdawn.util"

mock_list = [MagicMock(), MagicMock(), MagicMock(),
             MagicMock(), MagicMock(), MagicMock()]


class InitTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburDetector("test-server", [1, 2, 3, 4, 5, 6], 1)

    def test_class_attributes_set(self):
        for node in range(6):
            self.assertIsInstance(self.e.Nodes[node], ExcaliburNode)
            self.assertEqual(node + 1, self.e.Nodes[node].fem)

        self.assertEqual(self.e.MasterNode, self.e.Nodes[0])

    def test_given_invalid_node_then_error(self):
        with self.assertRaises(ValueError):
            ExcaliburDetector("test-server", [10], 10)

    def test_given_duplicate_node_then_error(self):
        with self.assertRaises(ValueError):
            ExcaliburDetector("test-server", [1, 1], 1)

    def test_given_master_node_not_in_nodes_then_error(self):
        with self.assertRaises(ValueError):
            ExcaliburDetector("test-server", [1, 2], 3)


class SetVoltageTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburDetector("test-server", [1, 2, 3, 4, 5, 6], 1)
        self.e.Nodes = mock_list
        self.e.MasterNode = self.e.Nodes[0]

    def test_enable_lv(self):

        self.e.enable_lv()

        self.e.Nodes[0].enable_lv.assert_called_once_with()
        for node in self.e.Nodes[1:]:
            self.assertFalse(node.enable_lv.call_count)

    def test_disable_lv(self):

        self.e.disable_lv()

        self.e.Nodes[0].disable_lv.assert_called_once_with()
        for node in self.e.Nodes[1:]:
            self.assertFalse(node.disable_lv.call_count)

    def test_initialise_lv(self):

        self.e.initialise_lv()

        self.e.Nodes[0].initialise_lv.assert_called_once_with()
        for node in self.e.Nodes[1:]:
            self.assertFalse(node.initialise_lv.call_count)

    def test_enable_hv(self):

        self.e.enable_hv()

        self.e.Nodes[0].enable_hv.assert_called_once_with()
        for node in self.e.Nodes[1:]:
            self.assertFalse(node.enable_hv.call_count)

    def test_disable_hv(self):

        self.e.disable_hv()

        self.e.Nodes[0].disable_hv.assert_called_once_with()
        for node in self.e.Nodes[1:]:
            self.assertFalse(node.disable_hv.call_count)

    def test_set_hv_bias(self):

        self.e.set_hv_bias(120)

        self.e.Nodes[0].set_hv_bias.assert_called_once_with(120)
        for node in self.e.Nodes[1:]:
            self.assertFalse(node.set_hv_bias.call_count)

    def test_disable(self):

        self.e.disable()

        self.e.Nodes[0].disable.assert_called_once_with()
        for node in self.e.Nodes[1:]:
            node.assert_not_called()


class FunctionsTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburDetector("test-server", [1, 2, 3, 4, 5, 6], 1)
        self.e.Nodes = mock_list
        self.e.MasterNode = self.e.Nodes[0]

    def tearDown(self):
        self.e.MasterNode.reset_mock()

    def test_read_chip_ids(self):
        self.e.read_chip_ids()

        for node in self.e.Nodes:
            node.read_chip_ids.assert_called_once_with()

    def test_set_quiet(self):
        self.e.set_quiet(True)

        for node in self.e.Nodes:
            node.set_quiet.assert_called_once_with(True)

    def test_monitor(self):
        self.e.monitor()

        for node in self.e.Nodes:
            node.monitor.assert_called_once_with()

    def test_load_config(self):
        self.e.load_config()

        for node in self.e.Nodes:
            node.load_config.assert_called_once_with()

    def test_display_status(self):
        self.e.display_status()

        for node in self.e.Nodes:
            node.display_status.assert_called_once_with()

    def test_setup(self):
        self.e.setup()

        self.e.MasterNode.initialise_lv.assert_called_once_with()
        self.e.MasterNode.set_hv_bias.assert_called_once_with(120)

        for node in self.e.Nodes:
            node.setup.assert_called_once_with()

    def test_threshold_equalization(self):
        self.e.threshold_equalization([[0], [0], [0], [0], [0], [0]])

        for node in self.e.Nodes:
            node.threshold_equalization.assert_called_once_with([0])

    def test_threshold_equalization_given_invalid_chips(self):

        with self.assertRaises(ValueError):
            self.e.threshold_equalization([0, 1, 2, 3, 4, 5, 6, 7])

    @patch(util_patch_path + '.get_time_stamp',
           return_value="2016-10-21_16:42:50")
    @patch(DAWN_patch_path + '.plot_image')
    def test_expose(self, plot_mock, _):
        mock_array = np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                               [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]])

        for node in self.e.Nodes:
            node.expose.return_value = mock_array

        expected_array = np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                                   [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                                   [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                                   [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                                   [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                                   [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                                   [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                                   [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                                   [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                                   [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                                   [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                                   [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]])

        self.e.expose(100)

        for node in self.e.Nodes:
            node.expose.assert_called_once_with(100)

        plot_mock.assert_called_once_with(ANY, "Excalibur Detector Image - 2016-10-21_16:42:50")
        np.testing.assert_array_equal(expected_array, plot_mock.call_args[0][0])


class UtilTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburDetector("test-server", [1, 2, 3, 4, 5, 6], 1)

    @patch(util_patch_path + '.grab_slice')
    @patch(Detector_patch_path + '._generate_node_range',
           return_value=["start", "stop"])
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
