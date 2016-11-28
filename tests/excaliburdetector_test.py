import unittest

from pkg_resources import require
require("mock")
from mock import patch, MagicMock, ANY, call

import numpy as np

from excaliburcalibrationdawn import ExcaliburDetector, ExcaliburNode, Range
Detector_patch_path = "excaliburcalibrationdawn.excaliburdetector" \
                      ".ExcaliburDetector"
Node_patch_path = "excaliburcalibrationdawn.excaliburdetector.ExcaliburNode"
DAWN_patch_path = "excaliburcalibrationdawn.excaliburdetector.ExcaliburDAWN"
util_patch_path = "excaliburcalibrationdawn.util"

mock_list = [MagicMock(), MagicMock(), MagicMock(),
             MagicMock(), MagicMock(), MagicMock()]

detector = MagicMock(name="test-detector", nodes=[1, 2, 3, 4, 5, 6],
                     master_node=1, servers=["test-server{}".format(i + 1)
                                             for i in range(6)],
                     ip_addresses=["192.168.0.10{}".format(i + 1)
                                   for i in range(6)])
mock_config = MagicMock(detector=detector)


class InitTest(unittest.TestCase):

    @patch('logging.getLogger')
    def test_attributes_set(self, get_mock):

        e = ExcaliburDetector(mock_config)

        for node in range(6):
            self.assertIsInstance(e.Nodes[node], ExcaliburNode)
            self.assertEqual(node + 1, e.Nodes[node].id)

        self.assertEqual(e.MasterNode, e.Nodes[0])
        self.assertEqual(e.calib_root, e.MasterNode.calib_root)
        self.assertIn(call("ExcaliburDetector"), get_mock.mock_calls)
        self.assertEqual(get_mock.return_value, e.logger)

    def test_given_invalid_node_then_error(self):
        detector_ = MagicMock(name="test-detector", nodes=[10],
                              master_node=10, servers=["test-server10"],
                              ip_addresses=["192.168.0.101"])
        mock_config_ = MagicMock(detector=detector_)

        with self.assertRaises(ValueError):
            ExcaliburDetector(mock_config_)

    def test_given_duplicate_node_then_error(self):
        detector_ = MagicMock(name="test-detector", nodes=[1, 1],
                              master_node=1, servers=["test-server1",
                                                      "test-server2"],
                              ip_addresses=["192.168.0.101",
                                            "192.168.0.102"])
        mock_config_ = MagicMock(detector=detector_)

        with self.assertRaises(ValueError):
            ExcaliburDetector(mock_config_)

    def test_given_master_node_not_in_nodes_then_error(self):
        detector_ = MagicMock(name="test-detector", nodes=[1, 2],
                              master_node=3, servers=["test-server1",
                                                      "test-server2"],
                              ip_addresses=["192.168.0.101",
                                            "192.168.0.102"])
        mock_config_ = MagicMock(detector=detector_)

        with self.assertRaises(ValueError):
            ExcaliburDetector(mock_config_)

    def test_given_mismatched_lengths_then_error(self):
        detector_ = MagicMock(name="test-detector", nodes=[1, 2, 3],
                              master_node=3, servers=["test-server1",
                                                      "test-server2"],
                              ip_addresses=["192.168.0.101",
                                            "192.168.0.102",
                                            "192.168.0.103"])
        mock_config_ = MagicMock(detector=detector_)

        with self.assertRaises(ValueError):
            ExcaliburDetector(mock_config_)

        detector_ = MagicMock(name="test-detector", nodes=[1, 2, 3],
                              master_node=3, servers=["test-server1",
                                                      "test-server2",
                                                      "test-server3"],
                              ip_addresses=["192.168.0.101",
                                            "192.168.0.102"])
        mock_config_ = MagicMock(detector=detector_)

        with self.assertRaises(ValueError):
            ExcaliburDetector(mock_config_)


class SetVoltageTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburDetector(mock_config)
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
        self.e = ExcaliburDetector(mock_config)
        self.e.Nodes = mock_list
        self.e.MasterNode = self.e.Nodes[0]

    def tearDown(self):
        for node in self.e.Nodes:
            node.reset_mock()

    @patch('os.path.isdir', return_value=True)
    def test_setup_new_detector(self, _):
        with self.assertRaises(IOError):
            self.e.setup_new_detector()

    @patch('os.path.isdir', return_value=False)
    def test_setup_new_detector_exists_then_error(self, _):
        self.e.setup_new_detector()

        for node in self.e.Nodes:
            node.create_calib.assert_called_once_with()

    def test_read_chip_ids(self):
        self.e.read_chip_ids()

        for node in self.e.Nodes:
            node.read_chip_ids.assert_called_once_with()

    def test_set_dac(self):
        self.e.Nodes[0].id = 1

        self.e.set_dac(1, "Threshold0", 5)

        self.e.Nodes[0].set_dac.assert_called_once_with(range(8),
                                                        "Threshold0", 5)

    def test_read_dac(self):
        self.e.Nodes[0].id = 1

        self.e.read_dac(1, "Threshold0")

        self.e.Nodes[0].read_dac.assert_called_once_with("Threshold0")

    def test_set_quiet(self):
        self.e.set_quiet(True)

        for node in self.e.Nodes:
            node.set_quiet.assert_called_once_with(True)

    def test_monitor(self):
        self.e.monitor()

        for node in self.e.Nodes:
            node.monitor.assert_called_once_with()

    @patch(util_patch_path + '.wait_for_threads')
    @patch(util_patch_path + '.spawn_thread')
    def test_load_config(self, spawn_mock, wait_mock):
        self.e.load_config()

        spawn_mock.assert_has_calls([call(node.load_config)
                                     for node in self.e.Nodes])
        wait_mock.assert_called_once_with([spawn_mock.return_value] * 6)

    @patch(util_patch_path + '.wait_for_threads')
    @patch(util_patch_path + '.spawn_thread')
    def test_unequalise_pixels(self, spawn_mock, wait_mock):
        self.e.Nodes[0].id = 1

        self.e.unequalise_pixels(1, [0])

        spawn_mock.assert_called_once_with(
            self.e.Nodes[0].unequalize_pixels, [0])
        wait_mock.assert_called_once_with([spawn_mock.return_value])

    @patch(util_patch_path + '.wait_for_threads')
    @patch(util_patch_path + '.spawn_thread')
    def test_unmask_pixels(self, spawn_mock, wait_mock):
        self.e.Nodes[0].id = 1

        self.e.unmask_pixels(1, [0])

        spawn_mock.assert_called_once_with(
            self.e.Nodes[0].unmask_pixels, [0])
        wait_mock.assert_called_once_with([spawn_mock.return_value])

    def test_display_status(self):
        self.e.display_status()

        for node in self.e.Nodes:
            node.display_status.assert_called_once_with()

    @patch(util_patch_path + '.wait_for_threads')
    @patch(util_patch_path + '.spawn_thread')
    def test_setup(self, spawn_mock, wait_mock):
        self.e.setup()

        self.e.MasterNode.initialise_lv.assert_called_once_with()
        self.e.MasterNode.set_hv_bias.assert_called_once_with(120)

        spawn_mock.assert_has_calls([call(node.setup)
                                     for node in self.e.Nodes])
        wait_mock.assert_called_once_with([spawn_mock.return_value] * 6)

    @patch(util_patch_path + '.wait_for_threads')
    @patch(util_patch_path + '.spawn_thread')
    def test_set_gnd_fbk_cas(self, spawn_mock, wait_mock):
        self.e.Nodes[0].id = 1

        self.e.set_gnd_fbk_cas(node_id=1, chips=[0])

        spawn_mock.assert_called_once_with(self.e.Nodes[0].set_gnd_fbk_cas,
                                           [0])
        wait_mock.assert_called_once_with([spawn_mock.return_value])

    @patch(util_patch_path + '.wait_for_threads')
    @patch(util_patch_path + '.spawn_thread')
    def test_set_gnd_fbk_cas_default(self, spawn_mock, wait_mock):
        self.e.set_gnd_fbk_cas()

        spawn_mock.assert_has_calls([call(node.set_gnd_fbk_cas, [0, 1, 2, 3,
                                                                 4, 5, 6, 7])
                                     for node in self.e.Nodes])
        wait_mock.assert_called_once_with([spawn_mock.return_value] * 6)

    @patch(util_patch_path + '.wait_for_threads')
    @patch(util_patch_path + '.spawn_thread')
    def test_threshold_equalization(self, spawn_mock, wait_mock):
        self.e.Nodes[0].id = 1

        self.e.set_gnd_fbk_cas(node_id=1, chips=[0])

        spawn_mock.assert_called_once_with(self.e.Nodes[0].set_gnd_fbk_cas,
                                           [0])
        wait_mock.assert_called_once_with([spawn_mock.return_value])

    @patch(util_patch_path + '.wait_for_threads')
    @patch(util_patch_path + '.spawn_thread')
    def test_threshold_equalization_default(self, spawn_mock, wait_mock):
        self.e.threshold_equalization()

        spawn_mock.assert_has_calls(
            [call(self.e._try_node_threshold_equalization, node,
                  [0, 1, 2, 3, 4, 5, 6, 7])
             for node in self.e.Nodes])
        wait_mock.assert_called_once_with([spawn_mock.return_value] * 6)

    def test_try_node_threshold_equalization(self):
        node_mock = MagicMock()
        self.e._try_node_threshold_equalization(node_mock,
                                                [0, 1, 2, 3, 4, 5, 6, 7])

        node_mock.threshold_equalization.assert_called_once_with(
            [0, 1, 2, 3, 4, 5, 6, 7])

    def test_try_node_threshold_equalization_error_raised_then_log(self):
        node_mock = MagicMock()
        error = IOError("Bad things happened.")
        node_mock.threshold_equalization.side_effect = error

        with self.assertRaises(IOError):
            self.e._try_node_threshold_equalization(node_mock,
                                                    [0, 1, 2, 3, 4, 5, 6, 7])

        self.assertEqual([("Threshold equalization failed for node %s.\n"
                           "Error:\n%s", [node_mock.id, error])],
                         self.e.errors)

    def test_threshold_equalization_given_invalid_chips(self):

        with self.assertRaises(ValueError):
            self.e.threshold_equalization([0, 1, 2, 3, 4, 5, 6, 7])

    @patch(util_patch_path + '.tag_plot_name')
    @patch(Detector_patch_path + '._combine_images')
    @patch(DAWN_patch_path + '.plot_image')
    @patch(util_patch_path + '.spawn_thread')
    def test_acquire_tp_image(self, spawn_mock, plot_mock, combine_mock, tag_mock):

        mock_image = MagicMock()

        for node in self.e.Nodes:
            node.acquire_tp_image.return_value = mock_image

        self.e.acquire_tp_image("triangles.mask")

        spawn_mock.assert_has_calls([call(node.acquire_tp_image,
                                          "triangles.mask")
                                     for node in self.e.Nodes])

        combine_mock.assert_called_once_with(
            [spawn_mock.return_value.join.return_value] * 6)
        plot_mock.assert_called_once_with(combine_mock.return_value,
                                          tag_mock.return_value)

    @patch(util_patch_path + '.tag_plot_name')
    @patch(Detector_patch_path + '._combine_images')
    @patch(DAWN_patch_path + '.plot_image')
    @patch(util_patch_path + '.spawn_thread')
    def test_expose(self, spawn_mock, plot_mock, combine_mock, tag_mock):

        mock_image = MagicMock()

        for node in self.e.Nodes:
            node.expose.return_value = mock_image

        self.e.expose(100)

        spawn_mock.assert_has_calls([call(node.expose, 100)
                                     for node in self.e.Nodes])

        combine_mock.assert_called_once_with(
            [spawn_mock.return_value.join.return_value] * 6)
        plot_mock.assert_called_once_with(combine_mock.return_value,
                                          tag_mock.return_value)

    def test_scan_dac(self):

        self.e.Nodes[3].id = 6

        self.e.scan_dac(6, [0], "Threshold0", Range(0, 10, 1))

        self.e.Nodes[3].scan_dac.assert_called_once_with([0], "Threshold0",
                                                         Range(0, 10, 1))
        other_nodes = list(self.e.Nodes)
        other_nodes.remove(self.e.Nodes[3])
        for node in other_nodes:
            node.scan_dac.assert_not_called()

    def test_combine_images(self):
        images = [np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]])] * 6

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

        detector_image = self.e._combine_images(images)

        np.testing.assert_array_equal(expected_array, detector_image)

    @patch('shutil.copytree')
    def test_rotate_configs(self, copy_mock):

        self.e.rotate_configs()

        copy_mock.assert_called_once_with(self.e.calib_root,
                                          self.e.calib_root + "_epics")
        for node in self.e.Nodes[1:2:6]:
            node.rotate_config.assert_called_once_with()
        for node in self.e.Nodes[0:2:6]:
            node.rotate_config.assert_not_called()


class UtilTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburDetector(mock_config)

    def test_validate_return(self):
        self.e.Nodes[0].id = 1

        nodes, chips = self.e._validate(1, [0, 1, 2, 3, 4, 5, 6, 7])

        self.assertEqual([self.e.Nodes[0]], nodes)
        self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7], chips)

    def test_validate_chips_invalid_then_error(self):

        with self.assertRaises(ValueError):
            self.e._validate(None, [8])

    def test_validate_none_then_generate(self):

        nodes, chips = self.e._validate(None, None)

        self.assertEqual(self.e.Nodes, nodes)
        self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7], chips)

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

    def test_find_node(self):

        node = self.e._find_node(1)

        self.assertEqual(self.e.Nodes[0], node)

    def test_find_node_not_found_then_error(self):

        with self.assertRaises(ValueError):
            self.e._find_node(7)
