import unittest

from pkg_resources import require
require("mock")
from mock import patch, MagicMock, ANY, call, PropertyMock

import numpy as np

from excaliburcalibrationdawn import ExcaliburNode, Range
Node_patch_path = "excaliburcalibrationdawn.excaliburnode.ExcaliburNode"
ETAI_patch_path = "excaliburcalibrationdawn.excaliburnode.ExcaliburTestAppInterface"
DAWN_patch_path = "excaliburcalibrationdawn.excaliburnode.ExcaliburDAWN"
util_patch_path = "excaliburcalibrationdawn.util"

detector = MagicMock(name="testdetector", nodes=[1],
                     master_node=1, servers=["test-server"],
                     ip_addresses=["192.168.0.1"])
mock_config = MagicMock(detector=detector)
# Force MagicMock to have name attribute
type(detector).name = PropertyMock(return_value="testdetector")


class InitTest(unittest.TestCase):

    @patch(ETAI_patch_path)
    @patch(DAWN_patch_path)
    @patch('logging.getLogger')
    def test_attributes_set(self, get_mock, DAWN_mock, ETAI_mock):
        node = 3
        e = ExcaliburNode(node, mock_config)

        ETAI_mock.assert_called_once_with(3, "192.168.0.104", 6969, None)
        DAWN_mock.assert_called_once_with()
        self.assertEqual(ETAI_mock.return_value, e.app)
        self.assertEqual(DAWN_mock.return_value, e.dawn)

        self.assertEqual(e.id, node)
        self.assertEqual(e.ip_address, "192.168.0.104")

        self.assertEqual(e.dac_target, 10)
        self.assertEqual(e.num_sigma, 3.2)
        self.assertEqual(e.allowed_delta, 4)
        self.assertEqual(e.calib_root,
                         "/dls/detectors/support/silicon_pixels/excaliburRX/"
                         "3M-RX001/testdetector/calib")
        self.assertEqual(e.config_dir,
                         "/dls/detectors/support/silicon_pixels/excaliburRX/"
                         "TestApplication_15012015/config")
        self.assertEqual(e.settings, {'mode': 'spm',
                                      'gain': 'slgm',
                                      'bitdepth': 12,
                                      'readmode': 'sequential',
                                      'counter': 0,
                                      'disccsmspm': 'discL',
                                      'equalization': 0,
                                      'trigmode': 0,
                                      'exposure': 100,
                                      'frames': 1})
        self.assertEqual(e.dac_number, {'Threshold0': 1,
                                        'Threshold1': 2,
                                        'Threshold2': 3,
                                        'Threshold3': 4,
                                        'Threshold4': 5,
                                        'Threshold5': 6,
                                        'Threshold6': 7,
                                        'Threshold7': 8,
                                        'Preamp': 9,
                                        'Ikrum': 10,
                                        'Shaper': 11,
                                        'Disc': 12,
                                        'DiscLS': 13,
                                        'ShaperTest': 14,
                                        'DACDiscL': 15,
                                        'DACTest': 16,
                                        'DACDiscH': 17,
                                        'Delay': 18,
                                        'TPBuffIn': 19,
                                        'TPBuffOut': 20,
                                        'RPZ': 21,
                                        'GND': 22,
                                        'TPREF': 23,
                                        'FBK': 24,
                                        'Cas': 25,
                                        'TPREFA': 26,
                                        'TPREFB': 27})
        self.assertEqual(e.chip_range, [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(e.plot_name, '')

        get_mock.assert_called_once_with("Node3")

    def test_server_used(self):
        e = ExcaliburNode(1, mock_config, "test-server")
        self.assertEqual("test-server", e.server_name)
        self.assertEqual("test-server.diamond.ac.uk", e.app.server_path)

    def test_given_node_invalid_node_raises(self):
        with self.assertRaises(ValueError):
            ExcaliburNode(0, mock_config)
        with self.assertRaises(ValueError):
            ExcaliburNode(7, mock_config)


class SetVoltageTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode(1, mock_config)
        self.app_mock = MagicMock()
        self.e.app = self.app_mock

    def test_enable_lv(self):

        self.e.enable_lv()

        self.app_mock.set_lv_state.assert_called_once_with(1)

    def test_disable_lv(self):

        self.e.disable_lv()

        self.app_mock.set_lv_state.assert_called_once_with(0)

    @patch(Node_patch_path + '.enable_lv')
    @patch(Node_patch_path + '.disable_lv')
    def test_initialise_lv(self, disable_mock, enable_mock):

        self.e.initialise_lv()

        self.assertEqual(2, enable_mock.call_count)
        self.assertEqual(1, disable_mock.call_count)

    def test_enable_hv(self):

        self.e.enable_hv()

        self.app_mock.set_hv_state.assert_called_once_with(1)

    def test_disable_hv(self):

        self.e.disable_hv()

        self.app_mock.set_hv_state.assert_called_once_with(0)

    def test_set_hv_bias(self):

        self.e.set_hv_bias(120)

        self.app_mock.set_hv_bias.assert_called_once_with(120)

    @patch(Node_patch_path + '.disable_lv')
    @patch(Node_patch_path + '.disable_hv')
    @patch(Node_patch_path + '.set_hv_bias')
    def test_disable(self, set_mock, hv_mock, lv_mock):

        self.e.disable()

        set_mock.assert_called_once_with(0)
        hv_mock.assert_called_once_with()
        lv_mock.assert_called_once_with()


class SimpleFunctionsTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode(1, mock_config)
        self.app_mock = MagicMock()
        self.e.app = self.app_mock

    def test_set_quiet(self):
        self.e.set_quiet(True)
        self.assertTrue(self.app_mock.quiet)
        self.e.set_quiet(False)
        self.assertFalse(self.app_mock.quiet)

    def test_set_quiet_invalid(self):
        with self.assertRaises(ValueError):
            self.e.set_quiet(1)
        with self.assertRaises(ValueError):
            self.e.set_quiet(None)
        with self.assertRaises(ValueError):
            self.e.set_quiet("True")

    @patch(Node_patch_path + '._create_calib_structure')
    @patch(Node_patch_path + '._save_chip_ids')
    @patch('shutil.copy')
    @patch(Node_patch_path + '.save_discbits')
    @patch(Node_patch_path + '.copy_slgm_into_other_gain_modes')
    def test_create_calib(self, copy_gain, save_mock, copy_mock,
                                log_mock, create_mock):
        expected_source = "/dls/detectors/support/silicon_pixels/excaliburRX" \
                          "/TestApplication_15012015/config/Default_SPM.dacs"
        expected_target = "/dls/detectors/support/silicon_pixels/excaliburRX" \
                          "/3M-RX001/testdetector/calib/fem1/spm/slgm/dacs"
        self.e.create_calib()

        create_mock.assert_called_once_with()
        log_mock.assert_called_once_with()
        copy_mock.assert_called_once_with(expected_source, expected_target)
        save_mock.assert_called_once_with(range(8), ANY, "discLbits")
        np.testing.assert_array_equal(np.zeros(shape=[256, 8*256]),
                                      save_mock.call_args[0][1])
        copy_gain.assert_called_once_with()

    @patch('os.makedirs')
    def test_create_calib_structure(self, makedirs_mock):
        expected_path = "/dls/detectors/support/silicon_pixels/excaliburRX" \
                        "/3M-RX001/testdetector/calib/fem1/spm/"
        self.e._create_calib_structure()

        expected_calls = [call(expected_path + gain)
                               for gain in ["shgm", "hgm", "lgm", "slgm"]]
        makedirs_mock.assert_has_calls(expected_calls)

    @patch(Node_patch_path + '.copy_slgm_into_other_gain_modes')
    @patch(Node_patch_path + '.load_config')
    @patch(Node_patch_path + '.check_chip_ids')
    def test_setup(self, check_mock, load_config_mock, copy_mock):

        self.e.setup()

        check_mock.assert_called_once_with()
        self.app_mock.load_dacs.assert_called_once_with(
            range(8), '/dls/detectors/support/silicon_pixels/excaliburRX/'
                      'TestApplication_15012015/config/Default_SPM.dacs')
        load_config_mock.assert_called_once_with(range(8))
        copy_mock.assert_called_once_with()

    @patch(util_patch_path + '.get_time_stamp', return_value="20161020~154548")
    def test_generate_file_name(self, _):
        file_name = self.e.generate_file_name("TestImage")

        self.assertEqual("20161020~154548_TestImage_1.hdf5", file_name)


@patch(DAWN_patch_path + '.plot_image')
@patch(util_patch_path + '.tag_plot_name')
@patch(Node_patch_path + '.generate_file_name')
@patch(DAWN_patch_path + '.load_image_data')
@patch(util_patch_path + '.wait_for_file')
@patch(ETAI_patch_path + '.grab_remote_file')
@patch(ETAI_patch_path + '.configure_test_pulse')
@patch(ETAI_patch_path + '.acquire_tp_image')
class CaptureTPImageTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode(1, mock_config)
        self.e.chip_range = [0]
        self.e.remote_node = True

    @patch('os.path.isfile', return_value=True)
    def test_file_exist_then_capture_image(self, isfile_mock, acquire_mock,
                                           configure_mock, grab_mock,
                                           wait_mock, load_mock, gen_file_mock,
                                           gen_plot_mock, plot_mock):
        expected_mask = "/dls/detectors/support/silicon_pixels/excaliburRX/" \
                        "TestApplication_15012015/config/triangles.mask"
        expected_dacs = "/dls/detectors/support/silicon_pixels/excaliburRX/" \
                        "3M-RX001/testdetector/calib/fem1/spm/slgm/dacs"
        expected_config = dict(discL=self.e.discL_bits[0],
                               discH=self.e.discH_bits[0],
                               pixel_mask=self.e.pixel_mask[0])

        self.e.acquire_tp_image("triangles.mask", 100, 1000)

        isfile_mock.assert_called_once_with(expected_mask)
        configure_mock.assert_called_once_with(0, expected_mask,
                                               expected_dacs, expected_config)
        acquire_mock.assert_called_once_with(
            [0], 100, 1000, hdf_file=gen_file_mock.return_value)
        grab_mock.assert_called_once_with(gen_file_mock.return_value)
        wait_mock.assert_called_once_with(gen_file_mock.return_value, 10)
        load_mock.assert_called_once_with(gen_file_mock.return_value)
        plot_mock.assert_called_once_with(load_mock.return_value,
                                          gen_plot_mock.return_value)

    @patch('os.path.isfile', return_value=False)
    def test_file_doesnt_exist_then_capture_image(self, isfile_mock,
                                                  acquire_mock, configure_mock,
                                                  grab_mock, wait_mock,
                                                  load_mock, gen_file_mock,
                                                  gen_plot_mock, plot_mock):

        with self.assertRaises(IOError):
            self.e.acquire_tp_image("none.mask")


@patch(Node_patch_path + '.backup_calib_dir')
@patch(Node_patch_path + '.check_chip_ids')
@patch(Node_patch_path + '.set_dacs')
@patch(Node_patch_path + '.set_gnd_fbk_cas')
@patch(Node_patch_path + '.calibrate_disc_l')
class ThresholdEqualizationTest(unittest.TestCase):

    def test_correct_calls_made(self, calibrate_mock, set_gnd_mock,
                                set_dacs_mock, check_mock, backup_mock):
        e = ExcaliburNode(1, mock_config)
        chips = [1, 4, 6, 7]

        e.threshold_equalization(chips)

        self.assertEqual('slgm', e.settings['gain'])
        self.assertEqual('spm', e.settings['mode'])

        check_mock.assert_called_once_with()
        backup_mock.assert_called_once_with()
        set_dacs_mock.assert_called_once_with(chips)
        set_gnd_mock.assert_called_once_with(chips)
        calibrate_mock.assert_called_once_with(chips)


@patch(Node_patch_path + '.backup_calib_dir')
@patch(Node_patch_path + '.save_kev2dac_calib')
class ThresholdCalibrationTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode(1, mock_config)

    def test_threshold_calibration_shgm(self, save_mock, check_mock):
        expected_slope = np.array([8.81355932203]*8)
        expected_offset = np.array([10.0]*8)

        self.e.settings['gain'] = 'shgm'
        self.e.threshold_calibration(0)

        check_mock.assert_called_once_with()
        self.assertEqual(save_mock.call_args[0][0], 0)
        np.testing.assert_array_almost_equal(expected_slope, save_mock.call_args[0][1][0])
        self.assertTrue((save_mock.call_args[0][2] == expected_offset).all())

    def test_threshold_calibration_hgm(self, save_mock, check_mock):
        expected_slope = np.array([6.61016949]*8)
        expected_offset = np.array([10.0]*8)

        self.e.settings['gain'] = 'hgm'
        self.e.threshold_calibration(0)

        check_mock.assert_called_once_with()
        self.assertEqual(save_mock.call_args[0][0], 0)
        np.testing.assert_array_almost_equal(expected_slope, save_mock.call_args[0][1][0])
        self.assertTrue((save_mock.call_args[0][2] == expected_offset).all())

    def test_threshold_calibration_lgm(self, save_mock, check_mock):
        expected_slope = np.array([4.40677966]*8)
        expected_offset = np.array([10.0]*8)

        self.e.settings['gain'] = 'lgm'
        self.e.threshold_calibration(0)

        check_mock.assert_called_once_with()
        self.assertEqual(save_mock.call_args[0][0], 0)
        np.testing.assert_array_almost_equal(expected_slope, save_mock.call_args[0][1][0])
        self.assertTrue((save_mock.call_args[0][2] == expected_offset).all())

    def test_threshold_calibration_slgm(self, save_mock, check_mock):
        expected_slope = np.array([2.20338983]*8)
        expected_offset = np.array([10.0]*8)

        self.e.settings['gain'] = 'slgm'
        self.e.threshold_calibration(0)

        check_mock.assert_called_once_with()
        self.assertEqual(save_mock.call_args[0][0], 0)
        np.testing.assert_array_almost_equal(expected_slope, save_mock.call_args[0][1][0])
        self.assertTrue((save_mock.call_args[0][2] == expected_offset).all())

    @patch(Node_patch_path + '.threshold_calibration')
    def test_threshold_calibration_all_gains(self, calibration_mock, _, _2):
        self.e.threshold_calibration_all_gains()

        self.assertEqual(4, calibration_mock.call_count)

    def test_one_energy_thresh_calib(self, save_mock, _):
        expected_slope = np.array([8.6666666]*8)
        expected_offset = np.array([10.0]*8)
        self.e.config.E1_DAC = dict(slgm=np.array([[62]*8]).astype(float))

        self.e.settings['gain'] = 'slgm'
        self.e.one_energy_thresh_calib()

        self.assertEqual(save_mock.call_args[0][0], 0)
        np.testing.assert_array_almost_equal(expected_slope, save_mock.call_args[0][1][0])
        self.assertTrue((save_mock.call_args[0][2] == expected_offset).all())

    @patch(util_patch_path + '.tag_plot_name')
    @patch(DAWN_patch_path + '.plot_linear_fit', return_value=(10, 2))
    def test_multiple_energy_thresh_calib(self, plot_mock, tag_mock, save_mock,
                                          _):
        self.e.config.E1_DAC = dict(slgm=np.array([[62]]))
        self.e.config.E2_DAC = dict(slgm=np.array([[62]]))
        self.e.config.E3_DAC = dict(slgm=np.array([[62]]))
        expected_slope = np.array([2.0] + [0.0]*7)
        expected_offset = np.array([10.0] + [0.0]*7)
        expected_array_1 = np.array([6, 12, 24])
        expected_array_2 = np.array([62, 62, 62])

        self.e.settings['gain'] = 'slgm'
        self.e.multiple_energy_thresh_calib([0])

        tag_mock.assert_called_once_with("DAC vs Energy", "Node 1")
        plot_mock.assert_called_once_with(ANY, ANY, [0, 1], "DAC Value",
                                          "Energy", tag_mock.return_value,
                                          "Chip 0")
        np.testing.assert_array_equal(expected_array_1, plot_mock.call_args[0][0])
        np.testing.assert_array_equal(expected_array_2, plot_mock.call_args[0][1])

        np.testing.assert_array_almost_equal(expected_slope, save_mock.call_args[0][1])
        np.testing.assert_array_almost_equal(expected_offset, save_mock.call_args[0][2])


class SaveKev2DacCalibTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode(1, mock_config)
        self.threshold = 0
        self.gain = [1.1, 0.7, 1.1, 1.3, 1.0, 0.9, 1.2, 0.9]
        self.offset = [0.2, -0.7, 0.1, 0.0, 0.3, -0.1, 0.2, 0.5]
        self.expected_array = np.array([self.gain, self.offset])
        self.expected_filename = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/testdetector/calib/fem1/spm/slgm/threshold0'

    @patch('numpy.savetxt')
    @patch('os.chmod')
    def test_given_file_then_save(self, chmod_mock, save_mock):

        self.e.save_kev2dac_calib(self.threshold, self.gain, self.offset)

        self.assertEqual(self.expected_filename, save_mock.call_args[0][0])
        np.testing.assert_array_equal(self.expected_array, save_mock.call_args[0][1])
        self.assertEqual(dict(fmt='%.2f'), save_mock.call_args[1])
        chmod_mock.assert_called_once_with(self.expected_filename, 0777)


@patch(Node_patch_path + '.load_config')
class FindXrayEnergyDacTest(unittest.TestCase):

    mock_scan_data = np.random.randint(250, size=(3, 256, 8*256))
    mock_scan_range = Range(110, 30, 2)

    @patch(DAWN_patch_path + '.fit_dac_scan')
    @patch(Node_patch_path + '.display_dac_scan',
           return_value=[MagicMock(), MagicMock()])
    @patch(Node_patch_path + '.scan_dac',
           return_value=mock_scan_data.copy())
    def test_correct_calls_made(self, scan_mock, display_mock, fit_mock,
                                load_mock):
        e = ExcaliburNode(1, mock_config)
        e.settings['gain'] = "shgm"
        chips = [0]
        expected_array = self.mock_scan_data
        expected_array[expected_array > 200] = 0

        values = e.find_xray_energy_dac(chips)

        load_mock.assert_called_once_with(chips)
        scan_mock.assert_called_once_with(chips, "Threshold0", (110, 30, 2))

        display_mock.assert_called_once_with(chips, ANY, self.mock_scan_range)
        np.testing.assert_array_equal(expected_array,
                                      display_mock.call_args[0][1])

        fit_mock.assert_called_once_with(
            [display_mock.return_value[0].__getitem__.return_value],
            display_mock.return_value[1])
        self.assertEqual(tuple(display_mock.return_value), values)


@patch('time.sleep')
@patch('numpy.genfromtxt')
@patch(Node_patch_path + '.set_dac')
@patch(Node_patch_path + '.expose')
class SetThreshEnergyTest(unittest.TestCase):

    def test_correct_calls_made(self, expose_mock, set_dac_mock, gen_mock,
                                sleep_mock):
        e = ExcaliburNode(1, mock_config)
        expected_filepath = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/testdetector/calib/fem1/spm/slgm/threshold0'

        e.set_thresh_energy(0, 7.0)

        gen_mock.assert_called_once_with(expected_filepath)
        self.assertEqual(8, set_dac_mock.call_count)
        self.assertEqual(2, expose_mock.call_count)


class SetDacsTest(unittest.TestCase):

    @patch(ETAI_patch_path + '.load_dacs')
    @patch(Node_patch_path + '._write_dac')
    def test_correct_calls_made(self, write_mock, load_mock):
        mock_config.DACS = dict(Threshold1=0, Threshold2=0)
        e = ExcaliburNode(1, mock_config)
        chips = [0]

        e.set_dacs(chips)

        write_mock.assert_has_calls([call(0, "Threshold1", 0),
                                     call(0, "Threshold2", 0)])
        load_mock.assert_called_once_with(chips, e.dacs_file)


class TestAppCallsTest(unittest.TestCase):

    file_mock = MagicMock()

    def setUp(self):
        self.e = ExcaliburNode(1, mock_config)
        self.e.file_index = 5

    @patch(ETAI_patch_path + '.read_chip_ids')
    def test_read_chip_ids(self, read_mock):

        self.e.read_chip_ids()

        read_mock.assert_called_once_with()

    @patch(ETAI_patch_path + '.read_chip_ids')
    @patch('__builtin__.open', return_value=file_mock)
    def test_log_chip_ids(self, open_mock, read_mock):

        self.e._save_chip_ids()

        open_mock.assert_called_once_with(
            "/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001"
            "/testdetector/calib/fem1/efuseIDs", "w")
        read_mock.assert_called_once_with(
            stdout=self.file_mock.__enter__.return_value)

    @patch(util_patch_path + '.files_match', return_value=True)
    @patch(ETAI_patch_path + '.read_chip_ids')
    @patch('__builtin__.open', return_value=file_mock)
    def test_check_chip_ids(self, open_mock, read_mock, match_mock):
        expected_temp = "/tmp/temp_id.txt"
        expected_target = "/dls/detectors/support/silicon_pixels/excaliburRX" \
                          "/3M-RX001/testdetector/calib/fem1/efuseIDs"

        self.e.check_chip_ids()

        open_mock.assert_called_once_with(expected_temp, "w")
        read_mock.assert_called_once_with(
            stdout=self.file_mock.__enter__.return_value)
        match_mock.assert_called_once_with(expected_temp, expected_target)

    @patch(util_patch_path + '.files_match', return_value=False)
    @patch(ETAI_patch_path + '.read_chip_ids')
    @patch('__builtin__.open', return_value=file_mock)
    def test_check_chip_ids_unmatched(self, open_mock, read_mock, match_mock):

        with self.assertRaises(IOError):
            self.e.check_chip_ids()

    @patch(ETAI_patch_path + '.read_slow_control_parameters')
    def test_monitor(self, read_mock):

        self.e.monitor()

        read_mock.assert_called_once_with()

    @patch(DAWN_patch_path + '.load_image_data')
    @patch(ETAI_patch_path + '.acquire')
    @patch(util_patch_path + '.wait_for_file')
    @patch(Node_patch_path + '.generate_file_name')
    @patch(ETAI_patch_path + '.grab_remote_file')
    def test_acquire(self, grab_mock, gen_mock, wait_mock, acquire_mock,
                     load_mock):

        self.e._acquire(10, 100, burst=True)

        gen_mock.assert_called_once_with("Image")
        acquire_mock.assert_called_once_with(
            [0, 1, 2, 3, 4, 5, 6, 7], 10, 100, trig_mode=0, gain_mode='slgm',
            burst=True, pixel_mode='spm', counter=0, equalization=0,
            disc_mode='discL', hdf_file=gen_mock.return_value, path='/tmp',
            depth=12, read_mode='sequential')
        grab_mock.assert_not_called()
        wait_mock.assert_called_once_with(gen_mock.return_value, 10)
        load_mock.assert_called_once_with(gen_mock.return_value)

    @patch(DAWN_patch_path + '.load_image_data')
    @patch(ETAI_patch_path + '.acquire')
    @patch(util_patch_path + '.wait_for_file')
    @patch(Node_patch_path + '.generate_file_name')
    @patch(ETAI_patch_path + '.grab_remote_file')
    def test_acquire_with_remote_node(self, grab_mock, gen_mock,
                                      wait_mock, acquire_mock, load_mock):
        self.e.remote_node = True

        self.e._acquire(10, 100, burst=True)

        gen_mock.assert_called_once_with("Image")
        acquire_mock.assert_called_once_with(
            [0, 1, 2, 3, 4, 5, 6, 7], 10, 100, trig_mode=0, gain_mode='slgm',
            burst=True, pixel_mode='spm', counter=0, equalization=0,
            disc_mode='discL', hdf_file=gen_mock.return_value, path='/tmp',
            depth=12, read_mode='sequential')
        grab_mock.assert_called_once_with(gen_mock.return_value)
        wait_mock.assert_called_once_with(gen_mock.return_value, 10)
        load_mock.assert_called_once_with(gen_mock.return_value)

    @patch(util_patch_path + '.tag_plot_name')
    @patch(DAWN_patch_path + '.plot_image')
    @patch(Node_patch_path + '._acquire')
    def test_expose(self, acquire_mock, plot_mock, tag_mock):

        self.e.expose()

        acquire_mock.assert_called_once_with(1, 100)
        tag_mock.assert_called_once_with("Image", "Node 1")
        plot_mock.assert_called_once_with(acquire_mock.return_value,
                                          tag_mock.return_value)

    @patch(util_patch_path + '.tag_plot_name')
    @patch(DAWN_patch_path + '.plot_image')
    @patch(Node_patch_path + '._acquire')
    def test_expose_with_exposure(self, acquire_mock, plot_mock, tag_mock):

        self.e.expose(200)

        acquire_mock.assert_called_once_with(1, 200)
        tag_mock.assert_called_once_with("Image", "Node 1")
        plot_mock.assert_called_once_with(acquire_mock.return_value,
                                          tag_mock.return_value)

    @patch(Node_patch_path + '._acquire')
    def test_burst(self, acquire_mock):

        self.e.burst(10, 100)

        acquire_mock.assert_called_once_with(10, 100, burst=True)

    @patch(util_patch_path + '.tag_plot_name',
           return_value="Node 1 - Image - 20161124~113442")
    @patch(DAWN_patch_path + '.plot_image')
    @patch(Node_patch_path + '._acquire')
    def test_cont(self, acquire_mock, plot_mock, tag_mock):

        self.e.cont(100, 10)

        self.assertEqual("continuous", self.e.settings['readmode'])
        acquire_mock.assert_called_once_with(100, 10)
        tag_mock.assert_called_once_with("Image", "Node 1")
        plot_mock.assert_has_calls(
            [call(acquire_mock.return_value.__getitem__(),
                  "Node 1 - Image - 20161124~113442 - {}".format(idx))
             for idx in range(5)])
        self.assertEqual(5, plot_mock.call_count)

    @patch(Node_patch_path + '._acquire')
    def test_cont_burst(self, acquire_mock):

        self.e.cont_burst(100, 10)

        self.assertEqual("continuous", self.e.settings['readmode'])
        acquire_mock.assert_called_once_with(100, 10, burst=True)

    @patch(util_patch_path + '.tag_plot_name')
    @patch(DAWN_patch_path + '.plot_image')
    @patch(Node_patch_path + '.expose')
    def test_correct_calls_made(self, expose_mock, plot_mock, tag_mock):
        e = ExcaliburNode(1, mock_config)

        e.loop(1)

        expose_mock.assert_called_once_with()
        plot_mock.assert_called_once_with(expose_mock.return_value.__add__(),
                                          tag_mock.return_value)

    @patch(ETAI_patch_path + '.load_dacs')
    @patch(Node_patch_path + '._write_dac')
    def test_set_dac(self, write_mock, load_mock):
        dac_file = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/testdetector/calib/fem1/spm/slgm/dacs'
        expected_lines = ['Heading', 'Threshold0 = 40\r\n', 'Line2']
        readlines_mock = ['Heading', 'Line1', 'Line2']
        self.file_mock.__enter__.return_value.readlines.return_value = readlines_mock[:]  # Don't pass by reference
        chips = [0]

        self.e.set_dac(chips, "Threshold0", 40)

        write_mock.assert_called_with(0, "Threshold0", 40)
        load_mock.assert_called_once_with(chips, dac_file)

    @patch('__builtin__.open', return_value=file_mock)
    def test_write_dac(self, open_mock):
        dac_file = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/testdetector/calib/fem1/spm/slgm/dacs'
        expected_lines = ['Heading', 'Threshold0 = 40\r\n', 'Line2']
        readlines_mock = ['Heading', 'Line1', 'Line2']
        self.file_mock.__enter__.return_value.readlines.return_value = readlines_mock[:]  # Don't pass by reference
        chips = [0]

        self.e._write_dac(0, "Threshold0", 40)

        open_mock.assert_called_with(dac_file, 'w')
        self.assertEqual(2 * len(chips), open_mock.call_count)
        self.file_mock.__enter__.return_value.writelines.assert_called_once_with(expected_lines)

    @patch(ETAI_patch_path + '.sense')
    def test_read_dac(self, sense_mock):
        expected_file = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/testdetector/calib/fem1/spm/slgm/dacs'
        chips = range(8)

        self.e.read_dac('Threshold0')

        sense_mock.assert_called_once_with(chips, 'Threshold0', expected_file)

    @patch(ETAI_patch_path + '.load_config')
    @patch(util_patch_path + '.save_array')
    def test_load_temp_config(self, save_mock, load_mock):
        filepath = '/dls/detectors/support/silicon_pixels/excaliburRX' \
                   '/3M-RX001/testdetector/calib/fem1/spm/slgm/'
        expected_file_1 = '/dls/detectors/support/silicon_pixels/excaliburRX' \
                          '/3M-RX001/testdetector/calib/fem1/spm/slgm' \
                          '/discLbits.tmp'
        expected_file_2 = '/dls/detectors/support/silicon_pixels/excaliburRX' \
                          '/3M-RX001/testdetector/calib/fem1/spm/slgm' \
                          '/discHbits.tmp'
        expected_file_3 = '/dls/detectors/support/silicon_pixels/excaliburRX' \
                          '/3M-RX001/testdetector/calib/fem1/spm/slgm' \
                          '/pixelmask.tmp'
        chips = [0]
        discLbits = MagicMock()
        discHbits = MagicMock()
        maskbits = MagicMock()

        self.e.load_temp_config(chips, discLbits, discHbits, maskbits)

        save_mock.assert_has_calls(
            [call(filepath + 'discLbits.tmp', discLbits),
             call(filepath + 'discHbits.tmp', discHbits),
             call(filepath + 'pixelmask.tmp', maskbits)])

        load_mock.assert_called_once_with([0], expected_file_1,
                                          expected_file_2, expected_file_3)

    # Make sure we always get the same random numbers
    rand = np.random.RandomState(123)

    @patch(Node_patch_path + '.load_temp_config')
    @patch(util_patch_path + '.grab_chip_slice')
    @patch(Node_patch_path + '._load_discbits',
           side_effect=[rand.randint(2, size=[256, 256]),
                        rand.randint(2, size=[256, 256])])
    def test_load_all_discbits_L(self, open_mock, grab_mock, load_mock):
        chips = [0]
        temp_bits_mock = MagicMock()
        mask_mock = MagicMock()

        self.e.load_all_discbits(chips, "discL", temp_bits_mock, mask_mock)

        self.assertFalse(open_mock.call_count)
        self.assertEqual((temp_bits_mock, 0), grab_mock.call_args_list[0][0])
        self.assertEqual((ANY, 0), grab_mock.call_args_list[1][0])
        self.assertEqual((mask_mock, 0), grab_mock.call_args_list[2][0])
        load_mock.assert_called_once_with(0, grab_mock.return_value,
                                          grab_mock.return_value,
                                          grab_mock.return_value)

    @patch(Node_patch_path + '.load_temp_config')
    @patch(util_patch_path + '.grab_chip_slice')
    @patch(Node_patch_path + '._load_discbits')
    def test_load_all_discbits_H(self, open_mock, grab_mock, load_mock):
        chips = [0]
        temp_bits_mock = MagicMock()
        mask_mock = MagicMock()

        self.e.load_all_discbits(chips, "discH", temp_bits_mock, mask_mock)

        open_mock.assert_called_once_with(chips, "discLbits")
        self.assertEqual((open_mock.return_value, 0), grab_mock.call_args_list[0][0])
        self.assertEqual((ANY, 0), grab_mock.call_args_list[1][0])
        self.assertEqual((mask_mock, 0), grab_mock.call_args_list[2][0])
        load_mock.assert_called_once_with(0, grab_mock.return_value,
                                          grab_mock.return_value,
                                          grab_mock.return_value)

    @patch(Node_patch_path + '.load_temp_config')
    @patch(util_patch_path + '.grab_chip_slice')
    @patch(Node_patch_path + '._load_discbits')
    def test_load_all_discbits_error(self, _, _1, _2):
        chips = [0]
        temp_bits_mock = MagicMock()
        mask_mock = MagicMock()

        with self.assertRaises(ValueError):
            self.e.load_all_discbits(chips, "discX", temp_bits_mock, mask_mock)


@patch(ETAI_patch_path + '.load_config')
@patch(Node_patch_path + '.set_dac')
@patch(Node_patch_path + '.expose')
class LoadConfigTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode(1, mock_config)
        self.chips = [0]

    def test_correct_calls_made(self, expose_mock, set_mock, load_mock):
        expected_file_1 = '/dls/detectors/support/silicon_pixels/excaliburRX/' \
                          '3M-RX001/testdetector/calib/fem1/spm/slgm/discLbits.chip0'
        expected_file_2 = '/dls/detectors/support/silicon_pixels/excaliburRX/' \
                          '3M-RX001/testdetector/calib/fem1/spm/slgm/discHbits.chip0'
        expected_file_3 = '/dls/detectors/support/silicon_pixels/excaliburRX/' \
                          '3M-RX001/testdetector/calib/fem1/spm/slgm/pixelmask.chip0'

        self.e.load_config(self.chips)

        # Check first set_dac call
        call_args = set_mock.call_args_list[0]
        self.assertEqual((range(8), "Threshold1", 100), call_args[0])
        # Check second set_dac call
        call_args = set_mock.call_args_list[1]
        self.assertEqual((range(8), "Threshold0", 40), call_args[0])

        load_mock.assert_called_once_with(self.chips[0],expected_file_1,
                                          expected_file_2, expected_file_3)


@patch('time.sleep')
@patch(Node_patch_path + '.load_config')
@patch(Node_patch_path + '.set_dac')
@patch(Node_patch_path + '.expose')
class Fe55ImageRX001Test(unittest.TestCase):

    def test_correct_calls_made(self, expose_mock, set_mock,
                                load_mock, sleep_mock):
        e = ExcaliburNode(1, mock_config)
        chips = [1, 4, 6, 7]
        exposure_time = 10000

        e.fe55_image_rx001(chips, exposure_time)

        self.assertEqual('shgm', e.settings['gain'])
        self.assertEqual('spm', e.settings['mode'])

        load_mock.assert_called_once_with(chips)
        set_mock.assert_called_once_with(chips, "Threshold0", 40)
        sleep_mock.assert_called_once_with(0.5)
        expose_mock.assert_called_once_with(exposure_time)


@patch(Node_patch_path + '.display_dac_scan')
@patch(DAWN_patch_path + '.load_image_data')
@patch(Node_patch_path + '.generate_file_name',
       return_value="20161020~154548_TestImage.hdf5")
@patch(ETAI_patch_path + '.perform_dac_scan')
@patch(util_patch_path + '.wait_for_file')
@patch(ETAI_patch_path + '.grab_remote_file')
class ScanDacTest(unittest.TestCase):

    def setUp(self):
        self.dac_file = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/testdetector/calib/fem1/spm/slgm/dacs'
        self.e = ExcaliburNode(1, mock_config)
        self.chips = [0]
        self.dac_range = Range(1, 10, 1)
        self.e.file_index = 5

    def test_given_start_lower_than_stop(self, grab_mock, wait_mock, scan_mock,
                                         gen_mock, load_mock, display_mock):
        self.e.scan_dac(self.chips, 'Threshold0', self.dac_range)

        gen_mock.assert_called_once_with("DACScan")
        scan_mock.assert_called_once_with(self.chips, 'Threshold0',
                                          self.dac_range, 5, self.dac_file,
                                          '/tmp', gen_mock.return_value)
        grab_mock.assert_not_called()
        wait_mock.assert_called_once_with(
            "/tmp/20161020~154548_TestImage.hdf5", 10)
        load_mock.assert_called_once_with(
            "/tmp/20161020~154548_TestImage.hdf5")
        display_mock.assert_called_once_with(self.chips,
                                             load_mock.return_value,
                                             self.dac_range)

    def test_given_remote_node_then_grab_file(self, grab_mock, wait_mock,
                                              scan_mock, gen_mock,
                                              load_mock, display_mock):
        self.e.remote_node = True
        expected_output = "/tmp/20161020~154548_TestImage.hdf5"

        self.e.scan_dac(self.chips, 'Threshold0', self.dac_range)

        gen_mock.assert_called_once_with("DACScan")
        scan_mock.assert_called_once_with(self.chips, 'Threshold0',
                                          self.dac_range, 5, self.dac_file,
                                          '/tmp', gen_mock.return_value,)
        grab_mock.assert_called_once_with(expected_output)
        wait_mock.assert_called_once_with(expected_output, 10)
        load_mock.assert_called_once_with(expected_output)
        display_mock.assert_called_once_with(self.chips,
                                             load_mock.return_value,
                                             self.dac_range)


class DisplayDacScanTest(unittest.TestCase):

    @patch(DAWN_patch_path + '.plot_dac_scan')
    def test_display_dac_scan_low_to_high(self, plot_mock):
        e = ExcaliburNode(1, mock_config)
        chips = [0]
        mock_array = np.random.randint(10, size=(10, 256, 8 * 256))
        expected_plot_data = [mock_array[:, 0:256, 0:256].mean(2).mean(1)]
        expected_dac_axis = range(1, 11)

        plot_data, dac_axis = e.display_dac_scan(chips, mock_array,
                                                 Range(1, 10, 1))

        plot_mock.assert_called_once_with(ANY, chips, ANY, "Node 1 - DAC Scan")
        np.testing.assert_array_equal(expected_plot_data, plot_mock.call_args[0][0])
        np.testing.assert_array_equal(dac_axis, plot_mock.call_args[0][2])
        np.testing.assert_array_equal(expected_dac_axis, dac_axis)
        np.testing.assert_array_equal(expected_plot_data, plot_data)

    @patch(DAWN_patch_path + '.plot_dac_scan')
    def test_display_dac_scan_high_to_low(self, plot_mock):
        e = ExcaliburNode(1, mock_config)
        chips = [0]
        mock_array = np.random.randint(10, size=(10, 256, 8 * 256))
        expected_plot_data = [mock_array[:, 0:256, 0:256].mean(2).mean(1)]
        expected_dac_axis = range(10, 0, -1)

        plot_data, dac_axis = e.display_dac_scan(chips, mock_array,
                                                 Range(10, 1, 1))

        plot_mock.assert_called_once_with(ANY, chips, ANY, "Node 1 - DAC Scan")
        np.testing.assert_array_equal(expected_plot_data, plot_mock.call_args[0][0])
        np.testing.assert_array_equal(dac_axis, plot_mock.call_args[0][2])
        np.testing.assert_array_equal(expected_dac_axis, dac_axis)
        np.testing.assert_array_equal(expected_plot_data, plot_data)


class AcquireFFTest(unittest.TestCase):

    # Make sure we always get the same random numbers
    rand = np.random.RandomState(1234)
    mock_array = rand.randint(10, size=(256, 8 * 256))

    @patch(Node_patch_path + '.expose',
           return_value=mock_array)
    @patch(DAWN_patch_path + '.plot_image')
    def test_acquire_ff(self, plot_mock, expose_mock):
        e = ExcaliburNode(1, mock_config)
        mean = self.mock_array[0:256, 3*256:4*256].mean()
        array = np.copy(self.mock_array)
        array[array == 0] = mean
        expected_array = (np.ones([256, 8*256]) * mean) / array
        expected_array[expected_array > 2] = 1
        expected_array[expected_array < 0] = 1

        ff_coeff = e.acquire_ff(1, 0.1)

        expose_mock.assert_called_once_with()
        plot_mock.assert_called_once_with(ANY, name='Flat Field coefficients')
        np.testing.assert_array_almost_equal(expected_array, ff_coeff, 5)


class ApplyFFCorrectionTest(unittest.TestCase):

    # Make sure we always get the same random numbers
    rand = np.random.RandomState(1234)

    @patch(DAWN_patch_path + '.plot_image')
    @patch(DAWN_patch_path + '.load_image_data')
    @patch('time.sleep')
    @patch(Node_patch_path + '.generate_file_name')
    def test_apply_ff_correction(self, gen_mock, sleep_mock, load_mock,
                                 plot_mock):
        e = ExcaliburNode(1, mock_config)
        e.file_index = 5

        e.apply_ff_correction(1, 0.1)

        gen_mock.assert_called_once_with("FFImage")
        load_mock.assert_called_once_with(gen_mock.return_value)
        plot_mock.assert_called_once_with(load_mock.return_value.__getitem__().__mul__().__getitem__(), name='Image data Cor')

        # TODO: Finish once function completed


class SaveDiscbitsTest(unittest.TestCase):

    @patch(util_patch_path + '.save_array')
    def test_correct_call_made(self, save_mock):
        expected_file = "/dls/detectors/support/silicon_pixels/excaliburRX/" \
                        "3M-RX001/testdetector/calib/fem1/spm/slgm/test.chip0"
        discbits = np.random.randint(10, size=(256, 8*256))
        expected_subarray = discbits[0:256, 0:256]
        e = ExcaliburNode(1, mock_config)

        e.save_discbits([0], discbits, 'test')

        save_mock.assert_called_once_with(expected_file, ANY)
        np.testing.assert_array_equal(expected_subarray,
                                      save_mock.call_args[0][1])


class MaskUnmaskTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode(1, mock_config)

    @patch(DAWN_patch_path + '.plot_image')
    @patch(Node_patch_path + '._apply_mask')
    def test_mask_columns(self, apply_mock, plot_mock):
        self.e.full_array_shape = [4, 16]  # Make testing easier
        self.e.chip_size = 4  # Make testing easier
        self.e.num_chips = 4  # Make testing easier
        chips = [1, 2]
        expected_array = np.array([[0., 0., 0., 0., 0., 1., 1., 0.,
                                    0., 1., 1., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 1., 1., 0.,
                                    0., 1., 1., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 1., 1., 0.,
                                    0., 1., 1., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 1., 1., 0.,
                                    0., 1., 1., 0., 0., 0., 0., 0.]])

        self.e.mask_columns(chips, 1, 2)

        apply_mock.assert_called_once_with(chips, ANY)
        np.testing.assert_array_equal(expected_array,
                                      apply_mock.call_args[0][1])
        plot_mock.assert_called_once_with(ANY, "Mask")
        np.testing.assert_array_equal(expected_array,
                                      plot_mock.call_args[0][0])

    @patch(util_patch_path + '.tag_plot_name')
    @patch(DAWN_patch_path + '.plot_image')
    @patch(Node_patch_path + '._apply_mask')
    def test_mask_rows(self, apply_mock, plot_mock, tag_mock):

        self.e.full_array_shape = [4, 16]  # Make testing easier
        self.e.chip_size = 4  # Make testing easier
        self.e.num_chips = 4  # Make testing easier
        chips = [1, 2]
        expected_array = np.array([[0., 0., 0., 0., 0., 0., 0., 0.,
                                    0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 1., 1., 1., 1.,
                                    1., 1., 1., 1., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 1., 1., 1., 1.,
                                    1., 1., 1., 1., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0.,
                                    0., 0., 0., 0., 0., 0., 0., 0.]])

        self.e.mask_rows(chips, 1, 2)

        apply_mock.assert_called_once_with(chips, ANY)
        np.testing.assert_array_equal(expected_array, apply_mock.call_args[0][1])
        tag_mock.assert_called_once_with("Mask", "Node 1")
        plot_mock.assert_called_once_with(ANY, tag_mock.return_value)
        np.testing.assert_array_equal(expected_array, plot_mock.call_args[0][0])

    @patch(Node_patch_path + '._mask_noisy_pixels')
    def test_mask_pixels_using_image(self, mask_mock):
        image_data = np.random.randint(10, size=(256, 8*256))
        expected_mask = image_data > 7

        self.e.mask_pixels_using_image([0], image_data, 7)

        mask_mock.assert_called_once_with([0], ANY)
        np.testing.assert_array_equal(expected_mask, mask_mock.call_args[0][1])

    mock_scan_data = np.random.randint(3, size=(3, 256, 8*256))

    @patch(Node_patch_path + '._mask_noisy_pixels')
    @patch(Node_patch_path + '.scan_dac', return_value=mock_scan_data)
    def test_mask_pixels_using_dac_scan(self, scan_mock, mask_mock):
        expected_mask = self.mock_scan_data.sum(0) > 1

        self.e.mask_pixels_using_dac_scan([0])

        scan_mock.assert_called_once_with([0], "Threshold0", (20, 120, 2))
        mask_mock.assert_called_once_with([0], ANY)
        np.testing.assert_array_equal(expected_mask, mask_mock.call_args[0][1])

    @patch(DAWN_patch_path + '.plot_image')
    @patch(Node_patch_path + '._apply_mask')
    def test_mask_noisy_pixels(self, apply_mock, plot_mock):
        image_data = np.random.randint(2, size=(256, 8*256))

        self.e._mask_noisy_pixels([0], image_data)

        plot_mock.assert_called_once_with(ANY, name="Noisy Pixels")
        np.testing.assert_array_equal(image_data, plot_mock.call_args[0][0])

        apply_mock.assert_called_once_with([0], ANY)
        np.testing.assert_array_equal(image_data, apply_mock.call_args[0][1])

    @patch(Node_patch_path + '.load_config')
    @patch(util_patch_path + '.save_array')
    def test_apply_mask(self, save_mock, load_mock):
        mask = np.random.randint(10, size=(256, 8*256))
        expected_mask = mask[:, 0:256]
        expected_file = "/dls/detectors/support/silicon_pixels/excaliburRX/" \
                        "3M-RX001/testdetector/calib/fem1/spm/slgm/" \
                        "pixelmask.chip0"

        self.e._apply_mask([0], mask)

        save_mock.assert_called_once_with(expected_file, ANY)
        np.testing.assert_array_equal(expected_mask, save_mock.call_args[0][1])

        load_mock.assert_called_once_with([0])

    @patch(ETAI_patch_path + '.load_config')
    @patch(util_patch_path + '.save_array')
    def test_unmask_all_pixels(self, save_mock, load_mock):
        expected_file_1 = "/dls/detectors/support/silicon_pixels/excaliburRX" \
                          "/3M-RX001/testdetector/calib/fem1/spm/slgm" \
                          "/discLbits.chip0"
        expected_file_2 = "/dls/detectors/support/silicon_pixels/excaliburRX" \
                          "/3M-RX001/testdetector/calib/fem1/spm/slgm" \
                          "/discHbits.chip0"
        expected_file_3 = "/dls/detectors/support/silicon_pixels/excaliburRX" \
                          "/3M-RX001/testdetector/calib/fem1/spm/slgm" \
                          "/pixelmask.chip0"
        zeros = np.zeros([256, 256])

        self.e.unmask_pixels([0])

        save_mock.assert_called_once_with(expected_file_3, ANY)
        np.testing.assert_array_equal(zeros, save_mock.call_args[0][1])
        load_mock.assert_called_once_with(0, expected_file_1, expected_file_2,
                                          expected_file_3)

    @patch(ETAI_patch_path + '.load_config')
    @patch(util_patch_path + '.save_array')
    def test_unequalize_pixels(self, save_mock, load_mock):
        expected_file_1 = "/dls/detectors/support/silicon_pixels/excaliburRX" \
                          "/3M-RX001/testdetector/calib/fem1/spm/slgm/discLbits.chip0"
        expected_file_2 = "/dls/detectors/support/silicon_pixels/excaliburRX" \
                          "/3M-RX001/testdetector/calib/fem1/spm/slgm/discHbits.chip0"
        expected_file_3 = "/dls/detectors/support/silicon_pixels/excaliburRX" \
                          "/3M-RX001/testdetector/calib/fem1/spm/slgm/pixelmask.chip0"
        zeros = np.zeros([256, 256])

        self.e.unequalize_pixels([0])

        save_mock.assert_has_calls([call(expected_file_1, ANY),
                                    call(expected_file_2, ANY)])
        np.testing.assert_array_equal(zeros, save_mock.call_args_list[0][0][1])
        np.testing.assert_array_equal(zeros, save_mock.call_args_list[1][0][1])
        load_mock.assert_called_once_with(0, expected_file_1,
                                          expected_file_2,
                                          expected_file_3)


@patch('shutil.copytree')
class CheckCalibDirTest(unittest.TestCase):

    @patch(util_patch_path + '.get_time_stamp',
           return_value="20161020~154548")
    def test_does_exist_then_backup(self, _, copy_tree_mock):
        e = ExcaliburNode(1, mock_config)

        e.backup_calib_dir()

        copy_tree_mock.assert_called_once_with(
            '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001'
            '/testdetector/calib',
            '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001'
            '/testdetector/calib_20161020~154548')


@patch('shutil.copytree')
@patch('shutil.rmtree')
class CopySLGMIntoOtherGainModesTest(unittest.TestCase):

    @patch('os.path.exists', return_value=True)
    def test_exist_then_rm_and_copy(self, exists_mock, rm_mock, copytree_mock):
        expected_path = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/testdetector/calib/fem1/spm/'
        e = ExcaliburNode(1, mock_config)

        e.copy_slgm_into_other_gain_modes()

        # Check exists calls
        self.assertEqual(expected_path + 'lgm', exists_mock.call_args_list[0][0][0])
        self.assertEqual(expected_path + 'hgm', exists_mock.call_args_list[1][0][0])
        self.assertEqual(expected_path + 'shgm', exists_mock.call_args_list[2][0][0])
        # Check rm calls
        self.assertEqual(expected_path + 'lgm', rm_mock.call_args_list[0][0][0])
        self.assertEqual(expected_path + 'hgm', rm_mock.call_args_list[1][0][0])
        self.assertEqual(expected_path + 'shgm', rm_mock.call_args_list[2][0][0])
        # Check copytree calls
        self.assertEqual((expected_path + 'slgm', expected_path + 'lgm'), copytree_mock.call_args_list[0][0])
        self.assertEqual((expected_path + 'slgm', expected_path + 'hgm'), copytree_mock.call_args_list[1][0])
        self.assertEqual((expected_path + 'slgm', expected_path + 'shgm'), copytree_mock.call_args_list[2][0])

    @patch('os.path.exists', return_value=False)
    def test_dont_exist_then_just_copy(self, exists_mock, rm_mock,
                                       copytree_mock):
        expected_path = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/testdetector/calib/fem1/spm/'
        e = ExcaliburNode(1, mock_config)

        e.copy_slgm_into_other_gain_modes()

        # Check exists calls
        self.assertEqual(expected_path + 'lgm', exists_mock.call_args_list[0][0][0])
        self.assertEqual(expected_path + 'hgm', exists_mock.call_args_list[1][0][0])
        self.assertEqual(expected_path + 'shgm', exists_mock.call_args_list[2][0][0])
        # Check rm calls
        self.assertFalse(rm_mock.call_count)
        # Check copytree calls
        self.assertEqual((expected_path + 'slgm', expected_path + 'lgm'), copytree_mock.call_args_list[0][0])
        self.assertEqual((expected_path + 'slgm', expected_path + 'hgm'), copytree_mock.call_args_list[1][0])
        self.assertEqual((expected_path + 'slgm', expected_path + 'shgm'), copytree_mock.call_args_list[2][0])


class LoadDiscbitsTest(unittest.TestCase):

    mock_array = np.random.randint(10, size=(256, 256))

    @patch('numpy.loadtxt', return_value=mock_array)
    def test_correct_call_made(self, load_mock):
        expected_path = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/testdetector/calib/fem1/spm/slgm/test_file.chip0'
        e = ExcaliburNode(1, mock_config)

        value = e._load_discbits([0], 'test_file')

        load_mock.assert_called_once_with(expected_path)
        self.assertTrue((value[0:256, 0:256] == self.mock_array).all())


class CombineROIsTest(unittest.TestCase):

    # Make sure we always get the same random numbers
    rand = np.random.RandomState(123)

    @patch(DAWN_patch_path + '.plot_image')
    @patch(Node_patch_path + '.roi',
           return_value=np.ones([256, 256]))
    @patch(Node_patch_path + '.save_discbits')
    @patch(Node_patch_path + '._load_discbits',
           side_effect=[rand.randint(2, size=[256, 256]),
                        rand.randint(2, size=[256, 256])])
    def test_correct_calls_made(self, open_mock, save_mock, roi_mock,
                               plot_mock):
        expected_path = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/testdetector/calib/fem1/spm/slgm/test_file.chip0'
        e = ExcaliburNode(1, mock_config)

        value = e.combine_rois([0], 'test_file', 1, 'rect')

        roi_mock.assert_called_once_with([0], 0, 1, 'rect')
        # plot_mock.assert_called_once_with()
        # save_mock.assert_called_once_with()
        # plot_mock.assert_called_once_with()

        # TODO: Finish once sure what function should do


@patch(DAWN_patch_path + '.plot_image')
class FindTest(unittest.TestCase):  # TODO: Improve

    def setUp(self):
        self.e = ExcaliburNode(1, mock_config)

        # Make sure we always get the same random numbers
        self.rand = np.random.RandomState(123)

    # TODO: Why do these get the same expected array?

    @patch(Node_patch_path + '._display_histogram')
    def test_find_edge_low_to_high(self, display_mock, plot_mock):
        dac_scan_data = self.rand.randint(10, size=(3, 3, 3))
        dac_range = MagicMock(start=1, stop=10, step=1)
        expected_array = [[10, 9, 10], [10, 9, 8], [10, 10, 10]]

        value = self.e.find_edge([0], dac_scan_data, dac_range, 7)

        plot_mock.assert_called_once_with(ANY, name="Node 1 - Noise Edges")
        np.testing.assert_array_equal(expected_array, plot_mock.call_args[0][0])
        display_mock.assert_called_once_with([0], ANY,
                                             "Node 1 - Histogram of NEdge",
                                             "Edge Location")
        np.testing.assert_array_equal(expected_array, display_mock.call_args[0][1])
        np.testing.assert_array_equal(expected_array, value)

    @patch(Node_patch_path + '._display_histogram')
    def test_find_edge_high_to_low(self, display_mock, plot_mock):
        dac_scan_data = self.rand.randint(10, size=(3, 3, 3))
        dac_range = MagicMock(start=10, stop=1, step=1)
        expected_array = [[10, 9, 10], [10, 9, 10], [10, 10, 10]]

        value = self.e.find_edge([0], dac_scan_data, dac_range, 7)

        display_mock.assert_called_once_with([0], ANY,
                                             "Node 1 - Histogram of NEdge",
                                             "Edge Location")
        np.testing.assert_array_equal(expected_array, display_mock.call_args[0][1])
        np.testing.assert_array_equal(expected_array, value)

    @patch(Node_patch_path + '._display_histogram')
    def test_find_max(self, display_mock, plot_mock):
        dac_scan_data = self.rand.randint(10, size=(3, 3, 3))
        dac_range = MagicMock(start=1, stop=10, step=1)
        expected_array = [[10, 9, 10], [10, 9, 8], [10, 10, 10]]

        value = self.e.find_max([0], dac_scan_data, dac_range)

        display_mock.assert_called_once_with([0], ANY,
                                             "Node 1 - Histogram of NMax",
                                             "Max Location")
        np.testing.assert_array_equal(expected_array, display_mock.call_args[0][1])
        np.testing.assert_array_equal(expected_array, value)

    @patch(util_patch_path + '.grab_chip_slice')
    @patch(DAWN_patch_path + '.plot_histogram')
    def test_display_histogram(self, plot_histo_mock, grab_mock, _):
        mock_array = MagicMock()

        self.e._display_histogram([0], mock_array, "Test", "X-Axis")

        plot_histo_mock.assert_called_once_with([grab_mock.return_value],
                                                "Test", "X-Axis")


class OptimizeDacDiscTest(unittest.TestCase):

    rand = np.random.RandomState(1234)

    def setUp(self):
        self.e = ExcaliburNode(1, mock_config)

    @patch(Node_patch_path + '._display_optimization_results')
    @patch(Node_patch_path + '.set_dac')
    @patch(DAWN_patch_path + '.clear_plot')
    @patch(DAWN_patch_path + '.plot_linear_fit', return_value=(1, 1))
    @patch(DAWN_patch_path + '.add_plot_line')
    @patch(Node_patch_path + '.load_all_discbits')
    @patch(Node_patch_path + '._dac_scan_fit', return_value=([1], [1]))
    def test_optimize_dac_disc(self, scan_mock, load_mock, add_mock, fit_mock,
                               clear_mock, set_mock, display_mock):
        chips = [0]
        zeros = np.zeros([256, 8*256])
        ones = np.ones([256, 8*256])

        self.e._optimize_dac_disc(chips, "discL", ones)

        load_mock.assert_has_calls([call(chips, "discL", ANY, ANY),
                                    call(chips, "discL", ANY, ANY)])
        np.testing.assert_array_equal(zeros, load_mock.call_args_list[0][0][2])
        np.testing.assert_array_equal(ones, load_mock.call_args_list[0][0][3])
        np.testing.assert_array_equal(15 * ones, load_mock.call_args_list[1][0][2])
        np.testing.assert_array_equal(ones, load_mock.call_args_list[1][0][3])

        r = Range(0, 150, 5)
        p = [5000, 50, 30]
        scan_mock.assert_has_calls([call(chips, "Threshold0", 30, r, 0, p),
                                    call(chips, "Threshold0", 80, r, 0, p),
                                    call(chips, "Threshold0", 130, r, 0, p),
                                    call(chips, "Threshold0", 80, r, 15,
                                         [5000, 0, 30])])

        name = "Node 1 - Mean edge shift in Threshold DACs as a function of" \
               " DAC_disc for discbit = 0"
        clear_mock.assert_called_once_with(name)

        add_mock.assert_called_once_with(ANY, ANY, "Disc Value", "Edges",
                                         name, label="Chip 0")
        fit_mock.assert_called_once_with(ANY, ANY, [0, -1],
                                         "Disc Value", "Edges",
                                         fit_name=name, label="Chip 0")
        np.testing.assert_array_equal(np.array([30, 80, 130]), fit_mock.call_args[0][0])
        np.testing.assert_array_equal(np.array([1., 1., 1.]), fit_mock.call_args[0][1])

        set_mock.assert_called_once_with(chips, "Threshold0", 3)
        display_mock.assert_called_once_with(chips, ANY, ANY, ANY, ANY)
        np.testing.assert_array_equal(np.array([1.]), display_mock.call_args[0][1])
        np.testing.assert_array_equal(np.array([1.]), display_mock.call_args[0][2])
        np.testing.assert_array_equal(np.array([1., 0., 0., 0., 0., 0., 0., 0.]),
                                      display_mock.call_args[0][3])

    @patch(DAWN_patch_path + '.plot_gaussian_fit', return_value=(0, 0))
    @patch(Node_patch_path + '.set_dac')
    @patch(Node_patch_path + '.scan_dac',
           return_value=rand.randint(10, size=(3, 256, 8*256)))
    @patch(Node_patch_path + '.find_max')
    def test_chip_dac_scan(self, find_mock, scan_mock, set_mock, plot_mock):
        chips = [0]
        expected_message = "Node 1 - Histogram of edges when scanning " \
                           "DAC_disc for discbit = 0"
        range = MagicMock(start=0, stop=150, step=50)

        self.e._dac_scan_fit(chips, "discL", 1, range, 0, [5000, 0, 30])

        set_mock.assert_called_once_with(chips, "discL", 1)
        scan_mock.assert_called_once_with(chips, "discL", range)
        find_mock.assert_called_once_with(chips, scan_mock.return_value,
                                          range)
        plot_mock.assert_called_once_with([find_mock.return_value.__getitem__.return_value] + [None]*7,
                                          expected_message, [5000, 0, 30], 3)

    def test_display_optimization_results(self):
        chips = [0]

        self.e._display_optimization_results(chips, MagicMock(), MagicMock(),
                                             MagicMock(), [1]*8)


class EqualizeDiscbitsTest(unittest.TestCase):

    rand = np.random.RandomState(1234)

    def setUp(self):
        self.e = ExcaliburNode(1, mock_config)
        self.e.chip_size = 4
        self.e.num_chips = 4
        self.e.full_array_shape = [4, 16]

    @patch(Node_patch_path + '.load_config')
    @patch(DAWN_patch_path + '.plot_histogram_with_mask')
    @patch(DAWN_patch_path + '.plot_image')
    @patch(Node_patch_path + '.find_max',
           return_value=rand.randint(2, size=(4, 16)))
    @patch(Node_patch_path + '.scan_dac',
           return_value=rand.randint(10, size=(3, 4, 16)))
    @patch(Node_patch_path + '.load_all_discbits')
    def test_equalize_discbits(self, load_mock, scan_mock, find_mock,
                               plot_mock, histo_mock, load_config_mock):
        chips = [0]
        roi = np.zeros([4, 16])

        self.e._equalise_discbits(chips, "DACDiscL", "Threshold0", roi,
                                  "Stripes")

        self.assertEqual(32 + 1, scan_mock.call_count)
        self.assertEqual(([0], "DACDiscL", ANY, ANY),
                         load_mock.call_args_list[0][0])
        np.testing.assert_array_equal(roi, load_mock.call_args_list[0][0][3])
        self.assertEqual(([0], "Threshold0", (0, 20, 2)),
                         scan_mock.call_args_list[0][0])
        self.assertEqual(([0], scan_mock.return_value, Range(0, 20, 2)),
                         find_mock.call_args_list[0][0])
        self.assertEqual((ANY,), plot_mock.call_args_list[0][0])
        self.assertEqual(dict(name="Node 1 - Discriminator Bits"),
                         plot_mock.call_args_list[0][1])
        self.assertEqual((chips, ANY, ANY,
                          "Node 1 - Histogram of Final Discbits", "Bit Value"),
                         histo_mock.call_args_list[0][0])
        load_config_mock.assert_called_once_with(chips)

    def test_equalize_discbits_given_invalid_method_raises(self):

        with self.assertRaises(NotImplementedError):
            self.e._equalise_discbits([], "DACDiscL", "Threshold0", MagicMock(), "Not a Method")


class CheckCalibTest(unittest.TestCase):

    rand = np.random.RandomState(1234)

    @patch(Node_patch_path + '.find_max',
           return_value=rand.randint(2, size=(256, 8*256)))
    @patch(Node_patch_path + '.load_config')
    def test_correct_call_made(self, load_mock, find_mock):
        pass  # TODO: Function doesn't work
        e = ExcaliburNode(1, mock_config)
        chips = [0]

        # e.check_calib(chips, [0, 10, 1])
        #
        # load_mock.assert_called_once_with(chips)


@patch(DAWN_patch_path + '.plot_image')
class ROITest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode(1, mock_config)

    def test_rect(self, plot_mock):
        chips = range(8)
        expected_array = np.zeros([256, 8*256])
        # expected_array = np.ones([256, 8*256])

        roi = self.e.roi(chips, 1, 1, 'rect')
        # roi = self.e.roi(chips, 0, 1, 'rect')

        np.testing.assert_array_equal(expected_array, plot_mock.call_args[0][0])
        np.testing.assert_array_equal(expected_array, roi)

    def test_spacing(self, plot_mock):
        chips = range(8)
        expected_array = np.ones([256, 8*256])

        for c in chips:
            expected_array[:, c*256] = 0
            expected_array[:, (c+1)*256 - 1] = 0

        roi = self.e.roi(chips, 1, 1, 'spacing')

        np.testing.assert_array_equal(expected_array, plot_mock.call_args[0][0])
        np.testing.assert_array_equal(expected_array, roi)

    def test_given_invalid_method_then_error(self, _):

        with self.assertRaises(NotImplementedError):
            self.e.roi([0], 1, 1, 'not a method')


class CalibrateDiscTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode(1, mock_config)

    @patch(Node_patch_path + '.set_dac')
    @patch(Node_patch_path + '._calibrate_disc')
    def test_optimize_disc_l(self, calibrate_mock, set_mock):
        chips = [0]

        self.e.calibrate_disc_l(chips)

        set_mock.assert_called_once_with(chips, "Threshold1", 0)
        self.assertEqual('discL', self.e.settings['disccsmspm'])
        calibrate_mock.assert_called_once_with(chips, "discL")

    @patch(Node_patch_path + '.set_dac')
    @patch(Node_patch_path + '._calibrate_disc')
    def test_optimize_disc_h(self, calibrate_mock, set_mock):
        chips = [0]

        self.e.calibrate_disc_h(chips)

        set_mock.assert_called_once_with(chips, "Threshold0", 60)
        self.assertEqual('discH', self.e.settings['disccsmspm'])
        calibrate_mock.assert_called_once_with(chips, "discH")

    @patch(Node_patch_path + '._optimize_dac_disc')
    @patch(Node_patch_path + '.roi')
    @patch(Node_patch_path + '._equalise_discbits')
    @patch(Node_patch_path + '.save_discbits')
    @patch(Node_patch_path + '.combine_rois')
    @patch(Node_patch_path + '.load_config')
    @patch(Node_patch_path + '.copy_slgm_into_other_gain_modes')
    def test_correct_calls_made(self, copy_mock, load_mock, combine_rois,
                                save_mock, equalize_mock, roi_mock, opt_mock):
        chips = [0]

        self.e._calibrate_disc(chips, 'discL', 1, 'rect')

        opt_mock.assert_called_once_with(chips, 'discL', 1 - roi_mock.return_value)
        equalize_mock.assert_called_once_with(chips, 'discL', 'Threshold0', 1 - roi_mock.return_value, 'stripes')
        self.assertEqual(save_mock.call_args_list[0][0], (chips, equalize_mock.return_value, 'discLbits_roi_0'))
        combine_rois.assert_called_once_with(chips, 'discL', 1, 'rect')
        self.assertEqual(save_mock.call_args_list[1][0], (chips, combine_rois.return_value, 'discLbits'))
        load_mock.assert_called_once_with(chips)
        copy_mock.assert_called_once_with()


class CSMTest(unittest.TestCase):

    @patch(Node_patch_path + '.expose')
    @patch(Node_patch_path + '.load_config')
    @patch(Node_patch_path + '.set_dac')
    def test_correct_calls_made(self, set_mock, load_mock, expose_mock):
        e = ExcaliburNode(1, mock_config)
        chips = range(8)

        e.csm()

        self.assertEqual('csm', e.settings['mode'])
        self.assertEqual('slgm', e.settings['gain'])
        self.assertEqual(1, e.settings['counter'])

        self.assertEqual(set_mock.call_args_list[0][0], (range(8), 'Threshold0', 200))
        self.assertEqual(set_mock.call_args_list[1][0], (range(8), 'Threshold1', 200))
        load_mock.assert_called_once_with(chips)
        self.assertEqual(set_mock.call_args_list[2][0], (range(8), 'Threshold0', 45))
        self.assertEqual(set_mock.call_args_list[3][0], (range(8), 'Threshold1', 100))
        self.assertEqual(2, expose_mock.call_count)


class SetGNDFBKCasExcaliburRX001Test(unittest.TestCase):

    @patch(ETAI_patch_path + '.load_dacs')
    @patch(Node_patch_path + '._write_dac')
    def test_correct_calls_made(self, write_mock, load_mock):
        e = ExcaliburNode(1, mock_config)
        mock_config.GND_DAC = np.array([[141]])
        mock_config.FBK_DAC = np.array([[190]])
        mock_config.CAS_DAC = np.array([[178]])
        chips = [0]

        e.set_gnd_fbk_cas(chips)

        self.assertEqual(write_mock.call_args_list[0][0], (0, 'GND', 141))
        self.assertEqual(write_mock.call_args_list[1][0], (0, 'FBK', 190))
        self.assertEqual(write_mock.call_args_list[2][0], (0, 'Cas', 178))
        load_mock.assert_called_once_with(chips, e.dacs_file)


class RotateTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode(1, mock_config)

    @patch(util_patch_path + '.rotate_array')
    @patch('os.path.isfile', return_value=True)
    def test_rotate_config_files_exist(self, _, rotate_mock):
        self.e.chip_range = [0]  # Make test easier
        root_path = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/'
        expected_epics_path = root_path + 'testdetector/calib_epics'
        expected_discL_path = expected_epics_path + '/fem1/spm/slgm/discLbits.chip0'
        expected_discH_path = expected_epics_path + '/fem1/spm/slgm/discHbits.chip0'
        expected_mask_path = expected_epics_path + '/fem1/spm/slgm/pixelmask.chip0'

        self.e.rotate_config()

        rotate_mock.assert_has_calls([call(expected_discL_path),
                                      call(expected_discH_path),
                                      call(expected_mask_path)])

    @patch(util_patch_path + '.rotate_array')
    @patch('os.path.isfile', return_value=False)
    def test_rotate_config_files_dont_exist(self, _, rotate_mock):
        self.e.chip_range = [0]  # Make test easier

        self.e.rotate_config()

        rotate_mock.assert_not_called()


@patch('__builtin__.print')
class DisplayTest(unittest.TestCase):

    @patch("os.listdir",
           return_value=["diagonal.mask", "stfcinverted.mask", "triangle.mask",
                         "zeros.mask", "Default_SPM.dacs", "noise.dacs"])
    def test_display_masks(self, _, print_mock):
        expected_call = "Available masks: diagonal.mask, stfcinverted.mask, " \
                        "triangle.mask, zeros.mask"
        e = ExcaliburNode(1, mock_config)

        e.display_masks()

        self.assertEqual(expected_call, print_mock.call_args_list[0][0][0])

    @patch("os.listdir",
           return_value=["diagonal.mask", "stfcinverted.mask", "triangle.mask",
                         "zeros.mask", "Default_SPM.dacs", "noise.dacs"])
    def test_display_dacs(self, _, print_mock):
        expected_call = "Available DAC files: Default_SPM.dacs, noise.dacs"
        e = ExcaliburNode(1, mock_config)

        e.display_dac_files()

        self.assertEqual(expected_call, print_mock.call_args_list[0][0][0])

    def test_display_status(self, print_mock):
        expected_calls = ['Status for Node 1',
                          'LV: 0',
                          'HV: 0',
                          'HV Bias: 0',
                          'DACs Loaded: None',
                          'Initialised: False']
        e = ExcaliburNode(1, mock_config)

        e.display_status()

        calls = [call[0][0] for call in print_mock.call_args_list]
        self.assertEqual(expected_calls, calls)
