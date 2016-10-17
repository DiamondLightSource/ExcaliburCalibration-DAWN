import unittest

from pkg_resources import require
require("mock")
from mock import patch, MagicMock, ANY

from excaliburcalibrationdawn.excaliburnode import ExcaliburNode, np, Range
Node_patch_path = "excaliburcalibrationdawn.excaliburnode.ExcaliburNode"
ETAI_patch_path = "excaliburcalibrationdawn.excaliburnode.ExcaliburTestAppInterface"
DAWN_patch_path = "excaliburcalibrationdawn.excaliburnode.ExcaliburDAWN"
util_patch_path = "excaliburcalibrationdawn.arrayutil"


class InitTest(unittest.TestCase):

    def setUp(self):
        self.node = 3
        self.e = ExcaliburNode(self.node)

    def test_class_attributes_set(self):
        self.assertEqual(self.e.dac_target, 10)
        self.assertEqual(self.e.num_sigma, 3.2)
        self.assertEqual(self.e.allowed_delta, 4)
        self.assertEqual(self.e.calib_dir, "/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib")
        self.assertEqual(self.e.config_dir, "/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/config")
        self.assertEqual(self.e.settings, {'mode': 'spm',
                                           'gain': 'shgm',
                                           'bitdepth': 12,
                                           'readmode': 'sequential',
                                           'counter': 0,
                                           'disccsmspm': 'discL',
                                           'equalization': 0,
                                           'trigmode': 0,
                                           'acqtime': 100,
                                           'frames': 1,
                                           'imagepath': '/tmp/',
                                           'filename': 'image',
                                           'filenameIndex': ''})
        self.assertEqual(self.e.dac_number, {'Threshold0': 1,
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
        self.assertEqual(self.e.chip_range, [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(self.e.plot_name, '')

    def test_instance_attributes_set(self):
        self.assertEqual(self.e.fem, self.node)
        self.assertEqual(self.e.ipaddress, "192.168.0.104")

    def test_server_used(self):
        e = ExcaliburNode(1, "test-server")
        self.assertEqual("test-server1", e.server_name)
        self.assertEqual("test-server1", e.app.server_name)

    def test_given_node_invalid_node_raises(self):
        with self.assertRaises(ValueError):
            ExcaliburNode(0)
        with self.assertRaises(ValueError):
            ExcaliburNode(7)


class SetVoltageTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode(1)
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


@patch(Node_patch_path + '.check_calib_dir')
@patch(Node_patch_path + '.log_chip_id')
@patch(Node_patch_path + '.set_dacs')
@patch(Node_patch_path + '.set_gnd_fbk_cas_excalibur_rx001')
@patch(Node_patch_path + '.calibrate_disc')
class ThresholdEqualizationTest(unittest.TestCase):

    def test_correct_calls_made(self, cal_disc_mock, set_gnd_mock,
                                set_dacs_mock, log_mock, check_mock):
        e = ExcaliburNode(1)
        chips = [1, 4, 6, 7]

        e.threshold_equalization(chips)

        self.assertEqual('slgm', e.settings['gain'])
        self.assertEqual('spm', e.settings['mode'])

        check_mock.assert_called_once_with()
        log_mock.assert_called_once_with()
        set_dacs_mock.assert_called_once_with(chips)
        set_gnd_mock.assert_called_once_with(chips, e.fem)
        cal_disc_mock.assert_called_once_with(chips, 'discL')


@patch(Node_patch_path + '.check_calib_dir')
@patch(Node_patch_path + '.save_kev2dac_calib')
class ThresholdCalibrationTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode(1)

    def test_threshold_calibration_shgm(self, save_mock, check_mock):
        expected_slope = np.array([8.81355932203]*8)
        expected_offset = np.array([10.0]*8)

        self.e.settings['gain'] = 'shgm'
        self.e.threshold_calibration('0.1')

        check_mock.assert_called_once_with()
        self.assertEqual(save_mock.call_args[0][0], '0.1')
        np.testing.assert_array_almost_equal(expected_slope, save_mock.call_args[0][1][0])
        self.assertTrue((save_mock.call_args[0][2] == expected_offset).all())

    def test_threshold_calibration_hgm(self, save_mock, check_mock):
        expected_slope = np.array([6.61016949]*8)
        expected_offset = np.array([10.0]*8)

        self.e.settings['gain'] = 'hgm'
        self.e.threshold_calibration('0.1')

        check_mock.assert_called_once_with()
        self.assertEqual(save_mock.call_args[0][0], '0.1')
        np.testing.assert_array_almost_equal(expected_slope, save_mock.call_args[0][1][0])
        self.assertTrue((save_mock.call_args[0][2] == expected_offset).all())

    def test_threshold_calibration_lgm(self, save_mock, check_mock):
        expected_slope = np.array([4.40677966]*8)
        expected_offset = np.array([10.0]*8)

        self.e.settings['gain'] = 'lgm'
        self.e.threshold_calibration('0.1')

        check_mock.assert_called_once_with()
        self.assertEqual(save_mock.call_args[0][0], '0.1')
        np.testing.assert_array_almost_equal(expected_slope, save_mock.call_args[0][1][0])
        self.assertTrue((save_mock.call_args[0][2] == expected_offset).all())

    def test_threshold_calibration_slgm(self, save_mock, check_mock):
        expected_slope = np.array([2.20338983]*8)
        expected_offset = np.array([10.0]*8)

        self.e.settings['gain'] = 'slgm'
        self.e.threshold_calibration('0.1')

        check_mock.assert_called_once_with()
        self.assertEqual(save_mock.call_args[0][0], '0.1')
        np.testing.assert_array_almost_equal(expected_slope, save_mock.call_args[0][1][0])
        self.assertTrue((save_mock.call_args[0][2] == expected_offset).all())

    @patch(Node_patch_path + '.threshold_calibration')
    def test_threshold_calibration_all_gains(self, calibration_mock, _, _2):
        self.e.threshold_calibration_all_gains()

        self.assertEqual(4, calibration_mock.call_count)

    def test_one_energy_thresh_calib(self, save_mock, _):
        expected_slope = np.array([8.6666666]*8)
        expected_offset = np.array([10.0]*8)

        self.e.settings['gain'] = 'slgm'
        self.e.one_energy_thresh_calib()

        self.assertEqual(save_mock.call_args[0][0], '0')
        np.testing.assert_array_almost_equal(expected_slope, save_mock.call_args[0][1][0])
        self.assertTrue((save_mock.call_args[0][2] == expected_offset).all())

    @patch(DAWN_patch_path + '.plot_linear_fit', return_value=(10, 2))
    def test_multiple_energy_thresh_calib(self, plot_mock, save_mock, _):
        expected_slope = np.array([2.0] + [0.0]*7)
        expected_offset = np.array([10.0] + [0.0]*7)
        expected_array_1 = np.array([6, 12, 24])
        expected_array_2 = np.array([62, 62, 62])

        self.e.settings['gain'] = 'slgm'
        self.e.multiple_energy_thresh_calib([0])

        plot_mock.assert_called_once_with(ANY, ANY, [0, 1], name='DAC vs Energy', clear=True)
        np.testing.assert_array_equal(expected_array_1, plot_mock.call_args[0][0])
        np.testing.assert_array_equal(expected_array_2, plot_mock.call_args[0][1])

        np.testing.assert_array_almost_equal(expected_slope, save_mock.call_args[0][1])
        np.testing.assert_array_almost_equal(expected_offset, save_mock.call_args[0][2])


class SaveKev2DacCalibTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode(1)
        self.threshold = '0'
        self.gain = [1.1, 0.7, 1.1, 1.3, 1.0, 0.9, 1.2, 0.9]
        self.offset = [0.2, -0.7, 0.1, 0.0, 0.3, -0.1, 0.2, 0.5]
        self.expected_array = np.array([self.gain, self.offset])
        self.expected_filename = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/threshold0'

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
    @patch(DAWN_patch_path + '.plot_dac_scan',
           return_value=[MagicMock(), MagicMock()])
    @patch(Node_patch_path + '.scan_dac',
           return_value=mock_scan_data.copy())
    def test_correct_calls_made(self, scan_mock, plot_mock, fit_mock,
                                load_mock):
        e = ExcaliburNode(1)
        chips = [0]
        expected_array = self.mock_scan_data
        expected_array[expected_array > 200] = 0

        values = e.find_xray_energy_dac(chips)

        load_mock.assert_called_once_with(chips)
        scan_mock.assert_called_once_with(chips, "Threshold0", (110, 30, 2))

        plot_mock.assert_called_once_with(chips, ANY, self.mock_scan_range)
        np.testing.assert_array_equal(expected_array,
                                      plot_mock.call_args[0][1])

        fit_mock.assert_called_once_with(
            [plot_mock.return_value[0].__getitem__.return_value],
            plot_mock.return_value[1])
        self.assertEqual(tuple(plot_mock.return_value), values)


class MaskRowBlockTest(unittest.TestCase):

    @patch(Node_patch_path + '.load_config')
    @patch('numpy.savetxt')
    @patch(DAWN_patch_path + '.plot_image')
    def test_correct_calls_made(self, plot_mock, save_mock, load_mock):

        e = ExcaliburNode(1)
        e.full_array_shape = [4, 16]  # Make testing easier
        e.chip_size = 4  # Make testing easier
        e.num_chips = 4  # Make testing easier
        chips = [1, 2]
        expected_array = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        expected_subarray = np.array([[0., 0., 0., 0.],
                                      [1., 1., 1., 1.],
                                      [1., 1., 1., 1.],
                                      [0., 0., 0., 0.]])
        expected_filename = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/pixelmask.chip'

        e.mask_row_block(chips, 1, 2)

        # Check first save call
        self.assertEqual(expected_filename + '1', save_mock.call_args_list[0][0][0])
        np.testing.assert_array_equal(expected_subarray, save_mock.call_args_list[0][0][1])
        self.assertEqual(dict(delimiter=' ', fmt='%.18g'), save_mock.call_args_list[0][1])
        # Check second save call
        self.assertEqual(expected_filename + '2', save_mock.call_args_list[1][0][0])
        np.testing.assert_array_equal(expected_subarray, save_mock.call_args_list[1][0][1])
        self.assertEqual(dict(delimiter=' ', fmt='%.18g'), save_mock.call_args_list[1][1])
        # Check plot call
        np.testing.assert_array_equal(expected_array, plot_mock.call_args[0][0])

        load_mock.assert_called_once_with(chips)


@patch(Node_patch_path + '.set_thresh_energy')
class SetThreshold0Test(unittest.TestCase):

    def test_correct_calls_made(self, set_thresh_energy_mock):
        e = ExcaliburNode(1)

        e.set_threshold0('7')

        set_thresh_energy_mock.assert_called_once_with('0', 7.0)

    def test_correct_calls_made_with_default_param(self,
                                                   set_thresh_energy_mock):
        e = ExcaliburNode(1)

        e.set_threshold0()

        set_thresh_energy_mock.assert_called_once_with('0', 5.0)


@patch(Node_patch_path + '.expose')
@patch(Node_patch_path + '.set_dac')
class SetThreshold0DacTest(unittest.TestCase):

    def test_correct_calls_made(self, set_dac_mock, expose_mock):
        e = ExcaliburNode(1)
        chips = [1, 2, 3]

        e.set_threshold0_dac(chips, 1)

        set_dac_mock.assert_called_once_with(chips, 'Threshold0', 1)
        expose_mock.assert_called_once_with()

    def test_correct_calls_made_with_default_param(self, set_dac_mock,
                                                   expose_mock):
        e = ExcaliburNode(1)
        chips = [0, 1, 2, 3, 4, 5, 6, 7]

        e.set_threshold0_dac()

        set_dac_mock.assert_called_once_with(chips, 'Threshold0', 40)
        expose_mock.assert_called_once_with()


@patch('time.sleep')
@patch('numpy.genfromtxt')
@patch(Node_patch_path + '.set_dac')
@patch(Node_patch_path + '.expose')
class SetThreshEnergyTest(unittest.TestCase):

    def test_correct_calls_made(self, expose_mock, set_dac_mock, gen_mock,
                                sleep_mock):
        e = ExcaliburNode(1)
        expected_filepath = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/threshold0'

        e.set_thresh_energy('0', 7.0)

        gen_mock.assert_called_once_with(expected_filepath)
        self.assertEqual(8, set_dac_mock.call_count)
        self.assertEqual(2, expose_mock.call_count)

    def test_correct_calls_made_with_default_param(self, expose_mock,
                                                   set_dac_mock, gen_mock,
                                                   sleep_mock):
        e = ExcaliburNode(1)
        expected_filepath = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/threshold0'

        e.set_thresh_energy()

        gen_mock.assert_called_once_with(expected_filepath)
        self.assertEqual(8, set_dac_mock.call_count)
        sleep_mock.assert_called_with(0.2)
        self.assertEqual(2, expose_mock.call_count)


class SetDacsTest(unittest.TestCase):

    @patch(Node_patch_path + '.set_dac')
    def test_correct_calls_made(self, set_dac_mock):
        e = ExcaliburNode(1)
        chips = [0]
        expected_calls = [('Threshold1', 0), ('Threshold2', 0),
                          ('Threshold3', 0), ('Threshold4', 0),
                          ('Threshold5', 0), ('Threshold6', 0),
                          ('Threshold7', 0), ('Preamp', 175), ('Ikrum', 10),
                          ('Shaper', 150), ('Disc', 125), ('DiscLS', 100),
                          ('ShaperTest', 0), ('DACDiscL', 90), ('DACTest', 0),
                          ('DACDiscH', 90), ('Delay', 30), ('TPBuffIn', 128),
                          ('TPBuffOut', 4), ('RPZ', 255), ('TPREF', 128),
                          ('TPREFA', 500), ('TPREFB', 500)]

        e.set_dacs(chips)

        for index, call_args in enumerate(set_dac_mock.call_args_list):
            self.assertEqual((chips,) + expected_calls[index], call_args[0])


class TestAppCallsTest(unittest.TestCase):

    file_mock = MagicMock()

    def setUp(self):
        self.e = ExcaliburNode(1)

    @patch(ETAI_patch_path + '.read_chip_ids')
    def test_read_chip_id(self, read_mock):

        self.e.read_chip_ids()

        read_mock.assert_called_once_with()

    @patch(ETAI_patch_path + '.read_chip_ids')
    @patch('__builtin__.open', return_value=file_mock)
    def test_log_chip_id(self, _, read_mock):

        self.e.log_chip_id()

        read_mock.assert_called_once_with(stdout=self.file_mock.__enter__.return_value)

    @patch(ETAI_patch_path + '.read_slow_control_parameters')
    def test_monitor(self, read_mock):

        self.e.monitor()

        read_mock.assert_called_once_with()

    @patch(DAWN_patch_path + '.load_image_data')
    @patch(ETAI_patch_path + '.acquire')
    @patch('time.sleep')
    @patch(Node_patch_path + '.update_filename_index')
    def test_acquire(self, update_idx_mock, sleep_mock, acquire_mock,
                     load_mock):

        self.e._acquire(10, 100, burst=True)

        update_idx_mock.assert_called_once_with()
        acquire_mock.assert_called_once_with(
            [0, 1, 2, 3, 4, 5, 6, 7], 10, 100, trig_mode=0, gain_mode='shgm',
            burst=True, pixel_mode='spm', counter=0, equalization=0,
            disc_mode='discL', hdf_file='image_.hdf5', depth=12,
            read_mode='sequential')
        sleep_mock.assert_called_once_with(0.5)
        load_mock.assert_called_once_with('/tmp/image_.hdf5')

    @patch('time.asctime', return_value='Tue Sep 27 10:12:27 2016')
    @patch(DAWN_patch_path + '.plot_image')
    @patch(Node_patch_path + '._acquire')
    def test_expose(self, acquire_mock, plot_mock, _):

        self.e.expose()

        acquire_mock.assert_called_once_with(1, 100)
        plot_mock.assert_called_once_with(acquire_mock.return_value, name='Image_Tue Sep 27 10:12:27 2016')

    @patch('time.asctime', return_value='Tue Sep 27 10:12:27 2016')
    @patch(DAWN_patch_path + '.plot_image')
    @patch(Node_patch_path + '._acquire')
    def test_expose_with_exposure(self, acquire_mock, plot_mock, _):

        self.e.expose(200)

        acquire_mock.assert_called_once_with(1, 200)
        plot_mock.assert_called_once_with(acquire_mock.return_value, name='Image_Tue Sep 27 10:12:27 2016')

    @patch(Node_patch_path + '._acquire')
    def test_burst(self, acquire_mock):

        self.e.burst(10, 0.1)

        acquire_mock.assert_called_once_with(10, 0.1, burst=True)

    @patch('time.asctime', return_value='Tue Sep 27 10:12:27 2016')
    @patch(DAWN_patch_path + '.plot_image')
    @patch(Node_patch_path + '._acquire')
    def test_cont(self, acquire_mock, plot_mock, _):

        self.e.cont(100, 10)

        self.assertEqual("continuous", self.e.settings['readmode'])
        acquire_mock.assert_called_once_with(100, 10)
        plot_mock.assert_called_with(acquire_mock.return_value.__getitem__(), name=ANY)
        self.assertEqual(5, plot_mock.call_count)

    @patch(Node_patch_path + '._acquire')
    def test_cont_burst(self, acquire_mock):

        self.e.cont_burst(100, 10)

        self.assertEqual("continuous", self.e.settings['readmode'])
        acquire_mock.assert_called_once_with(100, 10, burst=True)

    @patch(ETAI_patch_path + '.load_dacs')
    @patch('__builtin__.open', return_value=file_mock)
    def test_set_dac(self, open_mock, load_mock):
        dac_file = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/dacs'
        expected_lines = ['Heading', 'Threshold0 = 40\r\n', 'Line2']
        readlines_mock = ['Heading', 'Line1', 'Line2']
        self.file_mock.__enter__.return_value.readlines.return_value = readlines_mock[:]  # Don't pass by reference
        chips = [0]

        self.e.set_dac(chips)

        load_mock.assert_called_once_with(chips, dac_file)
        # Check file writing calls
        open_mock.assert_called_with(dac_file, 'w')
        self.assertEqual(2 * len(chips), open_mock.call_count)
        self.file_mock.__enter__.return_value.writelines.assert_called_once_with(expected_lines)

    @patch(ETAI_patch_path + '.sense')
    def test_read_dac(self, sense_mock):
        expected_file = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/dacs'
        chips = [0]

        self.e.read_dac(chips, 'Threshold0')

        sense_mock.assert_called_once_with(chips, 'Threshold0', expected_file)

    @patch(ETAI_patch_path + '.load_config')
    @patch('numpy.savetxt')
    def test_load_config_bits(self, save_mock, load_mock):
        filepath = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/'
        expected_kwargs = dict(fmt='%.18g', delimiter=' ')
        expected_file_1 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/discLbits.tmp'
        expected_file_2 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/discHbits.tmp'
        expected_file_3 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/maskbits.tmp'
        chips = [0]
        discLbits = MagicMock()
        discHbits = MagicMock()
        maskbits = MagicMock()

        self.e.load_config_bits(chips, discLbits, discHbits, maskbits)

        # Check first save call
        call_args = save_mock.call_args_list[0]
        self.assertEqual((filepath + 'discLbits.tmp', discLbits), call_args[0])
        self.assertEqual(expected_kwargs, call_args[1])
        # Check second save call
        call_args = save_mock.call_args_list[1]
        self.assertEqual((filepath + 'discHbits.tmp', discHbits), call_args[0])
        self.assertEqual(expected_kwargs, call_args[1])
        # Check third save call
        call_args = save_mock.call_args_list[2]
        self.assertEqual((filepath + 'maskbits.tmp', maskbits), call_args[0])
        self.assertEqual(expected_kwargs, call_args[1])

        load_mock.assert_called_once_with([0], expected_file_1, expected_file_2, expected_file_3)

    # Make sure we always get the same random numbers
    rand = np.random.RandomState(123)

    @patch(Node_patch_path + '.load_config_bits')
    @patch(Node_patch_path + '._grab_chip_slice')
    @patch(Node_patch_path + '.open_discbits_file',
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
        load_mock.assert_called_once_with([0], grab_mock.return_value,
                                          grab_mock.return_value,
                                          grab_mock.return_value)

    @patch(Node_patch_path + '.load_config_bits')
    @patch(Node_patch_path + '._grab_chip_slice')
    @patch(Node_patch_path + '.open_discbits_file')
    def test_load_all_discbits_H(self, open_mock, grab_mock, load_mock):
        chips = [0]
        temp_bits_mock = MagicMock()
        mask_mock = MagicMock()

        self.e.load_all_discbits(chips, "discH", temp_bits_mock, mask_mock)

        open_mock.assert_called_once_with(chips, "discLbits")
        self.assertEqual((open_mock.return_value, 0), grab_mock.call_args_list[0][0])
        self.assertEqual((ANY, 0), grab_mock.call_args_list[1][0])
        self.assertEqual((mask_mock, 0), grab_mock.call_args_list[2][0])
        load_mock.assert_called_once_with([0], grab_mock.return_value,
                                          grab_mock.return_value,
                                          grab_mock.return_value)

    @patch(Node_patch_path + '.load_config_bits')
    @patch(Node_patch_path + '._grab_chip_slice')
    @patch(Node_patch_path + '.open_discbits_file')
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
        self.e = ExcaliburNode(1)
        self.chips = [0]

    def test_correct_calls_made(self, expose_mock, set_mock, load_mock):
        expected_file_1 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/discLbits.chip0'
        expected_file_2 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/discHbits.chip0'
        expected_file_3 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/pixelmask.chip0'

        self.e.load_config(self.chips)

        # Check first set_dac call
        call_args = set_mock.call_args_list[0]
        self.assertEqual((range(8), "Threshold1", 100), call_args[0])
        # Check second set_dac call
        call_args = set_mock.call_args_list[1]
        self.assertEqual((range(8), "Threshold0", 40), call_args[0])

        load_mock.assert_called_once_with(self.chips, expected_file_1, expected_file_2, expected_file_3)
        expose_mock.assert_called_once_with()


@patch('time.sleep')
@patch(Node_patch_path + '.load_config')
@patch(Node_patch_path + '.set_threshold0_dac')
@patch(Node_patch_path + '.expose')
class Fe55ImageRX001Test(unittest.TestCase):

    def test_correct_calls_made(self, expose_mock, set_threshhold_mock,
                                load_mock, sleep_mock):
        e = ExcaliburNode(1)
        chips = [1, 4, 6, 7]
        exposure_time = 10000

        e.fe55_image_rx001(chips, exposure_time)

        self.assertEqual('shgm', e.settings['gain'])
        self.assertEqual('spm', e.settings['mode'])

        load_mock.assert_called_once_with(chips)
        set_threshhold_mock.assert_called_once_with(chips, 40)
        sleep_mock.assert_called_once_with(0.5)
        expose_mock.assert_called_once_with(exposure_time)

    def test_correct_calls_made_with_default_params(self, expose_mock,
                                                    set_threshhold_mock,
                                                    load_mock, sleep_mock):
        e = ExcaliburNode(1)
        chips = [0, 1, 2, 3, 4, 5, 6, 7]
        exposure_time = 60000

        e.fe55_image_rx001()

        self.assertEqual('shgm', e.settings['gain'])
        self.assertEqual('spm', e.settings['mode'])

        load_mock.assert_called_once_with(chips)
        set_threshhold_mock.assert_called_once_with(chips, 40)
        sleep_mock.assert_called_once_with(0.5)
        expose_mock.assert_called_once_with(exposure_time)


@patch(DAWN_patch_path + '.load_image_data')
@patch(Node_patch_path + '.update_filename_index')
@patch(ETAI_patch_path + '.perform_dac_scan')
@patch('time.sleep')
class ScanDacTest(unittest.TestCase):

    def test_given_start_lower_than_stop(self, sleep_mock, scan_mock,
                                         update_mock, load_mock):
        dac_file = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem/spm/shgm/dacs'
        save_file = 'image_dacscan.hdf5'
        e = ExcaliburNode(1)
        chips = [0]
        dac_range = Range(1, 10, 1)

        e.scan_dac(chips, 'Threshold0', dac_range)

        update_mock.assert_called_once_with()
        scan_mock.assert_called_once_with(chips, 'Threshold0', Range(1, 10, 1),
                                          dac_file, save_file)
        sleep_mock.assert_called_once_with(1)
        load_mock.assert_called_once_with('/tmp/' + save_file)


class UpdateFilenameIndexTest(unittest.TestCase):

    file_mock = MagicMock()

    def setUp(self):
        self.e = ExcaliburNode(1)
        self.file_mock.read.return_value = 1

    def tearDown(self):
        self.file_mock.reset_mock()

    @patch('__builtin__.open', return_value=file_mock)
    @patch('os.path.isfile', return_value=True)
    def test_increment(self, _, open_mock):
        self.file_mock.read.return_value = 1

        self.e.update_filename_index()

        open_mock.assert_called_with('/tmp/image.idx', 'w')
        self.file_mock.__enter__.return_value.read.assert_called_once_with()
        self.file_mock.__enter__.return_value.write.assert_called_once_with('2')

    @patch('os.chmod', return_value=file_mock)
    @patch('__builtin__.open', return_value=file_mock)
    @patch('os.path.isfile', return_value=False)
    def test_create(self, _, open_mock, chmod_mock):
        self.file_mock.read.return_value = 1

        self.e.update_filename_index()

        open_mock.assert_called_with('/tmp/image.idx', 'w')
        self.assertFalse(self.file_mock.read.call_count)
        self.file_mock.__enter__.return_value.write.assert_called_once_with('0')
        self.assertEqual('0', self.e.settings['filenameIndex'])
        chmod_mock.assert_called_once_with('/tmp/image.idx', 0777)


class AcquireFFTest(unittest.TestCase):

    # Make sure we always get the same random numbers
    rand = np.random.RandomState(1234)
    mock_array = rand.randint(10, size=(256, 8 * 256))

    @patch(Node_patch_path + '.expose',
           return_value=mock_array)
    @patch(DAWN_patch_path + '.plot_image')
    def test_acquire_ff(self, plot_mock, expose_mock):
        e = ExcaliburNode(1)
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
    def test_apply_ff_correction(self, sleep_mock, load_mock, plot_mock):
        e = ExcaliburNode(1)

        e.apply_ff_correction(1, 0.1)

        load_mock.assert_called_once_with('/tmp/image_.hdf5')
        plot_mock.assert_called_once_with(load_mock.return_value.__getitem__().__mul__().__getitem__(), name='Image data Cor')
        sleep_mock.assert_called_once_with(1)

        # TODO: Finish once function completed


@patch('time.sleep')
@patch('time.asctime', return_value='Mon Sep 26 17:04:31 2016')
@patch('numpy.savetxt')
@patch(DAWN_patch_path + '.plot_image')
@patch(DAWN_patch_path + '.load_image_data')
@patch(ETAI_patch_path + '.acquire')
@patch(Node_patch_path + '.set_dac')
@patch(Node_patch_path + '.expose')
class LogoTestTest(unittest.TestCase):

    rand = np.random.RandomState(123)
    mock_array = rand.randint(2, size=(243, 1598))

    def setUp(self):
        self.e = ExcaliburNode(1)
        self.e.chip_range = [0]  # Make test easier - only run for one chip

        logo_tp = np.ones([256, 8*256])
        logo_tp[7:250, 225:1823] = self.mock_array
        logo_tp[logo_tp > 0] = 1
        self.logo_tp = 1 - logo_tp

    @patch(ETAI_patch_path + '.configure_test_pulse')
    @patch('numpy.loadtxt', return_value=mock_array)
    @patch('os.path.isfile', return_value=True)
    def test_logo_test_files_exist(self, _, load_mock, configure_mock,
                                   shoot_mock, set_mock, acquire_mock,
                                   load_image_mock, plot_mock, save_mock, _2,
                                   sleep_mock):
        expected_dac_file = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/dacs'
        mask_file = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/Logo_chip0_mask'
        disc_files = dict(disch='/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/discHbits.chip0',
                          pixelmask='/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/pixelmask.chip0',
                          discl='/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/discLbits.chip0')
        chips = [0]

        self.e.logo_test()

        set_mock.assert_called_once_with(chips, "Threshold0", 40)
        shoot_mock.assert_called_once_with(10)
        load_mock.assert_called_once_with('/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/config/logo.txt')

        configure_mock.assert_called_once_with(chips, expected_dac_file, mask_file, disc_files)
        acquire_mock.assert_called_once_with(chips, '1', '100', tp_count='100', hdf_file='image_.hdf5')
        sleep_mock.assert_called_once_with(0.2)

        self.assertEqual(save_mock.call_args[0][0], '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/Logo_chip0_mask')
        np.testing.assert_array_equal(self.logo_tp[0:256, 0:256], save_mock.call_args[0][1])
        self.assertEqual(save_mock.call_args[1], dict(fmt='%.18g', delimiter=' '))

        load_image_mock.assert_called_once_with('/tmp/image_.hdf5')
        plot_mock.assert_called_once_with(load_image_mock.return_value, name='Image_Mon Sep 26 17:04:31 2016')

    @patch(ETAI_patch_path + '.configure_test_pulse')
    @patch('numpy.loadtxt', return_value=mock_array)
    @patch('os.path.isfile', return_value=False)
    def test_logo_test_files_dont_exist(self, _, load_mock, configure_mock,
                                        shoot_mock, set_mock, acquire_mock,
                                        load_image_mock, plot_mock, save_mock,
                                        _2, sleep_mock):
        expected_dac_file = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/dacs'
        mask_file = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/Logo_chip0_mask'
        chips = [0]

        self.e.logo_test()

        set_mock.assert_called_once_with(chips, "Threshold0", 40)
        shoot_mock.assert_called_once_with(10)
        load_mock.assert_called_once_with('/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/config/logo.txt')

        configure_mock.assert_called_once_with(chips, expected_dac_file, mask_file, None)
        acquire_mock.assert_called_once_with(chips, '1', '100', tp_count='100', hdf_file='image_.hdf5')
        sleep_mock.assert_called_once_with(0.2)

        self.assertEqual(save_mock.call_args[0][0], '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/Logo_chip0_mask')
        np.testing.assert_array_equal(self.logo_tp[0:256, 0:256], save_mock.call_args[0][1])
        self.assertEqual(save_mock.call_args[1], dict(fmt='%.18g', delimiter=' '))

        load_image_mock.assert_called_once_with('/tmp/image_.hdf5')
        plot_mock.assert_called_once_with(load_image_mock.return_value, name='Image_Mon Sep 26 17:04:31 2016')


class TestPulseTest(unittest.TestCase):

    # TODO: Test other if branch?

    def setUp(self):
        self.e = ExcaliburNode(1)

    @patch('time.asctime', return_value='Tue Sep 27 10:43:52 2016')
    @patch(DAWN_patch_path + '.plot_image')
    @patch(DAWN_patch_path + '.load_image_data')
    @patch(ETAI_patch_path + '.acquire')
    @patch(ETAI_patch_path + '.configure_test_pulse')
    @patch(Node_patch_path + '.update_filename_index')
    @patch('numpy.savetxt')
    def test_test_pulse(self, save_mock,  update_mock, configure_mock,
                        acquire_mock, load_mock, plot_mock, _):
        expected_dac_file = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/dacs'
        mask_file = 'excaliburRx/config/triangle.mask'
        chips = [0]

        self.e.test_pulse(chips, mask_file, 1000)

        configure_mock.assert_called_once_with(chips, expected_dac_file, mask_file)
        update_mock.assert_called_once_with()
        acquire_mock.assert_called_once_with(chips, '1', '100', tp_count='1000',
                                             hdf_file='image_.hdf5')
        load_mock.assert_called_once_with('/tmp/image_.hdf5')

        plot_mock.assert_called_once_with(load_mock.return_value, name='Image_Tue Sep 27 10:43:52 2016')


class SaveDiscbitsTest(unittest.TestCase):

    @patch('numpy.savetxt')
    def test_correct_call_made(self, save_mock):
        discbits = np.random.randint(10, size=(256, 8*256))
        expected_subarray = discbits[0:256, 0:256]
        e = ExcaliburNode(1)

        e.save_discbits([0], discbits, 'test')

        call_args = save_mock.call_args[0]
        call_kwargs = save_mock.call_args[1]
        self.assertEqual('/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/test.chip0',
                         call_args[0])
        self.assertTrue((expected_subarray == call_args[1]).all())
        self.assertEqual(dict(fmt='%.18g', delimiter=' '), call_kwargs)


@patch(DAWN_patch_path + '.plot_image')
@patch('numpy.savetxt')
@patch(ETAI_patch_path + '.load_config')
class MaskUnmaskTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode(1)

    def test_mask_col(self, load_mock, save_mock, plot_mock):
        expected_file_1 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/discLbits.chip0'
        expected_file_2 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/pixelmask.chip0'
        expected_mask = np.zeros([256, 8*256])
        expected_mask[:, 1] = 1
        expected_submask = expected_mask[0:256, 0:256]

        self.e.mask_col(0, 1)

        self.assertEqual('/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/pixelmask.chip0', save_mock.call_args[0][0])
        np.testing.assert_array_equal(expected_submask, save_mock.call_args[0][1])
        self.assertEqual(dict(fmt='%.18g', delimiter=' '), save_mock.call_args[1])

        load_mock.assert_called_once_with([0], expected_file_1, pixelmask=expected_file_2)

        np.testing.assert_array_equal(expected_mask, plot_mock.call_args[0][0])
        self.assertEqual(dict(name='Bad pixels'), plot_mock.call_args[1])

    def test_mask_sup_col(self, load_mock, save_mock, plot_mock):
        expected_file_1 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/discLbits.chip0'
        expected_file_2 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/pixelmask.chip0'
        expected_mask = np.zeros([256, 8*256])
        expected_mask[:, 32:32 + 64] = 1
        expected_submask = expected_mask[0:256, 0:256]

        self.e.mask_super_column(0, 1)

        self.assertEqual('/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/pixelmask.chip0', save_mock.call_args[0][0])
        np.testing.assert_array_equal(expected_submask, save_mock.call_args[0][1])
        self.assertEqual(dict(fmt='%.18g', delimiter=' '), save_mock.call_args[1])

        load_mock.assert_called_once_with([0], expected_file_1, pixelmask=expected_file_2)

        np.testing.assert_array_equal(expected_mask, plot_mock.call_args[0][0])
        self.assertEqual(dict(name='Bad Pixels'), plot_mock.call_args[1])

    def test_mask_pixels(self, load_mock, save_mock, plot_mock):
        expected_file_1 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/discLbits.chip0'
        expected_file_2 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/pixelmask.chip0'
        image_data = np.random.randint(10, size=(256, 8*256))
        expected_mask = image_data > 7

        self.e.mask_pixels([0], image_data, 7)

        np.testing.assert_array_equal(expected_mask, plot_mock.call_args[0][0])
        self.assertEqual(dict(name='Bad pixels'), plot_mock.call_args[1])

        self.assertEqual('/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/pixelmask.chip0', save_mock.call_args[0][0])
        np.testing.assert_array_equal(expected_mask[0:256, 0:256], save_mock.call_args[0][1])
        self.assertEqual(dict(fmt='%.18g', delimiter=' '), save_mock.call_args[1])

        load_mock.assert_called_once_with([0], expected_file_1, pixelmask=expected_file_2)

    mock_scan_data = np.random.randint(3, size=(3, 256, 8*256))

    @patch(Node_patch_path + '.scan_dac',
           return_value=mock_scan_data)
    def test_mask_pixels_using_dac_scan(self, scan_mock, load_mock, save_mock,
                                        plot_mock):
        mask = self.mock_scan_data.sum(0) > 1

        self.e.mask_pixels_using_dac_scan([0])

        scan_mock.assert_called_once_with([0], "Threshold0", (20, 120, 2))

        np.testing.assert_array_equal(mask, plot_mock.call_args[0][0])
        self.assertEqual(dict(name='Bad Pixels'), plot_mock.call_args[1])

        self.assertEqual(save_mock.call_args[0][0], '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/pixelmask.chip0')
        np.testing.assert_array_equal(mask[0:256, 0:256], save_mock.call_args[0][1])
        self.assertEqual(dict(fmt='%.18g', delimiter=' '), save_mock.call_args[1])

    def test_unmask_all_pixels(self, load_mock, save_mock, _):
        expected_file_1 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/discLbits.chip0'
        expected_file_2 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/pixelmask.chip0'
        expected_mask = np.zeros([256, 256])

        self.e.unmask_all_pixels([0])

        self.assertEqual(save_mock.call_args[0][0], '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/pixelmask.chip0')
        np.testing.assert_array_equal(expected_mask, save_mock.call_args[0][1])
        self.assertEqual(dict(fmt='%.18g', delimiter=' '), save_mock.call_args[1])

        load_mock.assert_called_once_with([0], expected_file_1, pixelmask=expected_file_2)

    def test_unequalize_pixels(self, load_mock, save_mock, _):
        expected_file_1 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/discL_bits.chip0'
        expected_file_2 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/pixelmask.chip0'
        expected_mask = np.zeros([256, 256])

        self.e.unequalize_all_pixels([0])

        self.assertEqual(save_mock.call_args[0][0], '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/discL_bits.chip0')
        np.testing.assert_array_equal(expected_mask, save_mock.call_args[0][1])
        self.assertEqual(dict(fmt='%.18g', delimiter=' '), save_mock.call_args[1])

        load_mock.assert_called_once_with([0], expected_file_1, pixelmask=expected_file_2)


@patch('shutil.copytree')
@patch('shutil.copy')
@patch('os.makedirs')
class CheckCalibDirTest(unittest.TestCase):

    @patch('os.path.isfile', return_value=False)
    @patch('os.path.isdir', return_value=False)
    def test_doesnt_exist_then_create(self, isdir_mock, isfile_mock,
                                      make_mock, copy_mock, copy_tree_mock):
        expected_path = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm'
        e = ExcaliburNode(1)

        e.check_calib_dir()

        isdir_mock.assert_called_once_with(expected_path)
        isfile_mock.assert_called_once_with(expected_path + '/dacs')
        make_mock.assert_called_once_with(expected_path)
        copy_mock.assert_called_once_with('/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/config/dacs',
                                          expected_path)
        self.assertFalse(copy_tree_mock.call_count)

    @patch('time.asctime', return_value='Fri Sep 16 14:59:18 2016')
    @patch('os.path.isfile', return_value=True)
    @patch('os.path.isdir', return_value=True)
    def test_does_exist_then_backup(self, isdir_mock, isfile_mock, _,
                                    make_mock, copy_mock, copy_tree_mock):
        expected_path = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm'
        e = ExcaliburNode(1)

        e.check_calib_dir()

        isdir_mock.assert_called_once_with(expected_path)
        isfile_mock.assert_called_once_with(expected_path + '/dacs')
        self.assertFalse(make_mock.call_count)
        copy_tree_mock.assert_called_once_with('/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib',
                                               '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib_backup_Fri Sep 16 14:59:18 2016')
        self.assertFalse(copy_mock.call_count)


@patch('shutil.copytree')
@patch('shutil.rmtree')
class CopySLGMIntoOtherGainModesTest(unittest.TestCase):

    @patch('os.path.exists', return_value=True)
    def test_exist_then_rm_and_copy(self, exists_mock, rm_mock, copytree_mock):
        expected_path = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/'
        e = ExcaliburNode(1)

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
        expected_path = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/'
        e = ExcaliburNode(1)

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


class OpenDiscbitsFileTest(unittest.TestCase):

    mock_array = np.random.randint(10, size=(256, 256))

    @patch('numpy.loadtxt', return_value=mock_array)
    def test_correct_call_made(self, load_mock):
        expected_path = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/test_file.chip0'
        e = ExcaliburNode(1)

        value = e.open_discbits_file([0], 'test_file')

        load_mock.assert_called_once_with(expected_path)
        self.assertTrue((value[0:256, 0:256] == self.mock_array).all())


class CombineROIsTest(unittest.TestCase):

    # Make sure we always get the same random numbers
    rand = np.random.RandomState(123)

    @patch(DAWN_patch_path + '.plot_image')
    @patch(Node_patch_path + '.roi',
           return_value=np.ones([256, 256]))
    @patch(Node_patch_path + '.save_discbits')
    @patch(Node_patch_path + '.open_discbits_file',
           side_effect=[rand.randint(2, size=[256, 256]),
                        rand.randint(2, size=[256, 256])])
    def test_correct_calls_made(self, open_mock, save_mock, roi_mock,
                               plot_mock):
        expected_path = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem1/spm/shgm/test_file.chip0'
        e = ExcaliburNode(1)

        value = e.combine_rois([0], 'test_file', 1, 'rect')

        roi_mock.assert_called_once_with([0], 0, 1, 'rect')
        # plot_mock.assert_called_once_with()
        # save_mock.assert_called_once_with()
        # plot_mock.assert_called_once_with()

        # TODO: Finish once sure what function should do


class FindTest(unittest.TestCase):  # TODO: Improve

    def setUp(self):
        self.e = ExcaliburNode(1)

        # Make sure we always get the same random numbers
        self.rand = np.random.RandomState(123)

    # TODO: Why do these get the same expected array?

    @patch(Node_patch_path + '._display_histogram')
    def test_find_edge_low_to_high(self, display_mock):
        dac_scan_data = self.rand.randint(10, size=(3, 3, 3))
        dac_range = MagicMock(start=1, stop=10, step=1)
        expected_array = [[10, 9, 10], [10, 9, 8], [10, 10, 10]]

        value = self.e.find_edge([0], dac_scan_data, dac_range, 7)

        display_mock.assert_called_once_with([0], ANY)
        np.testing.assert_array_equal(expected_array, display_mock.call_args[0][1])
        np.testing.assert_array_equal(expected_array, value)

    @patch(Node_patch_path + '._display_histogram')
    def test_find_edge_high_to_low(self, display_mock):
        dac_scan_data = self.rand.randint(10, size=(3, 3, 3))
        dac_range = MagicMock(start=10, stop=1, step=1)
        expected_array = [[10, 9, 10], [10, 9, 10], [10, 10, 10]]

        value = self.e.find_edge([0], dac_scan_data, dac_range, 7)

        display_mock.assert_called_once_with([0], ANY)
        np.testing.assert_array_equal(expected_array, display_mock.call_args[0][1])
        np.testing.assert_array_equal(expected_array, value)

    @patch(Node_patch_path + '._display_histogram')
    def test_find_max(self, display_mock):
        dac_scan_data = self.rand.randint(10, size=(3, 3, 3))
        dac_range = MagicMock(start=1, stop=10, step=1)
        expected_array = [[10, 9, 10], [10, 9, 8], [10, 10, 10]]

        value = self.e.find_max([0], dac_scan_data, dac_range)

        display_mock.assert_called_once_with([0], ANY)
        np.testing.assert_array_equal(expected_array, display_mock.call_args[0][1])
        np.testing.assert_array_equal(expected_array, value)

    @patch(Node_patch_path + '._grab_chip_slice')
    @patch(DAWN_patch_path + '.plot_histogram')
    @patch(DAWN_patch_path + '.plot_image')
    def test_display_histogram(self, plot_mock, plot_histo_mock, grab_mock):
        mock_array = MagicMock()

        self.e._display_histogram([0], mock_array)

        np.testing.assert_array_equal(mock_array, plot_mock.call_args[0][0])
        self.assertEqual(dict(name="noise edges"), plot_mock.call_args[1])
        self.assertEqual([grab_mock.return_value], plot_histo_mock.call_args[0][0])


class OptimizeDacDiscTest(unittest.TestCase):

    rand = np.random.RandomState(1234)

    def setUp(self):
        self.e = ExcaliburNode(1)

    @patch(Node_patch_path + '.set_dac')
    @patch(Node_patch_path + '._optimize_dac_disc')
    def test_optimize_disc_l(self, optimize_mock, set_mock):
        chips = [0]
        roi_mock = np.ones([256, 8*256])

        self.e.optimize_disc_l(chips, roi_mock)

        set_mock.assert_called_once_with(chips, "Threshold1", 0)
        self.assertEqual('0', self.e.settings['disccsmspm'])
        optimize_mock.assert_called_once_with(chips, "DACDiscL", roi_mock)

    @patch(Node_patch_path + '.set_dac')
    @patch(Node_patch_path + '._optimize_dac_disc')
    def test_optimize_disc_h(self, optimize_mock, set_mock):
        chips = [0]
        roi_mock = np.ones([256, 8*256])

        self.e.optimize_disc_h(chips, roi_mock)

        set_mock.assert_called_once_with(chips, "Threshold0", 60)
        self.assertEqual('1', self.e.settings['disccsmspm'])
        optimize_mock.assert_called_once_with(chips, "DACDiscH", roi_mock)

    @patch(Node_patch_path + '._display_optimization_results')
    @patch(Node_patch_path + '.set_dac')
    @patch(DAWN_patch_path + '.clear_plot')
    @patch(DAWN_patch_path + '.plot_linear_fit', return_value=(1, 1))
    @patch(DAWN_patch_path + '.add_plot_line')
    @patch(Node_patch_path + '.load_all_discbits')
    @patch(Node_patch_path + '._chip_dac_scan', return_value=([1], [1]))
    def test_optimize_dac_disc(self, scan_mock, load_mock, add_mock, plot_mock,
                               clear_mock, set_mock, display_mock):
        chips = [0]
        roi_mock = np.ones([256, 8*256])

        self.e._optimize_dac_disc(chips, "DACDiscL", roi_mock)

        self.assertEqual(2, load_mock.call_count)
        self.assertEqual(4, scan_mock.call_count)
        self.assertEqual(3, clear_mock.call_count)
        self.assertEqual(3, add_mock.call_count)
        self.assertEqual(1, plot_mock.call_count)
        self.assertEqual(1, set_mock.call_count)
        self.assertEqual(1, display_mock.call_count)

    @patch(DAWN_patch_path + '.plot_gaussian_fit', return_value=(0, 0))
    @patch(Node_patch_path + '.set_dac')
    @patch(Node_patch_path + '.scan_dac',
           return_value=rand.randint(10, size=(3, 256, 8*256)))
    @patch(Node_patch_path + '.find_max')
    def test_chip_dac_scan(self, find_mock, scan_mock, set_mock, plot_mock):
        chips = [0]
        expected_message = "Histogram of edges when scanning DAC_disc for discbit = 0"
        range = MagicMock(start=0, stop=150, step=50)

        self.e._chip_dac_scan(chips, "DACDiscL", 1, range, 0, [5000, 0, 30])

        set_mock.assert_called_once_with(chips, "DACDiscL", 1)
        scan_mock.assert_called_once_with(chips, "DACDiscL", range)
        find_mock.assert_called_once_with(chips, scan_mock.return_value,
                                          range)
        plot_mock.assert_called_once_with([find_mock.return_value.__getitem__.return_value], expected_message, [5000, 0, 30], 3)

    def test_display_optimization_results(self):
        chips = [0]

        self.e._display_optimization_results(chips, MagicMock(), MagicMock(),
                                             MagicMock(), [1]*8)


class EqualizeDiscbitsTest(unittest.TestCase):

    rand = np.random.RandomState(1234)

    def setUp(self):
        self.e = ExcaliburNode(1)
        self.e.chip_size = 4
        self.e.num_chips = 4
        self.e.full_array_shape = [4, 16]

    @patch(Node_patch_path + '._equalise_discbits')
    @patch(Node_patch_path + '.set_dac')
    def test_equalize_disc_l(self, set_mock, equalise_mock):
        roi = MagicMock()
        self.e.equalize_disc_l([0], roi, "Method")

        set_mock.assert_called_once_with([0], "Threshold1", 0)
        equalise_mock.assert_called_once_with([0], "DACDiscL", "Threshold0", roi, "Method")
        self.assertEqual("0", self.e.settings['disccsmspm'])

    @patch(Node_patch_path + '._equalise_discbits')
    @patch(Node_patch_path + '.set_dac')
    def test_equalize_disc_h(self, set_mock, equalise_mock):
        roi = MagicMock()
        self.e.equalize_disc_h([0], roi, "Method")

        set_mock.assert_called_once_with([0], "Threshold0", 60)
        equalise_mock.assert_called_once_with([0], "DACDiscH", "Threshold1", roi, "Method")
        self.assertEqual("1", self.e.settings['disccsmspm'])

    @patch(Node_patch_path + '.load_config')
    @patch(Node_patch_path + '._display_histogram')
    @patch(DAWN_patch_path + '.clear_plot')
    @patch(DAWN_patch_path + '.plot_image')
    @patch(Node_patch_path + '.find_max',
           return_value=rand.randint(2, size=(4, 16)))
    @patch(Node_patch_path + '.scan_dac',
           return_value=rand.randint(10, size=(3, 4, 16)))
    @patch(Node_patch_path + '.load_all_discbits')
    def test_equalize_discbits(self, load_mock, scan_mock, find_mock,
                               plot_mock, clear_mock, histo_mock,
                               load_config_mock):
        chips = [0]
        roi = np.zeros([4, 16])

        self.e._equalise_discbits(chips, "DACDiscL", "Threshold0", roi, "Stripes")

        self.assertEqual(32 + 1, scan_mock.call_count)
        self.assertEqual(([0], "DACDiscL", ANY, ANY), load_mock.call_args_list[0][0])
        np.testing.assert_array_equal(roi, load_mock.call_args_list[0][0][3])
        self.assertEqual(([0], "Threshold0", (0, 20, 2)), scan_mock.call_args_list[0][0])
        self.assertEqual(([0], scan_mock.return_value, Range(0, 20, 2)), find_mock.call_args_list[0][0])
        self.assertEqual((ANY,), plot_mock.call_args_list[0][0])
        self.assertEqual(dict(name="discbits"), plot_mock.call_args_list[0][1])
        self.assertEqual(("Histogram of Final Discbits",), clear_mock.call_args_list[0][0])
        self.assertEqual(("Histogram of Final Discbits",), clear_mock.call_args_list[0][0])
        self.assertEqual((chips, ANY), histo_mock.call_args_list[0][0])
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
        e = ExcaliburNode(1)
        chips = [0]

        # e.check_calib(chips, [0, 10, 1])
        #
        # load_mock.assert_called_once_with(chips)


@patch(DAWN_patch_path + '.plot_image')
class ROITest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode(1)

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


class CalibrateDiscTest(unittest.TestCase):

    @patch(Node_patch_path + '._optimize_dac_disc')
    @patch(Node_patch_path + '.roi')
    @patch(Node_patch_path + '._equalise_discbits')
    @patch(Node_patch_path + '.save_discbits')
    @patch(Node_patch_path + '.combine_rois')
    @patch(Node_patch_path + '.load_config')
    @patch(Node_patch_path + '.copy_slgm_into_other_gain_modes')
    def test_correct_calls_made(self, copy_mock, load_mock, combine_rois,
                                save_mock, equalize_mock, roi_mock, opt_mock):
        e = ExcaliburNode(1)
        chips = [0]

        e.calibrate_disc(chips, 'discL', 1, 'rect')

        opt_mock.assert_called_once_with(chips, 'discL', roi_full_mask=1 - roi_mock.return_value)
        equalize_mock.assert_called_once_with(chips, 'discL', 1 - roi_mock.return_value, 'stripes')
        self.assertEqual(save_mock.call_args_list[0][0], (chips, equalize_mock.return_value, 'discLbits_roi_0'))
        combine_rois.assert_called_once_with(chips, 'discL', 1, 'rect')
        self.assertEqual(save_mock.call_args_list[1][0], (chips, combine_rois.return_value, 'discLbits'))
        load_mock.assert_called_once_with(chips)
        copy_mock.assert_called_once_with()


class LoopTest(unittest.TestCase):

    @patch(DAWN_patch_path + '.plot_image')
    @patch(Node_patch_path + '.expose')
    def test_correct_calls_made(self, expose_mock, plot_mock):
        e = ExcaliburNode(1)

        e.loop(1)

        expose_mock.assert_called_once_with()
        plot_mock.assert_called_once_with(expose_mock.return_value.__add__(), name='Sum')


class CSMTest(unittest.TestCase):

    @patch(Node_patch_path + '.expose')
    @patch(Node_patch_path + '.load_config')
    @patch(Node_patch_path + '.set_dac')
    def test_correct_calls_made(self, set_mock, load_mock, expose_mock):
        e = ExcaliburNode(1)
        chips = range(8)

        e.csm()

        self.assertEqual('csm', e.settings['mode'])
        self.assertEqual('slgm', e.settings['gain'])
        self.assertEqual('1', e.settings['counter'])

        self.assertEqual(set_mock.call_args_list[0][0], (range(8), 'Threshold0', 200))
        self.assertEqual(set_mock.call_args_list[1][0], (range(8), 'Threshold1', 200))
        load_mock.assert_called_once_with(chips)
        self.assertEqual(set_mock.call_args_list[2][0], (range(8), 'Threshold0', 45))
        self.assertEqual(set_mock.call_args_list[3][0], (range(8), 'Threshold1', 100))
        self.assertEqual(2, expose_mock.call_count)


class SetGNDFBKCasExcaliburRX001Test(unittest.TestCase):

    @patch(Node_patch_path + '.set_dac')
    def test_correct_calls_made(self, set_mock):
        e = ExcaliburNode(1)
        chips = [0]

        e.set_gnd_fbk_cas_excalibur_rx001(chips, 1)

        self.assertEqual(set_mock.call_args_list[0][0], ([0], 'GND', 141))
        self.assertEqual(set_mock.call_args_list[1][0], ([0], 'FBK', 190))
        self.assertEqual(set_mock.call_args_list[2][0], ([0], 'Cas', 178))


class RotateTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode(1)

    @patch('numpy.rot90')
    @patch('numpy.savetxt')
    @patch('numpy.loadtxt')
    def test_rotate_config(self, load_mock, save_mock, rotate_mock):
        test_path = 'path/to/config'

        self.e.rotate_config(test_path)

        load_mock.assert_called_once_with(test_path)
        rotate_mock.assert_called_once_with(load_mock.return_value, 2)
        save_mock.assert_called_once_with(test_path, rotate_mock.return_value,
                                          fmt='%.18g', delimiter=' ')

    @patch('shutil.copytree')
    @patch(Node_patch_path + '.rotate_config')
    @patch('os.path.isfile', return_value=True)
    def test_rotate_config_files_exist(self, _, rotate_mock, copy_mock):
        self.e.num_chips = 1  # Make test easier
        root_path = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/'
        expected_calib_path = root_path + 'calib'
        expected_epics_path = root_path + 'calib_epics'
        expected_discL_path = expected_epics_path + '/fem1/spm/slgm/discLbits.chip0'
        expected_discH_path = expected_epics_path + '/fem1/spm/slgm/discHbits.chip0'
        expected_mask_path = expected_epics_path + '/fem1/spm/slgm/pixelmask.chip0'

        self.e.rotate_all_configs()

        copy_mock.assert_called_once_with(expected_calib_path, expected_epics_path)
        self.assertEqual(rotate_mock.call_args_list[0][0][0], expected_discL_path)
        self.assertEqual(rotate_mock.call_args_list[1][0][0], expected_discH_path)
        self.assertEqual(rotate_mock.call_args_list[2][0][0], expected_mask_path)
        self.assertEqual(9, rotate_mock.call_count)


class SliceGrabSetTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode(1)

    @patch(util_patch_path + '.grab_slice')
    @patch(Node_patch_path + '._generate_chip_range')
    def test_grab_chip_slice(self, generate_mock, grab_mock):
        array = MagicMock()
        generate_mock.return_value = MagicMock(), MagicMock

        value = self.e._grab_chip_slice(array, 1)

        generate_mock.assert_called_once_with(1)
        grab_mock.assert_called_once_with(array, generate_mock.return_value[0], generate_mock.return_value[1])
        self.assertEqual(grab_mock.return_value, value)

    @patch(util_patch_path + '.set_slice')
    @patch(Node_patch_path + '._generate_chip_range')
    def test_set_chip_slice(self, generate_mock, set_mock):
        array = np.array([[1, 2, 3, 4, 5],
                          [10, 20, 30, 40, 50],
                          [100, 200, 300, 400, 500]])
        generate_mock.return_value = MagicMock(), MagicMock

        self.e._set_chip_slice(array, 1, 0)

        generate_mock.assert_called_once_with(1)
        set_mock.assert_called_once_with(array, generate_mock.return_value[0], generate_mock.return_value[1], 0)

    def test_generate_chip_range(self):
        expected_start = [0, 256]
        expected_stop = [255, 511]

        start, stop = self.e._generate_chip_range(1)

        np.testing.assert_array_equal(expected_start, start)
        np.testing.assert_array_equal(expected_stop, stop)


class ToListTest(unittest.TestCase):

    def test_to_list_given_value_then_return_list(self):
        response = ExcaliburNode._to_list(1)

        self.assertEqual([1], response)

    def test_to_list_given_list_then_return(self):
        response = ExcaliburNode._to_list([1])

        self.assertEqual([1], response)
