import unittest

from pkg_resources import require
require("mock")
from mock import patch, MagicMock, ANY

from excaliburcalibrationdawn.excaliburnode import ExcaliburNode, np, Range
ERX_patch_path = "excaliburcalibrationdawn.excaliburnode.ExcaliburNode"
ETAI_patch_path = "excaliburcalibrationdawn.excaliburnode.ExcaliburTestAppInterface"
ED_patch_path = "excaliburcalibrationdawn.excaliburnode.ExcaliburDAWN"


class InitTest(unittest.TestCase):

    def setUp(self):
        self.node = 2
        self.e = ExcaliburNode(self.node)

    def test_class_attributes_set(self):
        self.assertEqual(self.e.dac_target, 10)
        self.assertEqual(self.e.num_sigma, 3.2)
        self.assertEqual(self.e.allowed_delta, 4)
        self.assertEqual(self.e.calib_dir, "/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib")
        self.assertEqual(self.e.config_dir, "/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/config")
        self.assertEqual(self.e.settings, {'mode': 'spm',
                                           'gain': 'shgm',
                                           'bitdepth': '12',
                                           'readmode': '0',
                                           'counter': '0',
                                           'disccsmspm': '0',
                                           'equalization': '0',
                                           'trigmode': '0',
                                           'acqtime': '100',
                                           'frames': '1',
                                           'imagepath': '/tmp/',
                                           'filename': 'image',
                                           'Threshold': 'Not set',
                                           'filenameIndex': ''})
        self.assertEqual(self.e.dac_number, {'Threshold0': '1',
                                             'Threshold1': '2',
                                             'Threshold2': '3',
                                             'Threshold3': '4',
                                             'Threshold4': '5',
                                             'Threshold5': '6',
                                             'Threshold6': '7',
                                             'Threshold7': '8',
                                             'Preamp': '9',
                                             'Ikrum': '10',
                                             'Shaper': '11',
                                             'Disc': '12',
                                             'DiscLS': '13',
                                             'ShaperTest': '14',
                                             'DACDiscL': '15',
                                             'DACTest': '16',
                                             'DACDiscH': '17',
                                             'Delay': '18',
                                             'TPBuffIn': '19',
                                             'TPBuffOut': '20',
                                             'RPZ': '21',
                                             'GND': '22',
                                             'TPREF': '23',
                                             'FBK': '24',
                                             'Cas': '25',
                                             'TPREFA': '26',
                                             'TPREFB': '27'})
        self.assertEqual(self.e.chip_range, [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(self.e.plot_name, '')

    def test_instance_attributes_set(self):
        self.assertEqual(self.e.fem, self.node)
        self.assertEqual(self.e.ipaddress, "192.168.0.105")

    def test_given_node_0_then_ip_overidden(self):
        e = ExcaliburNode(0)
        self.assertEqual(e.ipaddress, "192.168.0.106")

    def test_given_node_invalid_node(self):
        e = ExcaliburNode(8)
        self.assertEqual(e.ipaddress, "192.168.0.10-1")


@patch(ERX_patch_path + '.check_calib_dir')
@patch(ERX_patch_path + '.log_chip_id')
@patch(ERX_patch_path + '.set_dacs')
@patch(ERX_patch_path + '.set_gnd_fbk_cas_excalibur_rx001')
@patch(ERX_patch_path + '.calibrate_disc')
class ThresholdEqualizationTest(unittest.TestCase):

    def test_correct_calls_made(self, cal_disc_mock, set_gnd_mock,
                                set_dacs_mock, log_mock, check_mock):
        e = ExcaliburNode()
        chips = [1, 4, 6, 7]

        e.threshold_equalization(chips)

        self.assertEqual('slgm', e.settings['gain'])
        self.assertEqual('spm', e.settings['mode'])

        check_mock.assert_called_once_with()
        log_mock.assert_called_once_with()
        set_dacs_mock.assert_called_once_with(chips)
        set_gnd_mock.assert_called_once_with(chips, e.fem)
        cal_disc_mock.assert_called_once_with(chips, 'discL')


@patch(ERX_patch_path + '.check_calib_dir')
@patch(ERX_patch_path + '.save_kev2dac_calib')
class ThresholdCalibrationTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode()

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

    @patch(ERX_patch_path + '.threshold_calibration')
    def test_threshold_calibration_all_gains(self, calibration_mock, _, _2):
        self.e.threshold_calibration_all_gains()

        self.assertEqual(4, calibration_mock.call_count)

    def test_one_energy_thresh_calib(self, save_mock, _):
        # TODO: Test actual values used once not hard-coded in
        expected_slope = np.array([9.0]*8)
        expected_offset = np.array([10.0]*8)

        self.e.settings['gain'] = 'slgm'
        self.e.one_energy_thresh_calib('0.1')

        self.assertEqual(save_mock.call_args[0][0], '0.1')
        np.testing.assert_array_almost_equal(expected_slope, save_mock.call_args[0][1][0])
        self.assertTrue((save_mock.call_args[0][2] == expected_offset).all())

    @patch(ED_patch_path + '.plot_linear_fit', return_value=(10, 2))
    def test_multiple_energy_thresh_calib(self, plot_mock, save_mock, _):
        expected_slope = np.array([2.0] + [0.0]*7)
        expected_offset = np.array([10.0] + [0.0]*7)

        self.e.settings['gain'] = 'slgm'
        self.e.multiple_energy_thresh_calib([0], '0.1')

        plot_mock.assert_called_once_with(ANY, ANY, [0, 1], name='DAC vs Energy', clear=True)

        np.testing.assert_array_almost_equal(expected_slope, save_mock.call_args[0][1])
        np.testing.assert_array_almost_equal(expected_offset, save_mock.call_args[0][2])


class SaveKev2DacCalibTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode()
        self.threshold = '0'
        self.gain = [1.1, 0.7, 1.1, 1.3, 1.0, 0.9, 1.2, 0.9]
        self.offset = [0.2, -0.7, 0.1, 0.0, 0.3, -0.1, 0.2, 0.5]
        self.expected_array = np.array([self.gain, self.offset])
        self.expected_filename = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/threshold0'

    @patch('numpy.savetxt')
    @patch('os.chmod')
    def test_given_file_then_save(self, chmod_mock, save_mock):

        self.e.save_kev2dac_calib(self.threshold, self.gain, self.offset)

        self.assertEqual(self.expected_filename, save_mock.call_args[0][0])
        np.testing.assert_array_equal(self.expected_array, save_mock.call_args[0][1])
        self.assertEqual(dict(fmt='%.2f'), save_mock.call_args[1])
        chmod_mock.assert_called_once_with(self.expected_filename, 0777)


@patch(ERX_patch_path + '.load_config')
class FindXrayEnergyDacTest(unittest.TestCase):

    mock_scan_data = np.random.randint(250, size=(3, 256, 8*256))
    mock_scan_range = MagicMock()

    @patch(ED_patch_path + '.fit_dac_scan')
    @patch(ED_patch_path + '.plot_dac_scan',
           return_value=[MagicMock(), MagicMock()])
    @patch(ERX_patch_path + '.scan_dac',
           return_value=[mock_scan_data.copy(), mock_scan_range])
    def test_correct_calls_made(self, scan_mock, plot_mock, fit_mock,
                                load_mock):
        e = ExcaliburNode()
        chips = range(8)
        expected_array = self.mock_scan_data
        expected_array[expected_array > 200] = 0

        values = e.find_xray_energy_dac()

        load_mock.assert_called_once_with(chips)
        scan_mock.assert_called_once_with(chips, "Threshold0", (110, 30, 2))

        plot_mock.assert_called_once_with(chips, ANY, self.mock_scan_range)
        np.testing.assert_array_equal(expected_array, plot_mock.call_args[0][1])

        fit_mock.assert_called_once_with(chips, plot_mock.return_value[0], plot_mock.return_value[1])
        self.assertEqual(tuple(plot_mock.return_value), values)


class MaskRowBlockTest(unittest.TestCase):

    @patch(ERX_patch_path + '.load_config')
    @patch('numpy.savetxt')
    @patch(ED_patch_path + '.plot_image')
    def test_correct_calls_made(self, plot_mock, save_mock, load_mock):

        e = ExcaliburNode(0)
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
        expected_filename = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/pixelmask.chip'

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


@patch(ERX_patch_path + '.set_thresh_energy')
class SetThreshold0Test(unittest.TestCase):

    def test_correct_calls_made(self, set_thresh_energy_mock):
        e = ExcaliburNode(0)

        e.set_threshold0('7')

        set_thresh_energy_mock.assert_called_once_with('0', 7.0)

    def test_correct_calls_made_with_default_param(self,
                                                   set_thresh_energy_mock):
        e = ExcaliburNode(0)

        e.set_threshold0()

        set_thresh_energy_mock.assert_called_once_with('0', 5.0)


@patch(ERX_patch_path + '.expose')
@patch(ERX_patch_path + '.set_dac')
class SetThreshold0DacTest(unittest.TestCase):

    def test_correct_calls_made(self, set_dac_mock, expose_mock):
        e = ExcaliburNode(0)
        chips = [1, 2, 3]

        e.set_threshold0_dac(chips, 1)

        set_dac_mock.assert_called_once_with(chips, 'Threshold0', 1)
        expose_mock.assert_called_once_with()

    def test_correct_calls_made_with_default_param(self, set_dac_mock,
                                                   expose_mock):
        e = ExcaliburNode(0)
        chips = [0, 1, 2, 3, 4, 5, 6, 7]

        e.set_threshold0_dac()

        set_dac_mock.assert_called_once_with(chips, 'Threshold0', 40)
        expose_mock.assert_called_once_with()


@patch('time.sleep')
@patch('numpy.genfromtxt')
@patch(ERX_patch_path + '.set_dac')
@patch(ERX_patch_path + '.expose')
class SetThreshEnergyTest(unittest.TestCase):

    def test_correct_calls_made(self, expose_mock, set_dac_mock, gen_mock,
                                sleep_mock):
        e = ExcaliburNode(0)
        expected_filepath = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/threshold0'

        e.set_thresh_energy('0', 7.0)

        gen_mock.assert_called_once_with(expected_filepath)
        self.assertEqual(8, set_dac_mock.call_count)
        self.assertEqual(2, expose_mock.call_count)

    def test_correct_calls_made_with_default_param(self, expose_mock,
                                                   set_dac_mock, gen_mock,
                                                   sleep_mock):
        e = ExcaliburNode(0)
        expected_filepath = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/threshold0'

        e.set_thresh_energy()

        gen_mock.assert_called_once_with(expected_filepath)
        self.assertEqual(8, set_dac_mock.call_count)
        sleep_mock.assert_called_with(0.2)
        self.assertEqual(2, expose_mock.call_count)


class SetDacsTest(unittest.TestCase):

    @patch(ERX_patch_path + '.set_dac')
    def test_correct_calls_made(self, set_dac_mock):
        e = ExcaliburNode(0)
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


class SubprocessCallsTest(unittest.TestCase):  # TODO: Rename

    file_mock = MagicMock()

    def setUp(self):
        self.e = ExcaliburNode(0)

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

    @patch('time.sleep')
    @patch(ERX_patch_path + '.update_filename_index')
    @patch(ETAI_patch_path + '.acquire')
    def test_burst(self, acquire_mock, update_idx_mock, sleep_mock):
        expected_file = 'image_.hdf5'
        chips = range(8)

        self.e.burst(10, 0.1)

        update_idx_mock.assert_called_once_with()
        acquire_mock.assert_called_once_with(chips, 10, 0.1, burst=True, hdffile=expected_file)
        sleep_mock.assert_called_once_with(0.5)

    @patch('time.asctime', return_value='Tue Sep 27 10:12:27 2016')
    @patch(ED_patch_path + '.load_image_data')
    @patch(ED_patch_path + '.plot_image')
    @patch('time.sleep')
    @patch(ETAI_patch_path + '.acquire')
    @patch(ERX_patch_path + '.update_filename_index')
    def test_expose(self, update_idx_mock, acquire_mock, sleep_mock, plot_mock,
                    load_mock, _):
        expected_file = 'image_.hdf5'
        chips = range(8)

        self.e.expose()

        update_idx_mock.assert_called_once_with()
        acquire_mock.assert_called_once_with(chips, '1', '100', hdffile=expected_file)
        sleep_mock.assert_called_once_with(0.5)
        load_mock.assert_called_once_with('/tmp/image_.hdf5')
        plot_mock.assert_called_once_with(load_mock.return_value, name='Image_Tue Sep 27 10:12:27 2016')

    @patch('time.asctime', return_value='Tue Sep 27 10:12:27 2016')
    @patch(ED_patch_path + '.load_image_data')
    @patch(ED_patch_path + '.plot_image')
    @patch('time.sleep')
    @patch(ETAI_patch_path + '.acquire')
    @patch(ERX_patch_path + '.update_filename_index')
    def test_shoot(self, update_idx_mock, acquire_mock, sleep_mock, plot_mock,
                   load_mock, _):
        expected_file = 'image_.hdf5'
        chips = range(8)

        self.e.shoot(10)

        update_idx_mock.assert_called_once_with()
        acquire_mock.assert_called_once_with(chips, '1', '10', hdffile=expected_file)
        sleep_mock.assert_called_once_with(0.2)
        load_mock.assert_called_once_with('/tmp/image_.hdf5')
        plot_mock.assert_called_once_with(load_mock.return_value, name='Image_Tue Sep 27 10:12:27 2016')

    @patch('time.asctime', return_value='Tue Sep 27 10:12:27 2016')
    @patch(ED_patch_path + '.load_image_data')
    @patch(ED_patch_path + '.plot_image')
    @patch('time.sleep')
    @patch(ETAI_patch_path + '.acquire')
    @patch(ERX_patch_path + '.update_filename_index')
    def test_cont(self, update_idx_mock, acquire_mock, sleep_mock, plot_mock,
                  load_mock, _):
        expected_file = 'image_.hdf5'
        chips = range(8)

        self.e.cont(100, 10)

        update_idx_mock.assert_called_once_with()
        acquire_mock.assert_called_once_with(chips, '100', '10', readmode='1', hdffile=expected_file)
        sleep_mock.assert_called_once_with(0.2)
        load_mock.assert_called_once_with('/tmp/image_.hdf5')
        plot_mock.assert_called_with(load_mock.return_value.__getitem__(), name=ANY)
        self.assertEqual(5, plot_mock.call_count)

    @patch(ETAI_patch_path + '.acquire')
    @patch(ERX_patch_path + '.update_filename_index')
    def test_cont_burst(self, update_idx_mock, acquire_mock):
        expected_file = 'image_.hdf5'
        chips = range(8)

        self.e.cont_burst(100, 10)

        update_idx_mock.assert_called_once_with()
        acquire_mock.assert_called_once_with(chips, 100, 10, burst=True, readmode='1', hdffile=expected_file)

    @patch(ETAI_patch_path + '.load_dacs')
    @patch('__builtin__.open', return_value=file_mock)
    def test_set_dac(self, open_mock, load_mock):
        dac_file = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/dacs'
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
        expected_file = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/dacs'
        chips = [0]

        self.e.read_dac(chips, 'Threshold0')

        sense_mock.assert_called_once_with(chips, 'Threshold0', expected_file)

    @patch(ETAI_patch_path + '.load_config')
    @patch('numpy.savetxt')
    def test_load_config_bits(self, save_mock, load_mock):
        filepath = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/'
        expected_kwargs = dict(fmt='%.18g', delimiter=' ')
        expected_file_1 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/discLbits.tmp'
        expected_file_2 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/discHbits.tmp'
        expected_file_3 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/maskbits.tmp'
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


@patch(ETAI_patch_path + '.load_config')
@patch(ERX_patch_path + '.set_dac')
@patch(ERX_patch_path + '.expose')
class LoadConfigTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode(0)
        self.chips = [0]

    def test_correct_calls_made(self, expose_mock, set_mock, load_mock):
        expected_file_1 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/discLbits.chip0'
        expected_file_2 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/discHbits.chip0'
        expected_file_3 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/pixelmask.chip0'

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
@patch(ERX_patch_path + '.load_config')
@patch(ERX_patch_path + '.set_threshold0_dac')
@patch(ERX_patch_path + '.expose')
class Fe55ImageRX001Test(unittest.TestCase):

    def test_correct_calls_made(self, expose_mock, set_threshhold_mock,
                                load_mock, sleep_mock):
        e = ExcaliburNode(0)
        chips = [1, 4, 6, 7]
        exposure_time = 10000

        e.fe55_image_rx001(chips, exposure_time)

        self.assertEqual('shgm', e.settings['gain'])
        self.assertEqual('spm', e.settings['mode'])

        load_mock.assert_called_once_with(chips)
        set_threshhold_mock.assert_called_once_with(chips, 40)
        sleep_mock.assert_called_once_with(0.5)
        expose_mock.assert_called_once_with()

    def test_correct_calls_made_with_default_params(self, expose_mock,
                                                    set_threshhold_mock,
                                                    load_mock, sleep_mock):
        e = ExcaliburNode(0)
        chips = [0, 1, 2, 3, 4, 5, 6, 7]
        exposure_time = 60000

        e.fe55_image_rx001()

        self.assertEqual('shgm', e.settings['gain'])
        self.assertEqual('spm', e.settings['mode'])

        load_mock.assert_called_once_with(chips)
        set_threshhold_mock.assert_called_once_with(chips, 40)
        sleep_mock.assert_called_once_with(0.5)
        expose_mock.assert_called_once_with()


@patch(ED_patch_path + '.load_image_data')
@patch(ERX_patch_path + '.update_filename_index')
@patch(ETAI_patch_path + '.perform_dac_scan')
@patch('time.sleep')
class ScanDacTest(unittest.TestCase):

    def test_given_start_lower_than_stop(self, sleep_mock, scan_mock,
                                         update_mock, load_mock):
        dac_file = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem/spm/shgm/dacs'
        save_file = 'image_dacscan.hdf5'
        e = ExcaliburNode(0)
        chips = [0]
        dac_range = [1, 10, 1]

        e.scan_dac(chips, 'Threshold0', dac_range)

        update_mock.assert_called_once_with()
        scan_mock.assert_called_once_with(chips, 'Threshold0', Range(1, 10, 1), dac_file, save_file)
        sleep_mock.assert_called_once_with(1)
        load_mock.assert_called_once_with('/tmp/' + save_file)


class UpdateFilenameIndexTest(unittest.TestCase):

    file_mock = MagicMock()

    def setUp(self):
        self.e = ExcaliburNode(0)
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

    @patch(ERX_patch_path + '.expose',
           return_value=mock_array)
    @patch(ED_patch_path + '.plot_image')
    def test_acquire_ff(self, plot_mock, expose_mock):
        e = ExcaliburNode(0)
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

    @patch(ED_patch_path + '.plot_image')
    @patch(ED_patch_path + '.load_image_data')
    @patch('time.sleep')
    def test_apply_ff_correction(self, sleep_mock, load_mock, plot_mock):
        e = ExcaliburNode(0)

        e.apply_ff_correction(1, 0.1)

        load_mock.assert_called_once_with('/tmp/image_.hdf5')
        plot_mock.assert_called_once_with(load_mock.return_value.__getitem__().__mul__().__getitem__(), name='Image data Cor')
        sleep_mock.assert_called_once_with(1)

        # TODO: Finish once function completed


@patch('time.sleep')
@patch('time.asctime', return_value='Mon Sep 26 17:04:31 2016')
@patch('numpy.savetxt')
@patch(ED_patch_path + '.plot_image')
@patch(ED_patch_path + '.load_image_data')
@patch(ETAI_patch_path + '.acquire')
@patch(ERX_patch_path + '.set_dac')
@patch(ERX_patch_path + '.shoot')
class LogoTestTest(unittest.TestCase):

    rand = np.random.RandomState(123)
    mock_array = rand.randint(2, size=(243, 1598))

    def setUp(self):
        self.e = ExcaliburNode(0)
        self.e.num_chips = 1  # Make test easier - only run for one chip

        logo_tp = np.ones([256, 8*256])
        logo_tp[7:250, 225:1823] = self.mock_array
        logo_tp[logo_tp > 0] = 1
        self.logo_tp = 1 - logo_tp

    @patch(ETAI_patch_path + '.configure_test_pulse_with_disc')
    @patch('numpy.loadtxt', return_value=mock_array)
    @patch('os.path.isfile', return_value=True)
    def test_logo_test_files_exist(self, _, load_mock, configure_mock,
                                   shoot_mock, set_mock, acquire_mock,
                                   load_image_mock, plot_mock, save_mock, _2,
                                   sleep_mock):
        expected_dac_file = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/dacs'
        mask_file = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/Logo_chip0_mask'
        disc_files = dict(disch='/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/discHbits.chip0',
                          pixelmask='/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/pixelmask.chip0',
                          discl='/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/discLbits.chip0')
        chips = [0]

        self.e.logo_test()

        set_mock.assert_called_once_with(chips, "Threshold0", 40)
        shoot_mock.assert_called_once_with(10)
        load_mock.assert_called_once_with('/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/config/logo.txt')

        configure_mock.assert_called_once_with(chips, expected_dac_file, mask_file, disc_files)
        acquire_mock.assert_called_once_with(chips, '1', '100', tpcount='100', hdffile='image_.hdf5')
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

        configure_mock.assert_called_once_with(chips, expected_dac_file, mask_file)
        acquire_mock.assert_called_once_with(chips, '1', '100', tpcount='100', hdffile='image_.hdf5')
        sleep_mock.assert_called_once_with(0.2)

        self.assertEqual(save_mock.call_args[0][0], '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/Logo_chip0_mask')
        np.testing.assert_array_equal(self.logo_tp[0:256, 0:256], save_mock.call_args[0][1])
        self.assertEqual(save_mock.call_args[1], dict(fmt='%.18g', delimiter=' '))

        load_image_mock.assert_called_once_with('/tmp/image_.hdf5')
        plot_mock.assert_called_once_with(load_image_mock.return_value, name='Image_Mon Sep 26 17:04:31 2016')


class TestPulseTest(unittest.TestCase):

    # TODO: Test other if branch?

    def setUp(self):
        self.e = ExcaliburNode(0)

    @patch('time.asctime', return_value='Tue Sep 27 10:43:52 2016')
    @patch(ED_patch_path + '.plot_image')
    @patch(ED_patch_path + '.load_image_data')
    @patch(ETAI_patch_path + '.acquire')
    @patch(ETAI_patch_path + '.configure_test_pulse')
    @patch('numpy.savetxt')
    def test_test_pulse(self, save_mock, configure_mock, acquire_mock,
                        load_mock, plot_mock, _):
        expected_dac_file = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/dacs'
        mask_file = 'excaliburRx/config/triangle.mask'
        chips = [0]

        self.e.test_pulse(chips, mask_file, 1000)

        configure_mock.assert_called_once_with(chips, expected_dac_file, mask_file)
        acquire_mock.assert_called_once_with(chips, '1', '100', tpcount='1000', hdffile='image_.hdf5')
        load_mock.assert_called_once_with('/tmp/image_.hdf5')

        plot_mock.assert_called_once_with(load_mock.return_value, name='Image_Tue Sep 27 10:43:52 2016')


class SaveDiscbitsTest(unittest.TestCase):

    @patch('numpy.savetxt')
    def test_correct_call_made(self, save_mock):
        discbits = np.random.randint(10, size=(256, 8*256))
        expected_subarray = discbits[0:256, 0:256]
        e = ExcaliburNode(0)

        e.save_discbits([0], discbits, 'test')

        call_args = save_mock.call_args[0]
        call_kwargs = save_mock.call_args[1]
        self.assertEqual('/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/test.chip0',
                         call_args[0])
        self.assertTrue((expected_subarray == call_args[1]).all())
        self.assertEqual(dict(fmt='%.18g', delimiter=' '), call_kwargs)


@patch(ED_patch_path + '.plot_image')
@patch('numpy.savetxt')
@patch(ETAI_patch_path + '.load_config')
class MaskUnmaskTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode(0)

    def test_mask_col(self, load_mock, save_mock, plot_mock):
        expected_file_1 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/discLbits.chip0'
        expected_file_2 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/pixelmask.chip0'
        expected_mask = np.zeros([256, 8*256])
        expected_mask[:, 1] = 1
        expected_submask = expected_mask[0:256, 0:256]

        self.e.mask_col(0, 1)

        self.assertEqual('/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/pixelmask.chip0', save_mock.call_args[0][0])
        np.testing.assert_array_equal(expected_submask, save_mock.call_args[0][1])
        self.assertEqual(dict(fmt='%.18g', delimiter=' '), save_mock.call_args[1])

        load_mock.assert_called_once_with([0], expected_file_1, pixelmask=expected_file_2)

        np.testing.assert_array_equal(expected_mask, plot_mock.call_args[0][0])
        self.assertEqual(dict(name='Bad pixels'), plot_mock.call_args[1])

    def test_mask_sup_col(self, load_mock, save_mock, plot_mock):
        expected_file_1 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/discLbits.chip0'
        expected_file_2 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/pixelmask.chip0'
        expected_mask = np.zeros([256, 8*256])
        expected_mask[:, 32:32 + 64] = 1
        expected_submask = expected_mask[0:256, 0:256]

        self.e.mask_super_column(0, 1)

        self.assertEqual('/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/pixelmask.chip0', save_mock.call_args[0][0])
        np.testing.assert_array_equal(expected_submask, save_mock.call_args[0][1])
        self.assertEqual(dict(fmt='%.18g', delimiter=' '), save_mock.call_args[1])

        load_mock.assert_called_once_with([0], expected_file_1, pixelmask=expected_file_2)

        np.testing.assert_array_equal(expected_mask, plot_mock.call_args[0][0])
        self.assertEqual(dict(name='Bad Pixels'), plot_mock.call_args[1])

    def test_mask_pixels(self, load_mock, save_mock, plot_mock):
        expected_file_1 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/discLbits.chip0'
        expected_file_2 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/pixelmask.chip0'
        image_data = np.random.randint(10, size=(256, 8*256))
        expected_mask = image_data > 7

        self.e.mask_pixels([0], image_data, 7)

        np.testing.assert_array_equal(expected_mask, plot_mock.call_args[0][0])
        self.assertEqual(dict(name='Bad pixels'), plot_mock.call_args[1])

        self.assertEqual('/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/pixelmask.chip0', save_mock.call_args[0][0])
        np.testing.assert_array_equal(expected_mask[0:256, 0:256], save_mock.call_args[0][1])
        self.assertEqual(dict(fmt='%.18g', delimiter=' '), save_mock.call_args[1])

        load_mock.assert_called_once_with([0], expected_file_1, pixelmask=expected_file_2)

    mock_scan_data = np.random.randint(3, size=(3, 256, 8*256))

    @patch(ERX_patch_path + '.scan_dac',
           return_value=[mock_scan_data, None])
    def test_mask_pixels_using_dac_scan(self, scan_mock, load_mock, save_mock,
                                        plot_mock):
        mask = self.mock_scan_data.sum(0) > 1

        self.e.mask_pixels_using_dac_scan([0])

        scan_mock.assert_called_once_with([0], "Threshold0", (20, 120, 2))

        np.testing.assert_array_equal(mask, plot_mock.call_args[0][0])
        self.assertEqual(dict(name='Bad Pixels'), plot_mock.call_args[1])

        self.assertEqual(save_mock.call_args[0][0], '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/pixelmask.chip0')
        np.testing.assert_array_equal(mask[0:256, 0:256], save_mock.call_args[0][1])
        self.assertEqual(dict(fmt='%.18g', delimiter=' '), save_mock.call_args[1])

    def test_unmask_all_pixels(self, load_mock, save_mock, _):
        expected_file_1 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/discLbits.chip0'
        expected_file_2 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/pixelmask.chip0'
        expected_mask = np.zeros([256, 256])

        self.e.unmask_all_pixels([0])

        self.assertEqual(save_mock.call_args[0][0], '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/pixelmask.chip0')
        np.testing.assert_array_equal(expected_mask, save_mock.call_args[0][1])
        self.assertEqual(dict(fmt='%.18g', delimiter=' '), save_mock.call_args[1])

        load_mock.assert_called_once_with([0], expected_file_1, pixelmask=expected_file_2)

    def test_unequalize_pixels(self, load_mock, save_mock, _):
        expected_file_1 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/discL_bits.chip0'
        expected_file_2 = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/pixelmask.chip0'
        expected_mask = np.zeros([256, 256])

        self.e.unequalize_all_pixels([0])

        self.assertEqual(save_mock.call_args[0][0], '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/discL_bits.chip0')
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
        expected_path = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm'
        e = ExcaliburNode(0)

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
        expected_path = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm'
        e = ExcaliburNode(0)

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
        expected_path = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/'
        e = ExcaliburNode(0)

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
        expected_path = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/'
        e = ExcaliburNode(0)

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
        expected_path = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/test_file.chip0'
        e = ExcaliburNode(0)

        value = e.open_discbits_file([0], 'test_file')

        load_mock.assert_called_once_with(expected_path)
        self.assertTrue((value[0:256, 0:256] == self.mock_array).all())


class CombineROIsTest(unittest.TestCase):

    # Make sure we always get the same random numbers
    rand = np.random.RandomState(123)

    @patch(ED_patch_path + '.plot_image')
    @patch(ERX_patch_path + '.roi',
           return_value=np.ones([256, 256]))
    @patch(ERX_patch_path + '.save_discbits')
    @patch(ERX_patch_path + '.open_discbits_file',
           side_effect=[rand.randint(2, size=[256, 256]),
                        rand.randint(2, size=[256, 256])])
    def test_correct_calls_made(self, open_mock, save_mock, roi_mock,
                               plot_mock):
        expected_path = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/test_file.chip0'
        e = ExcaliburNode(0)

        value = e.combine_rois([0], 'test_file', 1, 'rect')

        roi_mock.assert_called_once_with([0], 0, 1, 'rect')
        # plot_mock.assert_called_once_with()
        # save_mock.assert_called_once_with()
        # plot_mock.assert_called_once_with()

        # TODO: Finish once sure what function should do


@patch(ED_patch_path + '.plot_histogram')
@patch(ED_patch_path + '.plot_image')
class FindTest(unittest.TestCase):  # TODO: Improve

    def setUp(self):
        self.e = ExcaliburNode(0)

        # Make sure we always get the same random numbers
        self.rand = np.random.RandomState(123)

    # TODO: Why do these get the same expected array?

    def test_find_edge(self, plot_mock, plot_histo_mock):
        dac_scan_data = self.rand.randint(10, size=(3, 3, 3))
        dac_range = [1, 10, 1]
        expected_array = [[10, 9, 10], [10, 9, 8], [10, 10, 10]]

        value = self.e.find_edge([0], dac_scan_data, dac_range, 7)

        np.testing.assert_array_equal(expected_array, plot_mock.call_args[0][0])
        self.assertEqual(dict(name="noise edges"), plot_mock.call_args[1])
        self.assertEqual([0], plot_histo_mock.call_args[0][0])
        np.testing.assert_array_equal(expected_array, plot_histo_mock.call_args[0][1])
        np.testing.assert_array_equal(expected_array, value)

    def test_find_max(self, plot_mock, plot_histo_mock):
        dac_scan_data = self.rand.randint(10, size=(3, 3, 3))
        dac_range = [1, 10, 1]
        expected_array = [[10, 9, 10], [10, 9, 8], [10, 10, 10]]

        value = self.e.find_max([0], dac_scan_data, dac_range)

        np.testing.assert_array_equal(expected_array, plot_mock.call_args[0][0])
        self.assertEqual(dict(name="noise edges"), plot_mock.call_args[1])
        self.assertEqual([0], plot_histo_mock.call_args[0][0])
        np.testing.assert_array_equal(expected_array, plot_histo_mock.call_args[0][1])
        np.testing.assert_array_equal(expected_array, value)


class OptimizeDacDiscTest(unittest.TestCase):

    rand = np.random.RandomState(1234)

    @patch(ED_patch_path + '.clear_plot')
    @patch(ED_patch_path + '.plot_image')
    @patch('numpy.histogram')
    @patch('numpy.asarray')
    @patch('excaliburcalibrationdawn.excaliburdawn.curve_fit',
           return_value=[[1, 2, 3], None])
    @patch(ED_patch_path + '.gauss_function')
    @patch(ERX_patch_path + '.set_dac')
    @patch(ERX_patch_path + '.scan_dac',
           return_value=[rand.randint(10, size=(3, 256, 8*256)), None])
    @patch(ERX_patch_path + '.open_discbits_file')
    @patch(ERX_patch_path + '.load_config_bits')
    @patch(ERX_patch_path + '.find_max')
    def test_(self, find_mock, load_mock, open_mock, scan_mock, set_mock, gauss_mock,
              fit_mock, asarray_mock, histo_mock, plot_mock, clear_mock):
        # TODO: Write proper tests once function is split up
        e = ExcaliburNode()
        chips = [0]
        roi_mock = np.ones([256, 8*256])

        # e.optimize_dac_disc(chips, 'discL', roi_mock)


class EqualizeDiscbitsTest(unittest.TestCase):

    rand = np.random.RandomState(1234)

    @patch(ED_patch_path + '.plot_image')
    @patch(ED_patch_path + '.clear_plot')
    @patch(ERX_patch_path + '.set_dac')
    @patch(ERX_patch_path + '.open_discbits_file')
    @patch(ERX_patch_path + '.load_config_bits')
    @patch(ERX_patch_path + '.load_config')
    @patch(ERX_patch_path + '.find_max',
           return_value=rand.randint(2, size=(256, 8*256)))
    @patch(ERX_patch_path + '.scan_dac',
           return_value=[rand.randint(10, size=(3, 256, 8*256)), None])
    # @unittest.skip("Takes too long")
    def test_correct_calls_made(self, scan_mock, find_mock, load_mock,
                                load_bits_mock, open_mock, set_mock,
                                clear_mock, plot_mock):
        # TODO: Finish once run on real system, know what it does and
        # TODO: maybe split up / refactored.
        e = ExcaliburNode(0)
        chips = [0]
        roi = np.zeros([256, 8*256])

        # e.equalise_discbits(chips, 'discL', roi)

        # self.assertEqual(32 + 1, scan_mock.call_count)
        # find_mock.assert_called_once_with()
        # load_mock.assert_called_once_with()
        # load_bits_mock.assert_called_once_with()
        # open_mock.assert_called_once_with()
        # set_mock.assert_called_once_with()
        # plot_mock.assert_called_once_with()


class CheckCalibTest(unittest.TestCase):

    rand = np.random.RandomState(1234)

    @patch(ERX_patch_path + '.find_max',
           return_value=rand.randint(2, size=(256, 8*256)))
    @patch(ERX_patch_path + '.load_config')
    def test_correct_call_made(self, load_mock, find_mock):
        pass  # TODO: Function doesn't work
        e = ExcaliburNode(0)
        chips = [0]

        # e.check_calib(chips, [0, 10, 1])
        #
        # load_mock.assert_called_once_with(chips)


@patch(ED_patch_path + '.plot_image')
class ROITest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode(0)

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

    @patch(ERX_patch_path + '.optimize_dac_disc')
    @patch(ERX_patch_path + '.roi')
    @patch(ERX_patch_path + '.equalise_discbits')
    @patch(ERX_patch_path + '.save_discbits')
    @patch(ERX_patch_path + '.combine_rois')
    @patch(ERX_patch_path + '.load_config')
    @patch(ERX_patch_path + '.copy_slgm_into_other_gain_modes')
    def test_correct_calls_made(self, copy_mock, load_mock, combine_rois,
                                save_mock, equalize_mock, roi_mock, opt_mock):
        e = ExcaliburNode(0)
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

    @patch(ED_patch_path + '.plot_image')
    @patch(ERX_patch_path + '.expose')
    def test_correct_calls_made(self, expose_mock, plot_mock):
        e = ExcaliburNode(0)

        e.loop(1)

        expose_mock.assert_called_once_with()
        plot_mock.assert_called_once_with(expose_mock.return_value.__add__(), name='Sum')


class CSMTest(unittest.TestCase):

    @patch(ERX_patch_path + '.expose')
    @patch(ERX_patch_path + '.load_config')
    @patch(ERX_patch_path + '.set_dac')
    def test_correct_calls_made(self, set_mock, load_mock, expose_mock):
        e = ExcaliburNode(0)
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

    @patch(ERX_patch_path + '.set_dac')
    def test_correct_calls_made(self, set_mock):
        e = ExcaliburNode(0)
        chips = [0]

        e.set_gnd_fbk_cas_excalibur_rx001(chips, 1)

        self.assertEqual(set_mock.call_args_list[0][0], ([0], 'GND', 141))
        self.assertEqual(set_mock.call_args_list[1][0], ([0], 'FBK', 190))
        self.assertEqual(set_mock.call_args_list[2][0], ([0], 'Cas', 178))


class RotateTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburNode(0)

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
    @patch(ERX_patch_path + '.rotate_config')
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
        self.e = ExcaliburNode(0)

    def test_grab_slice(self):
        array = np.array([[1, 2, 3, 4, 5],
                          [10, 20, 30, 40, 50],
                          [100, 200, 300, 400, 500]])
        expected_subarray = np.array([[2, 3, 4],
                                      [20, 30, 40],
                                      [200, 300, 400]])

        value = self.e._grab_slice(array, [0, 1], [2, 3])

        np.testing.assert_array_equal(expected_subarray, value)

    @patch(ERX_patch_path + '._grab_slice')
    @patch(ERX_patch_path + '._generate_chip_range')
    def test_grab_chip_slice(self, generate_mock, grab_mock):
        array = MagicMock()
        generate_mock.return_value = MagicMock(), MagicMock

        value = self.e._grab_chip_slice(array, 1)

        generate_mock.assert_called_once_with(1)
        grab_mock.assert_called_once_with(array, generate_mock.return_value[0], generate_mock.return_value[1])
        self.assertEqual(grab_mock.return_value, value)

    def test_set_slice(self):
        array = np.array([[1, 2, 3, 4, 5],
                          [10, 20, 30, 40, 50],
                          [100, 200, 300, 400, 500]])
        array_copy = array.copy()
        expected_array = np.array([[1, 0, 0, 0, 5],
                                   [10, 0, 0, 0, 50],
                                   [100, 0, 0, 0, 500]])
        subarray = np.array([[2, 3, 4],
                             [20, 30, 40],
                             [200, 300, 400]])

        self.e._set_slice(array, [0, 1], [2, 3], 0)
        np.testing.assert_array_equal(expected_array, array)
        self.e._set_slice(array, [0, 1], [2, 3], subarray)
        np.testing.assert_array_equal(array_copy, array)

    @patch(ERX_patch_path + '._set_slice')
    @patch(ERX_patch_path + '._generate_chip_range')
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
