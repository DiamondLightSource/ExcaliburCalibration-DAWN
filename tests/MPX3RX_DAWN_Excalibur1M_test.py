import unittest

from pkg_resources import require
require("mock")
from mock import patch, MagicMock

from scripts.MPX3RX_DAWN_Excalibur1M import ExcaliburRX, np


class InitTest(unittest.TestCase):

    def setUp(self):
        self.node = 2
        self.e = ExcaliburRX(self.node)

    def test_class_attributes_set(self):
        self.assertEqual(self.e.command,
                         "/dls/detectors/support/silicon_pixels/excaliburRX/"
                         "TestApplication_15012015/excaliburTestApp")
        self.assertEqual(self.e.port, "6969")
        self.assertEqual(self.e.dac_target, 10)
        self.assertEqual(self.e.nb_of_sigma, 3.2)
        self.assertEqual(self.e.edge_val, 10)
        self.assertEqual(self.e.acc_dist, 4)
        self.assertEqual(self.e.calib_settings, {'calibDir': "/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/",
                                                'configDir': "/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/config/",
                                                'dacfilename': "dacs",
                                                'dacfile': "",
                                                'noiseEdge': "10"})
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
        self.assertEqual(self.e.mode_code, {'spm': '0', 'csm': '1'})
        self.assertEqual(self.e.gain_code, {'shgm': '0', 'hgm': '1', 'lgm': '2',
                                           'slgm': '3'})
        self.assertEqual(self.e.dac_code, {'Threshold0': '1',
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
                                           'DACTest': '30',
                                           'DACDiscH': '31',
                                           'Delay': '16',
                                           'TPBuffIn': '17',
                                           'TPBuffOut': '18',
                                           'RPZ': '19',
                                           'GND': '20',
                                           'TPREF': '21',
                                           'FBK': '22',
                                           'Cas': '23',
                                           'TPREFA': '24',
                                           'TPREFB': '25'})
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
        e = ExcaliburRX(0)
        self.assertEqual(e.ipaddress, "192.168.0.106")

    def test_given_node_invalid_node(self):
        e = ExcaliburRX(8)
        self.assertEqual(e.ipaddress, "192.168.0.10-1")


@patch('scripts.MPX3RX_DAWN_Excalibur1M.ExcaliburRX.check_calib_dir')
@patch('scripts.MPX3RX_DAWN_Excalibur1M.ExcaliburRX.log_chip_id')
@patch('scripts.MPX3RX_DAWN_Excalibur1M.ExcaliburRX.set_dacs')
@patch('scripts.MPX3RX_DAWN_Excalibur1M.ExcaliburRX.set_gnd_fbk_cas_excalibur_rx001')
@patch('scripts.MPX3RX_DAWN_Excalibur1M.ExcaliburRX.calibrate_disc')
class ThresholdEqualizationTest(unittest.TestCase):

    def test_correct_calls_made(self, cal_disc_mock, set_gnd_mock,
                                set_dacs_mock, log_mock, check_mock):
        e = ExcaliburRX(0)
        chips = [1, 4, 6, 7]

        e.threshold_equalization(chips)

        self.assertEqual('slgm', e.settings['gain'])
        self.assertEqual('spm', e.settings['mode'])

        check_mock.assert_called_once_with()
        log_mock.assert_called_once_with()
        set_dacs_mock.assert_called_once_with(chips)
        set_gnd_mock.assert_called_once_with(chips, e.fem)
        cal_disc_mock.assert_called_once_with(chips, 'discL')

    def test_correct_calls_made_using_default_param(self, cal_disc_mock,
                                                    set_gnd_mock,
                                                    set_dacs_mock,
                                                    log_mock,
                                                    check_mock):
        e = ExcaliburRX(0)
        chips = range(8)

        e.threshold_equalization()

        self.assertEqual('slgm', e.settings['gain'])
        self.assertEqual('spm', e.settings['mode'])

        check_mock.assert_called_once_with()
        log_mock.assert_called_once_with()
        set_dacs_mock.assert_called_once_with(chips)
        set_gnd_mock.assert_called_once_with(chips, e.fem)
        cal_disc_mock.assert_called_once_with(chips, 'discL')


class SaveKev2DacCalibTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburRX(0)
        self.threshold = '0'
        self.gain = [1.1, 0.7, 1.1, 1.3, 1.0, 0.9, 1.2, 0.9]
        self.offset = [0.2, -0.7, 0.1, 0.0, 0.3, -0.1, 0.2, 0.5]
        self.expected_array = np.array([self.gain, self.offset])
        self.expected_filename = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/threshold0'

    @patch('os.path.isfile', return_value=True)
    @patch('numpy.savetxt')
    @patch('os.chmod')
    def test_given_existing_file_then_save(self, chmod_mock, save_mock,
                                           isfile_mock):

        self.e.save_kev2dac_calib(self.threshold, self.gain, self.offset)

        isfile_mock.assert_called_once_with(self.expected_filename)
        call_args = save_mock.call_args[0]
        call_kwargs = save_mock.call_args[1]
        self.assertEqual(self.expected_filename, call_args[0])
        self.assertTrue((self.expected_array == call_args[1]).all())
        self.assertEqual(dict(fmt='%.2f'), call_kwargs)
        self.assertFalse(chmod_mock.call_count)

    @patch('os.path.isfile', return_value=False)
    @patch('numpy.savetxt')
    @patch('os.chmod')
    def test_given_no_file_then_create_and_chmod(self, chmod_mock, save_mock,
                                                 isfile_mock):

        self.e.save_kev2dac_calib(self.threshold, self.gain, self.offset)

        isfile_mock.assert_called_once_with(self.expected_filename)
        call_args = save_mock.call_args[0]
        call_kwargs = save_mock.call_args[1]
        self.assertEqual(self.expected_filename, call_args[0])
        self.assertTrue((self.expected_array == call_args[1]).all())
        self.assertEqual(dict(fmt='%.2f'), call_kwargs)
        chmod_mock.assert_called_once_with(self.expected_filename, 0777)


class MaskRowBlockTest(unittest.TestCase):

    @patch('scripts.MPX3RX_DAWN_Excalibur1M.ExcaliburRX.load_config')
    @patch('numpy.savetxt')
    @patch('scisoftpy.plot')
    def test_correct_calls_made(self, plot_mock, save_mock, load_mock):

        e = ExcaliburRX(0)
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
        call_args = save_mock.call_args_list[0][0]
        call_kwargs = save_mock.call_args_list[0][1]
        self.assertEqual(expected_filename + '1', call_args[0])
        self.assertTrue((expected_subarray == call_args[1]).all())
        self.assertEqual(dict(delimiter=' ', fmt='%.18g'), call_kwargs)
        # Check second save call
        call_args = save_mock.call_args_list[1][0]
        call_kwargs = save_mock.call_args_list[1][1]
        self.assertEqual(expected_filename + '2', call_args[0])
        self.assertTrue((expected_subarray == call_args[1]).all())
        self.assertEqual(dict(delimiter=' ', fmt='%.18g'), call_kwargs)
        # Check plot call
        call_args = plot_mock.image.call_args[0]
        self.assertTrue((expected_array == call_args[0]).all())

        load_mock.assert_called_once_with(chips)


@patch('scripts.MPX3RX_DAWN_Excalibur1M.ExcaliburRX.set_thresh_energy')
class SetThreshold0Test(unittest.TestCase):

    def test_correct_calls_made(self, set_thresh_energy_mock):
        e = ExcaliburRX(0)

        e.set_threshold0('7')

        set_thresh_energy_mock.assert_called_once_with('0', 7.0)

    def test_correct_calls_made_with_default_param(self,
                                                   set_thresh_energy_mock):
        e = ExcaliburRX(0)

        e.set_threshold0()

        set_thresh_energy_mock.assert_called_once_with('0', 5.0)


@patch('scripts.MPX3RX_DAWN_Excalibur1M.ExcaliburRX.expose')
@patch('scripts.MPX3RX_DAWN_Excalibur1M.ExcaliburRX.set_dac')
class SetThreshold0DacTest(unittest.TestCase):

    def test_correct_calls_made(self, set_dac_mock, expose_mock):
        e = ExcaliburRX(0)
        chips = [1, 2, 3]

        e.set_threshold0_dac(chips, 1)

        set_dac_mock.assert_called_once_with(chips, 'Threshold0', 1)
        expose_mock.assert_called_once_with()

    def test_correct_calls_made_with_default_param(self, set_dac_mock,
                                                   expose_mock):
        e = ExcaliburRX(0)
        chips = [0, 1, 2, 3, 4, 5, 6, 7]

        e.set_threshold0_dac()

        set_dac_mock.assert_called_once_with(chips, 'Threshold0', 40)
        expose_mock.assert_called_once_with()


@patch('numpy.genfromtxt')
@patch('scripts.MPX3RX_DAWN_Excalibur1M.ExcaliburRX.set_dac')
@patch('scripts.MPX3RX_DAWN_Excalibur1M.ExcaliburRX.expose')
class SetThreshEnergyTest(unittest.TestCase):

    def test_correct_calls_made(self, expose_mock, set_dac_mock, gen_mock):
        e = ExcaliburRX(0)
        expected_filepath = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/threshold0'

        e.set_thresh_energy('0', 7.0)

        gen_mock.assert_called_once_with(expected_filepath)
        self.assertEqual(8, set_dac_mock.call_count)
        self.assertEqual(2, expose_mock.call_count)

    def test_correct_calls_made_with_default_param(self, expose_mock,
                                                   set_dac_mock, gen_mock):
        e = ExcaliburRX(0)
        expected_filepath = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/threshold0'

        e.set_thresh_energy()

        gen_mock.assert_called_once_with(expected_filepath)
        self.assertEqual(8, set_dac_mock.call_count)
        self.assertEqual(2, expose_mock.call_count)


class SetDacsTest(unittest.TestCase):

    @patch('scripts.MPX3RX_DAWN_Excalibur1M.ExcaliburRX.set_dac')
    def test_correct_calls_made(self, set_dac_mock):
        e = ExcaliburRX(0)
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


class MaskTest(unittest.TestCase):

    def test_return_values(self):
        e = ExcaliburRX(0)

        value = e.mask([0])
        self.assertEqual('0x80', value)

        value = e.mask([0, 4])
        self.assertEqual('0x88', value)

        value = e.mask([0, 1, 4, 5, 7])
        self.assertEqual('0xcd', value)

        value = e.mask([0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual('0xff', value)

    def test_given_invalid_value_then_error(self):
        e = ExcaliburRX(0)

        with self.assertRaises(ValueError):
            e.mask([8])

    def test_given_duplicate_value_then_error(self):
        e = ExcaliburRX(0)

        with self.assertRaises(ValueError):
            e.mask([1, 1])


@patch('subprocess.call')
class SubprocessCallsTest(unittest.TestCase):

    file_mock = MagicMock()

    def setUp(self):
        self.e = ExcaliburRX(0)

    def test_read_chip_id(self, subp_mock):
        expected_call = ['/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp',
                         '-i', '192.168.0.106',
                         '-p', '6969',
                         '-m', '0xff',
                         '-r', '-e']

        self.e.read_chip_id()

        subp_mock.assert_called_once_with(expected_call)

    @patch('__builtin__.open', return_value=file_mock)
    def test_log_chip_id(self, _, subp_mock):
        expected_call = ['/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp',
                         '-i', '192.168.0.106',
                         '-p', '6969',
                         '-m', '0xff',
                         '-r', '-e']
        expected_kwargs = dict(stdout=self.file_mock.__enter__.return_value)

        self.e.log_chip_id()

        call_args = subp_mock.call_args
        self.assertEqual(expected_call, call_args[0][0])
        self.assertEqual(expected_kwargs, call_args[1])

    def test_monitor(self, subp_mock):
        expected_call = ['/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp',
                         '-i', '192.168.0.106',
                         '-p', '6969',
                         '-m', '0xff',
                         '--slow']

        self.e.monitor()

        subp_mock.assert_called_once_with(expected_call)

    @patch('__builtin__.open', return_value=file_mock)
    def test_set_dac(self, open_mock, subp_mock):
        dac_file = '/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/dacs'
        expected_call = ['/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp',
                         '-i', '192.168.0.106',
                         '-p', '6969',
                         '-m', '0x80',
                         '--dacs=' + dac_file]
        expected_lines = ['Heading', 'Threshold0 = 40\r\n', 'Line2']
        readlines_mock = ['Heading', 'Line1', 'Line2']
        self.file_mock.readlines.return_value = readlines_mock
        chips = [0]

        self.e.set_dac(chips)

        # Check subprocess calls
        call_args = subp_mock.call_args
        self.assertEqual(expected_call, call_args[0][0])
        self.assertEqual({}, call_args[1])
        # Check file writing calls
        open_mock.assert_called_with(dac_file, 'r+b')
        self.assertEqual(len(chips), open_mock.call_count)
        self.file_mock.writelines.assert_called_once_with(expected_lines)

    @patch('time.sleep')
    def test_read_dac(self, sleep_mock, subp_mock):
        expected_call_1 = ['/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp',
                           '-i', '192.168.0.106',
                           '-p', '6969',
                           '-m', '0x80',
                           '--sensedac=1',
                           '--dacs=/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/fem0/spm/shgm/dacs']
        expected_call_2 = ['/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp',
                           '-i', '192.168.0.106',
                           '-p', '6969',
                           '-m', '0xff',
                           '--sensedac=1', '--slow']
        chips = [0]

        self.e.read_dac(chips, 'Threshold0')

        # Check first subprocess call
        call_args = subp_mock.call_args_list[0]
        self.assertEqual(expected_call_1, call_args[0][0])
        self.assertEqual({}, call_args[1])
        # Check time.sleep call
        sleep_mock.assert_called_once_with(1)
        # Check second subprocess call
        call_args = subp_mock.call_args_list[1]
        self.assertEqual(expected_call_2, call_args[0][0])
        self.assertEqual({}, call_args[1])


@patch('scripts.MPX3RX_DAWN_Excalibur1M.ExcaliburRX.load_config')
@patch('scripts.MPX3RX_DAWN_Excalibur1M.ExcaliburRX.set_threshold0_dac')
@patch('scripts.MPX3RX_DAWN_Excalibur1M.ExcaliburRX.expose')
class ThresholdEqualizationTest(unittest.TestCase):

    def test_correct_calls_made(self, expose_mock, set_threshhold_mock,
                                load_mock):
        e = ExcaliburRX(0)
        chips = [1, 4, 6, 7]
        exposure_time = 10000

        e.fe55_image_rx001(chips, exposure_time)

        self.assertEqual('shgm', e.settings['gain'])
        self.assertEqual('spm', e.settings['mode'])

        load_mock.assert_called_once_with(chips)
        set_threshhold_mock.assert_called_once_with(chips, 40)
        expose_mock.assert_called_once_with()

    def test_correct_calls_made_with_default_params(self, expose_mock,
                                                    set_threshhold_mock,
                                                    load_mock):
        e = ExcaliburRX(0)
        chips = [0, 1, 2, 3, 4, 5, 6, 7]
        exposure_time = 60000

        e.fe55_image_rx001()

        self.assertEqual('shgm', e.settings['gain'])
        self.assertEqual('spm', e.settings['mode'])

        load_mock.assert_called_once_with(chips)
        set_threshhold_mock.assert_called_once_with(chips, 40)
        expose_mock.assert_called_once_with()


@patch('scisoftpy.plot')
class PlotDacScanTest(unittest.TestCase):

    def test_given_start_lower_than_stop(self, plot_mock):
        e = ExcaliburRX(0)
        chips = [0]
        dac_range = [1, 10, 1]
        dac_scan_data = np.random.randint(10, size=(10, 256, 8*256))

        e.plot_dac_scan(chips, dac_scan_data, dac_range)

        # TODO: Figure out how this functions works and finish tests


@patch('scripts.MPX3RX_DAWN_Excalibur1M.ExcaliburRX.update_filename_index')
@patch('scisoftpy.plot')
@patch('scisoftpy.io')
@patch('subprocess.call')
class ScanDacTest(unittest.TestCase):

    def test_given_start_lower_than_stop(self, subp_mock, plot_mock, io_mock,
                                         update_mock):
        e = ExcaliburRX(0)
        chips = [0]
        dac_range = [1, 10, 1]

        e.scan_dac(chips, 'Threshold0', dac_range)

        # TODO: Combine with duplicated plotdacscan before writing tests


class ShowPixelTest(unittest.TestCase):

    def test_correct_calls_made(self):
        pass

