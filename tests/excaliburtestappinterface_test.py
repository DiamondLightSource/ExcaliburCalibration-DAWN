import unittest
from subprocess import CalledProcessError

from pkg_resources import require
require("mock")
from mock import patch, MagicMock, ANY, call

from excaliburcalibrationdawn import ExcaliburTestAppInterface
ETAI_patch_path = "excaliburcalibrationdawn.excaliburtestappinterface.ExcaliburTestAppInterface"


@patch('logging.info')
class InitTest(unittest.TestCase):

    def test_attributes_set(self, info_mock):
        e = ExcaliburTestAppInterface(1, "test_ip", "test_port")
        expected_path = "/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp"

        self.assertEqual(expected_path, e.path)
        self.assertEqual("test_ip", e.ip_address)
        self.assertEqual("test_port", e.port)
        self.assertEqual(['/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp', '-i', 'test_ip', '-p', 'test_port'], e.base_cmd)
        self.assertEqual(e.dac_code, {'Threshold0': '1', 'Threshold1': '2', 'Threshold2': '3', 'Threshold3': '4', 'Threshold4': '5', 'Threshold5': '6', 'Threshold6': '7',
                                      'Threshold7': '8', 'Preamp': '9', 'Ikrum': '10', 'Shaper': '11', 'Disc': '12', 'DiscLS': '13', 'ShaperTest': '14', 'DACDiscL': '15',
                                      'DACTest': '30', 'DACDiscH': '31', 'Delay': '16', 'TPBuffIn': '17', 'TPBuffOut': '18', 'RPZ': '19', 'GND': '20', 'TPREF': '21',
                                      'FBK': '22', 'Cas': '23', 'TPREFA': '24', 'TPREFB': '25'})
        self.assertEqual(['/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp',
                          '-i', 'test_ip',
                          '-p', 'test_port'], e.base_cmd)
        info_mock.assert_called_once_with("Set self.quiet to False to display "
                                          "terminal output.")

    def test_base_cmd_with_server_given(self, _):
        e = ExcaliburTestAppInterface(1, "test_ip", "test_port", "test_server")
        self.assertEqual(['ssh',
                          'test_server.diamond.ac.uk',
                          '/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp',
                          '-i', 'test_ip',
                          '-p', 'test_port'], e.base_cmd)


@patch(ETAI_patch_path + '._mask')
class ConstructCommandTest(unittest.TestCase):

    def test_returns(self, mask_mock):
        e = ExcaliburTestAppInterface(1, "test_ip", "test_port")
        expected_command = ['/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp',
                            '-i', 'test_ip',
                            '-p', 'test_port',
                            '-m', mask_mock.return_value]
        chips = [0, 4, 5, 7]

        command = e._construct_command(chips)

        mask_mock.assert_called_once_with(chips)
        self.assertEqual(expected_command, command)

    def test_adds_args_and_returns(self, mask_mock):
        e = ExcaliburTestAppInterface(1, "test_ip", "test_port")
        expected_command = ['/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp',
                            '-i', 'test_ip',
                            '-p', 'test_port',
                            '-m', mask_mock.return_value,
                            '-r',
                            '-e']
        chips = [0, 4, 5, 7]

        command = e._construct_command(chips, "-r", "-e")

        mask_mock.assert_called_once_with(chips)
        self.assertEqual(expected_command, command)


@patch('logging.info')
@patch('logging.debug')
@patch('subprocess.check_output')
@patch('subprocess.check_call')
class SendCommandTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburTestAppInterface(1, "test_ip", "test_port")

    def test_quiet_subp_called_and_logged(self, call_mock, output_mock,
                                          debug_mock, _):
        expected_message = "Sending Command:\n'%s' with kwargs %s"
        output_mock.return_value = "Success"

        self.e._send_command(["test_command"], test=True)

        self.assertEqual((expected_message, "test_command", "{'test': True}"),
                         debug_mock.call_args_list[0][0])
        output_mock.assert_called_once_with(["test_command"], test=True)
        call_mock.assert_not_called()

    def test_quiet_error_raised_then_catch_and_log(self, call_mock, output_mock,
                                                   debug_mock, info_mock):
        expected_message = "Sending Command:\n'%s' with kwargs %s"
        output_mock.side_effect = CalledProcessError(1, "test_command",
                                                     output="Invalid command")

        self.e._send_command(["test_command"], test=True)

        self.assertEqual((expected_message, "test_command", "{'test': True}"),
                         debug_mock.call_args_list[0][0])
        output_mock.assert_called_once_with(["test_command"], test=True)
        call_mock.assert_not_called()
        self.assertEqual(("Error Output:\n%s", "Invalid command"),
                         debug_mock.call_args_list[1][0])
        info_mock.assert_called_once_with("Set self.quiet to False to display "
                                          "terminal output.")

    def test_subp_called_and_logged(self, call_mock, output_mock,
                                    debug_mock, _):
        expected_message = "Sending Command:\n'%s' with kwargs %s"
        call_mock.return_value = "Success"

        self.e.quiet = False
        self.e._send_command(["test_command"], test=True)

        self.assertEqual((expected_message, "test_command", "{'test': True}"),
                         debug_mock.call_args_list[0][0])
        call_mock.assert_called_once_with(["test_command"], test=True)

    def test_error_raised_then_catch_and_log(self, call_mock, output_mock,
                                             debug_mock, _):
        expected_message = "Sending Command:\n'%s' with kwargs %s"
        call_mock.side_effect = CalledProcessError(1, "test_command",
                                                   output="Invalid command")

        self.e.quiet = False
        self.e._send_command(["test_command"], test=True)

        self.assertEqual((expected_message, "test_command", "{'test': True}"),
                         debug_mock.call_args_list[0][0])
        call_mock.assert_called_once_with(["test_command"], test=True)
        self.assertEqual(("Error Output:\n%s", "Invalid command"),
                         debug_mock.call_args_list[1][0])


@patch(ETAI_patch_path + '._construct_command')
@patch(ETAI_patch_path + '._send_command')
class AcquireTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburTestAppInterface(1, "test_ip", "test_port")
        self.e.dacs_loaded = "dacs.txt"
        self.e.initialised = True
        self.chips = range(8)

    @patch('os.path.isfile', return_value=False)
    def test_acquire(self, _, send_mock, construct_mock):
        expected_params = ['-a', '-n', '100', '-t', '10']
        frames = 100
        acquire_time = 10

        self.e.acquire(self.chips, frames, acquire_time)

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_mock.assert_called_once_with(construct_mock.return_value)

    @patch('os.path.isfile', return_value=False)
    def test_acquire_all_args(self, _, send_mock, construct_mock):
        expected_params = ['-a', '-n', '100', '-t', '10', '--burst',
                           '--csmspm', '1', '--disccsmspm', '1',
                           '--depth', '1', '--counter', '1',
                           '--equalization', '1', '--gainmode', '3',
                           '--readmode', '1', '--trigmode', '2',
                           '--tpcount', '10', '--path=/scratch/RX_Images',
                           '--hdffile=test.hdf5']
        frames = 100
        acquire_time = 10

        self.e.acquire(self.chips, frames, acquire_time, burst=True,
                       pixel_mode="csm", disc_mode="discH", depth=1, counter=1,
                       equalization=1, gain_mode="slgm",
                       read_mode="continuous", trig_mode=2, tp_count=10,
                       path="/scratch/RX_Images", hdf_file="test.hdf5")

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_mock.assert_called_once_with(construct_mock.return_value)

    @patch('os.path.isfile', return_value=True)
    def test_given_existing_file_then_error(self, _, _2, _3):

        with self.assertRaises(IOError):
            self.e.acquire(self.chips, 1, 100, hdf_file="test.hdf5")

    def test_dacs_not_loaded_then_error(self, _, _2):

        self.e.dacs_loaded = None

        with self.assertRaises(ValueError):
            self.e.acquire(self.chips, 1, 100, hdf_file="test.hdf5")

    def test_not_initialised_then_error(self, _, _2):

        self.e.initialised = False

        with self.assertRaises(ValueError):
            self.e.acquire(self.chips, 1, 100, hdf_file="test.hdf5")


@patch(ETAI_patch_path + '._construct_command')
@patch(ETAI_patch_path + '._send_command')
class APICallsTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburTestAppInterface(1, "test_ip", "test_port")
        self.chips = range(8)

    def test_set_lv_state(self, send_mock, construct_mock):
        send_mock.return_value = True
        expected_params = ['--lvenable', '1']

        self.e.set_lv_state(1)

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_mock.assert_called_once_with(construct_mock.return_value)
        self.assertEqual(1, self.e.lv)

    def test_set_lv_state_fails(self, send_mock, construct_mock):
        send_mock.return_value = False
        expected_params = ['--lvenable', '1']

        self.e.set_lv_state(1)

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_mock.assert_called_once_with(construct_mock.return_value)
        self.assertEqual(0, self.e.lv)

    def test_set_lv_state_raises(self, send_mock, _2):
        send_mock.return_value = True

        with self.assertRaises(ValueError):
            self.e.set_lv_state(2)

    def test_set_hv_state(self, send_mock, construct_mock):
        send_mock.return_value = True
        expected_params = ['--hvenable', '1']

        self.e.set_hv_state(1)

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_mock.assert_called_once_with(construct_mock.return_value)
        self.assertEqual(1, self.e.hv)

    def test_set_hv_state_fails(self, send_mock, construct_mock):
        send_mock.return_value = False
        expected_params = ['--hvenable', '1']

        self.e.set_hv_state(1)

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_mock.assert_called_once_with(construct_mock.return_value)
        self.assertEqual(0, self.e.hv)

    def test_set_hv_state_raises(self, send_mock, _2):
        send_mock.return_value = True

        with self.assertRaises(ValueError):
            self.e.set_hv_state(2)

    def test_set_hv_bias(self, send_mock, construct_mock):
        send_mock.return_value = True
        expected_params = ['--hvbias', '120']

        self.e.set_hv_bias(120)

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_mock.assert_called_once_with(construct_mock.return_value)
        self.assertEqual(120, self.e.hv_bias)

    def test_set_hv_bias_fails(self, send_mock, construct_mock):
        send_mock.return_value = False
        expected_params = ['--hvbias', '120']

        self.e.set_hv_bias(120)

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_mock.assert_called_once_with(construct_mock.return_value)
        self.assertEqual(0, self.e.hv_bias)

    def test_set_hv_bias_raises(self, send_mock, _2):
        send_mock.return_value = True

        with self.assertRaises(ValueError):
            self.e.set_hv_bias(200)

    def test_sense(self, send_mock, construct_mock):
        expected_params_1 = ['--sensedac', '1', '--dacs=test_file']
        expected_params_2 = ['--sensedac', '1', '-s']

        self.e.sense(self.chips, "Threshold0", "test_file")

        construct_mock.assert_has_calls([call(self.chips, *expected_params_1),
                                         call(self.chips, *expected_params_2)])
        send_mock.assert_has_calls([call(construct_mock.return_value),
                                    call(construct_mock.return_value,
                                         loud_call=True)])

    def test_perform_dac_scan(self, send_mock, construct_mock):
        expected_params = ['--dacs=dac_file', '-t', '5', '--dacscan',
                           '1,0,10,1', '--path=path', '--hdffile=hdf_file']
        scan_range = MagicMock()
        scan_range.start = 0
        scan_range.stop = 10
        scan_range.step = 1

        self.e.perform_dac_scan(self.chips, "Threshold1", scan_range, 5,
                                "dac_file", "path", "hdf_file")

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_mock.assert_called_once_with(construct_mock.return_value)

    def test_read_chip_ids(self, send_cmd_mock, construct_mock):
        expected_params = ['-r', '-e']

        self.e.read_chip_ids()

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_cmd_mock.assert_called_once_with(construct_mock.return_value,
                                              loud_call=True)

    def test_read_chip_id_with_outfile(self, send_cmd_mock, construct_mock):
        expected_params = ['-r', '-e']
        mock_outfile = MagicMock()

        self.e.read_chip_ids(stdout=mock_outfile)

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_cmd_mock.assert_called_once_with(construct_mock.return_value,
                                              loud_call=True,
                                              stdout=mock_outfile)

    def test_read_slow_params(self, send_cmd_mock, construct_mock):
        expected_params = ['-s']

        self.e.read_slow_control_parameters()

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_cmd_mock.assert_called_once_with(construct_mock.return_value,
                                              loud_call=True)

    def test_load_dacs(self, send_cmd_mock, construct_mock):
        send_cmd_mock.return_value = True
        expected_params = ['--dacs=test_path/test_file']

        self.e.load_dacs(self.chips, "test_path/test_file")

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_cmd_mock.assert_called_once_with(construct_mock.return_value)
        self.assertEqual("test_file", self.e.dacs_loaded)

    def test_load_dacs_fails(self, send_cmd_mock, construct_mock):
        send_cmd_mock.return_value = False
        expected_params = ['--dacs=test_path/test_file']

        self.e.load_dacs(self.chips, "test_path/test_file")

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_cmd_mock.assert_called_once_with(construct_mock.return_value)
        self.assertIsNone(self.e.dacs_loaded)

    def test_configure_test_pulse(self, send_cmd_mock, construct_mock):
        tp_mask = MagicMock()
        expected_params = ['--dacs=test_file', '--tpmask=' + tp_mask]

        self.e.configure_test_pulse(self.chips, tp_mask, "test_file")

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_cmd_mock.assert_called_once_with(construct_mock.return_value)

    def test_configure_test_pulse_with_disc(self, send_cmd_mock, construct_mock):
        tp_mask = MagicMock()
        disc_l_mock = MagicMock()
        disc_h_mock = MagicMock()
        mask_mock = MagicMock()
        disc_files = dict(discL=disc_l_mock, discH=disc_h_mock,
                          pixel_mask=mask_mock)
        expected_params = ['--dacs=test_file', '--tpmask=' + tp_mask,
                           '--discl=' + disc_l_mock, '--disch=' + disc_h_mock,
                           '--pixelmask=' + mask_mock]

        self.e.configure_test_pulse(self.chips, tp_mask, "test_file", disc_files)

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_cmd_mock.assert_called_once_with(construct_mock.return_value)

    def test_load_tp_mask(self, send_cmd_mock, construct_mock):
        tp_mask = MagicMock()
        expected_params = ['--config', '--tpmask=' + tp_mask]

        self.e.load_tp_mask(self.chips, tp_mask)

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_cmd_mock.assert_called_once_with(construct_mock.return_value)


class CheckArgumentValidTest(unittest.TestCase):

    def test_given_None_then_False(self):

        response = ExcaliburTestAppInterface._arg_valid("Test", None, [])

        self.assertFalse(response)

    def test_given_invalid_then_raise(self):

        with self.assertRaises(ValueError):
            ExcaliburTestAppInterface._arg_valid("Test", 10, [1, 2, 3, 4, 5])

    def test_given_valid_then_True(self):

        response = ExcaliburTestAppInterface._arg_valid("Test", 5, [1, 2, 3, 4, 5])

        self.assertTrue(response)


class MaskTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburTestAppInterface(1, "test_ip", "test_port")

    def test_return_values(self):

        value = self.e._mask([0])
        self.assertEqual('0x80', value)

        value = self.e._mask([0, 4])
        self.assertEqual('0x88', value)

        value = self.e._mask([0, 1, 4, 5, 7])
        self.assertEqual('0xcd', value)

        value = self.e._mask([0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual('0xff', value)

    def test_given_invalid_value_then_error(self):

        with self.assertRaises(ValueError):
            self.e._mask([8])

    def test_given_duplicate_value_then_error(self):

        with self.assertRaises(ValueError):
            self.e._mask([1, 1])


@patch(ETAI_patch_path + '._send_command')
@patch(ETAI_patch_path + '._construct_command')
class LoadConfigTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburTestAppInterface(1, "test_ip", "test_port")
        self.discl = "discl"
        self.disch = "disch"
        self.pixelmask = "pixelmask"
        self.chips = [0]

    @patch('os.path.isfile', return_value=True)
    def test_load_config_all_exist(self, _, construct_mock, send_mock):
        expected_params = ['--config', '--discl=' + self.discl, '--disch=' + self.disch, '--pixelmask=' + self.pixelmask]

        self.e.load_config(self.chips, self.discl, self.disch, self.pixelmask)

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_mock.assert_called_once_with(construct_mock.return_value)

    @patch('os.path.isfile', side_effect=[True, True, False])
    def test_load_config_L_and_H(self, _, construct_mock, send_mock):
        expected_params = ['--config', '--discl=' + self.discl, '--disch=' + self.disch]

        self.e.load_config(self.chips, self.discl, self.disch, self.pixelmask)

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_mock.assert_called_once_with(construct_mock.return_value)

    @patch('os.path.isfile', side_effect=[True, False, True])
    def test_load_config_L_and_pixel(self, _, construct_mock, send_mock):
        expected_params = ['--config', '--discl=' + self.discl, '--pixelmask=' + self.pixelmask]

        self.e.load_config(self.chips, self.discl, self.disch, self.pixelmask)

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_mock.assert_called_once_with(construct_mock.return_value)

    @patch('os.path.isfile', side_effect=[True, False, False])
    def test_load_config_L(self, _, construct_mock, send_mock):
        expected_params = ['--config', '--discl=' + self.discl]

        self.e.load_config(self.chips, self.discl, self.disch, self.pixelmask)

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_mock.assert_called_once_with(construct_mock.return_value)

    @patch('os.path.isfile', return_value=False)
    def test_load_config_none_exist(self, _, construct_mock, send_mock):

        self.e.load_config(self.chips, self.discl, self.disch, self.pixelmask)

        self.assertFalse(construct_mock.call_count)
        self.assertFalse(send_mock.call_count)


class SimpleWrappersTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburTestAppInterface(1, "test_ip", "test_port",
                                           "test_server")
        self.chips = range(8)

    @patch(ETAI_patch_path + '.acquire')
    def test_acquire_tp_image(self, acquire_mock):
        expected_params = [1, 100]
        expected_kwargs = dict(hdf_file='test.hdf5', path='/scratch',
                               tp_count=1000)

        self.e.acquire_tp_image(self.chips, 100, 1000, "test.hdf5",
                                path="/scratch")

        acquire_mock.assert_called_once_with(self.chips, *expected_params,
                                             **expected_kwargs)


class GrabRemoteFile(unittest.TestCase):

    @patch(ETAI_patch_path + '._send_command')
    def test_correct_calls_made(self, send_mock):
        e = ExcaliburTestAppInterface(1, "test_ip", "test_port", "test_server")

        e.grab_remote_file("/tmp/test_file")

        send_mock.assert_called_once_with(
            ['scp', 'test_server.diamond.ac.uk:/tmp/test_file',
             '/tmp/test_file_fem1'])
