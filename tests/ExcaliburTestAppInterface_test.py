import unittest
from subprocess import CalledProcessError

from pkg_resources import require
require("mock")
from mock import patch, MagicMock, ANY

from scripts.ExcaliburTestAppInterface import ExcaliburTestAppInterface
ETAI_patch_path = "scripts.ExcaliburTestAppInterface.ExcaliburTestAppInterface"


class TestInit(unittest.TestCase):

    def test_attributes_set(self):
        e = ExcaliburTestAppInterface("test_ip", "test_port")
        expected_path = "/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp"

        self.assertEqual(expected_path, e.path)
        self.assertEqual("test_ip", e.ip)
        self.assertEqual("test_port", e.port)
        self.assertEqual(['/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp', '-i', 'test_ip', '-p', 'test_port'], e.base_cmd)


@patch(ETAI_patch_path + '._mask')
class TestConstructCommand(unittest.TestCase):

    def test_returns(self, mask_mock):
        e = ExcaliburTestAppInterface("test_ip", "test_port")
        expected_command = ['/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp',
                            '-i', 'test_ip',
                            '-p', 'test_port',
                            '-m', mask_mock.return_value]
        chips = [0, 4, 5, 7]

        command = e._construct_command(chips)

        mask_mock.assert_called_once_with(chips)
        self.assertEqual(expected_command, command)

    def test_adds_args_and_returns(self, mask_mock):
        e = ExcaliburTestAppInterface("test_ip", "test_port")
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


@patch('logging.debug')
@patch('subprocess.check_output')
class TestSendCommand(unittest.TestCase):

    def test_subp_called_and_logged(self, subp_mock, logging_mock):
        e = ExcaliburTestAppInterface("test_ip", "test_port")
        expected_message = "Sending command: 'test_command' with kwargs {'test': True}"
        subp_mock.return_value = "Success"

        e._send_command(["test_command"], test=True)

        self.assertEqual(expected_message, logging_mock.call_args_list[0][0][0])
        subp_mock.assert_called_once_with(["test_command"], test=True)
        self.assertEqual("Success", logging_mock.call_args_list[1][0][0])

    def test_error_raised_then_catch_and_log(self, subp_mock, logging_mock):
        e = ExcaliburTestAppInterface("test_ip", "test_port")
        expected_message = "Sending command: 'test_command' with kwargs {'test': True}"
        subp_mock.side_effect = CalledProcessError(1, "test_command", output="Invalid command")

        e._send_command(["test_command"], test=True)

        self.assertEqual(expected_message, logging_mock.call_args_list[0][0][0])
        subp_mock.assert_called_once_with(["test_command"], test=True)
        self.assertEqual("Invalid command", logging_mock.call_args_list[1][0][0])


@patch(ETAI_patch_path + '._construct_command')
@patch(ETAI_patch_path + '._send_command')
class TestAPICalls(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburTestAppInterface("test_ip", "test_port")
        self.chips = range(8)

    def test_acquire(self, send_mock, construct_mock):
        expected_params = ['-n', '100', '-t', '10', '--burst', '--readmode', '1', '--hdffile', 'test.hdf5']
        frames = 100
        acquire_time = 10

        self.e.acquire(self.chips, frames, acquire_time, burst=True, readmode=1, hdffile="test.hdf5")

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_mock.assert_called_once_with(construct_mock.return_value)

    @patch('time.sleep')
    def test_sense(self, sleep_mock, send_mock, construct_mock):
        expected_params_1 = ['--sensedac', '1', '--dacs', 'test_file']
        expected_params_2 = ['--sensedac', '1', '-s']

        self.e.sense(self.chips, 1, "test_file")

        self.assertEqual((self.chips, ) + tuple(expected_params_1), construct_mock.call_args_list[0][0])
        self.assertEqual(construct_mock.return_value, send_mock.call_args_list[0][0][0])
        sleep_mock.assert_called_once_with(1)
        self.assertEqual((self.chips, ) + tuple(expected_params_2), construct_mock.call_args_list[1][0])
        self.assertEqual(construct_mock.return_value, send_mock.call_args_list[1][0][0])

    def test_perform_dac_scan(self, send_mock, construct_mock):
        expected_params = ['--dacs', 'dac_file', '--dacscan', '4,0,10,1', '--hdffile', 'hdf_file']
        scan_range = MagicMock()
        scan_range.start = 0
        scan_range.stop = 10
        scan_range.step = 1

        self.e.perform_dac_scan(self.chips, 4, scan_range, "dac_file", "hdf_file")

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_mock.assert_called_once_with(construct_mock.return_value)

    def test_read_chip_ids(self, send_cmd_mock, construct_mock):
        expected_params = ['-r', '-e']

        self.e.read_chip_ids()

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_cmd_mock.assert_called_once_with(construct_mock.return_value)

    def test_read_chip_id_with_outfile(self, send_cmd_mock, construct_mock):
        expected_params = ['-r', '-e']
        mock_outfile = MagicMock()

        self.e.read_chip_ids(stdout=mock_outfile)

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_cmd_mock.assert_called_once_with(construct_mock.return_value,
                                              stdout=mock_outfile)

    def test_read_slow_params(self, send_cmd_mock, construct_mock):
        expected_params = ['-s']

        self.e.read_slow_control_parameters()

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_cmd_mock.assert_called_once_with(construct_mock.return_value)

    def test_load_dacs(self, send_cmd_mock, construct_mock):
        expected_params = ['--dacs', 'test_file']

        self.e.load_dacs(self.chips, "test_file")

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_cmd_mock.assert_called_once_with(construct_mock.return_value)

    def test_configure_test_pulse(self, send_cmd_mock, construct_mock):
        tp_mask = MagicMock()
        expected_params = ['--dacs', 'test_file', '--tpmask', tp_mask]

        self.e.configure_test_pulse(self.chips, "test_file", tp_mask)

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_cmd_mock.assert_called_once_with(construct_mock.return_value)

    def test_configure_test_pulse_with_disc(self, send_cmd_mock, construct_mock):
        tp_mask = MagicMock()
        disc_files = dict(discl="discl.txt", disch="disch.txt", pixelmask="mask.txt")
        expected_params = ['--dacs', 'test_file', '--tpmask', tp_mask, '--discl', 'discl.txt', '--disch', 'disch.txt', '--pixelmask', 'mask.txt']

        self.e.configure_test_pulse_with_disc(self.chips, "test_file", tp_mask, disc_files)

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_cmd_mock.assert_called_once_with(construct_mock.return_value)


class MaskTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburTestAppInterface("test_ip", "test_port")

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
        self.e = ExcaliburTestAppInterface("test_ip", "test_port")
        self.discl = "discl"
        self.disch = "disch"
        self.pixelmask = "pixelmask"
        self.chips = [0]

    @patch('os.path.isfile', return_value=True)
    def test_load_config_all_exist(self, _, construct_mock, send_mock):
        expected_params = ['--config', '--discl', self.discl, '--disch', self.disch, '--pixelmask', self.pixelmask]

        self.e.load_config(self.chips, self.discl, self.disch, self.pixelmask)

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_mock.assert_called_once_with(construct_mock.return_value)

    @patch('os.path.isfile', side_effect=[True, True, False])
    def test_load_config_L_and_H(self, _, construct_mock, send_mock):
        expected_params = ['--config', '--discl', self.discl, '--disch', self.disch]

        self.e.load_config(self.chips, self.discl, self.disch, self.pixelmask)

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_mock.assert_called_once_with(construct_mock.return_value)

    @patch('os.path.isfile', side_effect=[True, False, True])
    def test_load_config_L_and_pixel(self, _, construct_mock, send_mock):
        expected_params = ['--config', '--discl', self.discl, '--pixelmask', self.pixelmask]

        self.e.load_config(self.chips, self.discl, self.disch, self.pixelmask)

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_mock.assert_called_once_with(construct_mock.return_value)

    @patch('os.path.isfile', side_effect=[True, False, False])
    def test_load_config_L(self, _, construct_mock, send_mock):
        expected_params = ['--config', '--discl', self.discl]

        self.e.load_config(self.chips, self.discl, self.disch, self.pixelmask)

        construct_mock.assert_called_once_with(self.chips, *expected_params)
        send_mock.assert_called_once_with(construct_mock.return_value)

    @patch('os.path.isfile', return_value=False)
    def test_load_config_none_exist(self, _, construct_mock, send_mock):

        self.e.load_config(self.chips, self.discl, self.disch, self.pixelmask)

        self.assertFalse(construct_mock.call_count)
        self.assertFalse(send_mock.call_count)
