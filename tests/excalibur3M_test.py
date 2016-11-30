import unittest

from pkg_resources import require
require("mock")
from mock import patch, MagicMock, ANY

from excaliburcalibrationdawn import Excalibur3M
Detector_patch_path = "excaliburcalibrationdawn.excaliburdetector" \
                      ".ExcaliburDetector"


class InitTest(unittest.TestCase):

    @patch('logging.getLogger')
    @patch(Detector_patch_path + '.__init__')
    def test_super_called(self, excalibur_detector_mock, get_mock):
        detector = MagicMock(name="test-detector", nodes=[1], master_node=1,
                             servers=["test-server"],
                             ip_addresses=["192.168.0.1"])
        config = MagicMock(detector=detector)

        e = Excalibur3M(config)

        get_mock.assert_called_once_with("Excalibur3M")
        self.assertEqual(get_mock.return_value, e.logger)
        excalibur_detector_mock.assert_called_once_with(config)
