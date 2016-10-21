import unittest

from pkg_resources import require
require("mock")
from mock import patch, MagicMock, ANY

from excaliburcalibrationdawn.excalibur1M import Excalibur1M

Detector_patch_path = "excaliburcalibrationdawn.excaliburdetector" \
                      ".ExcaliburDetector"


class InitTest(unittest.TestCase):

    @patch(Detector_patch_path + '.__init__')
    def test_super_called(self, excalibur_detector_mock):
        Excalibur1M("test-server", [1, 2], 1)

        excalibur_detector_mock.assert_called_once_with("test-server",
                                                        [1, 2], 1)
