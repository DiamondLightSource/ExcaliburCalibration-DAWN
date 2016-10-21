import unittest

from pkg_resources import require
require("mock")
from mock import patch, MagicMock, ANY

from excaliburcalibrationdawn.excalibur3M import Excalibur3M
Detector_patch_path = "excaliburcalibrationdawn.excaliburdetector" \
                      ".ExcaliburDetector"


class InitTest(unittest.TestCase):

    @patch(Detector_patch_path + '.__init__')
    def test_super_called(self, excalibur_detector_mock):
        Excalibur3M("test-server", 1)

        excalibur_detector_mock.assert_called_once_with("test-server",
                                                        [1, 2, 3, 4, 5, 6], 1)
