import unittest

from pkg_resources import require
require("mock")
from mock import patch, MagicMock, ANY

from excaliburcalibrationdawn.excalibur1M import Excalibur1M
from excaliburcalibrationdawn.excaliburnode import ExcaliburNode
E1M_patch_path = "excaliburcalibrationdawn.excalibur1M.Excalibur1M"
EN_patch_path = "excaliburcalibrationdawn.excalibur1M.ExcaliburNode"


class InitTest(unittest.TestCase):

    def setUp(self):
        self.e = Excalibur1M()

    def test_class_attributes_set(self):
        self.assertIsInstance(self.e.nodes.Node1, ExcaliburNode)
        self.assertIsInstance(self.e.nodes.Node2, ExcaliburNode)

        self.assertEqual(0, self.e.nodes.Node1.fem)
        self.assertEqual(1, self.e.nodes.Node2.fem)
