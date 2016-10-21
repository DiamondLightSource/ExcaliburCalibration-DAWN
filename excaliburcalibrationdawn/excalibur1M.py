"""A 1M Excalibur detector."""
from inspect import getmembers, ismethod

from excaliburcalibrationdawn.excaliburnode import ExcaliburNode
from excaliburcalibrationdawn.excaliburdawn import ExcaliburDAWN
from excaliburcalibrationdawn import arrayutil as util

import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG)


class Excalibur1M(object):

    """A class representing a 1M Excalibur detector composed of two nodes."""

    node_shape = [256, 8*256]

    _node_methods = ["read_chip_ids"]
    _master_node_methods = ["initialise_lv", "enable_lv", "disable_lv",
                            "enable_hv", "disable_hv", "set_hv_bias"]

    def __init__(self, detector_name, master_node, node_2):
        """Initialise two ExcaliburNode instances as a 1M detector.

        Args:
            detector_name: Name of detector; string that gives the server name
                for each node if the suffix is added - e.g. p99-excalibur0
                where p99-excalibur01 is the server for node 6 (nodes reversed)
            master_node: Identifier for master node of detector
            node_2: Identifier for second node of detector

        """
        self._check_node_methods()
        for method in self._node_methods:
            self._create_node_method(method)
        for method in self._master_node_methods:
            self._create_master_node_method(method)

        logging.debug("Creating Excalibur1M with server %s and nodes %s, %s",
                      detector_name, master_node, node_2)

        self.server_root = detector_name
        self.MasterNode = ExcaliburNode(master_node, self.server_root)
        self.Nodes = [self.MasterNode,
                      ExcaliburNode(node_2, self.server_root)]

        self.dawn = ExcaliburDAWN()

    def _create_node_method(self, method):
        """Create method that calls `method` on all nodes.

        Args:
            method: Name of method

        """
        def _call_method_on_nodes(*params):
            for node in self.Nodes:
                node.__getattribute__(method)(*params)

        self.__setattr__(method, _call_method_on_nodes)

    def _create_master_node_method(self, method):
        """Create method that calls `method` on master node.

        Args:
            method: Name of method

        """
        def _call_method_on_master_node(*params):
            self.MasterNode.__getattribute__(method)(*params)

        self.__setattr__(method, _call_method_on_master_node)

    def _check_node_methods(self):
        """Check the methods in *_methods exist in ExcaliburNode."""
        asserted_methods = self._node_methods + self._master_node_methods
        methods = [name for name, _ in getmembers(ExcaliburNode, ismethod)]

        for method in asserted_methods:
            if method not in methods:
                raise AttributeError("ExcaliburNode does not have method {} "
                                     "asserted in '*_methods' lists".
                                     format(method))

    def threshold_equalization(self, chips):
        """Calibrate discriminator equalization for given chips in detector.

        Args:
            chips: List of lists of chips for each node

        """
        for node_idx, node in enumerate(self.Nodes):
            node.threshold_equalization(chips[node_idx])

    def optimize_dac_disc(self, chips, roi):
        """Optimize discriminators for given chips in detector.

        Args:
            chips: List of lists of chips for each node
            roi: Mask to apply during optimization

        """
        for node_idx, node in enumerate(self.Nodes):
            node_roi = self._grab_node_slice(roi, node_idx)
            node.optimize_disc_l(chips[node_idx], node_roi)
            node.optimize_disc_h(chips[node_idx], node_roi)

    def expose(self, exposure_time):
        """Acquire single image.

        Args:
            exposure_time: Acquire time for image

        Returns:
            numpy.array: Image data

        """
        for node in self.Nodes:
            node.settings['acquire_time'] = exposure_time

        image = self.Nodes[0].expose()
        for node_idx, node in enumerate(self.Nodes[1:]):
            image = np.concatenate((image, node.expose()), axis=0)

        self.dawn.plot_image(image, "Excalibur1M Image")

    def _grab_node_slice(self, array, node_idx):
        """Grab a node from a full array.

        Args:
            array: Array to grab from
            node_idx: Index of section of array to grab

        Returns:
            numpy.array: Sub array

        """
        start, stop = self._generate_node_range(node_idx)
        return util.grab_slice(array, start, stop)

    def _generate_node_range(self, node_idx):
        """Calculate start and stop coordinates of given node.

        Args:
            node_idx: Chip to calculate range for

        """
        height = self.node_shape[0]
        width = self.node_shape[1]

        start = [node_idx * height, 0]
        stop = [start[0] + height - 1, width - 1]
        return start, stop
