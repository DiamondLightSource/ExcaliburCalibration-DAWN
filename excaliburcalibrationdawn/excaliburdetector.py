"""An Excalibur RX detector."""
from excaliburcalibrationdawn.excaliburnode import ExcaliburNode
from excaliburcalibrationdawn.excaliburdawn import ExcaliburDAWN
from excaliburcalibrationdawn import arrayutil as util

import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG)


class ExcaliburDetector(object):

    """A abstract class representing an Excalibur RX detector."""

    node_shape = [256, 8*256]
    valid_nodes = [1, 2, 3, 4, 5, 6]

    def __init__(self, detector_name, nodes, master_node):
        """Initialise detector.

        Args:
            detector_name: Name of detector; string that gives the server name
                for each node if the suffix is added - e.g. p99-excalibur0
                where p99-excalibur01 is the server for node 6 (nodes reversed)
            nodes: List of identifiers for nodes of detector
            master_node: Identifier for master node of detector

        """
        self.server_root = detector_name

        if not set(nodes).issubset(self.valid_nodes):
            raise ValueError("Given nodes {nodes} not valid, should be in "
                             "{valid_nodes}".format(
                                 nodes=nodes, valid_nodes=self.valid_nodes))

        self.Nodes = []
        for node in nodes:
            self.Nodes.append(ExcaliburNode(node, self.server_root))
        self.MasterNode = self.Nodes[master_node - 1]

        self.dawn = ExcaliburDAWN()

    def read_chip_ids(self):
        """Read chip IDs for all chips in all nodes."""
        for node in self.Nodes:
            node.read_chip_ids()

    def initialise_lv(self):
        """Initialise LV."""
        self.MasterNode.initialise_lv()

    def enable_lv(self):
        """Enable LV."""
        self.MasterNode.enable_lv()

    def disable_lv(self):
        """Disable LV."""
        self.MasterNode.disable_lv()

    def enable_hv(self):
        """Enable HV."""
        self.MasterNode.enable_hv()

    def disable_hv(self):
        """Disable HV."""
        self.MasterNode.disable_hv()

    def set_hv_bias(self, hv_bias):
        """Set HV bias.

        Args:
            hv_bias: Voltage to set

        """
        self.MasterNode.set_hv_bias(hv_bias)

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
