"""A 1M Excalibur detector."""
from excaliburcalibrationdawn.excaliburnode import ExcaliburNode
from excaliburcalibrationdawn.excaliburdawn import ExcaliburDAWN
from excaliburcalibrationdawn import arrayutil as util

import numpy as np


class Excalibur1M(object):

    """A class representing a 1M Excalibur detector composed of two nodes."""

    node_shape = [256, 8*256]

    def __init__(self):
        """Initialise two ExcaliburNode instances as a 1M detector."""
        self.nodes = [ExcaliburNode(1), ExcaliburNode(2)]

        self.dawn = ExcaliburDAWN()

    def threshold_equalization(self, chips):
        """Calibrate discriminator equalization for given chips in detector.

        Args:
            chips: List of lists of chips for each node

        """
        for node_idx, node in enumerate(self.nodes):
            node.threshold_equalization(chips[node_idx])

    def optimize_dac_disc(self, chips, roi):
        """Optimize discriminators for given chips in detector.

        Args:
            chips: List of lists of chips for each node
            roi: Mask to apply during optimization

        """
        for node_idx, node in enumerate(self.nodes):
            node_roi = self._grab_node_slice(roi, node_idx)
            node.optimize_dac_disc(chips[node_idx], "DACDiscL", node_roi)
            node.optimize_dac_disc(chips[node_idx], "DACDiscH", node_roi)

    def expose(self, exposure_time):
        """Acquire single image using.

        Args:
            exposure_time: Acquire time for image

        Returns:
            numpy.array: Image data

        """
        for node in self.nodes:
            node.settings['acquire_time'] = exposure_time

        image = self.nodes[0].expose()
        for node_idx, node in enumerate(self.nodes[1:]):
            image = np.concatenate((image, node.expose()), axis=1)

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
