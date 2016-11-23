"""A 1M Excalibur detector."""
from excaliburcalibrationdawn.excaliburdetector import ExcaliburDetector

import logging


class Excalibur1M(ExcaliburDetector):

    """A class representing a 1M Excalibur detector composed of two nodes."""

    def __init__(self, detector_config):
        """Initialise detector.

        Args:
            detector_config(module): Module in config directory containing
                specifications of detector

        """
        detector = detector_config.detector

        if len(detector.nodes) != 2:
            raise ValueError("Excalibur1M requires two nodes, given "
                             "{nodes}".format(nodes=detector.nodes))

        self.logger = logging.getLogger("Excalibur1M")
        self.logger.debug("Creating Excalibur1M with nodes %s "
                          "(master node is %s) on servers %s with IPs %s",
                          detector.nodes, detector.master_node,
                          detector.servers, detector.ip_addresses)

        super(Excalibur1M, self).__init__(detector_config)
