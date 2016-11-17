"""A 3M Excalibur detector."""
from excaliburcalibrationdawn.excaliburdetector import ExcaliburDetector

import logging
logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)


class Excalibur3M(ExcaliburDetector):

    """A class representing a 3M Excalibur detector composed of six nodes."""

    nodes = [1, 2, 3, 4, 5, 6]

    def __init__(self, detector_config):
        """Initialise detector.

        Args:
            detector_config(module): Module in config directory containing
                specifications of detector

        """
        detector = detector_config.detector

        logging.debug("Creating Excalibur3M with nodes %s (master node is %s) "
                      "on servers %s with IPs %s",
                      detector.nodes, detector.master_node,
                      detector.servers, detector.ip_addresses)

        super(Excalibur3M, self).__init__(detector_config)
