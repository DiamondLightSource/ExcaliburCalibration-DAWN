"""A 1M Excalibur detector."""
from excaliburcalibrationdawn.excaliburdetector import ExcaliburDetector

import logging
logging.basicConfig(level=logging.DEBUG)


class Excalibur1M(ExcaliburDetector):

    """A class representing a 1M Excalibur detector composed of two nodes."""

    def __init__(self, detector_name, nodes, master_node):
        """Initialise two ExcaliburNode instances as a 1M detector.

        Args:
            detector_name: Name of detector; string that gives the server name
                for each node if the suffix is added - e.g. p99-excalibur0
                where p99-excalibur01 is the server for node 6 (nodes reversed)
            nodes: Identifier for second node of detector
            master_node: Identifier for master node of detector

        """
        super(Excalibur1M, self).__init__(detector_name, nodes, master_node)

        logging.debug("Creating Excalibur1M with server %s and nodes %s "
                      "(master node is %s)", detector_name, nodes, master_node)
