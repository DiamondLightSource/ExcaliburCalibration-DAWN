"""A 3M Excalibur detector."""
from excaliburcalibrationdawn.excaliburdetector import ExcaliburDetector

import logging
logging.basicConfig(level=logging.DEBUG)


class Excalibur3M(ExcaliburDetector):

    """A class representing a 3M Excalibur detector composed of six nodes."""

    nodes = [1, 2, 3, 4, 5, 6]

    def __init__(self, detector_name, master_node):
        """Initialise six ExcaliburNode instances as a 3M detector.

        Args:
            detector_name: Name of detector; string that gives the server name
                for each node if the suffix is added - e.g. p99-excalibur0
                where p99-excalibur01 is the server for node 6 (nodes reversed)
            master_node: Identifier for master node of detector

        """
        super(Excalibur3M, self).__init__(detector_name, self.nodes,
                                          master_node)

        logging.debug("Creating Excalibur3M with server %s and master node %s",
                      detector_name, master_node)
