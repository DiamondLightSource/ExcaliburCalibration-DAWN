"""A 1M Excalibur detector."""
from excaliburcalibrationdawn.excaliburdetector import ExcaliburDetector

import logging
logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)


class Excalibur1M(ExcaliburDetector):

    """A class representing a 1M Excalibur detector composed of two nodes."""

    def __init__(self, detector_name, nodes, master_node):
        """Initialise two ExcaliburNode instances as a 1M detector.

        Args:
            detector_name(str): Name of detector; string that gives the server
                name for each node if the suffix is added - e.g. p99-excalibur0
                where p99-excalibur01 is the server for node 6 (nodes reversed)
            nodes(list(int)): Two nodes making up 1M detector
            master_node(int): Node to assign as master

        """
        if len(nodes) != 2:
            raise ValueError("Excalibur1M requires two nodes, given "
                             "{nodes}".format(nodes=nodes))

        super(Excalibur1M, self).__init__(detector_name, nodes, master_node)

        logging.debug("Creating Excalibur1M with server %s and nodes %s "
                      "(master node is %s)", detector_name, nodes, master_node)
