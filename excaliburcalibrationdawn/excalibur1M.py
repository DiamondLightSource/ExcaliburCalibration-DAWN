"""A 1M Excalibur detector."""
from collections import namedtuple

from excaliburcalibrationdawn.excaliburnode import ExcaliburNode

Nodes = namedtuple("Nodes", "Node1 Node2")


class Excalibur1M(object):

    """A class representing a 1M Excalibur detector composed of two nodes."""

    def __init__(self):
        """Initialise two ExcaliburNode instances as a 1M detector."""
        self.nodes = Nodes(ExcaliburNode(0), ExcaliburNode(1))

    def optimize_dac_disc(self):
        """"""
        self.nodes.Node1.optimize_dac_disc()
        self.nodes.Node2.optimize_dac_disc()
