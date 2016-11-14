"""An Excalibur RX detector."""
import shutil
import posixpath

import numpy as np

from excaliburcalibrationdawn.excaliburnode import ExcaliburNode
from excaliburcalibrationdawn.excaliburdawn import ExcaliburDAWN
from excaliburcalibrationdawn import util

import logging
logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)


class ExcaliburDetector(object):

    """A abstract class representing an Excalibur RX detector."""

    node_shape = [256, 8*256]
    valid_nodes = [1, 2, 3, 4, 5, 6]

    root_path = '/dls/detectors/support/silicon_pixels/excaliburRX/'
    calib_dir = posixpath.join(root_path, '3M-RX001/calib')

    def __init__(self, detector_name, nodes, master_node):
        """Initialise detector.

        Args:
            detector_name(str): Name of detector; string that gives the server
                name for each node if the suffix is added - e.g. p99-excalibur0
                where p99-excalibur01 is the server for node 6 (nodes reversed)
            nodes(list(int)): List of identifiers for nodes of detector
            master_node(int): Node to assign as master

        """
        self.server_root = detector_name

        if len(nodes) > len(set(nodes)):
            raise ValueError("Given duplicate node in {nodes}".format(
                                 nodes=nodes))
        if not set(nodes).issubset(self.valid_nodes):
            raise ValueError("Given nodes {nodes} not valid, should be in "
                             "{valid_nodes}".format(
                                 nodes=nodes, valid_nodes=self.valid_nodes))
        if master_node not in nodes:
            raise ValueError("Master node {master} not in given nodes "
                             "{nodes}".format(master=master_node, nodes=nodes))

        self.MasterNode = ExcaliburNode(master_node, self.server_root)
        self.Nodes = [self.MasterNode]
        secondary_nodes = list(nodes)
        secondary_nodes.remove(master_node)
        for node in secondary_nodes:
            self.Nodes.append(ExcaliburNode(node, self.server_root))

        self.dawn = ExcaliburDAWN()

    @property
    def detector_range(self):
        return [range(8)] * len(self.Nodes)

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
            hv_bias(int): Voltage to set

        """
        self.MasterNode.set_hv_bias(hv_bias)

    def set_quiet(self, state):
        """Set the quiet state for each node to given state.

        Args:
            state(bool): True or False for whether terminal output silenced

        """
        for node in self.Nodes:
            node.set_quiet(state)

    def display_status(self):
        """Display status of node."""
        for node in self.Nodes:
            node.display_status()

    def setup(self):
        """Perform necessary initialisation."""
        self.MasterNode.initialise_lv()
        self.MasterNode.set_hv_bias(120)
        # self.MasterNode.enable_hv()
        for node in self.Nodes:
            node.setup()

    def disable(self):
        """Set HV bias to 0 and disable LV and HV."""
        self.MasterNode.disable()

    def monitor(self):
        """Monitor temperature, humidity, FEM voltage status and DAC out."""
        for node in self.Nodes:
            node.monitor()

    def load_config(self):
        """Load detector configuration files and default thresholds."""
        for node in self.Nodes:
            node.load_config()

    def set_gnd_fbk_cas(self, chips=None):
        """Set GND, FBK and CAS values from the config python script.

        Args:
            chips(list(list(int))): Chips to set

        """
        if chips is None:
            chips = self.detector_range
        elif not isinstance(chips[0], list):
            raise ValueError("Argument chips must be a list of lists of chips "
                             "for each node, got {}".format(chips))

        for node_idx, node in enumerate(self.Nodes):
            logging.info("Setting GND, FBK and Cas values from config script "
                         "for node %s", node_idx)
            node.set_gnd_fbk_cas(chips[node_idx])

    def threshold_equalization(self, chips=None):
        """Calibrate discriminator equalization for given chips in detector.

        Args:
            chips(list(list(int))): List of lists of chips for each node

        """
        if chips is None:
            chips = self.detector_range
        elif not isinstance(chips[0], list):
            raise ValueError("Argument chips must be a list of lists of chips "
                             "for each node, got {}".format(chips))

        for node_idx, node in enumerate(self.Nodes):
            logging.info("Equalizing node %s", node_idx)
            node.threshold_equalization(chips[node_idx])

    def acquire_tp_image(self, tp_mask):
        """Load the given test pulse mask and capture a tp image.

        Args:
            tp_mask(str): Mask file in config directory

        """
        images = []
        for node in self.Nodes:
            images.append(node.acquire_tp_image(tp_mask))

        detector_image = self._combine_images(images)

        plot_name = "Excalibur Detector TP Image - {time_stamp}".format(
            time_stamp=util.get_time_stamp())
        self.dawn.plot_image(detector_image, plot_name)

    def expose(self, exposure_time):
        """Acquire single image.

        Args:
            exposure_time(int): Acquire time for image

        Returns:
            numpy.array: Image data

        """
        images = []
        for node in self.Nodes:
            images.append(node.expose(exposure_time))

        detector_image = self._combine_images(images)

        plot_name = "Excalibur Detector Image - {time_stamp}".format(
            time_stamp=util.get_time_stamp())
        self.dawn.plot_image(detector_image, plot_name)

    def scan_dac(self, node_idx, chips, threshold, dac_range):
        """Perform a dac scan and plot the result (mean counts vs DAC values).

        Args:
            node_idx(int) Node to perform scan on
            chips(Any from self.dac_number keys): Chips to scan
            threshold(str): Threshold to scan (ThresholdX DACs - X: 0-7)
            dac_range(Range): Range of DAC values to scan over

        Returns:
            numpy.array: DAC scan data

        """
        logging.info("Performing DAC Scan on node %s", node_idx)
        scan_data = self.Nodes[node_idx].scan_dac(chips, threshold, dac_range)
        return scan_data

    @staticmethod
    def _combine_images(images):
        """Combine images from each node into a full detector image.

        Args:
            images(list): Images from each node

        Returns:
            np.array: Full detector image

        """
        detector_image = images[0]
        for image in images[1:]:
            detector_image = np.concatenate((detector_image, image), axis=0)

        return detector_image

    def rotate_configs(self):
        """Rotate arrays in config files for EPICS.

        Calibration files of node 1, 3 and 5 have to be rotated in order to be
        loaded correctly in EPICS. This routine copies calib into calib_epics
        and rotate discLbits, discHbits and maskbits files when they exist for
        nodes 1, 3, and 5

        """
        epics_calib_path = self.calib_dir + '_epics'
        shutil.copytree(self.calib_dir, epics_calib_path)
        logging.debug("EPICS calibration directory: %s", epics_calib_path)

        for node_idx in [1, 3, 5]:
            self.Nodes[node_idx].rotate_config()

    def _grab_node_slice(self, array, node_idx):
        """Grab a node from a full array.

        Args:
            array(numpy.array): Array to grab from
            node_idx(int): Index of section of array to grab

        Returns:
            numpy.array: Sub array

        """
        start, stop = self._generate_node_range(node_idx)
        return util.grab_slice(array, start, stop)

    def _generate_node_range(self, node_idx):
        """Calculate start and stop coordinates of given node.

        Args:
            node_idx(int): Chip to calculate range for

        """
        height = self.node_shape[0]
        width = self.node_shape[1]

        start = [node_idx * height, 0]
        stop = [start[0] + height - 1, width - 1]
        return start, stop
