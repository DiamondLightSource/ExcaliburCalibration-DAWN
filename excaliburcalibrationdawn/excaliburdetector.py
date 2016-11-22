"""An Excalibur RX detector."""
import shutil

import numpy as np

from excaliburcalibrationdawn.excaliburnode import ExcaliburNode
from excaliburcalibrationdawn.excaliburdawn import ExcaliburDAWN
from excaliburcalibrationdawn import util

import logging
logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)


class ExcaliburDetector(object):

    """A abstract class representing an Excalibur RX detector."""

    node_shape = [256, 8 * 256]
    valid_nodes = [1, 2, 3, 4, 5, 6]

    def __init__(self, detector_config):
        """Initialise detector.

        Args:
            detector_config(module): Module in config directory containing
                specifications of detector

        """
        nodes = detector_config.detector.nodes
        master_node = detector_config.detector.master_node
        servers = detector_config.detector.servers
        ip_addresses = detector_config.detector.ip_addresses

        if len(nodes) > len(set(nodes)):
            raise ValueError("Given duplicate node in {nodes}".format(
                nodes=nodes))
        if not set(nodes).issubset(self.valid_nodes):
            raise ValueError("Given nodes {nodes} not valid, should all be in "
                             "{valid_nodes}".format(
                                 nodes=nodes,
                                 valid_nodes=self.valid_nodes))
        if master_node not in nodes:
            raise ValueError("Master node {master} not in given nodes "
                             "{nodes}".format(master=master_node,
                                              nodes=nodes))
        if len(nodes) != len(servers) or len(nodes) != len(ip_addresses):
            raise ValueError("Nodes, servers and ip_addresses are different "
                             "lengths")

        self.Nodes = []
        for node_idx, server, ip_address in zip(nodes, servers, ip_addresses):
            node = ExcaliburNode(node_idx, detector_config, server, ip_address)

            self.Nodes.append(node)
            if node_idx == master_node:
                self.MasterNode = node

        self.dawn = ExcaliburDAWN("Detector")

    @property
    def calib_root(self):
        """Get calibration directory from MasterNode."""
        return self.MasterNode.calib_root

    @property
    def detector_range(self):
        """Generate detector range from range and number of nodes."""
        return [range(8)] * len(self.Nodes)

    def read_chip_ids(self):
        """Read chip IDs for all chips in all nodes."""
        for node in self.Nodes:
            util.spawn_thread(node.read_chip_ids)

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
        self.MasterNode.disable_hv()  # In case already setup and LV is enabled
        self.MasterNode.initialise_lv()
        self.MasterNode.set_hv_bias(120)
        # self.MasterNode.enable_hv()
        for node in self.Nodes:
            util.spawn_thread(node.setup)

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
            util.spawn_thread(node.load_config)

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
            util.spawn_thread(node.set_gnd_fbk_cas, chips[node_idx])

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
            util.spawn_thread(node.threshold_equalization, chips[node_idx])

    def acquire_tp_image(self, tp_mask):
        """Load the given test pulse mask and capture a tp image.

        Args:
            tp_mask(str): Mask file in config directory

        """
        node_threads = []
        for node in self.Nodes:
            node_threads.append(
                util.spawn_thread(node.acquire_tp_image, tp_mask))

        images = []
        for thread in node_threads:
            images.append(thread.join())

        detector_image = self._combine_images(images)

        plot_name = "TPImage - {time_stamp}".format(
            time_stamp=util.get_time_stamp())
        self.dawn.plot_image(detector_image, plot_name)

    def expose(self, exposure_time):
        """Acquire single image.

        Args:
            exposure_time(int): Acquire time for image

        Returns:
            numpy.array: Image data

        """
        node_threads = []
        for node in self.Nodes:
            node_threads.append(util.spawn_thread(node.expose, exposure_time))

        images = []
        for thread in node_threads:
            images.append(thread.join())

        detector_image = self._combine_images(images)

        plot_name = "Image - {time_stamp}".format(
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
        epics_calib_path = self.calib_root + '_epics'
        shutil.copytree(self.calib_root, epics_calib_path)
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
