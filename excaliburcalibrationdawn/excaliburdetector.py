"""An Excalibur RX detector."""
import os
import shutil
import logging

import numpy as np

from excaliburcalibrationdawn.excaliburnode import ExcaliburNode
from excaliburcalibrationdawn.excaliburdawn import ExcaliburDAWN
from excaliburcalibrationdawn import util


class ExcaliburDetector(object):

    """A abstract class representing an Excalibur RX detector."""

    node_shape = [256, 8 * 256]
    node_range = range(8)
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

        self.num_nodes = len(nodes)

        self.Nodes = []
        for node_idx, server, ip_address in zip(nodes, servers, ip_addresses):
            node = ExcaliburNode(node_idx, detector_config, server, ip_address)

            self.Nodes.append(node)
            if node_idx == master_node:
                self.MasterNode = node

        self.dawn = ExcaliburDAWN()
        self.logger = logging.getLogger("ExcaliburDetector")

    @property
    def calib_root(self):
        """Get calibration directory from MasterNode."""
        return self.MasterNode.calib_root

    def setup_new_detector(self):
        """Set up the calibration directories for the given detector config."""
        if os.path.isdir(self.calib_root):
            raise IOError("Calib directory {} already exists".format(
                self.calib_root))

        for node in self.Nodes:
            node.create_calib()

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

    def disable(self):
        """Set HV bias to 0 and disable LV and HV."""
        self.MasterNode.disable()

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

        node_threads = []
        for node in self.Nodes:
            node_threads.append(util.spawn_thread(node.setup))
        util.wait_for_threads(node_threads)

    def monitor(self):
        """Monitor temperature, humidity, FEM voltage status and DAC out."""
        for node in self.Nodes:
            node.monitor()

    def read_chip_ids(self):
        """Read chip IDs for all chips in all nodes."""
        for node in self.Nodes:
            node.read_chip_ids()

    def set_dac(self, node_idx, name, value):
        """Set DAC for given node.

        Args:
            node_idx(int): Node to set DACs for
            name(str): DAC to set (Any from self.dac_number keys)
            value(int): Value to set DAC to

        """
        node = self._find_node(node_idx)
        node.set_dac(self.node_range, name, value)

    def read_dac(self, node_idx, dac_name):
        """Read back DAC analogue voltage for given node.

        Args:
            node_idx(int): Node to read for
            dac_name(str): DAC value to read

        """
        node = self._find_node(node_idx)
        node.read_dac(dac_name)

    def load_config(self):
        """Load detector configuration files and default thresholds."""
        node_threads = []
        for node in self.Nodes:
            node_threads.append(util.spawn_thread(node.load_config))
        util.wait_for_threads(node_threads)

    def unequalise_pixels(self, node=None, chips=None):
        """Reset discriminator bits for the given node chips.

        Args:
            node(int): Node to unequalise - If None, all nodes included
            chips(list(int)): List of chips to include in equalisation - If
                None, all chips included

        """
        nodes, chips = self._validate(node, chips)

        node_threads = []
        for node in nodes:
            node_threads.append(
                util.spawn_thread(node.unequalize_pixels, chips))
        util.wait_for_threads(node_threads)

    def unmask_pixels(self, node=None, chips=None):
        """Reset pixelmask for the given chips.

        Args:
            node(int): Node to unmask - If None, all nodes included
            chips(list(int)): List of chips to include in equalisation - If
                None, all chips included

        """
        nodes, chips = self._validate(node, chips)

        node_threads = []
        for node in nodes:
            node_threads.append(util.spawn_thread(node.unmask_pixels, chips))
        util.wait_for_threads(node_threads)

    def set_gnd_fbk_cas(self, node_idx=None, chips=None):
        """Set GND, FBK and CAS values from the config python script.

        Args:
            node_idx(int): Node to equalise - If None, all nodes included
            chips(list(int)): List of chips to include in equalisation - If
                None, all chips included

        """
        nodes, chips = self._validate(node_idx, chips)

        node_threads = []
        for node in nodes:
            node_threads.append(util.spawn_thread(node.set_gnd_fbk_cas, chips))
        util.wait_for_threads(node_threads)

    def threshold_equalization(self, node=None, chips=None):
        """Calibrate discriminator equalization for given chips in detector.

        Args:
            node(int): Node to equalise - If None, all nodes included
            chips(list(int)): List of chips to include in equalisation - If
                None, all chips included

        """
        nodes, chips = self._validate(node, chips)

        node_threads = []
        for node_ in nodes:
            node_threads.append(
                util.spawn_thread(node_.threshold_equalization, chips))
        util.wait_for_threads(node_threads)

    def _validate(self, node, chips):
        """Check node and chips are valid and generate defaults if None.

        Args:
            node(int): Node to check
            chips(list(int)): Chips to check

        Returns:
            list(ExcaliburNode), list(int): Validated nodes and chips

        """
        if node is None:
            nodes = self.Nodes
        else:
            nodes = [self._find_node(node)]

        if chips is None:
            chips = self.node_range
        else:
            if not set(chips).issubset(self.node_range):
                raise ValueError("Chips should be in {valid}, got "
                                 "{actual}".format(valid=self.node_range,
                                                   actual=chips))

        return nodes, chips

    def acquire_tp_image(self, tp_mask):
        """Load the given test pulse mask and capture a tp image.

        Args:
            tp_mask(str): Mask file in config directory

        """
        node_threads = []
        for node in self.Nodes:
            node_threads.append(
                util.spawn_thread(node.acquire_tp_image, tp_mask))

        images = util.wait_for_threads(node_threads)

        detector_image = self._combine_images(images)

        plot_name = util.tag_plot_name("TPImage", "Detector")
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

        images = util.wait_for_threads(node_threads)

        detector_image = self._combine_images(images)

        plot_name = util.tag_plot_name("Image", "Detector")
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
        self.logger.info("Performing DAC Scan on node %s", node_idx)

        node = self._find_node(node_idx)

        scan_data = node.scan_dac(chips, threshold, dac_range)
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
        self.logger.debug("EPICS calibration directory: %s", epics_calib_path)

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

    def _find_node(self, node_idx):
        """Find the Node object corresponding to node_idx in self.Nodes list.

        Args:
            node_idx(int): Node to find

        Returns:
            ExcaliburNode: Node in self.Nodes with given node index

        """
        for node in self.Nodes:
            if node.fem == node_idx:
                return node

        raise ValueError("Node {} not found in detector nodes.".format(
            node_idx))
