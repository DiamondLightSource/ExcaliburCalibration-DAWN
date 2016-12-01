"""Python library for MPX3RX-based detectors calibration and testing."""
from __future__ import print_function  # To test what is printed
import os
import posixpath
import shutil
import time
import logging

from collections import namedtuple

import numpy as np

from excaliburcalibrationdawn.excaliburtestappinterface import \
    ExcaliburTestAppInterface
from excaliburcalibrationdawn.excaliburdawn import ExcaliburDAWN
from excaliburcalibrationdawn import util


Range = namedtuple("Range", ["start", "stop", "step"])


class ExcaliburNode(object):

    """Class to calibrate a node of an Excalibur-RX detector.

    ExcaliburNode is a class defining methods required to calibrate each node
    (8 MPX3-RX chips) of an EXCALIBUR-RX detector.
    These calibration scripts will work only inside the Python interpreter of
    DAWN software running on the PC server node connected to the FEM
    controlling the node that you wish to calibrate
    """

    # Threshold equalization will align pixel noise peaks at this DAC value
    dac_target = 10
    # Acceptable difference compared to dac_target
    allowed_delta = 4

    # Sigma value based on experimental data
    num_sigma = 3.2

    # Number of pixels per chip along each axis
    chip_size = 256
    # Number of chips in 1/2 module
    num_chips = 8
    # Shape of chip
    chip_shape = [chip_size, chip_size]
    # Shape of full row
    full_array_shape = [chip_size, num_chips * chip_size]

    root_path = "/dls/detectors/support/silicon_pixels/excaliburRX/"
    config_dir = posixpath.join(root_path, "TestApplication_15012015/config")
    default_dacs = posixpath.join(config_dir, "Default_SPM.dacs")

    output_folder = "/tmp"  # Location to save data files to
    file_name = "image"  # Default base name for data files

    # Line number used when editing dac file with new dac values
    dac_number = dict(Threshold0=1, Threshold1=2, Threshold2=3, Threshold3=4,
                      Threshold4=5, Threshold5=6, Threshold6=7, Threshold7=8,
                      Preamp=9, Ikrum=10, Shaper=11, Disc=12, DiscLS=13,
                      ShaperTest=14, DACDiscL=15, DACTest=16, DACDiscH=17,
                      Delay=18, TPBuffIn=19, TPBuffOut=20, RPZ=21, GND=22,
                      TPREF=23, FBK=24, Cas=25, TPREFA=26, TPREFB=27)

    chip_range = range(num_chips)
    plot_name = ''

    def __init__(self, node, detector_config, server=None, ip_address=None):
        """Initialize Excalibur node object.

        Args:
            node(int): Number of node (Between 1 and 6 for 3M)
            detector_config(module): Config for detector this node belongs to
            server(str): Server name
            ip_address(str): IP address of node on server

        """
        if node not in [1, 2, 3, 4, 5, 6]:
            raise ValueError("Node {node} is invalid. Should be 1-6.".format(
                node=node))

        self.id = node
        self.node_tag = "Node {}".format(node)
        if ip_address is not None:
            self.ip_address = ip_address
        else:
            self.ip_address = "192.168.0.10{}".format(7 - node)

        if server is not None:
            self.server_name = server
            self.remote_node = True
        else:
            self.server_name = None
            self.remote_node = False

        self.config = detector_config
        self.detector_root = posixpath.join(self.root_path,
                                            "3M-RX001/{}".format(
                                                detector_config.detector.name))
        self.calib_root = posixpath.join(self.detector_root, "calib")
        self.log_output = posixpath.join(self.detector_root, "logs")

        # Detector default settings - See excaliburtestappinterface for details
        self.settings = dict(mode="spm",  # spm or csm
                             gain="hgm",  # slgm, lgm, hgm or shgm
                             bitdepth=12,  # 1, 8, 12 or 24; 24 bits needs
                                           # disccsmspm set at 1 to use discL
                             # TODO: Fix above comment
                             readmode="sequential",  # 0 or 1
                             counter=0,  # 0 or 1
                             disccsmspm="discL",  # discL or discH
                             equalization=0,  # 0 or 1
                             trigmode=0,  # 0, 1 or 2
                             exposure=100,  # In milliseconds
                             frames=1)

        # Helper classes
        self.app = ExcaliburTestAppInterface(self.id, self.ip_address, 6969,
                                             self.server_name)
        self.dawn = ExcaliburDAWN()

        self.logger = logging.getLogger("Node{}".format(self.id))
        self.logger.info("Creating ExcaliburNode with server %s and ip %s",
                         self.server_name, self.ip_address)

    @property
    def current_calib(self):
        """Generate calibration directory for current mode and gain."""
        return posixpath.join(self.calib_root,
                              'fem{id}'.format(id=self.id),
                              self.settings['mode'],
                              self.settings['gain'])

    @property
    def discL_bits(self):
        """Generate discL_bits file paths for current template path."""
        template = posixpath.join(self.current_calib, '{disc}.chip{chip_idx}')
        return [template.format(disc="discLbits",
                                chip_idx=idx) for idx in self.chip_range]

    @property
    def discH_bits(self):
        """Generate discH_bits file paths for current template path."""
        template = posixpath.join(self.current_calib, '{disc}.chip{chip_idx}')
        return [template.format(disc="discHbits",
                                chip_idx=idx) for idx in self.chip_range]

    @property
    def pixel_mask(self):
        """Generate pixelmask file paths for current template path."""
        template = posixpath.join(self.current_calib, '{disc}.chip{chip_idx}')
        return [template.format(disc="pixelmask",
                                chip_idx=idx) for idx in self.chip_range]

    @property
    def dacs_file(self):
        """Generate DACs file path for current template path."""
        return posixpath.join(self.current_calib, "dacs")

    def create_calib(self):
        """Set up the folder structure and files."""
        self._create_calib_structure()
        self._save_chip_ids()
        # Reset to default mode and gain
        self.settings['gain'] = "hgm"
        self.settings['mode'] = "spm"
        # Create dacs file
        shutil.copy(self.default_dacs, self.dacs_file)
        # Create discLbits for each chip
        zeros = np.zeros(shape=self.full_array_shape)
        self.save_discbits(self.chip_range, zeros, "discLbits")
        self.copy_hgm_into_other_gain_modes()

    def _create_calib_structure(self):
        """Create the calibration directory structure."""
        self.logger.info("Creating calibration directory folder structure.")
        template = posixpath.join(self.calib_root, "fem{}".format(self.id),
                                  "spm/{gain}")
        paths = [template.format(gain=gain)
                 for gain in ["shgm", "hgm", "lgm", "slgm"]]
        for path_ in paths:
            os.makedirs(path_)

    def enable_logging_to_file(self):
        """Add a handler to log to file."""
        if not os.path.isdir(self.log_output):
            os.makedirs(self.log_output)
        file_name = "{time}_{node}".format(time=util.get_time_stamp(),
                                           node=self.id)
        file_path = posixpath.join(self.log_output, file_name)
        handler = logging.FileHandler(file_path)
        self.logger.addHandler(handler)
        self.dawn.logger.addHandler(handler)
        self.app.logger.addHandler(handler)

    def setup(self):
        """Perform necessary initialisation."""
        self.check_chip_ids()
        self.app.load_dacs(self.chip_range, self.default_dacs)
        self.load_config(self.chip_range)
        self.copy_hgm_into_other_gain_modes()

    def reset(self):
        """Reset/initialise FEMs"""
        self.read_chip_ids()

    def reboot(self):
        """Reboot FEMs."""
        self.app.reboot()

    def disable(self):
        """Set HV bias to 0 and disable LV and HV."""
        self.set_hv_bias(0)
        self.disable_hv()
        self.disable_lv()

    def enable_lv(self):
        """Enable LV."""
        self.app.set_lv_state(1)

    def disable_lv(self):
        """Disable LV."""
        self.app.set_lv_state(0)

    def initialise_lv(self):
        """Initialise LV; bug in ETA means LV doesn't turn on first time."""
        self.enable_lv()
        self.disable_lv()
        self.enable_lv()

    def enable_hv(self):
        """Enable HV."""
        self.app.set_hv_state(1)

    def disable_hv(self):
        """Disable HV."""
        self.app.set_hv_state(0)

    def set_hv_bias(self, hv_bias):
        """Set HV bias.

        Args:
            hv_bias(int): Voltage to set

        """
        self.app.set_hv_bias(hv_bias)

    def set_quiet(self, state):
        """Set the quiet state for calls to the ExcaliburTestApp.

        Args:
            state(bool): True or False for whether terminal output silenced

        """
        if not isinstance(state, bool):
            raise ValueError("Quiet state can be either True or False.")
        self.app.quiet = state

    def display_status(self):
        """Display status of node."""
        print("Status for Node {}".format(self.id))
        print("LV: {}".format(self.app.lv))
        print("HV: {}".format(self.app.hv))
        print("HV Bias: {}".format(self.app.hv_bias))
        print("DACs Loaded: {}".format(self.app.dacs_loaded))
        print("Initialised: {}".format(self.app.initialised))

    def check_chip_ids(self):
        """Read the eFuse IDs for the current detector and compare to config.

        Raises:
            IOError: If files don't match

        """
        temp_file = posixpath.join(self.output_folder, "temp_id.txt")
        with open(temp_file, "w") as output_file:
            self.app.read_chip_ids(stdout=output_file)

        node_id = posixpath.join(self.calib_root, "fem{}".format(self.id),
                                 "efuseIDs")

        if util.files_match(temp_file, node_id):
            self.logger.info("Detector config ID is a match.")
        else:
            raise IOError("Given detector config does not match current "
                          "detector ID")

    def acquire_tp_image(self, tp_mask, exposure=100, tp_count=1000):
        """Load the given test pulse mask and capture a tp image.

        Args:
            tp_mask(str): Mask file in config directory
            exposure(int) Exposure time for image
            tp_count(int): Test pulse count

        """
        mask_path = posixpath.join(self.config_dir, tp_mask)
        if not os.path.isfile(mask_path):
            raise IOError("Mask file {} does not exist".format(mask_path))

        output_file = self.generate_file_name("TPImage")
        output_path = posixpath.join(self.output_folder, output_file)

        for chip_idx in self.chip_range:
            config_files = dict(discL=self.discL_bits[chip_idx],
                                discH=self.discH_bits[chip_idx],
                                pixel_mask=self.pixel_mask[chip_idx])
            self.app.configure_test_pulse(chip_idx, mask_path,
                                          self.dacs_file, config_files)

        self.app.acquire_tp_image(self.chip_range, exposure, tp_count,
                                  hdf_file=output_file)

        if self.remote_node:
            self.app.grab_remote_file(output_path)
        util.wait_for_file(output_path, 10)
        image = self.dawn.load_image_data(output_path)
        self.dawn.plot_image(image, util.tag_plot_name("TPImage",
                                                       self.node_tag))
        return image

    def save_config_image(self, chips=range(8)):
        """
        Save hdf5 image versions of config files for ease of viewing
        """
        discLbits = np.zeros(self.full_array_shape)
        discHbits = np.zeros(self.full_array_shape)
        pixelmask = np.zeros(self.full_array_shape)
        for chip_idx in chips:
            if os.path.isfile(self.discL_bits[chip_idx]):
                util.set_chip_slice(discLbits, chip_idx,
                                    np.loadtxt(self.discL_bits[chip_idx]))
            if os.path.isfile(self.discH_bits[chip_idx]):
                util.set_chip_slice(discHbits, chip_idx,
                                    np.loadtxt(self.discH_bits[chip_idx]))
            if os.path.isfile(self.pixel_mask[chip_idx]):
                util.set_chip_slice(pixelmask, chip_idx,
                                    np.loadtxt(self.pixel_mask[chip_idx]))

        template = os.path.join(self.current_calib, "{}.hdf5")
        util.create_hdf5_file(discLbits, template.format("discLbits"))
        util.create_hdf5_file(discHbits, template.format("discHbits"))
        util.create_hdf5_file(pixelmask, template.format("pixelmask"))

    def full_calibration(self):
        """Perform the full calibration process."""
        self.setup()
        self.unequalize_pixels(range(8))
        self.enable_logging_to_file()
        self.optimise_gnd_fbk_cas()
        for dac in ["GND", "FBK", "Cas"]:
            self.read_dac(dac)
        self.app.load_dacs(range(8), self.dacs_file)
        self.threshold_equalization()
        self.app.load_dacs(range(8), self.dacs_file)
        self.load_config(range(8))
        self.mask_pixels_using_dac_scan(range(8), "Threshold0",
                                        Range(30, 220, 1))
        self.calibrate_disc_h(range(8))
        self.set_dac(range(8), "Threshold1", 100)
        self.copy_hgm_into_other_gain_modes()
        self.save_config_image()
        self.expose()
        self.expose()

    def threshold_equalization(self, chips=range(8)):
        """Calibrate discriminator equalization.

        You need to edit this function to define which mode (SPM or CSM) and
        which gains you want to calibrate during the threshold_equalization
        sequence.

        Args:
            chips(list(int)): Chip or list of chips to calibrate

        """
        chips = util.to_list(chips)
        self.settings['mode'] = "spm"
        self.settings['gain'] = "hgm"  # SJW - HGM for equalization

        self.check_chip_ids()
        self.backup_calib_dir()
        self.set_dacs(chips)

        self.calibrate_disc_l(chips)

        # self._calibrate_disc(chips, 'discH', 1, 'rect')
        # self.settings['mode'] = 'csm'
        # self.settings['gain'] = 'slgm'
        # self._calibrate_disc(chips, 'discL', 1, 'rect')
        # self._calibrate_disc(chips, 'discH', 1, 'rect')

    def threshold_calibration_all_gains(self, threshold="0"):
        """Calibrate equalization for all gain modes and chips.

        This will save a threshold calibration file called threshold0 or
        threshold1 in the calibration directory under each gain setting
        sub-folder.

        Args:
            threshold(int): Threshold to calibrate (0 or 1)

        """
        self.settings['gain'] = 'shgm'
        self.threshold_calibration(threshold)
        self.settings['gain'] = 'hgm'
        self.threshold_calibration(threshold)
        self.settings['gain'] = 'lgm'
        self.threshold_calibration(threshold)
        self.settings['gain'] = 'slgm'
        self.threshold_calibration(threshold)

    def threshold_calibration(self, threshold=0):
        """Calculate keV to DAC unit conversion for given threshold.

        This function produces threshold calibration data required to convert
        an X-ray energy detection threshold in keV into threshold DAC units
        It uses a first-approximation calibration data assuming that 6keV X-ray
        energy corresponds to dac code = 62 in SHGM. Dac scans showed that this
        was true +/- 2 DAC units in 98% of the chips tested when using
        threshold equalization function with default parameters
        (dacTarget = 10 and nbOfSigma = 3.2).

        Args:
            threshold(int): Threshold to calibrate (0 or 1)

        """
        gain_values = dict(shgm=1.0, hgm=0.75, lgm=0.5, slgm=0.25)
        default_6kev_dac = 62
        E0 = 0
        E1 = 5.9
        dac0 = self.dac_target * np.ones([6, 8]).astype('float')

        self.backup_calib_dir()

        gain = gain_values[self.settings['gain']]
        dac1 = dac0 + gain * (default_6kev_dac - dac0) * \
            np.ones([6, 8]).astype('float')

        self.logger.debug("E0: %s", E0)
        self.logger.debug("DAC0 Array: %s", dac0)
        self.logger.debug("E1: %s", E1)
        self.logger.debug("DAC1 Array: %s", dac1)

        slope = (dac1[self.id - 1, :] - dac0[self.id - 1, :]) / (E1 - E0)
        offset = dac0[self.id - 1, :]
        self.save_kev2dac_calib(threshold, slope, offset)
        self.logger.debug("Slope: %s, Offset: %s", slope, offset)

    def save_kev2dac_calib(self, threshold, gain, offset):
        """Save KeV conversion data to file.

        Args:
            threshold(int): Threshold calibration to save (0 or 1)
            gain(int/float): Conversion scale factor
            offset(int/float): Conversion offset

        """
        thresh_coeff = np.array([gain, offset])

        thresh_filename = posixpath.join(self.calib_root,
                                         'fem{id}',
                                         self.settings['mode'],
                                         self.settings['gain'],
                                         'threshold{threshold}'
                                         ).format(id=self.id,
                                                  threshold=threshold)

        self.logger.info("Saving calibration to: %s", thresh_filename)
        np.savetxt(thresh_filename, thresh_coeff, fmt='%.2f')
        os.chmod(thresh_filename, 0777)  # Allow anyone to overwrite

    def find_xray_energy_dac(self, chips=range(8), threshold=0, energy=5.9):
        """############## NOT TESTED.

        Perform a DAC scan and fit monochromatic spectra in order to find the
        DAC value corresponding to the X-ray energy

        Args:
            chips(list(int)): Chips to scan
            threshold(int): Threshold to scan (0 or 1)
            energy(float): X-ray energy for scan

        Returns:
            numpy.array, list: DAC scan array, DAC values of scan

        """
        self.settings['exposure'] = 100
        if self.settings['gain'] == 'shgm':
            dac_range = Range(self.dac_target + 100, self.dac_target + 20, 2)
        else:
            raise NotImplementedError()

        self.load_config(chips)
        filename = 'Threshold{threshold}Scan_{energy}keV'.format(
            threshold=threshold, energy=energy)
        self.file_name = filename

        dac_scan_data = self.scan_dac(chips, "Threshold" + str(threshold),
                                      dac_range)
        dac_scan_data[dac_scan_data > 200] = 0  # Set elements > 200 to 0
        [chip_dac_scan, dac_axis] = self.display_dac_scan(chips, dac_scan_data,
                                                          dac_range)
        scan_data = []
        for chip_idx in chips:
            scan_data.append(chip_dac_scan[chip_idx, :])
        self.dawn.fit_dac_scan(scan_data, dac_axis)

        # edge_dacs = self.find_edge(chips, dac_scan_data, dac_range, 2)
        # chip_edge_dacs = np.zeros(range(8))
        # for chip in chips:
        #     chip_edge_dacs[chip] = edge_dacs[0:256,
        #                            chip*256:chip*256 + 256].mean()

        return chip_dac_scan, dac_axis

    def mask_rows(self, chips, start, stop):
        """Mask a block of rows of pixels on the given chip(s).

        Args:
            chips(int/list(int)): Index(es) of chip(s) to mask
            start(int): Index of starting row of mask
            stop(int): Index of final row to be included in mask

        """
        chips = util.to_list(chips)

        mask = np.zeros(self.full_array_shape)
        for chip_idx in chips:
            # Subtract 1 to convert chip_size from number to index
            util.set_slice(mask, [start, chip_idx * self.chip_size],
                           [stop, (chip_idx + 1) * self.chip_size - 1], 1)

        self._apply_mask(chips, mask)
        self.dawn.plot_image(mask, util.tag_plot_name("Mask", self.node_tag))

    def one_energy_thresh_calib(self, threshold=0):
        """Plot single energy threshold calibration spectra.

        This function produces threshold calibration files using DAC scans
        performed with a monochromatic X-ray source. Dac scans and spectra are
        performed and plotted in dac scan and spectrum plots. You need to
        inspect the spectra and edit manually the array oneE_DAC with the DAC
        value corresponding to the X-ray energy for each chip

        This method will use 2 points for calculating Energy to DAC conversion
        coefficients:

        * The energy used in the dac scan
        * The noise peak at dacTarget (10 DAC units by default)

        Possible improvement: Fit dacs scans to populate automatically
        oneE_DAC array (cf fit dac scan)

        Args:
            threshold(int): Threshold to calibrate (0 or 1)

        """
        self.settings['exposure'] = 1000
        self.file_name = 'Single_energy_threshold_scan'

        # dacRange = (self.dacTarget + 80, self.dacTarget + 20, 2)
        # self.load_config(range(8))
        # [dacscanData, scanRange] = self.scan_dac(chips, "Threshold" +
        #                                         str(Threshold), dacRange)
        # edgeDacs = self.find_edge(chips, dacscanData, dacRange, 2)
        # chipEdgeDacs = np.zeros([len(chips)])
        # for chip in chips:
        #     chipEdgeDacs[chip] = \
        #         edgeDacs[0:256, chip*256:chip*256 + 256].mean()

        OneE_E = 6
        OneE_Dac = self.config.E1_DAC[self.settings['gain']]

        slope = (OneE_Dac[self.id - 1, :] - self.dac_target) / OneE_E
        offset = [self.dac_target] * 8
        self.save_kev2dac_calib(threshold, slope, offset)
        self.logger.debug("Slope: %s, Offset: %s", slope, offset)

        self.file_name = 'image'

    def multiple_energy_thresh_calib(self, chips=range(8), threshold=0):
        """Plot multiple energy threshold calibration spectra.

        This functions produces threshold calibration files using DAC scans
        performed with several monochromatic X-ray spectra. Dac scans and
        spectra are performed and plotted in dacscan and spectrum plots. You
        need to inspect the spectra and edit manually the DAC arrays with the
        DAC value corresponding to the X-ray energy for each chip

        This method will use several points for calculating Energy to DAC
        conversion coefficients:

        Args:
            chips(list(int)): Chips to calibrate
            threshold(int): Threshold to calibrate (0 or 1)

        """
        self.settings['exposure'] = 1000
        self.file_name = 'Single_energy_threshold_scan'

        # dacRange = (self.dacTarget + 80, self.dacTarget + 20, 2)
        # self.load_config(range(8))
        # [dacscanData, scanRange] = self.scan_dac(chips, "Threshold" +
        #                                         str(Threshold), dacRange)
        # edgeDacs = self.find_edge(chips, dacscanData, dacRange, 2)
        # chipEdgeDacs = np.zeros([len(chips)])
        # for chip in chips:
        #    chipEdgeDacs[chip] = edgeDacs[0:256, chip*256:chip*256+256].mean()

        E1_E = 6
        E1_Dac = self.config.E1_DAC[self.settings['gain']]

        E2_E = 12
        E2_Dac = self.config.E2_DAC[self.settings['gain']]

        E3_E = 24
        E3_Dac = self.config.E3_DAC[self.settings['gain']]

        offset = np.zeros(8)
        gain = np.zeros(8)

        plot_name = util.tag_plot_name("DAC vs Energy", self.node_tag)
        for chip_idx in chips:
            x = np.array([E1_E, E2_E, E3_E])
            y = np.array([E1_Dac[self.id - 1, chip_idx],
                          E2_Dac[self.id - 1, chip_idx],
                          E3_Dac[self.id - 1, chip_idx]])

            p1, p2 = self.dawn.plot_linear_fit(x, y, [0, 1], "DAC Value",
                                               "Energy", plot_name,
                                               "Chip {}".format(chip_idx))
            offset[chip_idx] = p1
            gain[chip_idx] = p2

        self.save_kev2dac_calib(threshold, gain, offset)

        self.logger.debug("Gain: %s, Offset: %s", gain, offset)
        self.logger.debug("Self.settings gain: %s", self.settings['gain'])

    def set_thresh_energy(self, threshold=0, thresh_energy=5.0):
        """Set given threshold in keV.

        Args:
            threshold(int): Threshold to set (0 or 1)
            thresh_energy(float): Energy to set

        """
        fname = posixpath.join(self.calib_root,
                               'fem{id}',
                               self.settings['mode'],
                               self.settings['gain'],
                               'threshold{threshold}'
                               ).format(id=self.id, threshold=threshold)

        thresh_coeff = np.genfromtxt(fname)
        self.logger.debug("Thresh coefficients 0: %s",
                          thresh_coeff[0, :].astype(np.int))

        thresh_DACs = (thresh_energy * thresh_coeff[0, :] +
                       thresh_coeff[1, :]).astype(np.int)
        for chip_idx in range(8):
            self.set_dac([chip_idx],
                         "Threshold" + str(threshold), thresh_DACs[chip_idx])

        time.sleep(0.2)
        self.settings['exposure'] = 100
        self.expose()
        time.sleep(0.2)
        self.expose()

        print("A threshold {threshold} of {energy}keV corresponds to {DACs} "
              "DAC units for each chip".format(threshold=threshold,
                                               energy=thresh_energy,
                                               DACs=thresh_DACs))

    def read_chip_ids(self):
        """Read chip IDs."""
        self.app.read_chip_ids()

    def _save_chip_ids(self):
        """Read chip IDs and save to eFuseIDs file in calibration directory."""
        log_filename = posixpath.join(self.calib_root,
                                      'fem{id}',
                                      'efuseIDs'
                                      ).format(id=self.id)

        with open(log_filename, "w") as outfile:
            self.app.read_chip_ids(stdout=outfile)

    def monitor(self):
        """Monitor temperature, humidity, FEM voltage status and DAC out."""
        self.app.read_slow_control_parameters()

    def fe55_image_rx001(self, chips=range(8), exposure=60000):
        """Save FE55 image.

        Args:
            chips(list(int)): Chips to use
            exposure(int): Exposure time of image

        Will save to:
        /dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/ DIRECTORY

        """
        img_path = self.output_folder

        self.settings['gain'] = 'shgm'
        self.load_config(chips)
        self.set_dac(chips, "Threshold0", 40)
        self.file_name = 'Fe55_image_node_{id}_{exposure}s'.format(
            id=self.id, exposure=self.settings['exposure'])
        self.output_folder = posixpath.join(self.root_path, "Fe55_images")
        time.sleep(0.5)

        self.expose(exposure)

        self.output_folder = img_path
        self.file_name = 'image'
        self.settings['exposure'] = 100

    def set_dac(self, chips, name, value):
        """Set any chip DAC at a given value in config file and then load.

        Args:
            chips(list(int)): Chips to set DACs for
            name(str): DAC to set (Any from self.dac_number keys)
            value(int): Value to set DAC to

        """
        self.logger.info("Setting DAC %s to %s for chips %s",
                         name, value, chips)
        for chip_idx in chips:
            self._write_dac(chip_idx, name, value)
        self.app.load_dacs(chips, self.dacs_file)

    def set_dacs(self, chips):
        """Set dacs to values recommended by MEDIPIX3-RX designers.

        Args:
            chips(list(int)): Chips to set DACs for

        """
        self.logger.info("Setting config script DACs for chips %s", chips)
        for dac, value in self.config.DACS.items():
            for chip_idx in chips:
                self._write_dac(chip_idx, dac, value)
        self.app.load_dacs(chips, self.dacs_file)

    def _write_dac(self, chip_idx, name, value):
        """Write a new DAC value to the dacs file in the calibration directory.

        Args:
            chip_idx(int): Chip to write DAC for
            name(str): Name of DAC
            value(str/int): Value to set DAC to

        """
        new_line = "{name} = {value}\r\n".format(name=name, value=value)

        with open(self.dacs_file, "r") as dac_file:
            f_content = dac_file.readlines()

        line_nb = chip_idx * 29 + np.int(self.dac_number[name])
        f_content[line_nb] = new_line
        with open(self.dacs_file, "w") as dac_file:
            dac_file.writelines(f_content)

    def read_dac(self, dac_name):
        """Read back DAC analogue voltage using chip sense DAC (DAC out).

        Args:
            dac_name(str): DAC value to read (Any from self.dac_number keys)

        """
        return self.app.sense(self.chip_range, dac_name, self.dacs_file)

    def read_dac_from_file(self, chips, name):
        """Read DAC value from the dacs file in the calibration directory.

        Args:
            chips(list(int)): Chips to read DAC for
            name(str): Name of DAC

        """
        with open(self.dacs_file, "r") as dac_file:
            f_content = dac_file.readlines()

        for chip_idx in chips:
            line_nb = chip_idx * 29 + np.int(self.dac_number[name])
            print(f_content[line_nb])

    def scan_dac(self, chips, threshold, dac_range, exposure=5):
        """Perform a dac scan and plot the result (mean counts vs DAC values).

        Args:
            chips(list(int)): Chips to scan
            threshold(str): Threshold to scan (ThresholdX DACs - X: 0-7)
            exposure(int): Exposure time for each acquisition of scan
            dac_range(Range): Range of DAC values to scan over

        Returns:
            numpy.array: DAC scan data

        """
        self.logger.info("Performing DAC Scan of %s; Start: %s, Stop: %s, "
                         "Step: %s", threshold, *dac_range.__dict__.values())

        dac_scan_file = self.generate_file_name("DACScan")

        dac_file = self.dacs_file
        self.app.perform_dac_scan(chips, threshold, dac_range, exposure,
                                  dac_file, self.output_folder, dac_scan_file,
                                  gain_mode=self.settings['gain'],
                                  disc_mode=self.settings['disccsmspm'],
                                  equalization=self.settings['equalization'])

        file_path = posixpath.join(self.output_folder, dac_scan_file)
        if self.remote_node:
            self.app.grab_remote_file(file_path)

        util.wait_for_file(file_path, 10)
        scan_data = self.dawn.load_image_data(file_path)

        self.display_dac_scan(chips, scan_data, dac_range)
        return scan_data

    def display_dac_scan(self, chips, scan_data, dac_range):
        """Analyse and plot the results of threshold dac scan.

        Args:
            chips(list(int)): Chips to plot for
            scan_data(numpy.array): Data from dac scan
            dac_range(Range): Scan range used for dac scan

        Returns:
            numpy.array, list: Averaged scan data, DAC values of scan

        """
        if dac_range.start > dac_range.stop:
            dac_axis = np.array(range(dac_range.start,
                                      dac_range.stop - dac_range.step,
                                      -dac_range.step))
        else:
            dac_axis = np.array(range(dac_range.start,
                                      dac_range.stop + dac_range.step,
                                      dac_range.step))

        plot_data = np.zeros([len(chips), dac_axis.size])
        for list_idx, chip_idx in enumerate(chips):
            chip = scan_data[:, 0:256, chip_idx * 256:(chip_idx + 1) * 256]
            plot_data[list_idx, :] = chip.sum(axis=1).sum(axis=1)

        plot_name = self.node_tag + " - DAC Scan"
        self.dawn.plot_dac_scan(plot_data, chips, dac_axis, plot_name)
        return plot_data, dac_axis

    def load_temp_config(self, chip_idx, discLbits=None, discHbits=None,
                         mask_bits=None):
        """Save the given disc configs to temporary files and load them.

        Args:
            chip_idx(list(int)): Chip to load config for
            discLbits(numpy.array): DiscL pixel config (256 x 256 : 0 to 31)
            discHbits(numpy.array): DiscH pixel config (256 x 256 : 0 to 31)
            mask_bits(numpy.array): Pixel mask (256 x 256 : 0 or 1)

        """
        self.logger.info("Loading numpy arrays as temporary config.")

        template_path = posixpath.join(self.current_calib, "{disc}.tmp")
        discH_bits_file = template_path.format(disc="discHbits")
        discL_bits_file = template_path.format(disc="discLbits")
        pixel_bits_file = template_path.format(disc="pixelmask")

        util.save_array(discL_bits_file, discLbits)
        util.save_array(discH_bits_file, discHbits)
        util.save_array(pixel_bits_file, mask_bits)

        self.app.load_config(chip_idx,
                             discL_bits_file, discH_bits_file, pixel_bits_file)

    def load_config(self, chips=range(8)):
        """Load detector configuration files and default thresholds.

        From calibration directory corresponding to selected mode and gain.

        Args:
            chips(list(int)): Chips to load config for

        """
        self.logger.info("Loading discriminator bits from config directory.")

        for chip_idx in chips:
            self.app.load_config(chip_idx, self.discL_bits[chip_idx],
                                 self.discH_bits[chip_idx],
                                 self.pixel_mask[chip_idx])

        self.set_dac(range(8), "Threshold1", 100)
        self.set_dac(range(8), "Threshold0", 40)

    def expose(self, exposure=None):
        """Acquire single frame using current detector settings.

        Args:
            exposure(int): Exposure time of image

        Returns:
            numpy.array: Image data

        """
        if exposure is None:
            exposure = self.settings['exposure']

        self.logger.info("Capturing image with %sms exposure", exposure)
        image = self._acquire(1, exposure)

        self.dawn.plot_image(image, util.tag_plot_name("Image", self.node_tag))
        return image

    def sum_exposure(self):
        """Acquire single frame and output the total counts.

        Returns:
            numpy.array: Image data

        """
        exposure = self.settings['exposure']

        self.logger.info("Capturing image with %sms exposure", exposure)
        image = self._acquire(1, exposure)

        self.dawn.plot_image(image, util.tag_plot_name("Image", self.node_tag))
        print("Total counts: {}".format(self.dawn.sum_image(image)))
        return image

    def burst(self, frames, exposure):
        """Acquire images in burst mode.

        Args:
            frames(int): Number of images to capture
            exposure(int): Exposure time for images

        """
        self._acquire(frames, exposure, burst=True)

    def cont(self, frames, exposure):
        """Acquire images in continuous mode.

        Args:
            frames(int): Number of images to capture
            exposure(int): Exposure time for images

        """
        self.settings['readmode'] = "continuous"

        image = self._acquire(frames, exposure)

        base_name = util.tag_plot_name("Image", self.node_tag)
        plots = min(frames, 5)  # Limit to 5 frames
        for plot in range(plots):
            name = "{base_name} - {idx}".format(base_name=base_name, idx=plot)
            self.dawn.plot_image(image[plot, :, :], name)

    def cont_burst(self, frames, exposure):
        """Acquire images in continuous burst mode.

        Args:
            frames(int): Number of images to capture
            exposure(int): Exposure time for images

        """
        self.settings['readmode'] = "continuous"
        self._acquire(frames, exposure, burst=True)

    def _acquire(self, frames, exposure, burst=False):
        """Acquire image(s) with given exposure and current settings.

        Args:
            frames(int): Number of frames to acquire
            exposure(int): Exposure time of each image
            burst(bool): Set burst mode for acquisition

        """
        file_name = self.generate_file_name("Image")

        self.app.acquire(self.chip_range, frames, exposure,
                         burst=burst,
                         pixel_mode=self.settings['mode'],
                         disc_mode=self.settings['disccsmspm'],
                         depth=self.settings['bitdepth'],
                         counter=self.settings['counter'],
                         equalization=self.settings['equalization'],
                         gain_mode=self.settings['gain'],
                         read_mode=self.settings['readmode'],
                         trig_mode=self.settings['trigmode'],
                         path=self.output_folder,
                         hdf_file=file_name)

        file_path = posixpath.join(self.output_folder, file_name)
        if self.remote_node:
            self.app.grab_remote_file(file_path)

        util.wait_for_file(file_path, 10)
        image = self.dawn.load_image_data(file_path)
        return image

    def generate_file_name(self, base_name):
        """Generate file name with a time stamp from base_name and node.

        Args:
            base_name(str): Base file name - e.g. Image, DAC Scan

        Returns:
            str: New file name

        """
        return "{tag}_{base_name}_{node}.hdf5".format(
            tag=util.get_time_stamp(), base_name=base_name, node=self.id)

    def acquire_ff(self, num, exposure):
        """Acquire and sum flat-field images.

        NOT TESTED
        Produces flat-field correction coefficients

        Args:
            num(int): Number of images to sum
            exposure(int): Exposure time for images

        Returns:
            numpy.array: Flat-field coefficients

        """
        self.settings['exposure'] = exposure

        ff_image = 0
        for _ in range(num):
            image = self.expose()
            ff_image += image

        chip = 3  # TODO: Why?
        chip_mean = util.grab_chip_slice(ff_image, chip).mean()
        # Set all zero elements in the chip to the mean value
        ff_image[ff_image == 0] = chip_mean
        # Coeff array is array of means divided by actual values
        ff_coeff = np.ones([256, 256 * 8]) * chip_mean
        ff_coeff = ff_coeff / ff_image

        self.dawn.plot_image(util.grab_chip_slice(ff_image, chip),
                             name="Flat Field coefficients")

        # Set any elements outside range 0-2 to 1 TODO: Why?
        ff_coeff[ff_coeff > 2] = 1
        ff_coeff[ff_coeff < 0] = 1

        return ff_coeff

    def apply_ff_correction(self, num_images, ff_coeff):
        """Apply flat-field correction.

        NOT TESTED

        Args:
            num_images(int): Number of images to plot (?)
            ff_coeff(numpy.array): Numpy array with calculated correction

        """
        # TODO: This just plots, but doesn't apply it
        file_name = self.generate_file_name("FFImage")

        image_path = posixpath.join(self.output_folder, file_name)
        images = self.dawn.load_image_data(image_path)

        for image_idx in range(num_images):
            ff_mask = images[image_idx, :, :] * ff_coeff
            ff_mask[ff_mask > 3000] = 0
            chip = 3
            self.dawn.plot_image(util.grab_chip_slice(ff_mask, chip),
                                 name="Image data Cor")

    def save_discbits(self, chips, discbits, disc):
        """Save discbit array into file in the calibration directory.

        Args:
            chips(list(int)): Chips to save for
            discbits(numpy.array): Numpy array to save
            disc(str): File name to save to (discL or discH)

        """
        for chip_idx in chips:
            discbits_file = posixpath.join(
                self.current_calib, '{disc}.chip{chip}'.format(
                    disc=disc, chip=chip_idx))

            self.logger.info("Saving discbits to %s", discbits_file)
            util.save_array(discbits_file,
                            util.grab_chip_slice(discbits, chip_idx))

    def mask_columns(self, chips, start, stop):
        """Mask a block of column of pixels on the given chip(s).

        Args:
            chips(int/list(int)): Index(es) of chip(s) to mask
            start(int): Index of starting row of mask
            stop(int): Index of final row to be included in mask

        """
        chips = util.to_list(chips)

        mask = np.zeros(self.full_array_shape)
        for chip_idx in chips:
            # Subtract 1 to convert chip_size from number to index
            offset = self.chip_size * chip_idx
            util.set_slice(mask, [0, offset + start],
                           [self.chip_size, offset + stop], 1)

        self._apply_mask(chips, mask)
        self.dawn.plot_image(mask, "Mask")

    def mask_pixels_using_image(self, chips, image_data, max_counts):
        """Mask pixels in image_data with counts above max_counts.

        Args:
            chips(list(int)): Chips to mask
            image_data(numpy.array): Numpy array image
            max_counts(int): Mask threshold

        """
        noisy_pixels = image_data > max_counts
        self._mask_noisy_pixels(chips, noisy_pixels)

    def mask_pixels_using_dac_scan(self, chips=range(8),
                                   threshold="Threshold0",
                                   dac_range=Range(20, 120, 2), add=False):
        """Perform threshold dac scan and mask pixels with counts > max_counts.

        Also updates the corresponding maskfile in the calibration directory

        Args:
            chips(list(int)): Chips to scan
            threshold(str): Threshold to scan (0 or 1)
            dac_range(Range): Range to scan over
            add(bool): Whether to add pixels to pixelmask file, otherwise file
                is replaced

        """
        max_counts = 1
        self.settings['exposure'] = 100
        dac_scan_data = self.scan_dac(chips, threshold, dac_range)

        total_noisy_pixels = np.zeros(8)
        noisy_pixels = np.zeros(self.full_array_shape)
        if add:  # Load from file and append counts from new image
            for chip_idx in chips:
                if os.path.isfile(self.pixel_mask[chip_idx]):
                    util.set_chip_slice(noisy_pixels, chip_idx,
                                        np.loadtxt(self.pixel_mask[chip_idx]))

                total_noisy_pixels[chip_idx] = \
                    util.grab_chip_slice(noisy_pixels, chip_idx).sum()

                percentage = 100 * total_noisy_pixels[chip_idx] / 256**2
                print("####### Loading {num} noisy pixels from chip {idx} "
                      "({perc}%)".format(num=int(total_noisy_pixels[chip_idx]),
                                         idx=chip_idx,
                                         perc=percentage))

            percentage = 100 * total_noisy_pixels.sum() / (8 * 256**2)
            print("####### {num} noisy pixels in half module "
                  "({perc}%)".format(num=int(total_noisy_pixels.sum()),
                                     perc=percentage))

            noisy_pixels += dac_scan_data.sum(0) >= max_counts

        else:  # Just sum image
            noisy_pixels = dac_scan_data.sum(0) >= max_counts

        self._mask_noisy_pixels(chips, noisy_pixels)

    def _mask_noisy_pixels(self, chips, data):
        """Mask the noisy pixels in given data on the given chips.

        Args:
            data(np.array): Data from DAC scan or image

        """
        self.dawn.plot_image(data, name="Noisy Pixels")

        bad_pix_tot = np.zeros(8)
        for chip_idx in chips:
            bad_pix_tot[chip_idx] = util.grab_chip_slice(data, chip_idx).sum()

            print('####### {pix} noisy pixels in chip {chip} ({tot}%)'.format(
                pix=str(bad_pix_tot[chip_idx]),
                chip=str(chip_idx),
                tot=str(100 * bad_pix_tot[chip_idx] / (256**2))))

        self._apply_mask(chips, data)

        print('####### {pix} noisy pixels in node ({tot}%)'.format(
            pix=str(bad_pix_tot.sum()),
            tot=str(100 * bad_pix_tot.sum() / (8 * 256**2))))

    def _apply_mask(self, chips, mask):
        """Save the given mask to the calibration to file and load to detector.

        Args:
            mask(np.array): Mask to apply

        """
        for chip_idx in chips:
            pixel_mask_file = self.pixel_mask[chip_idx]
            util.save_array(pixel_mask_file,
                            util.grab_chip_slice(mask, chip_idx))

        self.load_config(chips)

    def unmask_pixels(self, chips):
        """Reset pixelmask bits to zero.

        Args:
            chips(list(int)): Chips to unmask

        """
        zeros = np.zeros(self.chip_shape)

        for chip_idx in chips:
            util.save_array(self.pixel_mask[chip_idx], zeros)
            self.app.load_config(chip_idx, self.discL_bits[chip_idx],
                                 self.discH_bits[chip_idx],
                                 self.pixel_mask[chip_idx])

    def unequalize_pixels(self, chips):
        """Reset discL_bits to zero.

        Args:
            chips(list(int)): Chips to unequalize

        """
        zeros = np.zeros(self.chip_shape)

        for chip_idx in chips:
            util.save_array(self.discL_bits[chip_idx], zeros)
            util.save_array(self.discH_bits[chip_idx], zeros)
            self.app.load_config(chip_idx, self.discL_bits[chip_idx],
                                 self.discH_bits[chip_idx],
                                 self.pixel_mask[chip_idx])

    def backup_calib_dir(self):
        """Back up calibration directory with time stamp."""
        self.logger.info("Backing up calib directory.")
        backup_dir = "{calib}_{time_stamp}".format(
            calib=self.calib_root, time_stamp=util.get_time_stamp())
        shutil.copytree(self.calib_root, backup_dir)
        self.logger.debug("Backup directory: %s", backup_dir)

    def copy_hgm_into_other_gain_modes(self):
        """Copy hgm calibration folder into lgm, hgm and shgm folders.

        This function is used at the end of threshold equalization because
        threshold equalization is performed in the more favorable gain mode
        hgm and threshold equalization data is independent of the gain mode.

        """
        self.logger.info("Copying HGM calib to other gain modes")
        template_path = posixpath.join(self.calib_root,
                                       "fem{id}".format(id=self.id),
                                       self.settings['mode'],
                                       "{gain_mode}")

        lgm_dir = template_path.format(gain_mode="lgm")
        hgm_dir = template_path.format(gain_mode="hgm")
        slgm_dir = template_path.format(gain_mode="slgm")
        shgm_dir = template_path.format(gain_mode="shgm")

        if os.path.exists(slgm_dir):
            shutil.rmtree(slgm_dir)
        if os.path.exists(lgm_dir):
            shutil.rmtree(lgm_dir)
        if os.path.exists(shgm_dir):
            shutil.rmtree(shgm_dir)

        shutil.copytree(hgm_dir, slgm_dir)
        shutil.copytree(hgm_dir, lgm_dir)
        shutil.copytree(hgm_dir, shgm_dir)

    def _load_discbits(self, chips, discbits_filename):
        """Load discriminator bit array from calib directory into numpy array.

        Args:
            chips(list(int)): Chips to load
            discbits_filename(str): File to grab data from (discL or discH)

        Returns:
            numpy.array: Discbits array

        """
        self.logger.info("Reading discriminator bits from file.")
        discbits = np.zeros(self.full_array_shape)
        for chip_idx in chips:
            discbits_file = posixpath.join(
                self.current_calib, '{disc}.chip{chip}'.format(
                    disc=discbits_filename, chip=chip_idx))

            util.set_chip_slice(discbits, chip_idx, np.loadtxt(discbits_file))

        return discbits

    def combine_rois(self, chips, disc_name, steps, roi_type):
        """Combine multiple ROIs into one mask.

        Used to combine intermediate discbits_roi files produced when
        equalizing various ROIs into one discbits file

        Args:
            chips(list(int)): Chips to make ROIs for
            disc_name: Discriminator config file to edit (discL or discH)
            steps(int): Number of of ROIs to combine (steps during
                equalization)
            roi_type(str): Type of ROI (rect or spacing)

        """
        # TODO: Get rid of this if it isn't necessary anymore
        discbits = np.zeros(self.full_array_shape)
        for step in range(steps):
            roi_full_mask = self.roi(chips, step, steps, roi_type)
            discbits_roi = self._load_discbits(chips, disc_name +
                                               'bits_roi_' + str(step))
            discbits[roi_full_mask.astype(bool)] = \
                discbits_roi[roi_full_mask.astype(bool)]
            # TODO: Should this be +=? Currently just uses final ROI
            plot_name = self.node_tag + " - Disc Bits"
            self.dawn.plot_image(discbits_roi, name=plot_name)

        self.save_discbits(chips, discbits, disc_name + "bits")
        plot_name = self.node_tag + " - Disc Bits Total"
        self.dawn.plot_image(discbits, name=plot_name)

        return discbits

    def find_edge(self, chips, dac_scan_data, dac_range, edge_val):
        """Find noise or X-ray edge in threshold DAC scan.

        Args:
            chips(list(int)): Chips search
            dac_scan_data(numpy.array): Data from DAC scan
            dac_range(Range): Range DAC scan performed over
            edge_val(int): Threshold for edge

        Returns:
            numpy.array: Noise edge data

        """
        self.logger.info("Finding noise edges in DAC scan data.")
        # Reverse data for high to low scan
        if dac_range.stop > dac_range.start:
            is_reverse_scan = True
            dac_scan_data = dac_scan_data[::-1, :, :]
        else:
            is_reverse_scan = False

        over_threshold = dac_scan_data > edge_val
        threshold_edge = np.argmax(over_threshold, 0)
        if is_reverse_scan:
            edge_dacs = dac_range.stop - dac_range.step * threshold_edge
        else:
            edge_dacs = dac_range.start - dac_range.step * threshold_edge

        plot_name = self.node_tag + " - Noise Edges"
        self.dawn.plot_image(edge_dacs, name=plot_name)
        plot_name = self.node_tag + " - Histogram of NEdge"
        self._display_histogram(chips, edge_dacs, plot_name, "Edge Location")
        return edge_dacs

    def find_max(self, chips, dac_scan_data, dac_range):
        """Find noise max in threshold dac scan.

        Returns:
            numpy.array: Max noise data

        """
        self.logger.info("Finding max noise in DAC scan data.")
        max_dacs = dac_range.stop - dac_range.step * np.argmax(
            (dac_scan_data[::-1, :, :]), 0)
        # TODO: Assumes low to high scan? Does it matter?

        plot_name = self.node_tag + " - Noise Max"
        self.dawn.plot_image(max_dacs, name=plot_name)
        plot_name = self.node_tag + " - Histogram of NMax"
        self._display_histogram(chips, max_dacs, plot_name, "Max Location")
        return max_dacs

    def _display_histogram(self, chips, data, name, x_name):
        """Plot an image and a histogram of given data.

        Args:
            chips(list(int)): Chips to plot for
            data(numpy.array): Data to analyse
            x_name(str): Label for x-axis

        """
        image_data = []
        for chip_idx in chips:
            image_data.append(util.grab_chip_slice(data, chip_idx))
        self.dawn.plot_histogram(image_data, name, x_name)

    def _optimize_dac_disc(self, chips, disc_name, roi_mask):
        """Calculate optimum DAC disc values for given chips.

        Args:
            chips(list(int)): Chips to optimize
            disc_name(str): Discriminator to optimize (discL or discH)
            roi_mask(numpy.array): Mask to exclude pixels from
                optimization calculation

        Returns:
            numpy.array: Optimum DAC discriminator value for each chip

        """
        self.logger.info("Optimizing %s", disc_name)

        thresholds = dict(discL="Threshold0", discH="Threshold1")
        threshold = thresholds[disc_name]

        # Definition of parameters to be used for threshold scans
        self.settings['exposure'] = 5
        self.settings['counter'] = 0
        self.settings['equalization'] = 1  # Might not be necessary when
        # optimizing DAC Disc
        dac_range = Range(0, 150, 5)

        discbit = 0
        calib_plot_name = "Mean edge shift in Threshold DACs as a function " \
                          "of DAC_disc for discbit = {}".format(discbit)
        calib_plot_name = self.node_tag + " - " + calib_plot_name

        # Set discbits at 0
        discbits = discbit * np.ones(self.full_array_shape)
        self.load_all_discbits(chips, disc_name, discbits, roi_mask)

        # Threshold DAC scans, fitting and plotting
        p0 = [5000, 50, 30]
        dac_disc_range = range(30, 180, 50)
        x0 = np.zeros([8, len(dac_disc_range)])
        sigma = np.zeros([8, len(dac_disc_range)])
        a = np.zeros([8, len(dac_disc_range)])
        for idx, dac_value in enumerate(dac_disc_range):
            optimal_params = self._dac_scan_fit(chips, threshold, dac_value,
                                                dac_range, discbit, p0)
            x0[:, idx], sigma[:, idx], a[:, idx] = optimal_params

        self.dawn.clear_plot(calib_plot_name)
        for chip_idx in chips:
            self.dawn.add_plot_line(np.asarray(dac_disc_range),
                                    x0[chip_idx, :],
                                    "Disc Value", "Edges", calib_plot_name,
                                    label="Chip {}".format(chip_idx))

        # Plot mean noise edge vs DAC Disc for discbits set at 0
        offset = np.zeros(8)
        gain = np.zeros(8)
        for chip_idx in chips:
            results = self.dawn.plot_linear_fit(
                np.asarray(dac_disc_range), x0[chip_idx, :], [0, -1],
                "Disc Value", "Edges",
                fit_name=calib_plot_name, label="Chip {}".format(chip_idx))

            offset[chip_idx], gain[chip_idx] = results[0], results[1]

        # Calculate optimum DAC discriminator values
        opt_value_scaling = 1.05
        opt_dac_disc = np.zeros(self.num_chips)
        self.logger.info("Optimum DAC discriminator determination:\n")
        for chip_idx in chips:
            # TODO: Give these parameters descriptive names
            p1 = self.num_sigma * sigma[chip_idx, 1] / gain[chip_idx]
            p2 = self.dac_target / gain[chip_idx]
            chip_offset = offset[chip_idx]
            opt_value = int(p1 + chip_offset + p2 + chip_offset)
            self.set_dac([chip_idx], threshold, opt_value)
            opt_dac_disc[chip_idx] = opt_value * opt_value_scaling

            logging.info("Chip: {idx}, Gain: {gain}, Offset: {offset},"
                         "Optimum DAC Disc: {opt}".format(
                idx=chip_idx, gain=gain[chip_idx], offset=offset[chip_idx],
                opt=opt_dac_disc[chip_idx]))

        self._display_optimization_results(chips, x0, sigma, a, gain,
                                           opt_dac_disc)

    def _dac_scan_fit(self, chips, threshold, dac_value, dac_range, discbit,
                      p0):
        """Scan given DAC for given chips and perform a gaussian fit.

        Args:
            chips(list(int)): Chips to scan
            threshold(str): DAC to scan
            dac_value(int): Initial value to set dac to
            dac_range(Range): Range to scan over
            discbit(int): Current discbit value (for plot name)
            p0(list(int)): Initial estimate for gaussian fit parameters

        Returns:
            np.array, np.array: x0 and sigma values for fit
        """
        bins = (dac_range.stop - dac_range.start) / dac_range.step
        plot_name = "Histogram of edges when scanning DAC_disc for " \
                    "discbit = {discbit}".format(discbit=discbit)
        plot_name = self.node_tag + " - " + plot_name

        self.set_dac(chips, threshold, dac_value)
        # Scan threshold
        dac_scan_data = self.scan_dac(chips, threshold, dac_range)
        # Find noise edges
        edge_dacs = self.find_max(chips, dac_scan_data, dac_range)

        scan_data = [None] * 8
        for chip_idx in self.chip_range:
            if chip_idx in chips:
                scan_data[chip_idx] = util.grab_chip_slice(edge_dacs, chip_idx)

        return self.dawn.plot_gaussian_fit(scan_data, plot_name, p0, bins)

    def _display_optimization_results(self, chips, x0, sigma, a, gain,
                                      opt_dac_disc):
        """Print out optimization results."""
        self.logger.info("Linear fit results for edge shift calculation:\n")
        for chip_idx in chips:
            print("Chip {idx}: {shift}".format(
                idx=chip_idx, shift=str(round(gain[chip_idx], 2))))
            self.logger.info("Chip {idx} centroids: {c1}, {c2}, {c3}".format(
                idx=chip_idx,
                c1=x0[chip_idx, 0], c2=x0[chip_idx, 1], c3=x0[chip_idx, 2]))
            self.logger.info("Chip {idx} sigmas: {s1}, {s2}, {s3}".format(
                idx=chip_idx, s1=sigma[chip_idx, 0], s2=sigma[chip_idx, 1],
                s3=sigma[chip_idx, 2]))
            self.logger.info("Chip {idx} widths: {w1}, {w2}, {w3}".format(
                idx=chip_idx, w1=a[chip_idx, 0], w2=sigma[chip_idx, 1],
                w3=sigma[chip_idx, 2]))

        print("###############################################################"
              "########################")
        print("Optimum DACdisc value (DAC units) required to bring all pixels "
              "with an edge within +/- {num_sigma} sigma of the target, at "
              "the target of {target}".format(num_sigma=self.num_sigma,
                                              target=self.dac_target))
        for chip_idx in chips:
            print("Chip {idx}: {opt_value}".format(
                idx=chip_idx, opt_value=opt_dac_disc[chip_idx]))

        print("###############################################################"
              "########################")
        print("Edge shift (Threshold DAC Units) produced by 1 step of the "
              "32 discbit correction steps:")
        for chip_idx in chips:
            print("Chip {idx}: {shift}".format(
                idx=chip_idx, shift=opt_dac_disc[chip_idx] / 16))

    def _equalise_discbits(self, chips, disc_name, threshold, roi_full_mask,
                           method):
        """Equalize pixel discriminator.

        Uses stripes method as default (trimbits distributed across the matrix
        during each dacscan to avoid saturation the chips when pixels are in
        the noise at the same time)

        Args:
            chips(list(int)): Chips to equalize
            disc_name(str): Discriminator to equalize (discL or discH)
            threshold(str): Threshold to scan
            roi_full_mask(numpy.array): Mask to exclude pixels from
                equalization calculation
            method(str): Method to use (stripes)

        Returns:
            numpy.array: Equalised discriminator bits

        """
        self.logger.info("Equalising %s", disc_name)

        self.settings['exposure'] = 5
        self.settings['counter'] = 0
        self.settings['equalization'] = 1

        inv_mask = 1 - roi_full_mask
        if method.lower() == "stripes":
            dac_range = Range(0, 20, 2)
            discbits_tmp = np.zeros(self.full_array_shape) * inv_mask
            for row in range(self.chip_size):
                discbits_tmp[row, :] = row % 32
            for column in range(self.chip_size * self.num_chips):
                discbits_tmp[:, column] = \
                    (column % 32 + discbits_tmp[:, column]) % 32
            discbits_tmp *= inv_mask

            discbits = -10 * np.ones(self.full_array_shape) * inv_mask
            edge_dacs_stack = np.zeros([32] + self.full_array_shape)
            discbits_stack = np.zeros([32] + self.full_array_shape)
            for scan in range(0, 32, 1):
                self.logger.info("Equalize discbits step %s", scan)
                discbits_tmp = ((discbits_tmp + 1) % 32) * inv_mask
                discbits_stack[scan, :, :] = discbits_tmp

                self.load_all_discbits(chips, disc_name,
                                       discbits_tmp, roi_full_mask)

                dacscan_data = self.scan_dac(chips, threshold, dac_range)
                edge_dacs = self.find_max(chips, dacscan_data, dac_range)
                edge_dacs_stack[scan, :, :] = edge_dacs

                # TODO: Check if this can be done after for loop
                scan_nb = np.argmin(np.abs(edge_dacs_stack - self.dac_target),
                                    axis=0)
                for chip_idx in chips:
                    for x in range(self.chip_size):
                        for y in range(chip_idx * self.chip_size,
                                       (chip_idx + 1) * self.chip_size):
                            discbits[x, y] = \
                                discbits_stack[scan_nb[x, y], x, y]

                plot_name = self.node_tag + " - Discriminator Bits"
                self.dawn.plot_image(discbits, name=plot_name)

                plot_name = self.node_tag + " - Histogram of Final Discbits"
                self.dawn.plot_histogram_with_mask(chips, discbits, inv_mask,
                                                   plot_name, "Bit Value")
        else:
            raise NotImplementedError("Equalization method not implemented.")

        logging.info('\nHistogram of final discbits')
        for chip in chips:
            roiChipMask = 1 - roi_full_mask[0:256, chip * 256:chip * 256 + 256]
            discbitsChip = discbits[0:256, chip * 256:chip * 256 + 256]
            hist = np.histogram(discbitsChip[roiChipMask.astype(bool)],
                                bins=range(32))
            self.logger.info(str(hist))

        self.settings['disccsmspm'] = 'discL'
        self.settings['equalization'] = 0

        print("Pixel threshold equalization complete for node {}".format(
            self.id))

        self.load_config(chips)
        self.scan_dac(chips, threshold, Range(40, 10, 2))

        return discbits

    def load_all_discbits(self, chips, disc_name, temp_bits, mask):
        """Load the appropriate discriminator bits to calibrate the given disc.

        Args:
            chips(list(int)): Chips to load for
            disc_name(str): Discriminator being calibrated (discL or discH)
            temp_bits(numpy.array): Temporary array to be used for the
                non-calibratee
            mask(numpy.array): Pixel mask to load

        """
        if disc_name == "discH":
            discLbits = self._load_discbits(chips, "discLbits")
            discHbits = temp_bits
        elif disc_name == "discL":
            discLbits = temp_bits
            discHbits = np.zeros(self.full_array_shape)
        else:
            raise ValueError("Discriminator must be L or H, got {bad_disc}"
                             .format(bad_disc=disc_name))

        for chip_idx in chips:
            self.load_temp_config(chip_idx,
                                  util.grab_chip_slice(discLbits, chip_idx),
                                  util.grab_chip_slice(discHbits, chip_idx),
                                  util.grab_chip_slice(mask, chip_idx))

    def check_calib(self, chips, dac_range):
        """Check if dac scan looks OK after threshold calibration.

        NOT TESTED

        Args:
            chips(list(int)): Chips to check
            dac_range(Range): Range to scan DAC over

        """
        self.load_config(chips)
        equ_pix_tot = np.zeros(self.num_chips)
        self.file_name = 'dacscan'
        dac_scan_data = self.scan_dac(chips, "Threshold0", dac_range)
        self.plot_name = self.file_name
        edge_dacs = self.find_max(chips, dac_scan_data, dac_range)

        # Display statistics on equalization and save discLbit files for each
        # chip
        for chip in chips:
            equ_pix_tot[chip] = ((edge_dacs[roi, chip*256:chip*256 + 256] >
                                  self.dac_target - self.allowed_delta) &
                                 (edge_dacs[roi, chip*256:chip*256 + 256] <
                                  self.dac_target + self.allowed_delta)).sum()
            print('####### {pixels} equalized pixels in chip {chip} '
                  '({percentage})'.format(
                                   pixels=round(equ_pix_tot[chip], 0),
                                   chip=chip,
                                   percentage=round(100 * equ_pix_tot[chip] /
                                                    (256**2), 4)))

        # pixelsInTarget = (dacTarget - 5 < edge_dacs) & \
        #                      (edge_dacs < dacTarget + 5)

    def roi(self, chips, step, steps, roi_type):
        """Create a detector ROI to be used when equalizing thresholds.

        Using several ROIs during equalization was needed to avoid putting too
        many pixels in the noise at the same time. However the latest
        equalization scripts used for EXCALIBUR use the same technique as
        MERLIN to distribute equalization bits in diagonal across the matrix
        during equalization. Therefore the roi used by the latest scripts is
        always: roi = x.roi(range(8), 0, 1, 'rect')

        Args:
            chips(list(int)): Chips to create ROI for
            step(int): Current step in the equalization process
            steps(int): Total number of steps
            roi_type(str): ROI type to create
                rect: contiguous rectangles
                spacing: arrays of equally-spaced pixels distributed across
                the chip

        """
        if roi_type == "rect":
            roi_full_mask = np.zeros(self.full_array_shape)
            for chip_idx in chips:
                roi_full_mask[step*256/steps:step*256/steps + 256/steps,
                              chip_idx*256:chip_idx*256 + 256] = 1
        elif roi_type == "spacing":
            spacing = steps**0.5
            roi_full_mask = np.zeros(self.full_array_shape)
            offset = np.binary_repr(step, 2)
            for chip_idx in chips:
                roi_full_mask[int(offset[0]):
                              self.chip_size - int(offset[0]):
                              spacing,
                              chip_idx * self.chip_size + int(offset[1]):
                              (chip_idx + 1) * self.chip_size - int(offset[1]):
                              spacing] = 1
        else:
            raise NotImplementedError("Available methods are 'rect' and "
                                      "'spacing', got {}".format(roi_type))

        self.dawn.plot_image(roi_full_mask,
                             util.tag_plot_name("Roi Mask", self.node_tag))

        return roi_full_mask

    def calibrate_disc_l(self, chips):
        """Calibrate discriminator L for the given chips.

        Args:
            chips(list(int)): Chips to calibrate

        """
        self.set_dac(chips, "Threshold1", 0)
        self.settings['disccsmspm'] = "discL"
        self._calibrate_disc(chips, "discL")

    def calibrate_disc_h(self, chips):
        """Calibrate discriminator H for the given chips.

        Args:
            chips(list(int)): Chips to calibrate

        """
        self.set_dac(chips, "Threshold0", 255)  # To be above the noise
        self.settings['disccsmspm'] = "discH"
        self._calibrate_disc(chips, "discH")

    def _calibrate_disc(self, chips, disc_name, steps=1, roi_type='rect'):
        """Calibrate given discriminator for given chips.

        Args:
            chips(list(int)): Chips to calibrate
            disc_name(str): Discriminator to equalize (discL or discH)
            steps(int): Total number of steps for equalization
            roi_type(str): ROI type to use (rect or spacing - see roi)

        """
        thresholds = dict(discL="Threshold0", discH="Threshold1")
        threshold = thresholds[disc_name]

        roi_mask = 1 - self.roi(chips, 0, 1, 'rect')
        self._optimize_dac_disc(chips, disc_name, roi_mask)

        # Run threshold_equalization over each roi
        for step in range(steps):
            roi_full_mask = self.roi(chips, step, steps, roi_type)
            discbits = self._equalise_discbits(chips, disc_name, threshold,
                                               1 - roi_full_mask, 'stripes')
            self.save_discbits(chips, discbits,
                               disc_name + 'bits_roi_' + str(step))
        discbits = self.combine_rois(chips, disc_name, steps, roi_type)

        # TODO: Why does it save discbits 3 times, because of old method?

        self.save_discbits(chips, discbits, disc_name + 'bits')
        self.load_config(chips)  # Load threshold_equalization files created
        self.copy_hgm_into_other_gain_modes()  # Copy slgm threshold
        # equalization folder to other gain threshold_equalization folders

    def loop(self, num):
        """Acquire images and plot the sum.

        Args:
            num(int): Number of images to acquire

        """
        plot_name = util.tag_plot_name("Sum", self.node_tag)
        tmp = 0
        for _ in range(num):
            tmp = self.expose() + tmp
            self.dawn.plot_image(tmp, plot_name)

            return  # TODO: This will always stop after first loop

    def csm(self, chips=range(8), gain="slgm"):
        """Set charge summing mode and associated default settings.

        Args:
            chips(list(int)): Chips to load
            gain(str): Gain mode to set (slgm, lgm, hgm, shgm)

        """
        # TODO: Why does it have to call expose to set these??
        self.settings['mode'] = 'csm'
        # TODO: Get rid of gain or take away arg; not default if user provided
        self.settings['gain'] = gain
        self.settings['counter'] = 1
        # Make sure that unused have also Th0 and Th1 well above the noise
        self.set_dac(range(8), 'Threshold0', 200)
        self.set_dac(range(8), 'Threshold1', 200)
        self.load_config(chips)
        # Reset default thresholds
        self.set_dac(range(8), 'Threshold0', 45)
        self.set_dac(chips, 'Threshold1', 100)
        self.expose()
        self.expose()

    def check_dac_read_back(self, dac_name, chip_idx):
        """Read DAC voltage for the given dac and chip.

        Args:
            dac_name(str): DAC to read voltage for (GND, FBK or Cas)
            chip_idx(int): Chip to read voltage for

        Returns:
            float: Read-back voltage for DAC

        """
        response = self.app.sense(self.chip_range, dac_name, self.dacs_file,
                                  capture=True).split("\n")

        chip_id = chip_idx + 1
        target = "{} : OK (".format(chip_id)
        for line_idx, line in enumerate(response):
            if "Reading front-end DAC channels from FEM" in line:
                if chip_id in [1, 2, 3, 4]:
                    values = line
                else:
                    values = response[line_idx + 1]

                start = values.find(target) + 8
                voltage = float(values[start:start + 5])
                return voltage

    def optimise_gnd_fbk_cas(self, chips=range(8)):
        """Iterate GND, FBK and Cas DAV values to get optimum read-back values.

        Target DAC read-back values(recommended by Rafa in May 2015):
            GND: 0.65V
            FBK: 0.9V
            CAS: 0.85V

        Args:
            chips(list(int)): Chips to optimise

        """
        print("=========================================")
        print(" Beginning autoset of GND, FBK and Cas")
        print(" This might take a while")
        print(" FEM = {}".format(self.id))
        print("=========================================")

        dacs = ["GND", "FBK", "Cas"]
        start_values = dict(GND=145, FBK=190, Cas=180)
        targets = dict(GND=0.650, FBK=0.900, Cas=0.850)

        increments = [5, 2, 1]
        for chip_idx in chips:
            for dac in dacs:
                dac_value = start_values[dac]
                target = targets[dac]

                voltage = -1
                previous_voltage = -1
                failed = False
                state = None
                overshoots = 0
                while abs(voltage - target) > 0.004:
                    self.set_dac([chip_idx], dac, dac_value)
                    voltage = self.check_dac_read_back(dac, chip_idx)
                    if abs(voltage - previous_voltage) < 0.002:
                        failed = True
                        break
                    if voltage < target:
                        if state == "High":
                            overshoots += 1
                        state = "Low"
                    else:
                        if state == "Low":
                            overshoots += 1
                        state = "High"
                    try:
                        increment = increments[overshoots]
                    except IndexError:
                        increment = increments[-1]
                    if state == "High":
                        dac_value -= increment
                    elif state == "Low":
                        dac_value += increment
                    previous_voltage = voltage

                if failed:
                    print("==================================================="
                          "==================================================")
                    print(" Could not converge {dac} for chip {idx} "
                          "fem {id}; value = {val}".format(
                              dac=dac, idx=chip_idx, id=self.id,
                              val=voltage))
                    print(" Likely a stuck DAC")
                    print("==================================================="
                          "==================================================")
                else:
                    print("==================================================="
                          "==================================================")
                    print("{dac} converged for chip {idx} fem {id}; "
                          "voltage = {voltage}V for a DAC of {val}".format(
                              dac=dac, idx=chip_idx, id=self.id,
                              voltage=voltage, val=dac_value))
                    print("==================================================="
                          "==================================================")

    def rotate_config(self):
        """Rotate discbits files in EPICS calibration directory."""
        template_path = posixpath.join(self.calib_root + '_epics',
                                       'fem{fem}',
                                       'spm',
                                       'hgm',
                                       '{disc}.chip{chip}')

        for chip_idx in self.chip_range:
            discLbits_file = template_path.format(fem=self.id,
                                                  disc='discLbits',
                                                  chip=chip_idx)
            discHbits_file = template_path.format(fem=self.id,
                                                  disc='discHbits',
                                                  chip=chip_idx)
            pixel_mask_file = template_path.format(fem=self.id,
                                                   disc='pixelmask',
                                                   chip=chip_idx)
            if os.path.isfile(discLbits_file):
                util.rotate_array(discLbits_file)
                self.logger.info("%s rotated", discLbits_file)
            if os.path.isfile(discHbits_file):
                util.rotate_array(discHbits_file)
                self.logger.info("%s rotated", discHbits_file)
            if os.path.isfile(pixel_mask_file):
                util.rotate_array(pixel_mask_file)
                self.logger.info("%s rotated", pixel_mask_file)

    def display_masks(self):
        """Print list of masks in config directory."""
        mask_files = self._list_config_files(".mask")
        print("Available masks: " + ", ".join(mask_files))

    def display_dac_files(self):
        """Print list of DAC files in config directory."""
        dac_files = self._list_config_files(".dacs")
        print("Available DAC files: " + ", ".join(dac_files))

    def _list_config_files(self, suffix):
        """Print list of files with given extension in config directory.

        Args:
            suffix(str): File type to list

        """
        files = os.listdir(self.config_dir)
        return [file_ for file_ in files if file_.endswith(suffix)]
