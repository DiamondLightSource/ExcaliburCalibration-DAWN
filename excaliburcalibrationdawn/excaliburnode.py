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
from config import MPX3RX
from excaliburcalibrationdawn import util

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)


Range = namedtuple("Range", "start stop step")


class ExcaliburNode(object):

    """Class to calibrate a node of an Excalibur-RX detector.

    ExcaliburNode is a class defining methods required to calibrate each 1/2
    module (8 MPX3-RX chips) of an EXCALIBUR-RX detector.
    These calibration scripts will work only inside the Python interpreter of
    DAWN software running on the PC server node connected to the FEM
    controlling the half-module which you wish to calibrate
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
    # Shape of full 1/2 module array
    full_array_shape = [chip_size, num_chips * chip_size]

    root_path = '/dls/detectors/support/silicon_pixels/excaliburRX/'
    calib_dir = posixpath.join(root_path, '3M-RX001/calib')
    config_dir = posixpath.join(root_path, 'TestApplication_15012015/config')
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

    def __init__(self, node, server_root=None):
        """Initialize Excalibur node object.

        For example: On I13 the top FEM of EXCALIBUR-3M-RX001 is connected to
        node 1 (i13-1-excalibur01) and the bottom fem to node 6
        (i13-1-excalibur06).

        Args:
            server_root(str): Server name root; add node number to get
            real server
            node(int): PC node number of 1/2 module (Between 1 and 6 for 3M)

        """
        if node not in [1, 2, 3, 4, 5, 6]:
            raise ValueError("Node {node} is invalid. Should be 1-6.".format(
                node=node))
        suffix = 7 - node  # FEMs are numbered in reverse

        self.fem = node
        self.ipaddress = "192.168.0.10{}".format(suffix)

        if server_root is not None:
            self.server_name = "{root}{suffix}".format(root=server_root,
                                                       suffix=suffix)
            self.remote_node = True
        else:
            self.server_name = None
            self.remote_node = False

        logging.info("Creating ExcaliburNode with server %s and ip %s",
                     self.server_name, self.ipaddress)

        # Detector default settings - See excaliburtestappinterface for details
        self.settings = dict(mode="spm",  # spm or csm
                             gain="shgm",  # slgm, lgm, hgm or shgm
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

        # Commonly used file paths
        self.template_path = posixpath.join(self.calib_dir,
                                            'fem{fem}'.format(fem=self.fem),
                                            self.settings['mode'],
                                            self.settings['gain'])
        tp = posixpath.join(self.template_path, '{disc}.chip{chip}')
        self.discL_bits = [tp.format(disc="discLbits",
                                     chip=chip) for chip in self.chip_range]
        self.discH_bits = [tp.format(disc="discHbits",
                                     chip=chip) for chip in self.chip_range]
        self.pixel_mask = [tp.format(disc="pixelmask",
                                     chip=chip) for chip in self.chip_range]

        # Helper classes
        self.app = ExcaliburTestAppInterface(self.fem, self.ipaddress, 6969,
                                             self.server_name)
        self.dawn = ExcaliburDAWN()

    def setup(self):
        """Perform necessary initialisation."""
        self.read_chip_ids()
        self.app.load_dacs(range(8), self.default_dacs)

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
        if state not in [True, False]:
            raise ValueError("Quiet state can be either True or False.")
        self.app.quiet = state

    def display_status(self):
        """Display status of node."""
        print("Status for Node {}".format(self.fem))
        print("LV: {}".format(self.app.lv))
        print("HV: {}".format(self.app.hv))
        print("HV Bias: {}".format(self.app.hv_bias))
        print("DACs Loaded: {}".format(self.app.dacs_loaded))
        print("Initialised: {}".format(self.app.initialised))

    def threshold_equalization(self, chips=range(8)):
        """Calibrate discriminator equalization.

        You need to edit this function to define which mode (SPM or CSM) and
        which gains you want to calibrate during the threshold_equalization
        sequence

        Args:
            chips(list(int)): Chip or list of chips to calibrate

        """
        chips = util.to_list(chips)
        self.settings['mode'] = 'spm'
        self.settings['gain'] = 'slgm'

        self.check_calib_dir()
        self.log_chip_ids()
        self.set_dacs(chips)
        self.set_gnd_fbk_cas_excalibur_rx001(chips, self.fem)

        self.calibrate_disc_l(chips)

        # NOTE: Always equalize DiscL before DiscH since Threshold1 is set at 0
        # when equalizing DiscL. So if DiscH was equalized first, this would
        # induce noisy counts interfering with DiscL equalization

        # self._calibrate_disc(chips, 'discH', 1, 'rect')
        # self.settings['mode'] = 'csm'
        # self.settings['gain'] = 'slgm'
        # self._calibrate_disc(chips, 'discL', 1, 'rect')
        # self._calibrate_disc(chips, 'discH', 1, 'rect')

        # EG (13/06/2016) creates mask for horizontal noise
        # badPixels = self.mask_row_block(range(4), 256-20, 256)

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

        self.check_calib_dir()

        gain = gain_values[self.settings['gain']]
        dac1 = dac0 + gain * (default_6kev_dac - dac0) * \
            np.ones([6, 8]).astype('float')

        logging.debug("E0: %s", E0)
        logging.debug("DAC0 Array: %s", dac0)
        logging.debug("E1: %s", E1)
        logging.debug("DAC1 Array: %s", dac1)

        slope = (dac1[self.fem - 1, :] - dac0[self.fem - 1, :]) / (E1 - E0)
        offset = dac0[self.fem - 1, :]
        self.save_kev2dac_calib(threshold, slope, offset)
        logging.debug("Slope: %s, Offset: %s", slope, offset)

    def save_kev2dac_calib(self, threshold, gain, offset):
        """Save KeV conversion data to file.

        Args:
            threshold(int): Threshold calibration to save (0 or 1)
            gain(int/float): Conversion scale factor
            offset(int/float): Conversion offset

        """
        thresh_coeff = np.array([gain, offset])

        thresh_filename = posixpath.join(self.calib_dir,
                                         'fem{fem}',
                                         self.settings['mode'],
                                         self.settings['gain'],
                                         'threshold{threshold}'
                                         ).format(fem=self.fem,
                                                  threshold=threshold)

        logging.info("Saving calibration to: %s", thresh_filename)
        np.savetxt(thresh_filename, thresh_coeff, fmt='%.2f')
        os.chmod(thresh_filename, 0777)  # Allow anyone to overwrite

    def find_xray_energy_dac(self, chips=range(8), threshold=0, energy=5.9):
        """############## NOT TESTED

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

    def mask_row_block(self, chips, start, stop):
        """Mask a block of rows of pixels on the given chips.

        Args:
            chips(list(int)): List of indexes of chips to mask
            start(int): Index of starting row of mask
            stop(int): Index of final row to be included in mask

        Returns:
            numpy.array: Array mask that was written to config

        """
        bad_pixels = np.zeros(self.full_array_shape)

        for chip_idx in chips:
            # Subtract 1 to convert chip_size from number to index
            util.set_slice(bad_pixels, [start, chip_idx * self.chip_size],
                           [stop, (chip_idx + 1) * self.chip_size - 1], 1)

        for chip_idx in chips:
            pixel_mask_file = posixpath.join(self.calib_dir,
                                             'fem{fem}',
                                             self.settings['mode'],
                                             self.settings['gain'],
                                             'pixelmask.chip{idx}'
                                             ).format(fem=self.fem,
                                                      idx=chip_idx)
            np.savetxt(pixel_mask_file,
                       self._grab_chip_slice(bad_pixels, chip_idx),
                       fmt='%.18g', delimiter=' ')

        self.dawn.plot_image(bad_pixels, "Mask")
        self.load_config(chips)

        return bad_pixels

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
        OneE_Dac = MPX3RX.E1_DAC[self.settings['gain']]

        slope = (OneE_Dac[self.fem - 1, :] - self.dac_target) / OneE_E
        offset = [self.dac_target] * 8
        self.save_kev2dac_calib(threshold, slope, offset)
        logging.debug("Slope: %s, Offset: %s", slope, offset)

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
        E1_Dac = MPX3RX.E1_DAC[self.settings['gain']]

        E2_E = 12
        E2_Dac = MPX3RX.E2_DAC[self.settings['gain']]

        E3_E = 24
        E3_Dac = MPX3RX.E3_DAC[self.settings['gain']]

        offset = np.zeros(8)
        gain = np.zeros(8)

        for chip in chips:
            x = np.array([E1_E, E2_E, E3_E])
            y = np.array([E1_Dac[self.fem - 1, chip],
                          E2_Dac[self.fem - 1, chip],
                          E3_Dac[self.fem - 1, chip]])

            p1, p2 = self.dawn.plot_linear_fit(x, y, [0, 1],
                                               name="DAC vs Energy",
                                               label="Chip {}".format(chip))
            offset[chip] = p1
            gain[chip] = p2

        self.save_kev2dac_calib(threshold, gain, offset)

        logging.debug("Gain: %s, Offset: %s", gain, offset)
        logging.debug("Self.settings gain: %s", self.settings['gain'])

    def set_thresh_energy(self, threshold=0, thresh_energy=5.0):
        """Set given threshold in keV.

        ThresholdX DACs are calculated using thresholdX calibration file
        located in the calibration directory corresponding to the current mode
        and gain setting

        Args:
            threshold(int): Threshold to set (0 or 1)
            thresh_energy(float): Energy to set

        """  # TODO: Move docstring body somewhere more relevant
        fname = posixpath.join(self.calib_dir,
                               'fem{fem}',
                               self.settings['mode'],
                               self.settings['gain'],
                               'threshold{threshold}'
                               ).format(fem=self.fem, threshold=threshold)

        thresh_coeff = np.genfromtxt(fname)
        logging.debug("Thresh coefficients 0: %s",
                      thresh_coeff[0, :].astype(np.int))

        thresh_DACs = (thresh_energy * thresh_coeff[0, :] +
                       thresh_coeff[1, :]).astype(np.int)
        for chip in range(8):
            self.set_dac([chip],
                         "Threshold" + str(threshold), thresh_DACs[chip])

        time.sleep(0.2)
        self.settings['exposure'] = 100
        self.expose()
        time.sleep(0.2)
        self.expose()

        print("A threshold {threshold} of {energy}keV corresponds to {DACs} "
              "DAC units for each chip".format(threshold=threshold,
                                               energy=thresh_energy,
                                               DACs=thresh_DACs))

    def set_dacs(self, chips):
        """Set dacs to values recommended by MEDIPIX3-RX designers.

        Args:
            chips(list(int)): Chips to set DACs for

        """
        for dac, value in MPX3RX.DACS.items():
            self.set_dac(chips, dac, value)

    def read_chip_ids(self):
        """Read chip IDs."""
        self.app.read_chip_ids()

    def log_chip_ids(self):
        """Read chip IDs and logs chipIDs in calibration directory."""
        log_filename = posixpath.join(self.calib_dir,
                                      'fem{fem}',
                                      'efuseIDs'
                                      ).format(fem=self.fem)

        with open(log_filename, "w") as outfile:
            self.app.read_chip_ids(stdout=outfile)

    def monitor(self):
        """Monitor temperature, humidity, FEM voltage status and DAC out."""
        self.app.read_slow_control_parameters()

    def set_threshold0_dac(self, chips=range(8), dac_value=40):
        """Set threshold0 DAC to a selected value for given chips.

        Args:
            chips: Chips to set for
            dac_value: DAC value to set

        """
        self.set_dac(chips, 'Threshold0', dac_value)
        self.expose()

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
        self.set_threshold0_dac(chips, 40)
        self.file_name = 'Fe55_image_node_{fem}_{exposure}s'.format(
            fem=self.fem, exposure=self.settings['exposure'])
        self.output_folder = posixpath.join(self.root_path, "Fe55_images")
        time.sleep(0.5)

        self.expose(exposure)

        self.output_folder = img_path
        self.file_name = 'image'
        self.settings['exposure'] = 100

    def set_dac(self, chips, name="Threshold0", value=40):
        """Set any chip DAC at a given value in config file and then load.

        Args:
            chips(list(int)): Chips to set DACs for
            name(str): DAC to set (Any from self.dac_number keys)
            value(int): Value to set DAC to

        """
        logging.info("Setting DAC %s to %s", name, value)

        new_line = "{name} = {value}\r\n".format(name=name, value=value)
        dac_file_path = posixpath.join(self.calib_dir,
                                       'fem{fem}',
                                       self.settings['mode'],
                                       self.settings['gain'],
                                       'dacs'
                                       ).format(fem=self.fem)

        for chip in chips:
            with open(dac_file_path, 'r') as dac_file:
                f_content = dac_file.readlines()

            line_nb = chip * 29 + np.int(self.dac_number[name])
            f_content[line_nb] = new_line
            with open(dac_file_path, 'w') as dac_file:
                dac_file.writelines(f_content)

        self.app.load_dacs(chips, dac_file_path)

    def read_dac(self, dac_name):
        """Read back DAC analogue voltage using chip sense DAC (DAC out).

        Args:
            dac_name(str): DAC value to read (Any from self.dac_number keys)

        """
        dac_file = posixpath.join(self.calib_dir,
                                  'fem{fem}',
                                  self.settings['mode'],
                                  self.settings['gain'],
                                  'dacs'
                                  ).format(fem=self.fem)

        self.app.sense(self.chip_range, dac_name, dac_file)

    def scan_dac(self, chips, threshold, dac_range):
        """Perform a dac scan and plot the result (mean counts vs DAC values).

        Args:
            chips(Any from self.dac_number keys): Chips to scan
            threshold(str): Threshold to scan (ThresholdX DACs - X: 0-7)
            dac_range(Range): Range of DAC values to scan over

        Returns:
            numpy.array: DAC scan data

        """
        logging.info("Performing DAC Scan of %s; Start: %s, Stop: %s, "
                     "Step: %s", threshold, *dac_range.__dict__.values())

        dac_scan_file = util.generate_file_name("DACScan")

        dac_file = posixpath.join(self.calib_dir,
                                  "fem{fem}".format(fem=self.fem),
                                  self.settings['mode'],
                                  self.settings['gain'],
                                  "dacs")

        self.app.perform_dac_scan(chips, threshold, dac_range, dac_file,
                                  self.output_folder, dac_scan_file)

        file_path = posixpath.join(self.output_folder, dac_scan_file)
        if self.remote_node:
            file_path = self.app.grab_remote_file(file_path)

        util.wait_for_file(file_path, 5)
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
        for chip_idx in chips:
            chip = scan_data[:, 0:256, chip_idx * 256:(chip_idx + 1) * 256]
            plot_data[chip_idx, :] = chip.mean(2).mean(1)

        self.dawn.plot_dac_scan(plot_data, dac_axis)
        return plot_data, dac_axis

    def load_temp_config(self, chips, discLbits, discHbits, mask_bits):
        """Save the given disc configs to temporary files and load them.

        Args:
            chips(list(int)): Chips to load config for
            discLbits(numpy.array): DiscL pixel config (256 x 256 : 0 to 31)
            discHbits(numpy.array): DiscH pixel config (256 x 256 : 0 to 31)
            mask_bits(numpy.array): Pixel mask (256 x 256 : 0 or 1)

        """
        logging.info("Loading numpy arrays as temporary config.")

        template_path = posixpath.join(self.template_path,
                                       '{disc}.tmp')
        discH_bits_file = template_path.format(disc='discHbits')
        discL_bits_file = template_path.format(disc='discLbits')
        mask_bits_file = template_path.format(disc='maskbits')

        np.savetxt(discL_bits_file, discLbits, fmt='%.18g', delimiter=' ')
        np.savetxt(discH_bits_file, discHbits, fmt='%.18g', delimiter=' ')
        np.savetxt(mask_bits_file, mask_bits, fmt='%.18g', delimiter=' ')

        self.app.load_config(chips,
                             discL_bits_file, discH_bits_file, mask_bits_file)

    def load_config(self, chips=range(8)):
        """Load detector configuration files and default thresholds.

        From calibration directory corresponding to selected mode and gain.

        Args:
            chips(list(int)): Chips to load config for

        """
        logging.info("Loading discriminator bits from config directory.")

        for chip in chips:
            self.app.load_config([chip], self.discL_bits[chip],
                                 self.discH_bits[chip], self.pixel_mask[chip])

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

        logging.info("Capturing image with %sms exposure", exposure)
        image = self._acquire(1, exposure)

        plot_name = "Node Image - {time_stamp}".format(
            time_stamp=util.get_time_stamp())
        self.dawn.plot_image(image, plot_name)

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

        plots = min(frames, 5)  # Limit to 5 frames
        plot_tag = time.asctime()
        for plot in range(plots):
            self.dawn.plot_image(image[plot, :, :],
                                 name="Image_{tag}_{plot}".format(
                                 tag=plot_tag, plot=plot))

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
        file_name = util.generate_file_name("Image")

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
            file_path = self.app.grab_remote_file(file_path)

        util.wait_for_file(file_path, 5)
        image = self.dawn.load_image_data(file_path)
        return image

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
        file_name = "FlatField.hdf5"
        self.settings['exposure'] = exposure

        ff_image = 0
        for _ in range(num):
            image = self.expose()
            ff_image += image

        chip = 3  # TODO: Why?
        chip_mean = self._grab_chip_slice(ff_image, chip).mean()
        # Set all zero elements in the chip to the mean value
        ff_image[ff_image == 0] = chip_mean
        # Coeff array is array of means divided by actual values
        ff_coeff = np.ones([256, 256*8]) * chip_mean
        ff_coeff = ff_coeff/ff_image

        self.dawn.plot_image(self._grab_chip_slice(ff_image, chip),
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
        file_name = util.generate_file_name("FFImage")

        image_path = posixpath.join(self.output_folder, file_name)
        images = self.dawn.load_image_data(image_path)

        for image_idx in range(num_images):
            # TODO: Find better name than ff
            ff = images[image_idx, :, :] * ff_coeff
            ff[ff > 3000] = 0
            chip = 3
            self.dawn.plot_image(self._grab_chip_slice(ff, chip),
                                 name="Image data Cor")

    def logo_test(self):
        """Test the detector using test pulses representing excalibur logo."""
        # TODO: Make this call through test_pulse
        self.set_dac(self.chip_range, "Threshold0", 40)
        self.expose(10)  # TODO: Why does it need to set* and expose here?

        logo_tp = np.ones([256, 8*256])
        logo_file = posixpath.join(self.config_dir, "logo.txt")
        logo_small = np.loadtxt(logo_file)
        util.set_slice(logo_tp, [7, 225], [249, 1822], logo_small)
        logo_tp[logo_tp > 0] = 1
        logo_tp = 1 - logo_tp

        for chip in self.chip_range:
            dac_file = posixpath.join(self.calib_dir, 'dacs')

            # TODO: Why is this done for each chip?
            test_bits_file = posixpath.join(self.calib_dir,
                                            "Logo_chip{chip}_mask").format(
                                            chip=chip)
            np.savetxt(test_bits_file, self._grab_chip_slice(logo_tp, chip),
                       fmt='%.18g', delimiter=' ')

            # TODO: Do we really need to check these exist?
            if os.path.isfile(self.discL_bits[chip]) \
                and os.path.isfile(self.discH_bits[chip])  \
                    and os.path.isfile(self.pixel_mask[chip]):
                config_files = dict(discL=self.discL_bits[chip],
                                    discH=self.discH_bits[chip],
                                    pixel_mask=self.pixel_mask[chip])
            else:
                config_files = None

            self.app.configure_test_pulse([chip], dac_file, test_bits_file,
                                          config_files)

        time.sleep(0.2)

        file_name = util.generate_file_name("TPImage")

        self.app.acquire(self.chip_range,
                         self.settings['frames'],
                         self.settings['exposure'],
                         tp_count=100,
                         hdf_file=file_name)

        image_path = posixpath.join(self.output_folder, file_name)
        image = self.dawn.load_image_data(image_path)
        self.dawn.plot_image(image, name="Image_{}".format(time.asctime()))

    def test_pulse(self, chips, test_bits, pulses):
        """Test the detector using the given mask bits."""
        if type(test_bits) == str:
            test_bits_file = test_bits
        else:
            discLbits_file = self.calib_dir + 'fem' + \
                str(self.fem) + '/' + self.settings['mode'] + \
                '/' + self.settings['gain'] + '/' + 'testbits.tmp'
            np.savetxt(test_bits_file, test_bits, fmt='%.18g', delimiter=' ')

        file_name = util.generate_file_name("TPImage")

        for chip in chips:
            dac_file = posixpath.join(self.calib_dir, 'dacs')
            self.app.configure_test_pulse([chip], dac_file, test_bits_file)

        self.app.acquire(chips,
                         self.settings['frames'],
                         self.settings['exposure'],
                         tp_count=pulses,
                         hdf_file=file_name)

        image_path = posixpath.join(self.output_folder, file_name)
        image = self.dawn.load_image_data(image_path)
        self.dawn.plot_image(image, name="Image_{}".format(time.asctime()))

    def save_discbits(self, chips, discbits, discbitsFilename):
        """Save discbit array into file in the calibration directory.

        Args:
            chips(list(int)): Chips to save for
            discbits(numpy.array): Numpy array to save
            discbitsFilename(str): File name to save to (discL or discH)

        """
        for chip_idx in chips:
            discbits_file = posixpath.join(
                self.template_path, '{disc}.chip{chip}'.format(
                    disc=discbitsFilename, chip=chip_idx))

            logging.info("Saving discbits to %s", discbits_file)
            np.savetxt(discbits_file,
                       self._grab_chip_slice(discbits, chip_idx),
                       fmt='%.18g', delimiter=' ')

    def mask_super_column(self, chip, super_column):
        """Mask a super column (32 bits?) in a chip and update the maskfile.

        Args:
            chip(list(int)): Chip to mask
            super_column(int): Super column to mask (0 to 7)

        """
        bad_pixels = np.zeros(self.full_array_shape)
        bad_pixels[:, chip*256 + super_column * 32:chip * 256 +
                   super_column * 32 + 64] = 1  # TODO: Should it be 64 wide?

        pixel_mask_file = self.pixel_mask[chip]
        np.savetxt(pixel_mask_file, self._grab_chip_slice(bad_pixels, chip),
                   fmt='%.18g', delimiter=' ')
        self.app.load_config([chip],
                             self.discL_bits[chip], pixelmask=pixel_mask_file)

        self.dawn.plot_image(bad_pixels, name="Bad Pixels")

    def mask_col(self, chip, column):
        """Mask a column in a chip and update the maskfile.

        Args:
            chip(list(int)): Chip to mask
            column(int): Column to mask (0 to 255)

        """
        bad_pixels = np.zeros(self.full_array_shape)
        bad_pixels[:, column] = 1

        pixel_mask_file = self.pixel_mask[chip]
        np.savetxt(pixel_mask_file, self._grab_chip_slice(bad_pixels, chip),
                   fmt='%.18g', delimiter=' ')

        self.app.load_config([chip],
                             self.discL_bits[chip], pixelmask=pixel_mask_file)

        self.dawn.plot_image(bad_pixels, name="Bad pixels")

    def mask_pixels(self, chips, image_data, max_counts):
        """Mask pixels in image_data with counts above max_counts.

        Also updates the corresponding maskfile in the calibration directory

        Args:
            chips(list(int)): Chips to mask
            image_data(numpy.array): Numpy array image
            max_counts(int): Mask threshold

        """
        bad_pix_tot = np.zeros(8)
        bad_pixels = image_data > max_counts
        self.dawn.plot_image(bad_pixels, name="Bad pixels")
        for chip_idx in chips:
            bad_pix_tot[chip_idx] = \
                self._grab_chip_slice(bad_pixels, chip_idx).sum()

            print('####### {pix} noisy pixels in chip {chip} ({tot})'.format(
                pix=str(bad_pix_tot[chip_idx]),
                chip=str(chip_idx),
                tot=str(100 * bad_pix_tot[chip_idx] / (256**2))))

            pixel_mask_file = self.pixel_mask[chip_idx]
            np.savetxt(pixel_mask_file,
                       self._grab_chip_slice(bad_pixels, chip_idx),
                       fmt='%.18g', delimiter=' ')
            self.app.load_config([chip_idx], self.discL_bits[chip_idx],
                                 pixelmask=pixel_mask_file)

        print('####### {pix} noisy pixels in half module ({tot}%)'.format(
            pix=str(bad_pix_tot.sum()),
            tot=str(100 * bad_pix_tot.sum() / (8 * 256**2))))

    def mask_pixels_using_dac_scan(self, chips=range(8),
                                   threshold="Threshold0",
                                   dac_range=Range(20, 120, 2)):
        """Perform threshold dac scan and mask pixels with counts > max_counts.

        Also updates the corresponding maskfile in the calibration directory

        Args:
            chips(list(int)): Chips to scan
            threshold(str): Threshold to scan (0 or 1)
            dac_range(Range): Range to scan over

        """
        max_counts = 1
        bad_pix_tot = np.zeros(8)
        self.settings['exposure'] = 100
        dac_scan_data = self.scan_dac(chips, threshold, dac_range)
        bad_pixels = dac_scan_data.sum(0) > max_counts
        self.dawn.plot_image(bad_pixels, name="Bad Pixels")

        for chip_idx in chips:
            bad_pix_tot[chip_idx] = \
                self._grab_chip_slice(bad_pixels, chip_idx).sum()

            print('####### {pix} noisy pixels in chip {chip} ({tot})'.format(
                pix=str(bad_pix_tot[chip_idx]),
                chip=str(chip_idx),
                tot=str(100 * bad_pix_tot[chip_idx] / (256**2))))

            pixel_mask_file = self.pixel_mask[chip_idx]
            np.savetxt(pixel_mask_file,
                       self._grab_chip_slice(bad_pixels, chip_idx),
                       fmt='%.18g', delimiter=' ')
            # subprocess.call([self.command, "-i", self.ipaddress,
            #                  "-p", self.port,
            #                  "-m", self.mask(range(chip_idx, chip_idx + 1)),
            #                  "--config",
            #                  "--pixelmask=" + pixel_mask_file,
            #                  "--config",
            #                  "--discl=" + discLbitsFile])

        print('####### {pix} noisy pixels in half module ({tot}%)'.format(
            pix=str(bad_pix_tot.sum()),
            tot=str(100 * bad_pix_tot.sum() / (8 * 256**2))))

    def unmask_all_pixels(self, chips):
        """Unmask all pixels and update mask file in calibration directory.

        Args:
            chips(list(int)): Chips to unmask

        """

        bad_pixels = np.zeros(self.full_array_shape)

        for chip_idx in chips:
            pixel_mask_file = self.pixel_mask[chip_idx]
            np.savetxt(pixel_mask_file,
                       self._grab_chip_slice(bad_pixels, chip_idx),
                       fmt='%.18g', delimiter=' ')
            self.app.load_config([chip_idx], self.discL_bits[chip_idx],
                                 pixelmask=pixel_mask_file)

    def unequalize_all_pixels(self, chips):
        """Reset discL_bits and pixel_mask bits to 0.

        Args:
            chips(list(int)): Chips to unequalize

        """
        discL_bits = 31 * np.zeros(self.full_array_shape)  # TODO: Not 32?
        # TODO: Just have a single 256*256 array and save to all???
        for chip_idx in chips:
            discL_bits_file = self.discL_bits[chip_idx]
            np.savetxt(discL_bits_file,
                       self._grab_chip_slice(discL_bits, chip_idx),
                       fmt='%.18g', delimiter=' ')
            # TODO: Doesn't save pixelmask...
            self.app.load_config([chip_idx], discL_bits_file,
                                 pixelmask=self.pixel_mask[chip_idx])

    def check_calib_dir(self):
        """Check if calibration directory exists and backs it up."""
        calib_dir = posixpath.join(self.calib_dir,
                                   'fem{fem}',
                                   self.settings['mode'],
                                   self.settings['gain']).format(
            fem=self.fem)

        if (os.path.isdir(calib_dir)) == 0:
            os.makedirs(calib_dir)
        else:
            logging.info("Backing up calib directory.")
            backup_dir = "{calib}_backup_{time_stamp}".format(
                calib=self.calib_dir, time_stamp=util.get_time_stamp())
            shutil.copytree(self.calib_dir, backup_dir)
            logging.debug("Backup directory: %s", backup_dir)

        dac_file = posixpath.join(calib_dir, 'dacs')
        if os.path.isfile(dac_file) == 0:
            shutil.copy(posixpath.join(self.config_dir, 'dacs'), calib_dir)

        # if os.path.isfile(dac_file) == 0:
        #     shutil.copy(self.config_dir + 'zeros.mask',
        #                 calib_dir)

    def copy_slgm_into_other_gain_modes(self):
        """Copy slgm calibration folder into lgm, hgm and shgm folders.

        This function is used at the end of threshold equalization because
        threshold equalization is performed in the more favorable gain mode
        slgm and threshold equalization data is independent of the gain mode.

        """
        logging.info("Copying SLGM calib to other gain modes")
        template_path = posixpath.join(self.calib_dir,
                                       'fem{fem}'.format(fem=self.fem),
                                       self.settings['mode'],
                                       '{gain_mode}')

        lgm_dir = template_path.format(gain_mode='lgm')
        hgm_dir = template_path.format(gain_mode='hgm')
        slgm_dir = template_path.format(gain_mode='slgm')
        shgm_dir = template_path.format(gain_mode='shgm')

        if os.path.exists(lgm_dir):
            shutil.rmtree(lgm_dir)
        if os.path.exists(hgm_dir):
            shutil.rmtree(hgm_dir)
        if os.path.exists(shgm_dir):
            shutil.rmtree(shgm_dir)

        shutil.copytree(slgm_dir, lgm_dir)
        shutil.copytree(slgm_dir, hgm_dir)
        shutil.copytree(slgm_dir, shgm_dir)

    def open_discbits_file(self, chips, discbits_filename):
        """Open discriminator bit array stored in current calibration folder.

        Args:
            chips(list(int)): Chips to load
            discbits_filename(str): File to grab data from (discL or discH)

        Returns:
            numpy.array: Discbits array

        """
        logging.info("Reading discriminator bits from file.")
        discbits = np.zeros(self.full_array_shape)
        for chip_idx in chips:
            discbits_file = posixpath.join(
                self.template_path, '{disc}.chip{chip}'.format(
                    disc=discbits_filename, chip=chip_idx))

            self._set_chip_slice(discbits, chip_idx, np.loadtxt(discbits_file))

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
            discbits_roi = self.open_discbits_file(chips, disc_name +
                                                   'bits_roi_' + str(step))
            discbits[roi_full_mask.astype(bool)] = \
                discbits_roi[roi_full_mask.astype(bool)]
            # TODO: Should this be +=? Currently just uses final ROI
            self.dawn.plot_image(discbits_roi, name="Disc bits")

        self.save_discbits(chips, discbits, disc_name + "bits")
        self.dawn.plot_image(discbits, name="Disc bits total")

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
        logging.info("Finding noise edges in DAC scan data.")
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

        self.dawn.plot_image(edge_dacs, name="Noise Edges")
        self._display_histogram(chips, edge_dacs, "Noise Edges Histogram")
        return edge_dacs

    def find_max(self, chips, dac_scan_data, dac_range):
        """Find noise max in threshold dac scan.

        Returns:
            numpy.array: Max noise data

        """
        logging.info("Finding max noise in DAC scan data.")
        max_dacs = dac_range.stop - dac_range.step * np.argmax(
            (dac_scan_data[::-1, :, :]), 0)
        # TODO: Assumes low to high scan? Does it matter?

        self.dawn.plot_image(max_dacs, name="Noise Max")
        self._display_histogram(chips, max_dacs, "Histogram of Noise Max")
        return max_dacs

    def _display_histogram(self, chips, data, name):
        """Plot an image and a histogram of given data.

        Args:
            chips(list(int)): Chips to plot for
            data(numpy.array): Data to analyse

        """
        image_data = []
        for chip_idx in chips:
            image_data.append(self._grab_chip_slice(data, chip_idx))
        self.dawn.plot_histogram(image_data, name)

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
        logging.info("Optimizing %s", disc_name)

        thresholds = dict(discL="Threshold0", discH="Threshold1")
        threshold = thresholds[disc_name]

        # Definition of parameters to be used for threshold scans
        self.settings['exposure'] = 5
        self.settings['counter'] = 0
        self.settings['equalization'] = 1  # Might not be necessary when
        # optimizing DAC Disc
        dac_range = Range(0, 150, 5)

        ######################################################################
        # STEP 1
        # Perform threshold DAC scans for all discbits set at 0 and various
        # DACdisc values, discbits set at 0 shift DAC scans to the right
        # Plot noise edge position as a function of DACdisc
        # Calculate noise edge shift in threshold DAC units per DACdisc
        # DAC unit
        ######################################################################

        discbit = 0
        calib_plot_name = "Mean edge shift in Threshold DACs as a function " \
                          "of DAC_disc for discbit =" + str(discbit)

        # Set discbits at 0
        discbits = discbit * np.ones(self.full_array_shape)
        self.load_all_discbits(chips, disc_name, discbits, roi_mask)

        # Threshold DAC scans, fitting and plotting
        p0 = [5000, 50, 30]
        dac_disc_range = range(0, 150, 50)
        x0 = np.zeros([8, len(dac_disc_range)])
        for idx, dac_value in enumerate(dac_disc_range):
            x0[:, idx], _ = self._chip_dac_scan(chips, threshold, dac_value,
                                                dac_range, discbit, p0)

            self.dawn.clear_plot(calib_plot_name)
            for chip_idx in chips:
                # TODO: idx:(idx+1), not 0:idx+1; why doesn't it plot at end?
                self.dawn.add_plot_line(np.asarray(dac_disc_range[0:idx + 1]),
                                        x0[chip_idx, 0:idx + 1],
                                        calib_plot_name,
                                        label="Chip {}".format(chip_idx))

        # Plot mean noise edge vs DAC Disc for discbits set at 0
        offset = np.zeros(8)
        gain = np.zeros(8)
        for chip_idx in chips:
            results = self.dawn.plot_linear_fit(
                np.asarray(dac_disc_range), x0[chip_idx, :], [0, -1],
                fit_name=calib_plot_name, label="Chip {}".format(chip_idx))

            offset[chip_idx], gain[chip_idx] = results[0], results[1]

        # Fit range should be adjusted to remove outliers at 0 and max DAC 150

        ######################################################################
        # STEP 2
        # Perform threshold DAC scan for all discbits set at 15 (no correction)
        # Fit threshold scan and calculate width of noise edge distribution
        ######################################################################

        discbit = 15  # Centre of valid range - No correction
        dac_value = 80  # Value does not matter since no correction is applied

        discbits = discbit * np.ones(self.full_array_shape)
        self.load_all_discbits(chips, disc_name, discbits, roi_mask)

        p0 = [5000, 0, 30]
        x0, sigma = self._chip_dac_scan(chips, threshold, dac_value, dac_range,
                                        discbit, p0)

        opt_dac_disc = np.zeros(self.num_chips)
        for chip_idx in chips:
            opt_value = int(self.num_sigma * sigma[chip_idx] / gain[chip_idx])
            self.set_dac([chip_idx], threshold, opt_value)
            opt_dac_disc[chip_idx] = opt_value

        self._display_optimization_results(chips, x0, sigma, gain,
                                           opt_dac_disc)

    def _chip_dac_scan(self, chips, threshold, dac_value, dac_range, discbit,
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

        self.set_dac(chips, threshold, dac_value)
        # Scan threshold
        dac_scan_data = self.scan_dac(chips, threshold, dac_range)
        # Find noise edges
        edge_dacs = self.find_max(chips, dac_scan_data, dac_range)

        scan_data = [None]*8
        for chip_idx in self.chip_range:
            if chip_idx in chips:
                scan_data[chip_idx] = self._grab_chip_slice(edge_dacs,
                                                            chip_idx)
        x0, sigma = self.dawn.plot_gaussian_fit(scan_data, plot_name, p0, bins)

        return x0, sigma

    def _display_optimization_results(self, chips, x0, sigma, gain,
                                      opt_dac_disc):
        """Print out results of _chip_dac_scan."""
        print("Edge shift (in Threshold DAC units) produced by 1 DACdisc DAC"
              "unit for discbits=15:")
        for chip in chips:
            print("Chip {chip}: {shift}".format(
                chip=chip, shift=str(round(gain[chip], 2))))

        print("Mean noise edge (DAC Units) for discbits=15:")
        for chip in chips:
            print("Chip {chip}: {noise}".format(
                chip=chip, noise=round(x0[chip])))

        print("Sigma of noise edge (DAC units rms) distribution for "
              "discbits=15:")
        for chip in chips:
            print("Chip {chip}: {sigma}".format(
                chip=chip, sigma=round(sigma[chip])))

        ######################################################################
        # STEP 3
        # Calculate DAC disc required to bring all noise edges within X sigma
        # of the mean noise edge
        # X is defined by self.nbOfsigma
        ######################################################################

        print("Optimum equalization target (DAC units):")
        for chip in chips:
            print("Chip {chip}: {target}".format(
                chip=chip, target=round(x0[chip])))

        if abs(x0 - self.dac_target).any() > self.allowed_delta:  # To be checked
            print("########################### ONE OR MORE CHIPS NEED A"
                  "DIFFERENT EQUALIZATION TARGET")
        else:
            print("Default equalization target of {target} DAC units "
                  "can be used.".format(target=self.dac_target))

        print("DAC shift (DAC units) required to bring all pixels with an edge"
              "within +/- {num_sigma} sigma of the target, at the target "
              "of {target}:".format(num_sigma=self.num_sigma,
                                    target=self.dac_target))
        for chip in chips:
            print("Chip {chip}: {shift}".format(
                chip=chip, shift=int(self.num_sigma * sigma[chip])))

        print("Edge shift (Threshold DAC units) produced by 1 DACdisc DAC"
              " unit for discbits=0 (maximum shift):")
        for chip in chips:
            print("Chip {chip}: {shift}".format(
                chip=chip, shift=round(gain[chip], 2)))

        print("###############################################################"
              "########################")
        print("Optimum DACdisc value (DAC units) required to bring all pixels "
              "with an edge within +/- {num_sigma} sigma of the target, at "
              "the target of {target}".format(num_sigma=self.num_sigma,
                                              target=self.dac_target))
        for chip in chips:
            print("Chip {chip}: {opt_value}".format(
                chip=chip, opt_value=opt_dac_disc[chip]))

        print("###############################################################"
              "########################")
        print("Edge shift (Threshold DAC Units) produced by 1 step of the"
              "32 discbit correction steps:")
        for chip in chips:
            print("Chip {chip}: {shift}".format(
                chip=chip, shift=opt_dac_disc[chip] / 16))

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
        logging.info("Equalising %s", disc_name)

        self.settings['exposure'] = 5
        self.settings['counter'] = 0
        self.settings['equalization'] = 1

        inv_mask = 1 - roi_full_mask
        if method.lower() == "stripes":
            dac_range = Range(0, 20, 2)
            discbits_tmp = np.zeros(self.full_array_shape) * inv_mask
            for idx in range(self.chip_size):
                discbits_tmp[idx, :] = idx % 32
            for idx in range(self.chip_size * self.num_chips):
                discbits_tmp[:, idx] = (idx % 32 + discbits_tmp[:, idx]) % 32
            discbits_tmp *= inv_mask

            discbits = -10 * np.ones(self.full_array_shape) * inv_mask
            edge_dacs_stack = np.zeros([32] + self.full_array_shape)
            discbits_stack = np.zeros([32] + self.full_array_shape)
            for scan in range(0, 32, 1):
                logging.info("Equalize discbits step %s", scan)
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
                for chip in chips:
                    for x in range(self.chip_size):
                        for y in range(chip * self.chip_size,
                                       (chip + 1) * self.chip_size):
                            discbits[x, y] = \
                                discbits_stack[scan_nb[x, y], x, y]

                self.dawn.plot_image(discbits, name="Discriminator Bits")

                plot_name = "Histogram of Final Discbits"
                self.dawn.clear_plot(plot_name)
                self.dawn.plot_histogram_with_mask(chips, discbits, inv_mask,
                                                   plot_name)
        else:
            raise NotImplementedError("Equalization method not implemented.")

        self.settings['disccsmspm'] = 'discL'
        self.settings['equalization'] = 0

        print("Pixel threshold equalization complete")

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
        for chip in chips:
            if disc_name == 'discH':
                discLbits = self.open_discbits_file(chips, 'discLbits')
                discHbits = temp_bits
            elif disc_name == 'discL':
                discLbits = temp_bits
                discHbits = np.zeros(self.full_array_shape)
            else:
                raise ValueError("Discriminator must be L or H, got {bad_disc}"
                                 .format(bad_disc=disc_name))

            # TODO: Can we just call once with all chips?
            self.load_temp_config([chip],
                                  self._grab_chip_slice(discLbits, chip),
                                  self._grab_chip_slice(discHbits, chip),
                                  self._grab_chip_slice(mask, chip))

    def check_calib(self, chips, dac_range):
        """Check if dac scan looks OK after threshold calibration.

        NOT TESTED

        Args:
            chips(list(int)): Chips to check
            dac_range(Range): Range to scan DAC over (?)

        """
        pass  # TODO: Function doesn't work, what is `roi`?

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
            for chip in chips:
                roi_full_mask[step*256/steps:step*256/steps + 256/steps,
                              chip*256:chip*256 + 256] = 1
        elif roi_type == "spacing":
            spacing = steps**0.5
            roi_full_mask = np.zeros(self.full_array_shape)
            bin_repr = np.binary_repr(step, 2)
            for chip in chips:
                roi_full_mask[0 + int(bin_repr[0]):256 -
                                  int(bin_repr[0]):spacing,
                              chip*256 +
                                  int(bin_repr[1]):chip*256 +
                                  256 - int(bin_repr[1]):spacing] \
                    = 1
        else:
            raise NotImplementedError("Available methods are 'rect' and "
                                      "'spacing', got {}".format(roi_type))

        self.dawn.plot_image(roi_full_mask, name="Roi Mask")

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
        self.set_dac(chips, "Threshold0", 60)  # To be above the noise
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
        self.copy_slgm_into_other_gain_modes()  # Copy slgm threshold
        # equalization folder to other gain threshold_equalization folders

    def loop(self, num):
        """Acquire images and plot the sum.

        Args:
            num(int): Number of images to acquire

        """
        tmp = 0
        for _ in range(num):
            tmp = self.expose() + tmp
            self.dawn.plot_image(tmp, name="Sum")

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

    def set_gnd_fbk_cas_excalibur_rx001(self, chips, fem):
        """Set GND, FBK and CAS values.

        Args:
            chips(list(int)): Chips to set
            fem(int): FEM to set

        """
        logging.info("Setting GND, FBK and Cas values from ECD config.")
        for chip in chips:
            self.set_dac([chip], 'GND', MPX3RX.GND_DAC[fem - 1, chip])
            self.set_dac([chip], 'FBK', MPX3RX.FBK_DAC[fem - 1, chip])
            self.set_dac([chip], 'Cas', MPX3RX.CAS_DAC[fem - 1, chip])

        # self.read_dac(range(8), 'GND')
        # self.read_dac(range(8), 'FBK')
        # self.read_dac(range(8), 'Cas')

    def rotate_all_configs(self):
        """Rotate arrays in config files for EPICS.

        Calibration files of node 1, 3 and 5 have to be rotated in order to be
        loaded correctly in EPICS. This routine copies calib into calib_epics
        and rotate discLbits, discHbits and maskbits files when they exist for
        node 1, 3, and 5

        """
        chips = range(self.num_chips)
        EPICS_calib_path = self.calib_dir + '_epics'
        shutil.copytree(self.calib_dir, EPICS_calib_path)

        logging.debug("EPICS_calib_path: %s", EPICS_calib_path)

        template_path = posixpath.join(EPICS_calib_path,
                                       'fem{fem}',
                                       'spm',
                                       'slgm',
                                       '{disc}.chip{chip}')

        for fem in [1, 3, 5]:
            for chip_idx in chips:
                discLbits_file = template_path.format(fem=fem,
                                                      disc='discLbits',
                                                      chip=chip_idx)
                discHbits_file = template_path.format(fem=fem,
                                                      disc='discHbits',
                                                      chip=chip_idx)
                pixel_mask_file = template_path.format(fem=fem,
                                                       disc='pixelmask',
                                                       chip=chip_idx)
                if os.path.isfile(discLbits_file):
                    util.rotate_config(discLbits_file)
                    print("{file} rotated".format(file=discLbits_file))
                if os.path.isfile(discHbits_file):
                    util.rotate_config(discHbits_file)
                    print("{file} rotated".format(file=discHbits_file))
                if os.path.isfile(pixel_mask_file):
                    util.rotate_config(pixel_mask_file)
                    print("{file} rotated".format(file=pixel_mask_file))

    def _grab_chip_slice(self, array, chip_idx):
        """Grab a chip from a full array.

        Args:
            array(numpy.array): Array to grab from
            chip_idx(int): Index of section of array to grab

        Returns:
            numpy.array: Sub array

        """
        start, stop = self._generate_chip_range(chip_idx)
        return util.grab_slice(array, start, stop)

    def _set_chip_slice(self, array, chip_idx, value):
        """Grab a section of a 2D numpy array.

        Args:
            array(numpy.array): Array to grab from
            chip_idx(int): Index of section of array to grab
            value(numpy.array/int/float): Value to set slice to

        """
        start, stop = self._generate_chip_range(chip_idx)
        util.set_slice(array, start, stop, value)

    def _generate_chip_range(self, chip_idx):
        """Calculate start and stop coordinates of given chip.

        Args:
            chip_idx(int): Chip to calculate range for

        """
        start = [0, chip_idx * self.chip_size]
        stop = [self.chip_size - 1, (chip_idx + 1) * self.chip_size - 1]
        return start, stop

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
