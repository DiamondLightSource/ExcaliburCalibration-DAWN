"""Python library for MPX3RX-based detectors calibration and testing.

/dls/detectors/support/silicon_pixels/excaliburRX/PyScripts/MPX3RX-DAWN.py
DIAMOND LIGHT SOURCE 30-07-2015

Initially developed for I13 EXCALIBUR-3M-RX001 detector
EXCALIBUR-specific functions need to be extracted and copied into a separate
library. This will allow for the scripts to be usable with any MPX3-based
system provided that a library of control functions is available for each
type of detector:
# EXCALIBUR
# LANCELOT/MERLIN

===================== EXCALIBUR Test-Application =========================

NOTE: excaliburRX detector class communicates with FEMs via the
ExcaliburTestApplicationInterface Python class

NOTE: ExcaliburRX detector class requires configuration files copied in:
/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/
config/

========= To install libraries required by EXCALIBUR Test-Application

excalibur Test-Application requires libboost and libhdf5 libraries to be
installed locally. Use the following instructions to install the libraries:

cd /dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/
excaliburRxlib
mkdir /home/fedID/lib/
cp lib* /home/fedID/lib/
This will copy the required libraries:
[ktf91651@pc0066 /]$ ll /home/ktf91651/lib
total 17880
-rwxr-xr-x. 1 ktf91651 ktf91651   17974 Mar  7  2014 libboost_system.so
-rwxr-xr-x. 1 ktf91651 ktf91651   17974 Mar  7  2014 libboost_system.so.1.47.0
-rwxr-xr-x. 1 ktf91651 ktf91651  138719 Mar  7  2014 libboost_thread.so
-rwxr-xr-x. 1 ktf91651 ktf91651  138719 Mar  7  2014 libboost_thread.so.1.47.0
-rwxr-xr-x. 1 ktf91651 ktf91651 8946608 Mar  7  2014 libhdf5.so
-rwxr-xr-x. 1 ktf91651 ktf91651 8946608 Mar  7  2014 libhdf5.so.7

edit
/home/fedID/.bashrc_local
to add path to excalibur libraries

[ktf91651@p99-excalibur01 ~]$ more .bashrc_local
LIBDIR=$HOME/lib
if [ -d $LIBDIR ]; then
    export LD_LIBRARY_PATH=${LIBDIR}:${LD_LIBRARY_PATH}
fi

check path using
[ktf91651@p99-excalibur01 ~]$ echo $LD_LIBRARY_PATH
/home/ktf91651/lib:

======================== MODULE CALIBRATION USING PYTHON SCRIPTS

================= FRONT-END POWER-ON

To calibrate a 1/2 Module

ssh to the PC server node(standard DLS machine) connected to the MASTER FEM
card (the one interfaced with the I2C bus of the Power card)
On I13 EXCALIBUR-3M-RX001, this is the top FEM (192.168.0.106) connected to
node 1
###########################
> ssh i13-1-excalibur01
###########################
Enable LV and make sure that HV is set to 120V during calibration:
##########################################################################
> /dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/
excaliburTestApp -i 192.168.0.106 -p 6969 -m 0xff --lvenable 1
> /dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/
excaliburTestApp -i 192.168.0.106 -p 6969 -m 0xff --lvenable 0
> /dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/
excaliburTestApp -i 192.168.0.106 -p 6969 -m 0xff --lvenable 1
> /dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/
excaliburTestApp -i 192.168.0.106 -p 6969 -m 0xff --hvbias 120
> /dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/
excaliburTestApp -i 192.168.0.106 -p 6969 -m 0xff --hvenable 1
###########################################################################

ssh to the PC server node (standard DLS machine) connected to the FEM card
controlling the 1/2 module which you want to calibrate.
########################
> ssh i13-1-excalibur0x with x in [1-6] and x=1 corresponds to the top FEM
(Master FEM IP:192.168.0.106) connected to PC server node 1
########################

===================  DAWN START-UP

On the PC server node, start DAWN by typing in a shell:
######################
> module load dawn
> dawn &
#######################

================== PYTHON SCRIPT

Select the DExplore perspective
Open the file /dls/detectors/support/silicon_pixels/excaliburRX/PyScripts/
MPX3RX-DAWN.py
Run excaliburDAWN.py in the interactive console by clicking on the python
icon "activates the interactive console" (CTL Alt ENTER)
Select "Python Console" when the interactive interpreter console opens
(You might have to run the script twice)

Scripts were tested with /dls_sw/apps/python/anaconda/1.7.0/64/bin/python/
Python 2.7.10 |Anaconda 1.7.0 (64-bit)

Install Python interpreter using
    Window - Preferences - PyDev - Interpreters menus

In the Interactive console you need to create an excaliburRx object:
########################
> x = excaliburRX(node)
########################
were node is the EXCALIBUR PC node number (between 1 and 6 for
EXCALIBUR-3M) of the 1/2 module under test
For example: on I13 the top FEM of EXCALIBUR-3M-RX001 is connected to
node 1 (i13-1-excalibur01) and the bottom fem to node (i13-1-excalibur06).
When running Python calibration scripts on node i13-1-excalibur0X (with X
in [1:6]), you should use: x=excaliburRX(X)

For example when running the Python calibration scripts on node
i13-1-excalibur0X (with X in [1:6]), you should use: x=excaliburRX(X)
For I13 installation top FEM is connected to node 1 and bottom fem to
node 6

================ FBK GND and CAS DACs adjustment

The very first time you calibrate a module, you need to manually adjust
3 DACs: FBK, CAS and GND
The procedure is described in set_gnd_fbk_cas_excalibur_rx001
If you swap modules you also need to edit set_gnd_fbk_cas_excalibur_rx001
accordingly since set_gnd_fbk_cas_excalibur_rx001 contains DAC parameters
specific each 3 modules based on the position of the module

================= THRESHOLD EQUALIZATION

To run threshold_equalization scripts, just type in the interactive Python
console:
########################
> x.threshold_equalization()
########################
By default, threshold_equalization files will be created locally in a
temporary folder : /tmp/femX of the PC server node X.
You should copy the folder /femX in the path were EPICS expects
threshold_equalization files for all the fems/nodes

At the end of the threshold_equalization you should get the following
message in the interactive console:
Pixel threshold equalization complete

================= THRESHOLD CALIBRATION

To calibrate thresholds using default keV to DAC mapping
#####################################
> x.threshold_calibration_all_gains()
#####################################

To calibrate thresholds using X-rays:

Method 1: Brute-force:

Produce monochromatic X-rays or use Fe55
Perform a DAC scan using :
##################################
[dacscanData, scanRange] = self.scan_dac([0], "Threshold0", (80, 20, 1))
###################################
This will produce 2 plots:
dacscan with intergal spectrum
spectrum with differential spectrum

Inspect the spectrum and evaluate the position of the energy peak in DAC
units.
Example:
E = 6keV for energy peak DAC = 60
Since calibration is performed on noise peak, 0keV correspond to the
selected DACtarget (10 by default)
E = 0keV for DAC = 10
Perform a linear fit of the DAC (in DAC units) as a function of energy
(keV) and  edit threshold0 file in the calibration directory accordingly:

Each Threshold calibration file consists of 2 rows of 8 floating point
numbers:
# g0     g1   g2   g3   g4   g5   g6   g7
# Off0 Off1 Off2 Off3 Off4 Off5 Off6 Off7
# and the DAC value to apply to chip x for a requested threshold energy
value E in keV is given by:
# DACx= gx * E + Offx

Method 2: Using 1 energy and the noise peak dac

Method 3: Using several energies. Script not written yet.


============== ACQUIRE X_RAY TEST IMAGE WITH FE55

threshold_equalization data is then automatically loaded. And you can
acquire a 60s image from Fe55 X-rays using the following command:
####################
> x.fe55_image_rx001()
####################

To change the exposure time used during image acquisition:
##########################################
> x.fe55_image_rx001(range(8), exp_time_in_ms)
##########################################

============== ACQUIRE IMAGES

##########
> x.expose()
##########

To change acquisition time before exposure:
#############################################
> x.settings['acqtime']=1000 (for 1s exposure)
> x.expose()
#############################################
where acqtime is in ms

The image will be automatically saved in
    /dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/Fe55_images
The image is displayed in Image Data plot window which is opened by
selected Show Plot View in the Window tab

=====================PIXEL MASKING========================================

===================== SET THRESHOLD DAC
To change threshold:
######################################################################
> x.set_dac(range(8), "Threshold0", 40)
######################################################################
This will allow you to put your threshold just above the noise to check the
response of the 1/2 module to X-rays

================== LOAD DETECTOR CONFIGURATION
To load discbits, maskbits and default thresholds from current calibration
directory
###############
> x.load_config()
###############

================= TEST DETECTOR USING EXCALIBUR TEST PULSE LOGO
############
> x.testLogo()
############

==========================================================================
NOTE: chip 0 in this program correspond to the left chip of the bottom
half of a module or to the right chip of the top half of the module when
facing the front-surface of sensor

"""
import os
import posixpath
import shutil
import time

from collections import namedtuple

import numpy as np

from scripts.ExcaliburTestAppInterface import ExcaliburTestAppInterface
from scripts.ExcaliburDAWN import ExcaliburDAWN


Range = namedtuple("Range", "start stop step")


class ExcaliburRX(object):
    """Class to calibrate Excalibur-RX detectors.

    ExcaliburRX is a class defining methods required to calibrate each 1/2
    module (8 MPX3-RX chips) of an EXCALIBUR-RX detector.
    These calibration scripts will work only inside the Python interpreter of
    DAWN software running on the PC sever node connected to the FEM controlling
    the half-module which you wish to calibrate

    """
    # Threshold equalization will align pixel noise peaks at this DAC value
    dac_target = 10
    # Acceptable difference compared to dac_target
    allowed_delta = 4

    # Sigma value based on experimental data
    nb_of_sigma = 3.2

    # Number of pixels per chip along each axis
    chip_size = 256
    # Number of chips in 1/2 module
    num_chips = 8

    root_path = '/dls/detectors/support/silicon_pixels/excaliburRX/'
    calib_dir = posixpath.join(root_path, '3M-RX001/calib')
    config_dir = posixpath.join(root_path, 'TestApplication_15012015/config')

    # Line number used when editing dac file with new dac values
    dac_number = dict(Threshold0='1', Threshold1='2', Threshold2='3',
                      Threshold3='4', Threshold4='5', Threshold5='6',
                      Threshold6='7', Threshold7='8',
                      Preamp='9', Ikrum='10', Shaper='11', Disc='12',
                      DiscLS='13', ShaperTest='14', DACDiscL='15',
                      DACTest='16', DACDiscH='17', Delay='18', TPBuffIn='19',
                      TPBuffOut='20', RPZ='21', GND='22', TPREF='23', FBK='24',
                      Cas='25', TPREFA='26', TPREFB='27')

    chip_range = range(num_chips)
    plot_name = ''

    def __init__(self, node=0):
        """Initialize EXCALIBUR detector object.

        For example: On I13 the top FEM of EXCALIBUR-3M-RX001 is connected to
        node 1 (i13-1-excalibur01) and the bottom fem to node 6(?)
        (i13-1-excalibur06).

        Args:
            node: PC node number of 1/2 module (Between 1 and 6 for 3M)

        """
        # Detector default Settings
        self.settings = {'mode': 'spm',  # 'spm' or 'csm'
                         'gain': 'shgm',  # 'slgm', 'lgm', 'hgm' or 'shgm'
                         'bitdepth': '12',  # '1', '8', '12' or '24'; 24 bits
                         # needs disccsmspm set at 1 to use discL
                         'readmode': '0',  # '0' or '1'
                         'counter': '0',  # '0' or '1'
                         'disccsmspm': '0',  # '0' or '1'
                         'equalization': '0',  # '0' or '1'
                         'trigmode': '0',
                         'acqtime': '100',  # in ms
                         'frames': '1',  # Number of frames to acquire
                         'imagepath': '/tmp/',  # Image path
                         'filename': 'image',  # Image filename
                         'Threshold': 'Not set',  # Threshold in keV
                         'filenameIndex': ''}  # Image file index (used to
        # avoid overwriting files)

        self.fem = node
        self.ipaddress = "192.168.0.10" + str(7 - self.fem)
        if self.fem == 0:
            self.ipaddress = "192.168.0.106"

        self.app = ExcaliburTestAppInterface(self.ipaddress, port='6969')
        self.dawn = ExcaliburDAWN()

        # self.read_chip_ids()

    def threshold_equalization(self, chips=range(8)):
        """Calibrate discriminator equalization.

        You need to edit this function to define which mode (SPM or CSM) and
        which gains you want to calibrate during the threshold_equalization
        sequence

        Args:
            chips: Chips to calibrate

        """
        self.settings['mode'] = 'spm'
        self.settings['gain'] = 'slgm'

        # Checks whether a threshold_equalization directory exists, if not it
        # creates one with default dacs
        self.check_calib_dir()
        self.log_chip_id()  # Log chip IDs in threshold_equalization folder
        self.set_dacs(chips)  # Set DACs recommended by Rafa in May 2015
        self.set_gnd_fbk_cas_excalibur_rx001(chips, self.fem)  # This will load
        # the DAC values specific to each chip to have FBK, CAS and GND reading
        # back the recommended analogue value

        # IMPORTANT NOTE: These values of GND, FBK and CAS Dacs were adjusted
        # for the modules present in RX001 on 20 June 2015. If modules are
        # replaced, these DACs need to be re-adjusted and the FBK_DAC, GND_DAC
        # and Cas_DAC arrays in GND_FBK_CAS_ExcaliburRX001 have to be edited

        self.calibrate_disc(chips, 'discL')  # Calibrates DiscL Discriminator
        # connected to Threshold 0 using a rectangular ROI in 1

        # NOTE: Always equalize DiscL before DiscH since Threshold1 is set at 0
        # when equalizing DiscL. So if DiscH was equalized first, this would
        # induce noisy counts interfering with DiscL equalization

        # self.calibrate_disc(chips, 'discH', 1, 'rect')
        # self.settings['mode'] = 'csm'
        # self.settings['gain'] = 'slgm'
        # self.calibrate_disc(chips, 'discL', 1, 'rect')
        # self.calibrate_disc(chips, 'discH', 1, 'rect')

        # EG (13/06/2016) creates mask for horizontal noise
        # badPixels = self.mask_row_block(range(4), 256-20, 256)

    def threshold_calibration_all_gains(self, threshold="0"):
        """Calibrate equalization for all gain modes and chips.

        This will save a threshold calibration file called threshold0 or
        threshold1 in the calibration directory under each gain setting
        sub-folder. Each Threshold calibration file consists of 2 rows of 8
        floating point numbers:
        # g0     g1   g2   g3   g4   g5   g6   g7
        # Off0 Off1 Off2 Off3 Off4 Off5 Off6 Off7
        # and the DAC value to apply to chip x for a requested threshold energy
        # value E in keV is given by:
        # DACx= gx * E + Offx

        Args:
            threshold: Threshold to calibrate (0 or 1)

        """
        self.settings['gain'] = 'shgm'
        self.threshold_calibration(threshold)
        self.settings['gain'] = 'hgm'
        self.threshold_calibration(threshold)
        self.settings['gain'] = 'lgm'
        self.threshold_calibration(threshold)
        self.settings['gain'] = 'slgm'
        self.threshold_calibration(threshold)

    def threshold_calibration(self, threshold="0"):
        """Calculate keV to DAC unit conversion for given threshold.

        This function produces threshold calibration data required to convert
        an X-ray energy detection threshold in keV into threshold DAC units
        It uses a first-approximation calibration data assuming that 6keV X-ray
        energy corresponds to dac code = 62 in SHGM. Dac scans showed that this
        was true +/- 2 DAC units in 98% of the chips tested when using
        threshold equalization function with default parameters
        (dacTarget = 10 and nbOfSigma = 3.2).

        Args:
            threshold: Threshold to calibrate (0 or 1)

        """
        self.check_calib_dir()
        default_6kev_dac = 62

        E0 = 0
        E1 = 5.9  # keV
        Dac0 = self.dac_target * np.ones([6, 8]).astype('float')

        if self.settings['gain'] == 'shgm':
            Dac1 = Dac0 + 1*(default_6kev_dac - Dac0) * \
                          np.ones([6, 8]).astype('float')
        if self.settings['gain'] == 'hgm':
            Dac1 = Dac0 + 0.75*(default_6kev_dac - Dac0) * \
                          np.ones([6, 8]).astype('float')
        if self.settings['gain'] == 'lgm':
            Dac1 = Dac0 + 0.5*(default_6kev_dac - Dac0) * \
                          np.ones([6, 8]).astype('float')
        if self.settings['gain'] == 'slgm':
            Dac1 = Dac0 + 0.25*(default_6kev_dac - Dac0) * \
                          np.ones([6, 8]).astype('float')

        print(str(E0))
        print(str(Dac0))
        print(str(E1))
        print(str(Dac1))

        slope = (Dac1[self.fem - 1, :] - Dac0[self.fem - 1, :])/(E1 - E0)
        offset = Dac0[self.fem - 1, :]
        self.save_kev2dac_calib(threshold, slope, offset)
        print(str(slope) + str(offset))

    def save_kev2dac_calib(self, threshold, gain, offset):
        """Save KeV conversion data to file.

        Each threshold calibration file consists of 2 rows of 8 floating point
        numbers:
        # g0     g1   g2   g3   g4   g5   g6   g7
        # Off0 Off1 Off2 Off3 Off4 Off5 Off6 Off7
        # and the DAC value to apply to chip x for a requested threshold energy
        # value E in keV is given by:
        # DACx = gx * E + Offx

        Args:
            threshold: Threshold calibration to save (0 or 1)
            gain: Conversion scale factor
            offset: Conversion offset

        """
        thresh_coeff = np.zeros([2, 8])
        thresh_coeff[0, :] = gain
        thresh_coeff[1, :] = offset

        thresh_filename = posixpath.join(self.calib_dir,
                                         'fem{fem}',
                                         self.settings['mode'],
                                         self.settings['gain'],
                                         'threshold{threshold}'
                                         ).format(fem=self.fem,
                                                  threshold=threshold)
        print(thresh_filename)

        if os.path.isfile(thresh_filename):
            np.savetxt(thresh_filename, thresh_coeff, fmt='%.2f')
        else:
            np.savetxt(thresh_filename, thresh_coeff, fmt='%.2f')
            os.chmod(thresh_filename, 0777)  # First time the file is created,
            # permissions need to be changed to allow anyone to overwrite
            # calibration data

    def find_xray_energy_dac(self, chips=range(8), threshold="0", energy=5.9):
        """############## NOT TESTED

        Perform a DAC scan and fits monochromatic spectra in order to find the
        DAC value corresponding to the X-ray energy

        Args:
            chips: Chips to scan
            threshold: Threshold to scan (0 or 1)
            energy: Xray energy for scan

        Returns:
            numpy.array, list: DAC scan array, DAC values of scan

        """
        self.settings['acqtime'] = 100
        if self.settings['gain'] == 'shgm':
            dac_range = (self.dac_target + 100, self.dac_target + 20, 2)
        else:
            raise NotImplementedError()

        self.load_config(chips)
        filename = 'Threshold{threshold}Scan_{energy}keV'.format(
            threshold=threshold,
            energy=energy)
        self.settings['filename'] = filename

        [dac_scan_data, scan_range] = self.scan_dac(chips, "Threshold" +
                                                    str(threshold), dac_range)
        dac_scan_data[dac_scan_data > 200] = 0  # Set elements > 200 to 0
        [chip_dac_scan, dac_axis] = self.dawn.plot_dac_scan(chips,
                                                            dac_scan_data,
                                                            scan_range)
        self.dawn.fit_dac_scan(chips, chip_dac_scan, dac_axis)

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
            stop(int):  Index of final row to be included in mask

        Returns:
            numpy.array: Array mask that was written to config

        """
        bad_pixels = np.zeros([self.chip_size,
                               self.chip_size * self.num_chips])

        chip_size = self.chip_size
        for chip_idx in chips:
            # Slicing doesn't include the end, so add 1 to stop; chip_size is
            # in number, not index, so already has the necessary increment
            bad_pixels[start:stop + 1,
                       chip_idx*chip_size:(chip_idx + 1)*chip_size] = 1

        for chip_idx in chips:
            pixel_mask_file = posixpath.join(self.calib_dir,
                                             'fem{fem}',
                                             self.settings['mode'],
                                             self.settings['gain'],
                                             'pixelmask.chip{idx}'
                                             ).format(fem=self.fem,
                                                      idx=chip_idx)
            np.savetxt(pixel_mask_file,
                       bad_pixels[0:chip_size,
                                  chip_idx*chip_size:(chip_idx + 1)*chip_size],
                       fmt='%.18g', delimiter=' ')

        self.dawn.plot_image(bad_pixels, "Mask")
        self.load_config(chips)

        return bad_pixels

    def one_energy_thresh_calib(self, threshold="0"):
        """Plot single energy threshold calibration spectra.

        This function produces threshold calibration files using DAC scans
        performed with a monochromatic X-ray source. Dac scans and spectra are
        performed and plotted in dac scan and spectrum plots. You need to
        inspect the spectra and edit manually the array oneE_DAC with the DAC
        value corresponding to the X-ray energy for each chip

        This method will use 2 points for calculating Energy to DAC conversion
        coefficients:
            # The energy used in the dac scan
            # The noise peak at dacTarget (10 DAC units by default)

        Possible improvement: Fit dacs scans to populate automatically
        oneE_DAC array (cf fit dac scan)

        Args:
            threshold: Threshold to calibrate (0 or 1)

        """
        self.settings['acqtime'] = 1000
        self.settings['filename'] = 'Single_energy_threshold_scan'

        # dacRange = (self.dacTarget + 80, self.dacTarget + 20, 2)
        # self.load_config(range(8))
        # [dacscanData, scanRange] = self.scan_dac(chips, "Threshold" +
        #                                         str(Threshold), dacRange)
        # edgeDacs = self.find_edge(chips, dacscanData, dacRange, 2)
        # chipEdgeDacs = np.zeros([len(chips)])
        # for chip in chips:
        #     chipEdgeDacs[chip] = \
        #         edgeDacs[0:256, chip*256:chip*256 + 256].mean()

        OneE_Dac = np.ones([6, 8]).astype('float')
        OneE_E = 6  # keV

        # Edit Energy Dac array

        if self.settings['gain'] == 'shgm':
            OneE_Dac[0, :] = [20, 62, 62, 62, 62, 62, 62, 62]
            OneE_Dac[1, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            OneE_Dac[2, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            OneE_Dac[3, :] = [60, 35, 64, 64, 64, 64, 64, 64]
            OneE_Dac[4, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            OneE_Dac[5, :] = [64, 64, 64, 64, 64, 64, 64, 64]

        if self.settings['gain'] == 'hgm':
            OneE_Dac[0, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            OneE_Dac[1, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            OneE_Dac[2, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            OneE_Dac[3, :] = [60, 35, 64, 64, 64, 64, 64, 64]
            OneE_Dac[4, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            OneE_Dac[5, :] = [64, 64, 64, 64, 64, 64, 64, 64]

        if self.settings['gain'] == 'lgm':
            OneE_Dac[0, :] = [20, 62, 62, 62, 62, 62, 62, 62]
            OneE_Dac[1, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            OneE_Dac[2, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            OneE_Dac[3, :] = [60, 35, 64, 64, 64, 64, 64, 64]
            OneE_Dac[4, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            OneE_Dac[5, :] = [64, 64, 64, 64, 64, 64, 64, 64]

        if self.settings['gain'] == 'slgm':
            OneE_Dac[0, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            OneE_Dac[1, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            OneE_Dac[2, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            OneE_Dac[3, :] = [60, 35, 64, 64, 64, 64, 64, 64]
            OneE_Dac[4, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            OneE_Dac[5, :] = [64, 64, 64, 64, 64, 64, 64, 64]

        slope = (OneE_Dac[self.fem - 1, :] - self.dac_target) / OneE_E
        offset = [self.dac_target]*8
        self.save_kev2dac_calib(threshold, slope, offset)

        print(str(slope) + str(offset))

        self.settings['filename'] = 'image'

    def multiple_energy_thresh_calib(self, chips=range(8), threshold="0"):
        """Plot multiple energy threshold calibration spectra.

        This functions produces threshold calibration files using DAC scans
        performed with several monochromatic X-ray spectra. Dac scans and
        spectra are performed and plotted in dacscan and spectrum plots. You
        need to inspect the spectra and edit manually the DAC arrays with the
        DAC value corresponding to the X-ray energy for each chip

        This method will use several points for calculating Energy to DAC
        conversion coefficients:

        Args:
            chips: Chips to calibrate
            threshold: Threshold to calibrate (0 or 1)

        """
        self.settings['acqtime'] = 1000
        self.settings['filename'] = 'Single_energy_threshold_scan'

        # dacRange = (self.dacTarget + 80, self.dacTarget + 20, 2)
        # self.load_config(range(8))
        # [dacscanData, scanRange] = self.scan_dac(chips, "Threshold" +
        #                                         str(Threshold), dacRange)
        # edgeDacs = self.find_edge(chips, dacscanData, dacRange, 2)
        # chipEdgeDacs = np.zeros([len(chips)])
        # for chip in chips:
        #    chipEdgeDacs[chip] = edgeDacs[0:256, chip*256:chip*256+256].mean()

        # Edit Energy Dac arrays

        E1_Dac = np.ones([6, 8]).astype('float')
        E1_E = 6  # keV

        if self.settings['gain'] == 'shgm':
            E1_Dac[0, :] = [20, 62, 62, 62, 62, 62, 62, 62]
            E1_Dac[1, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E1_Dac[2, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            E1_Dac[3, :] = [60, 35, 64, 64, 64, 64, 64, 64]
            E1_Dac[4, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E1_Dac[5, :] = [64, 64, 64, 64, 64, 64, 64, 64]

        if self.settings['gain'] == 'hgm':
            E1_Dac[0, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            E1_Dac[1, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E1_Dac[2, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            E1_Dac[3, :] = [60, 35, 64, 64, 64, 64, 64, 64]
            E1_Dac[4, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E1_Dac[5, :] = [64, 64, 64, 64, 64, 64, 64, 64]

        if self.settings['gain'] == 'lgm':
            E1_Dac[0, :] = [20, 62, 62, 62, 62, 62, 62, 62]
            E1_Dac[1, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E1_Dac[2, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            E1_Dac[3, :] = [60, 35, 64, 64, 64, 64, 64, 64]
            E1_Dac[4, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E1_Dac[5, :] = [64, 64, 64, 64, 64, 64, 64, 64]

        if self.settings['gain'] == 'slgm':
            E1_Dac[0, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            E1_Dac[1, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E1_Dac[2, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            E1_Dac[3, :] = [60, 35, 64, 64, 64, 64, 64, 64]
            E1_Dac[4, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E1_Dac[5, :] = [64, 64, 64, 64, 64, 64, 64, 64]

        E2_Dac = np.ones([6, 8]).astype('float')
        E2_E = 12  # keV

        if self.settings['gain'] == 'shgm':
            E2_Dac[0, :] = [120, 110, 100, 90, 80, 70, 60, 50]
            E2_Dac[1, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E2_Dac[2, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            E2_Dac[3, :] = [60, 35, 64, 64, 64, 64, 64, 64]
            E2_Dac[4, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E2_Dac[5, :] = [64, 64, 64, 64, 64, 64, 64, 64]

        if self.settings['gain'] == 'hgm':
            E2_Dac[0, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            E2_Dac[1, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E2_Dac[2, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            E2_Dac[3, :] = [60, 35, 64, 64, 64, 64, 64, 64]
            E2_Dac[4, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E2_Dac[5, :] = [64, 64, 64, 64, 64, 64, 64, 64]

        if self.settings['gain'] == 'lgm':
            E2_Dac[0, :] = [20, 62, 62, 62, 62, 62, 62, 62]
            E2_Dac[1, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E2_Dac[2, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            E2_Dac[3, :] = [60, 35, 64, 64, 64, 64, 64, 64]
            E2_Dac[4, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E2_Dac[5, :] = [64, 64, 64, 64, 64, 64, 64, 64]

        if self.settings['gain'] == 'slgm':
            E2_Dac[0, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            E2_Dac[1, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E2_Dac[2, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            E2_Dac[3, :] = [60, 35, 64, 64, 64, 64, 64, 64]
            E2_Dac[4, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E2_Dac[5, :] = [64, 64, 64, 64, 64, 64, 64, 64]

        E3_Dac = np.ones([6, 8]).astype('float')
        E3_E = 24  # keV

        if self.settings['gain'] == 'shgm':
            E3_Dac[0, :] = [250, 110, 100, 90, 80, 70, 60, 50]
            E3_Dac[1, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E3_Dac[2, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            E3_Dac[3, :] = [60, 35, 64, 64, 64, 64, 64, 64]
            E3_Dac[4, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E3_Dac[5, :] = [64, 64, 64, 64, 64, 64, 64, 64]

        if self.settings['gain'] == 'hgm':
            E3_Dac[0, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            E3_Dac[1, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E3_Dac[2, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            E3_Dac[3, :] = [60, 35, 64, 64, 64, 64, 64, 64]
            E3_Dac[4, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E3_Dac[5, :] = [64, 64, 64, 64, 64, 64, 64, 64]

        if self.settings['gain'] == 'lgm':
            E3_Dac[0, :] = [20, 62, 62, 62, 62, 62, 62, 62]
            E3_Dac[1, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E3_Dac[2, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            E3_Dac[3, :] = [60, 35, 64, 64, 64, 64, 64, 64]
            E3_Dac[4, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E3_Dac[5, :] = [64, 64, 64, 64, 64, 64, 64, 64]

        if self.settings['gain'] == 'slgm':
            E3_Dac[0, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            E3_Dac[1, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E3_Dac[2, :] = [62, 62, 62, 62, 62, 62, 62, 62]
            E3_Dac[3, :] = [60, 35, 64, 64, 64, 64, 64, 64]
            E3_Dac[4, :] = [64, 64, 64, 64, 64, 64, 64, 64]
            E3_Dac[5, :] = [64, 64, 64, 64, 64, 64, 64, 64]

        offset = np.zeros(8)
        gain = np.zeros(8)
        clear = True

        for chip in chips:
            x = np.array([E1_E, E2_E, E3_E])
            y = np.array([E1_Dac[self.fem - 1, chip],
                          E2_Dac[self.fem - 1, chip],
                          E3_Dac[self.fem - 1, chip]])

            p1, p2 = self.dawn.plot_linear_fit(x, y, [0, 1],
                                               name='DAC vs Energy',
                                               clear=clear)
            offset[chip] = p1
            gain[chip] = p2

            clear = False  # Only clear plot the first time

        self.save_kev2dac_calib(threshold, gain, offset)

        print(str(gain) + str(offset))
        print(self.settings['gain'])

    def set_threshold0(self, thresh_energy='5'):
        """Set threshold0 energy in keV.

        Threshold0 DACs are calculated using threshold0 calibration file
        located in the calibration directory corresponding to the current mode
        and gain setting

        Args:
            thresh_energy: Energy to set

        """
        self.settings['Threshold'] = thresh_energy
        self.set_thresh_energy('0', float(self.settings['Threshold']))

    def set_thresh_energy(self, threshold="0", thresh_energy=5.0):
        """Set given threshold in keV.

        ThresholdX DACs are calculated using thresholdX calibration file
        located in the calibration directory corresponding to the current mode
        and gain setting

        Args:
            threshold: Threshold to set (0 or 1)
            thresh_energy: Energy to set

        """
        fname = posixpath.join(self.calib_dir,
                               'fem{fem}',
                               self.settings['mode'],
                               self.settings['gain'],
                               'threshold{threshold}'
                               ).format(fem=self.fem,
                                        threshold=threshold)

        thresh_coeff = np.genfromtxt(fname)
        print(thresh_coeff[0, :].astype(np.int))

        thresh_DACs = (thresh_energy * thresh_coeff[0, :] +
                       thresh_coeff[1, :]).astype(np.int)
        for chip in range(8):
            self.set_dac([chip],
                         "Threshold" + str(threshold), thresh_DACs[chip])

        time.sleep(0.2)
        self.settings['acqtime'] = '100'
        self.expose()
        time.sleep(0.2)
        self.expose()

        print("A Threshold" + str(threshold) + " of " + str(thresh_energy) +
              "keV corresponds to " + str(thresh_DACs) +
              " DAC units for each chip")

    def set_dacs(self, chips):
        """Set dacs to values recommended by MEDIPIX3-RX designers.

        Args:
            chips: Chips to set DACs for

        """
        self.set_dac(chips, 'Threshold1', 0)
        self.set_dac(chips, 'Threshold2', 0)
        self.set_dac(chips, 'Threshold3', 0)
        self.set_dac(chips, 'Threshold4', 0)
        self.set_dac(chips, 'Threshold5', 0)
        self.set_dac(chips, 'Threshold6', 0)
        self.set_dac(chips, 'Threshold7', 0)
        self.set_dac(chips, 'Preamp', 175)  # Could use 200
        self.set_dac(chips, 'Ikrum', 10)  # Low Ikrum for better low energy
        # X-ray sensitivity
        self.set_dac(chips, 'Shaper', 150)
        self.set_dac(chips, 'Disc', 125)
        self.set_dac(chips, 'DiscLS', 100)
        self.set_dac(chips, 'ShaperTest', 0)
        self.set_dac(chips, 'DACDiscL', 90)
        self.set_dac(chips, 'DACTest', 0)
        self.set_dac(chips, 'DACDiscH', 90)
        self.set_dac(chips, 'Delay', 30)
        self.set_dac(chips, 'TPBuffIn', 128)
        self.set_dac(chips, 'TPBuffOut', 4)
        self.set_dac(chips, 'RPZ', 255)  # RPZ is disabled at 255
        self.set_dac(chips, 'TPREF', 128)
        self.set_dac(chips, 'TPREFA', 500)
        self.set_dac(chips, 'TPREFB', 500)

    def read_chip_ids(self):
        """Read chip IDs."""
        self.app.read_chip_ids()
        print(str(self.chip_range))

    def log_chip_id(self):
        """Reads chip IDs and logs chipIDs in calibration directory."""
        log_filename = posixpath.join(self.calib_dir,
                                      'fem{fem}',
                                      '/efuseIDs'
                                      ).format(fem=self.fem)

        with open(log_filename, "w") as outfile:
            self.app.read_chip_ids(stdout=outfile)

        print(str(self.chip_range))

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

    def fe55_image_rx001(self, chips=range(8), acquire_time=60000):
        """Save FE55 image.

        Args:
            chips: Chips to use
            acquire_time: Exposure time of image

        Will save to:
        /dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/ DIRECTORY

        """
        img_path = self.settings['imagepath']
        self.settings['gain'] = 'shgm'
        self.load_config(chips)
        self.set_threshold0_dac(chips, 40)
        self.settings['acqtime'] = str(acquire_time)
        self.settings['filename'] = 'Fe55_image_node_{fem}_{acqtime}s'.format(
            fem=self.fem, acqtime=self.settings['acqtime'])
        self.settings['imagepath'] = posixpath.join(self.root_path,
                                                    'Fe55_images')

        print(self.settings['acqtime'])

        self.settings['acqtime'] = str(acquire_time)
        time.sleep(0.5)
        self.expose()
        self.settings['imagepath'] = img_path
        self.settings['filename'] = 'image'
        self.settings['acqtime'] = '100'

    def set_dac(self, chips, dac_name="Threshold0", dac_value=40):
        """Sets any chip DAC at a given value.

        Args:
            chips: Chips to set DACs for
            dac_name: DAC to set
            dac_value: Value to set DAC to

        """
        new_line = dac_name + " = " + str(dac_value) + "\r\n"
        dac_file_path = posixpath.join(self.calib_dir,
                                       'fem{fem}',
                                       self.settings['mode'],
                                       self.settings['gain'],
                                       'dacs'
                                       ).format(fem=self.fem)

        for chip in chips:
            with open(dac_file_path, 'r') as dac_file:
                f_content = dac_file.readlines()

            line_nb = chip*29 + np.int(self.dac_number[dac_name])
            f_content[line_nb] = new_line
            with open(dac_file_path, 'w') as dac_file:
                dac_file.writelines(f_content)

            self.app.load_dacs([chip], dac_file_path)  # TODO: Do this once?

    def read_dac(self, chips, dac_name):
        """Read back DAC analogue voltage using chip sense DAC (DAC out).

        Args:
            chips: Chips to read
            dac_name: DAC value to read

        """
        dac_file = posixpath.join(self.calib_dir,
                                  'fem{fem}',
                                  self.settings['mode'],
                                  self.settings['gain'],
                                  'dacs'
                                  ).format(fem=self.fem)

        self.app.sense(chips, dac_name, dac_file)

    def scan_dac(self, chips, dac_name, dac_range):  # ONLY FOR THRESHOLD DACS
        """Perform a dac scan and plot the result (mean counts vs DAC values).

        Args:
            chips: Chips to scan
            dac_name: DAC to scan
            dac_range: Range of DAC values to scan over

        Returns:
            numpy.array: DAC scan data

        """
        self.update_filename_index()

        dac_scan_file = '{name}_dacscan{index}.hdf5'.format(
            name=self.settings['filename'],
            index=self.settings['filenameIndex'])
        dac_file = posixpath.join(self.calib_dir,
                                  'fem'.format(fem=self.fem),
                                  self.settings['mode'],
                                  self.settings['gain'],
                                  'dacs')
        scan_range = Range(dac_range[0], dac_range[1], dac_range[2])

        self.app.perform_dac_scan(chips, dac_name, scan_range, dac_file,
                                  dac_scan_file)

        time.sleep(1)

        save_file = self.settings['imagepath'] + dac_scan_file
        dac_scan_data = self.dawn.load_image_data(save_file)

        return dac_scan_data

    def load_config_bits(self, chips, discLbits, discHbits, mask_bits):
        """Load specific detector configuration files (discbits, maskbits).

        Args:
            chips: Chips to load config for
            discLbits: DiscL pixel config (256 x 256 : 0 to 31)
            discHbits: DiscH pixel config (256 x 256 : 0 to 31)
            mask_bits: Pixel mask (256 x 256 : 0 or 1)

        """
        # TODO: Refactor to load one file at a time?
        template_path = posixpath.join(self.calib_dir,
                                       'fem{fem}'.format(fem=self.fem),
                                       self.settings['mode'],
                                       self.settings['gain'],
                                       '{disc}.tmp')

        discH_bits_file = template_path.format(disc='discHbits')
        discL_bits_file = template_path.format(disc='discLbits')
        mask_bits_file = template_path.format(disc='maskbits')

        np.savetxt(discL_bits_file, discLbits, fmt='%.18g', delimiter=' ')
        np.savetxt(discH_bits_file, discHbits, fmt='%.18g', delimiter=' ')
        np.savetxt(mask_bits_file, mask_bits, fmt='%.18g', delimiter=' ')

        for chip in chips:
            self.app.load_config([chip], discL_bits_file, discH_bits_file,
                                 mask_bits_file)

    def load_config(self, chips=range(8)):
        """Load detector configuration files and default thresholds.

        From calibration directory corresponding to selected mode and gain.

        Args:
            chips: Chips to load config for

        """
        template_path = posixpath.join(self.calib_dir,
                                       'fem{fem}'.format(fem=self.fem),
                                       self.settings['mode'],
                                       self.settings['gain'],
                                       '{disc}.chip{chip}')

        for chip in chips:
            discHbits_file = template_path.format(disc='discHbits', chip=chip)
            discLbits_file = template_path.format(disc='discLbits', chip=chip)
            pixel_mask_file = template_path.format(disc='pixelmask', chip=chip)

            self.app.load_config([chip], discLbits_file, discHbits_file,
                                 pixel_mask_file)

        self.set_dac(range(8), "Threshold1", 100)
        self.set_dac(range(8), "Threshold0", 40)
        self.expose()

    def update_filename_index(self):
        """Increments filename index in filename.idx file in image path.

        If the file doesn't exist then create it starting at 0.

        """
        idx_filename = self.settings['imagepath'] + \
            self.settings['filename'] + '.idx'

        new_file = False
        if os.path.isfile(idx_filename):
            with open(idx_filename, 'r') as idx_file:
                new_idx = int(idx_file.read()) + 1
        else:
            new_idx = 0
            new_file = True

        with open(idx_filename, 'w') as idx_file:
            idx_file.write(str(new_idx))

        self.settings['filenameIndex'] = str(new_idx)

        if new_file:
            os.chmod(idx_filename, 0777)

    def burst(self, frames, acquire_time):
        """Acquire images in burst mode.

        Args:
            frames: Number of images to capture
            acquire_time: Exposure time for images

        """
        self.settings['frames'] = frames
        self.settings['acqtime'] = acquire_time

        self.update_filename_index()
        self.settings['fullFilename'] = '{name}_{index}.hdf5'.format(
            name=self.settings['filename'],
            index=self.settings['filenameIndex'])

        self.app.acquire(self.chip_range,
                         frames,
                         acquire_time,
                         burst=True,
                         hdffile=self.settings['fullFilename'])

        time.sleep(0.5)

    def expose(self):
        """Acquire single image using current detector settings.

        Returns:
            numpy.array: Image data

        """
        # TODO: This is the same as shoot if the user doesn't edit the settings
        # TODO: manually from the console

        print(self.settings)

        self.update_filename_index()
        self.settings['frames'] = '1'
        self.settings['fullFilename'] = '{name}_{index}.hdf5'.format(
            name=self.settings['filename'],
            index=self.settings['filenameIndex'])

        self.app.acquire(self.chip_range,
                         str(self.settings['frames']),
                         str(self.settings['acqtime']),
                         hdffile=self.settings['fullFilename'])

        print(self.settings['filename'])

        time.sleep(0.5)

        image = self.dawn.load_image_data(self.settings['imagepath'] +
                                          self.settings['fullFilename'])
        self.dawn.plot_image(image, name="Image_{}".format(time.asctime()))

        return image

    def shoot(self, acquire_time):
        """Acquire an image with a given exposure time.

        Args:
            acquire_time: Exposure time for images

        """
        self.settings['acqtime'] = acquire_time
        self.settings['frames'] = '1'
        self.update_filename_index()
        self.settings['fullFilename'] = '{name}_{index}.hdf5'.format(
            name=self.settings['filename'],
            index=self.settings['filenameIndex'])

        self.app.acquire(self.chip_range,
                         str(self.settings['frames']),
                         str(acquire_time),
                         hdffile=self.settings['fullFilename'])

        print(self.settings['filename'])

        time.sleep(0.2)

        image = self.dawn.load_image_data(self.settings['imagepath'] +
                                          self.settings['fullFilename'])
        self.dawn.plot_image(image, name="Image_{}".format(time.asctime()))

    def cont(self, frames, acquire_time):
        """Acquire image in continuous mode.

        Args:
            frames: Number of images to capture
            acquire_time: Exposure time for images

        """
        self.settings['acqtime'] = acquire_time
        self.settings['frames'] = frames
        self.settings['readmode'] = '1'

        self.update_filename_index()
        self.settings['fullFilename'] = '{name}_{index}.hdf5'.format(
            name=self.settings['filename'],
            index=self.settings['filenameIndex'])

        self.app.acquire(self.chip_range,
                         str(frames),
                         str(acquire_time),
                         readmode='1',
                         hdffile=self.settings['fullFilename'])

        print(self.settings['filename'])

        time.sleep(0.2)

        image = self.dawn.load_image_data(self.settings['imagepath'] +
                                          self.settings['fullFilename'])

        plots = min(frames, 5)  # Limit to 5 frames
        plot_tag = time.asctime()
        for plot in range(plots):
            self.dawn.plot_image(image[plot, :, :],
                                 name="Image_{tag}_{plot}".format(
                                 tag=plot_tag, plot=plot))

    def cont_burst(self, frames, acquire_time):
        """Acquire images in continuous burst mode.

        Args:
            frames: Number of images to capture
            acquire_time: Exposure time for images

        """
        # TODO: Readmode used but not changed to 1 before?

        self.settings['acqtime'] = acquire_time
        self.update_filename_index()
        self.settings['frames'] = frames
        self.settings['fullFilename'] = '{name}_{index}.hdf5'.format(
            name=self.settings['filename'],
            index=self.settings['filenameIndex'])

        self.app.acquire(self.chip_range,
                         frames,
                         acquire_time,
                         burst=True,
                         readmode='1',
                         hdffile=self.settings['fullFilename'])

    def acquire_ff(self, ni, acquire_time):
        """Acquire and sum flat-field images.

        NOT TESTED
        Produces flat-field correction coefficients

        Args:
            ni: Number of images to sum
            acquire_time: Exposure time for images

        Returns:
            numpy.array: Flat-field coefficients

        """
        self.settings['fullFilename'] = "FlatField.hdf5"
        self.settings['acqtime'] = acquire_time

        ff_image = 0
        for n in range(ni):
            image = self.expose()
            ff_image += image

        chip = 3  # TODO: Why?
        chip_size = self.chip_size
        # Set all zero elements to the mean value
        ff_image[ff_image == 0] = \
            ff_image[0:chip_size, chip*chip_size:(chip + 1) * chip_size].mean()
        # Coeff array is array of means divided by actual values
        ff_coeff = np.ones([256, 256*8]) * \
            ff_image[0:chip_size,
                     chip*chip_size:(chip + 1) * chip_size].mean()
        ff_coeff = ff_coeff/ff_image

        self.dawn.plot_image(ff_coeff[0:256, chip*256:(chip + 1)*256],
                             name='Flat Field coefficients')

        # Set any elements outside range 0-2 to 1 TODO: Why?
        ff_coeff[ff_coeff > 2] = 1
        ff_coeff[ff_coeff < 0] = 1

        return ff_coeff

    def apply_ff_correction(self, num_images, ff_coeff):
        """Apply flat-field correction.

        NOT TESTED

        Args:
            num_images: Number of images to plot (?)
            ff_coeff: Numpy array with calculated correction

        """
        # TODO: This just plots, but doesn't apply it

        self.settings['fullFilename'] = '{name}_{index}.hdf5'.format(
            name=self.settings['filename'],
            index=self.settings['filenameIndex'])

        images = self.dawn.load_image_data(self.settings['imagepath'] +
                                           self.settings['fullFilename'])

        for image_idx in range(num_images):
            # TODO: Find better name than ff
            ff = images[image_idx, :, :] * ff_coeff
            ff[ff > 3000] = 0
            chip = 3
            self.dawn.plot_image(ff[0:256, chip*256:(chip + 1)*256],
                                 name='Image data Cor')  # TODO: 'Cor'?
            time.sleep(1)

    def logo_test(self):
        """Test the detector using test pulses representing excalibur logo."""
        chips = range(self.num_chips)
        self.set_dac(chips, "Threshold0", 40)
        self.shoot(10)
        logo_tp = np.ones([256, 8*256])
        logo_file = posixpath.join(self.config_dir,
                                   'logo.txt')
        logo_small = np.loadtxt(logo_file)
        logo_tp[7:250, 225:1823] = logo_small
        logo_tp[logo_tp > 0] = 1
        logo_tp = 1 - logo_tp

        for chip in chips:
            test_bits_file = posixpath.join(self.calib_dir,
                                            'Logo_chip{chip}_mask').format(
                                            chip=chip)
            np.savetxt(test_bits_file, logo_tp[0:256, chip*256:(chip + 1)*256],
                       fmt='%.18g', delimiter=' ')

            template_path = posixpath.join(self.calib_dir,
                                           'fem{fem}'.format(fem=self.fem),
                                           self.settings['mode'],
                                           self.settings['gain'],
                                           '{disc}.chip{chip}')

            discHbits_file = template_path.format(disc='discHbits', chip=chip)
            discLbits_file = template_path.format(disc='discLbits', chip=chip)
            pixel_mask_file = template_path.format(disc='pixelmask', chip=chip)

            dac_file = posixpath.join(self.calib_dir, 'dacs')

            if os.path.isfile(discLbits_file) \
                and os.path.isfile(discHbits_file)  \
                    and os.path.isfile(pixel_mask_file):
                disc_files = dict(discl=discLbits_file,
                                  disch=discHbits_file,
                                  pixelmask=pixel_mask_file)

                self.app.configure_test_pulse_with_disc([chip], dac_file,
                                                        test_bits_file,
                                                        disc_files)
            else:
                self.app.configure_test_pulse([chip], dac_file, test_bits_file)

        time.sleep(0.2)

        self.settings['fullFilename'] = "{name}_{index}.hdf5".format(
                                        name=self.settings['filename'],
                                        index=self.settings['filenameIndex'])

        self.app.acquire(chips,
                         str(self.settings['frames']),
                         str(self.settings['acqtime']),
                         tpcount='100',
                         hdffile=self.settings['fullFilename'])

        image = self.dawn.load_image_data(self.settings['imagepath'] +
                                          self.settings['fullFilename'])
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

        # self.update_filename_index()
        self.settings['fullFilename'] = "{name}_{index}.hdf5".format(
                                        name=self.settings['filename'],
                                        index=self.settings['filenameIndex'])

        for chip in chips:
            dac_file = posixpath.join(self.calib_dir, 'dacs')
            self.app.configure_test_pulse([chip], dac_file, test_bits_file)

        self.app.acquire(chips,
                         str(self.settings['frames']),
                         str(self.settings['acqtime']),
                         tpcount=str(pulses),
                         hdffile=self.settings['fullFilename'])

        print(self.settings['fullFilename'])

        image = self.dawn.load_image_data(self.settings['imagepath'] +
                                          self.settings['fullFilename'])
        self.dawn.plot_image(image, name="Image_{}".format(time.asctime()))

    def save_discbits(self, chips, discbits, discbitsFilename):
        """Save discbit array into file in the calibration directory.

        Args:
            chips: Chips to save for
            discbits: Numpy array to save
            discbitsFilename: File name to save to

        """
        for chip_idx in chips:

            discbits_file = posixpath.join(self.calib_dir,
                                           'fem{fem}',
                                           self.settings['mode'],
                                           self.settings['gain'],
                                           '{disc}.chip{chip}'
                                           ).format(fem=self.fem,
                                                    disc=discbitsFilename,
                                                    chip=chip_idx)

            np.savetxt(discbits_file,
                       discbits[0:256, chip_idx*256:chip_idx*256 + 256],
                       fmt='%.18g', delimiter=' ')

    def mask_super_column(self, chip, super_column):
        """Mask a super column (32 bits?) in a chip and update the maskfile.

        Args:
            chip: Chip to mask
            super_column: Super column to mask (0 to 7)

        """
        bad_pixels = np.zeros([self.chip_size,
                               self.chip_size * self.num_chips])
        bad_pixels[:, chip*256 + super_column * 32:chip * 256 +
                   super_column * 32 + 64] = 1  # TODO: Should it be 64 wide?

        template_path = posixpath.join(self.calib_dir,
                                       'fem{fem}'.format(fem=self.fem),
                                       self.settings['mode'],
                                       self.settings['gain'],
                                       '{disc}.chip{chip}')

        discLbits_file = template_path.format(disc='discLbits', chip=chip)
        pixel_mask_file = template_path.format(disc='pixelmask', chip=chip)

        np.savetxt(pixel_mask_file, bad_pixels[0:256, chip*256:(chip + 1)*256],
                   fmt='%.18g', delimiter=' ')
        self.app.load_config([chip], discLbits_file, pixelmask=pixel_mask_file)

        self.dawn.plot_image(bad_pixels, name='Bad Pixels')

    def mask_col(self, chip, column):
        """Mask a column in a chip and update the maskfile

        Args:
            chip: Chip to mask
            column: Column to mask (0 to 255)

        """
        bad_pixels = np.zeros([self.chip_size,
                               self.chip_size * self.num_chips])
        bad_pixels[:, column] = 1

        template_path = posixpath.join(self.calib_dir,
                                       'fem{fem}'.format(fem=self.fem),
                                       self.settings['mode'],
                                       self.settings['gain'],
                                       '{disc}.chip{chip}')

        discLbits_file = template_path.format(disc='discLbits', chip=chip)
        pixel_mask_file = template_path.format(disc='pixelmask', chip=chip)

        np.savetxt(pixel_mask_file, bad_pixels[0:256, chip*256:chip*256 + 256],
                   fmt='%.18g', delimiter=' ')

        self.app.load_config([chip], discLbits_file, pixelmask=pixel_mask_file)

        self.dawn.plot_image(bad_pixels, name='Bad pixels')

    def mask_pixels(self, chips, image_data, max_counts):
        """Mask pixels in image_data with counts above max_counts

        Also updates the corresponding maskfile in the calibration directory

        Args:
            chips: Chips to mask
            image_data: Numpy array image
            max_counts: Mask threshold

        """
        bad_pix_tot = np.zeros(8)
        bad_pixels = image_data > max_counts
        self.dawn.plot_image(bad_pixels, name='Bad pixels')
        for chip_idx in chips:
            template_path = posixpath.join(self.calib_dir,
                                           'fem{fem}'.format(fem=self.fem),
                                           self.settings['mode'],
                                           self.settings['gain'],
                                           '{disc}.chip{chip}')

            discLbits_file = template_path.format(disc='discLbits',
                                                  chip=chip_idx)
            pixel_mask_file = template_path.format(disc='pixelmask',
                                                   chip=chip_idx)

            bad_pix_tot[chip_idx] = \
                bad_pixels[0:256, chip_idx*256:(chip_idx + 1)*256].sum()

            print('####### ' + str(bad_pix_tot[chip_idx]) +
                  ' noisy pixels in chip ' + str(chip_idx) +
                  ' (' + str(100*bad_pix_tot[chip_idx]/(256**2)) + '%)')

            np.savetxt(pixel_mask_file,
                       bad_pixels[0:256, chip_idx*256:chip_idx*256 + 256],
                       fmt='%.18g', delimiter=' ')
            self.app.load_config([chip_idx], discLbits_file,
                                 pixelmask=pixel_mask_file)

        print('####### ' + str(bad_pix_tot.sum()) +
              ' noisy pixels in half module ' +
              ' (' + str(100 * bad_pix_tot.sum()/(8 * 256**2)) + '%)')

    def mask_pixels_using_dac_scan(self, chips=range(8),
                                   threshold="Threshold0",
                                   dac_range=(20, 120, 2)):
        """Perform threshold dac scan and mask pixels with counts > max_counts.

        Also updates the corresponding maskfile in the calibration directory

        Args:
            chips: Chips to scan
            threshold: Threshold to scan (0 or 1)
            dac_range: Range to scan over

        """
        max_counts = 1
        bad_pix_tot = np.zeros(8)
        self.settings['acqtime'] = 100
        [dac_scan_data, _] = self.scan_dac(chips, threshold, dac_range)
        bad_pixels = dac_scan_data.sum(0) > max_counts
        self.dawn.plot_image(bad_pixels, name='Bad Pixels')

        for chip_idx in chips:
            bad_pix_tot[chip_idx] = \
                bad_pixels[0:256, chip_idx*256:chip_idx*256 + 256].sum()

            print('####### ' + str(bad_pix_tot[chip_idx]) +
                  ' noisy pixels in chip_idx ' + str(chip_idx) +
                  ' (' + str(100 * bad_pix_tot[chip_idx]/(256**2)) + '%)')

            pixel_mask_file = posixpath.join(self.calib_dir,
                                             'fem{fem}'.format(fem=self.fem),
                                             self.settings['mode'],
                                             self.settings['gain'],
                                             '{disc}.chip{chip}').format(
                disc='pixelmask',
                chip=chip_idx)

            np.savetxt(pixel_mask_file,
                       bad_pixels[0:256, chip_idx*256:(chip_idx + 1)*256],
                       fmt='%.18g', delimiter=' ')
            # subprocess.call([self.command, "-i", self.ipaddress,
            #                  "-p", self.port,
            #                  "-m", self.mask(range(chip_idx, chip_idx + 1)),
            #                  "--config",
            #                  "--pixelmask=" + pixel_mask_file,
            #                  "--config",
            #                  "--discl=" + discLbitsFile])

        print('####### ' + str(bad_pix_tot.sum()) +
              ' noisy pixels in half module ' +
              ' (' + str(100 * bad_pix_tot.sum() / (8 * 256**2)) + '%)')

    def unmask_all_pixels(self, chips):
        """Unmask all pixels and update maskfile in calibration directory.

        Args:
            chips: Chips to unmask

        """

        bad_pixels = np.zeros([self.chip_size,
                               self.chip_size * self.num_chips])

        for chip_idx in chips:
            template_path = posixpath.join(self.calib_dir,
                                           'fem{fem}'.format(fem=self.fem),
                                           self.settings['mode'],
                                           self.settings['gain'],
                                           '{disc}.chip{chip}')

            discLbits_file = template_path.format(disc='discLbits',
                                                  chip=chip_idx)
            pixel_mask_file = template_path.format(disc='pixelmask',
                                                   chip=chip_idx)

            np.savetxt(pixel_mask_file,
                       bad_pixels[0:256, chip_idx*256:chip_idx*256 + 256],
                       fmt='%.18g', delimiter=' ')
            self.app.load_config([chip_idx], discLbits_file,
                                 pixelmask=pixel_mask_file)

    def unequalize_all_pixels(self, chips):
        """Reset discL_bits and pixel_mask bits to 0.

        Args:
            chips: Chips to unequalize

        """
        # TODO: Unequalize discH as well?
        # TODO: File path slightly different to above; discL_bits vs discLbits
        # TODO: Are they supposed to be the same?

        discL_bits = 31*np.zeros([self.chip_size,
                                  self.chip_size * self.num_chips])
        for chip_idx in chips:
            template_path = posixpath.join(self.calib_dir,
                                           'fem{fem}'.format(fem=self.fem),
                                           self.settings['mode'],
                                           self.settings['gain'],
                                           '{disc}.chip{chip}')

            discLbits_file = template_path.format(disc='discL_bits',
                                                  chip=chip_idx)
            pixel_mask_file = template_path.format(disc='pixelmask',
                                                   chip=chip_idx)

            np.savetxt(discLbits_file,
                       discL_bits[0:256, chip_idx*256:(chip_idx + 1)*256],
                       fmt='%.18g', delimiter=' ')
            self.app.load_config([chip_idx], discLbits_file,
                                 pixelmask=pixel_mask_file)

    def check_calib_dir(self):
        """Checks if calibration directory exists and backs it up."""
        calib_dir = posixpath.join(self.calib_dir,
                                   'fem{fem}',
                                   self.settings['mode'],
                                   self.settings['gain']).format(
            fem=self.fem)

        if (os.path.isdir(calib_dir)) == 0:
            os.makedirs(calib_dir)
        else:
            backup_dir = self.calib_dir + '_backup_' + \
                         time.asctime()
            shutil.copytree(self.calib_dir, backup_dir)

            print(backup_dir)

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
            chips: Chips to load
            discbits_filename: File to grab data from

        Returns:
            numpy.array: Discbits array

        """
        discbits = np.zeros([self.chip_size, self.chip_size * self.num_chips])
        for chip_idx in chips:
            discbits_file = posixpath.join(self.calib_dir,
                                           'fem{fem}',
                                           self.settings['mode'],
                                           self.settings['gain'],
                                           '{disc}.chip{chip}').format(
                fem=self.fem, disc=discbits_filename, chip=chip_idx)

            discbits[0:256, chip_idx*256:(chip_idx + 1)*256] = \
                np.loadtxt(discbits_file)

        return discbits

    def combine_rois(self, chips, disc_name, steps, roi_type):
        """Combine multiple ROIs into one mask.

        Used to combine intermediate discbits_roi files produced when
        equalizing various ROIs into one discbits file

        Args:
            chips: Chips to make ROIs for
            disc_name: Discriminator config file to edit ('DiscL' or 'DiscH')
            steps: Number of of ROIs to combine (steps during equalization)
            roi_type: Type of ROI ('rect' or 'spacing')

        """
        discbits = np.zeros([self.chip_size, self.chip_size * self.num_chips])
        for step in range(steps):
            roi_full_mask = self.roi(chips, step, steps, roi_type)
            discbits_roi = self.open_discbits_file(chips, disc_name +
                                                   'bits_roi_' + str(step))
            discbits[roi_full_mask.astype(bool)] = \
                discbits_roi[roi_full_mask.astype(bool)]
            # TODO: Should this be +=? Currently just uses final ROI
            self.dawn.plot_image(discbits_roi, name='Disc bits')

        self.save_discbits(chips, discbits, disc_name + 'bits')
        self.dawn.plot_image(discbits, name='Disc bits total')

        return discbits

    def find_edge(self, chips, dac_scan_data, dac_range, edge_val):
        """Find noise or X-ray edge in threshold DAC scan.

        Args:
            chips: Chips search
            dac_scan_data: Data from DAC scan
            dac_range: Range DAC scan performed over
            edge_val: Threshold for edge

        Returns:
            numpy.array(?): Noise edge data

        """

        if dac_range[1] > dac_range[0]:
            edge_dacs = dac_range[1] - dac_range[2] * np.argmax(
                (dac_scan_data[::-1, :, :] > edge_val), 0)
        else:
            edge_dacs = dac_range[0] - dac_range[2] * np.argmax(
                (dac_scan_data[:, :, :] > edge_val), 0)

        self.dawn.plot_image(edge_dacs, name="noise edges")
        self.dawn.plot_histogram(chips, edge_dacs)

        return edge_dacs

    def find_max(self, chips, dac_scan_data, dac_range):
        """Find noise max in threshold dac scan.

        Returns:
            numpy.array(?): Max noise data

        """
        edge_dacs = dac_range[1] - dac_range[2] * np.argmax(
            (dac_scan_data[::-1, :, :]), 0)
        # TODO: Assumes low to high scan? Does it matter?

        self.dawn.plot_image(edge_dacs, name="noise edges")
        self.dawn.plot_histogram(chips, edge_dacs)

        return edge_dacs

    def optimize_dac_disc(self, chips, disc_name, roi_full_mask):
        """Calculate optimum DAC disc values for given chips.

        Args:
            chips: Chips to optimize
            disc_name: Discriminator to optimize ('DiscL' or 'DiscH')
            roi_full_mask: Mask to exclude pixels from optimization calculation

        Returns:
            numpy.array: Optimum DAC discriminator value for each chip

        """
        # Definition of parameters to be used for threshold scans
        self.settings['acqtime'] = '5'
        self.settings['counter'] = '0'
        self.settings['equalization'] = '1'  # Might not be necessary when
        # optimizing DAC Disc
        dac_range = (0, 150, 5)

        # Setting variables depending on whether discL or discH is equalized
        # (Note: equalize discL before discH)
        if disc_name == 'discL':
            threshold = 'Threshold0'
            self.set_dac(chips, 'Threshold1', 0)
            self.settings['disccsmspm'] = '0'
            dac_disc_name = 'DACDiscL'
        if disc_name == 'discH':
            threshold = 'Threshold1'
            self.set_dac(chips, 'Threshold0', 60)  # To be above the noise
            # since DiscL is equalized before DiscH
            self.settings['disccsmspm'] = '1'
            dac_disc_name = 'DACDiscH'

        ######################################################################
        # STEP 1
        # Perform threshold DAC scans for all discbits set at 0 and various
        # DACdisc values, discbits set at 0 shift DAC scans to the right
        # Plot noise edge position as a function of DACdisc
        # Calculate noise edge shift in threshold DAC units per DACdisc
        # DAC unit
        ######################################################################

        discbit = 0
        dac_disc_range = range(0, 150, 50)  # range(50,150,50)
        bins = (dac_range[1] - dac_range[0]) / dac_range[2]

        # Initialization of fit parameters and plots
        opt_dac_disc = np.zeros(self.num_chips)
        sigma = np.zeros([8, len(dac_disc_range)])
        x0 = np.zeros([8, len(dac_disc_range)])
        a = np.zeros([8, len(dac_disc_range)])
        p0 = [5000, 50, 30]
        offset = np.zeros(8)
        gain = np.zeros(8)
        name = "Histogram of edges when scanning DAC_disc"
        plot_name = name + " for discbit =" + str(discbit)
        fit_plot_name = plot_name + " (fitted)"
        calib_plot_name = "Mean edge shift in Threshold DACs as a function" \
                          "of DAC_disc for discbit =" + str(discbit)
        self.dawn.clear_plot(plot_name)
        self.dawn.clear_plot(fit_plot_name)
        # dnp.plot.clear(calib_plot_name)

        # Set discbits at 0
        discbits = discbit*np.ones([self.chip_size,
                                    self.chip_size * self.num_chips])
        for chip in chips:
            if disc_name == 'discH':
                discLbits = self.open_discbits_file(chips, 'discLbits')
                self.load_config_bits(range(chip, chip + 1),
                                      discLbits[:, chip*256:chip*256 + 256],
                                      discbits[:, chip*256:chip*256 + 256],
                                      roi_full_mask[:,
                                                    chip*256:chip*256 + 256])
            if disc_name == 'discL':
                discHbits = np.zeros([self.chip_size,
                                      self.chip_size * self.num_chips])
                # Set discL and DiscH bits at 0 and unmask the whole matrix
                self.load_config_bits(range(chip, chip + 1),
                                      discbits[:, chip*256:chip*256 + 256],
                                      discHbits[:, chip*256:chip*256 + 256],
                                      roi_full_mask[:,
                                                    chip*256:chip*256 + 256])

        # Threshold DAC scans, fitting and plotting
        idx = 0
        for DAC_disc in dac_disc_range:
            self.set_dac(chips, dac_disc_name, DAC_disc)
            # Scan threshold
            [dac_scan_data, scan_range] = self.scan_dac(chips, threshold,
                                                        dac_range)
            # Find noise edges
            edge_dacs = self.find_max(chips, dac_scan_data, dac_range)
            for chip in chips:
                edge_histo = np.histogram(edge_dacs[0:256,
                                          chip*256:chip*256 + 256],
                                          bins=bins)
                dnp.plot.addline(edge_histo[1][0:-1], edge_histo[0],
                                 name=plot_name)
                popt, pcov = curve_fit(gauss_function, edge_histo[1][0:-2],
                                       edge_histo[0][0:-1], p0)

                x = edge_histo[1][0:-1]
                a[chip, idx] = popt[0]
                sigma[chip, idx] = popt[2]
                x0[chip, idx] = popt[1]
                dnp.plot.addline(x, gauss_function(x, a[chip, idx],
                                                   x0[chip, idx],
                                                   sigma[chip, idx]),
                                 name=fit_plot_name)
            idx += 1
            self.dawn.clear_plot(calib_plot_name)
            for chip in chips:
                dnp.plot.addline(np.asarray(dac_disc_range[0:idx]),
                                 x0[chip, 0:idx],
                                 name=calib_plot_name)

        # Plots mean noise edge as a function of DAC Disc for all discbits
        # set at 0
        for chip in chips:
            popt, pcov = curve_fit(lin_function, np.asarray(dac_disc_range),
                                   x0[chip, :], [0, -1])
            offset[chip] = popt[0]
            gain[chip] = popt[1]  # Noise edge shift in DAC units per DACdisc
            # DAC unit
            dnp.plot.addline(np.asarray(dac_disc_range),
                             lin_function(np.asarray(dac_disc_range),
                                          offset[chip], gain[chip]),
                             name=calib_plot_name)

        print("Edge shift (in Threshold DAC units) produced by 1 DACdisc DAC"
              "unit for discbits=15:")

        for chip in chips:
            print("Chip" + str(chip) + ' : ' + str(round(gain[chip], 2)) +
                  ' Threshold DAC units')

        # Fit range should be adjusted to remove outliers at 0 and max DAC 150

        ######################################################################
        # STEP 2
        # Perform threshold DAC scan for all discbits set at 15 (no correction)
        # Fit threshold scan and calculate width of noise edge distribution
        ######################################################################

        discbit = 15
        DAC_disc = 80  # Value does not matter since no correction is applied
        # when discbits = 15
        bins = (dac_range[1] - dac_range[0]) / dac_range[2]

        # Initialization of fit parameters and plots
        sigma = np.zeros([8])
        x0 = np.zeros([8])
        a = np.zeros([8])
        p0 = [5000, 0, 30]
        plot_name = name + " for discbit =" + str(discbit)
        fit_plot_name = plot_name + " (fitted)"
        self.dawn.clear_plot(plot_name)
        self.dawn.clear_plot(fit_plot_name)
        # dnp.plot.clear(calib_plot_name)

        # Set discbits at 15
        discbits = discbit*np.ones([self.chip_size,
                                    self.chip_size * self.num_chips])
        for chip in chips:
            if disc_name == 'discH':
                # discLbits = np.zeros([self.chipSize,
                #                       self.chipSize*self.nbOfChips])
                discLbits = self.open_discbits_file(chips, 'discLbits')
                self.load_config_bits(range(chip, chip + 1),
                                      discLbits[:, chip*256:chip*256 + 256],
                                      discbits[:, chip*256:chip*256 + 256],
                                      roi_full_mask[:,
                                                    chip*256:chip*256 + 256])
            if disc_name == 'discL':
                discHbits = np.zeros([self.chip_size,
                                      self.chip_size * self.num_chips])
                self.load_config_bits(range(chip, chip + 1),
                                      discbits[:, chip*256:chip*256 + 256],
                                      discHbits[:, chip*256:chip*256 + 256],
                                      roi_full_mask[:,
                                                    chip*256:chip*256 + 256])

        self.set_dac(chips, dac_disc_name, DAC_disc)
        [dac_scan_data, scan_range] = self.scan_dac(chips, threshold,
                                                    dac_range)
        edge_dacs = self.find_max(chips, dac_scan_data, dac_range)
        for chip in chips:
            edge_histo = np.histogram(edge_dacs[0:256,
                                                chip*256:chip*256 + 256],
                                      bins=bins)
            dnp.plot.addline(edge_histo[1][0:-1], edge_histo[0],
                             name=plot_name)
            popt, pcov = curve_fit(gauss_function, edge_histo[1][0:-2],
                                   edge_histo[0][0:-1], p0)
            x = edge_histo[1][0:-1]
            a[chip] = popt[0]
            sigma[chip] = popt[2]
            x0[chip] = popt[1]
            dnp.plot.addline(x,
                             gauss_function(x, a[chip], x0[chip], sigma[chip]),
                             name=fit_plot_name)

        print("Mean noise edge for discbits =15 :")
        for chip in chips:
            print("Chip" + str(chip) + ' : ' + str(round(x0[chip])) +
                  ' DAC units')

        print("sigma of Noise edge distribution for discbits =15 :")
        for chip in chips:
            print("Chip" + str(chip) + ' : ' + str(round(sigma[chip])) +
                  ' DAC units rms')

        ######################################################################
        # STEP 3
        # Calculate DAC disc required to bring all noise edges within X sigma
        # of the mean noise edge
        # X is defined by self.nbOfsigma
        ######################################################################

        print("Optimum equalization target :")
        for chip in chips:
            print("Chip" + str(chip) + ' : ' + str(round(x0[chip])) +
                  ' DAC units')

        if abs(x0 - self.dac_target).any() > self.allowed_delta:  # To be checked
            print("########################### ONE OR MORE CHIPS NEED A"
                  "DIFFERENT EQUALIZATION TARGET")
        else:
            print("Default equalization target of " + str(self.dac_target) +
                  " DAC units can be used.")

        print("DAC shift required to bring all pixels with an edge"
              "within +/- " + "sigma of the target, at the target of " +
              str(self.dac_target) + " DAC units : ")
        for chip in chips:
            print("Chip" + str(chip) + ' : ' +
                  str(int(self.nb_of_sigma * sigma[chip])) +
                  ' Threshold DAC units')

        print("Edge shift (in Threshold DAC units) produced by 1 DACdisc DAC"
              "unit for discbits=0 (maximum shift) :")
        for chip in chips:
            print("Chip" + str(chip) + ' : ' + str(round(gain[chip], 2)) +
                  ' Threshold DAC units')

        for chip in chips:
            opt_dac_disc[chip] = int(self.nb_of_sigma * sigma[chip] /
                                     gain[chip])

        print("###############################################################"
              "########################")
        print("Optimum DACdisc value required to bring all pixels with an edge"
              "within +/- " + str(self.nb_of_sigma) + " sigma of the target, "
              "at the target of " + str(self.dac_target) + " DAC units : ")

        for chip in chips:
            print("Chip" + str(chip) + ' : ' + str(opt_dac_disc[chip]) +
                  ' Threshold DAC units')

        print("###############################################################"
              "########################")

        print("Edge shift (in Threshold DAC Units) produced by 1 step of the"
              "32 discbit correction steps :")
        for chip in chips:
            print("Chip" + str(chip) + ' : ' + str(opt_dac_disc[chip]/16) +
                  ' Threshold DAC units')
        for chip in chips:
            self.set_dac(range(chip, chip + 1), dac_disc_name,
                         int(opt_dac_disc[chip]))

        return opt_dac_disc

    def equalise_discbits(self, chips, disc_name, roi_full_mask,
                          method='stripes'):
        """Equalize pixel discriminator.

        Uses stripes method as default (trimbits distributed across the matrix
        during each dacscan to avoid saturation the chips when pixels are in
        the noise at the same time)

        Args:
            chips: Chips to equalize
            disc_name: Discriminator to equalize ('DiscL' or 'DiscH')
            roi_full_mask: Mask to exclude pixels from equalization calculation
            method: Method to use ('dacscan', 'bitscan' or 'stripes')

        Returns:
            numpy.array: Equalised discriminator bits

        """
        self.settings['acqtime'] = '5'
        self.settings['counter'] = '0'
        self.settings['equalization'] = '1'
        if disc_name == 'discL':
            threshold = 'Threshold0'
            self.set_dac(chips, 'Threshold1', 0)
            self.settings['disccsmspm'] = '0'
        if disc_name == 'discH':
            threshold = 'Threshold1'
            self.set_dac(chips, 'Threshold0', 60)  # Well above noise since
            # discL bits are loaded
            self.settings['disccsmspm'] = '1'

        dnp.plot.image(roi_full_mask, name='roi')

        if method == 'stripes':
            dac_range = (0, 20, 2)
            discbits_tmp = np.zeros([self.chip_size,
                                     self.chip_size * self.num_chips]) * \
                           (1 - roi_full_mask)
            for idx in range(self.chip_size):
                discbits_tmp[idx, :] = idx % 32
            for idx in range(self.chip_size*self.num_chips):
                discbits_tmp[:, idx] = (idx % 32 + discbits_tmp[:, idx]) % 32
            edge_dacs_stack = np.zeros([32, self.chip_size,
                                        self.chip_size * self.num_chips])
            discbits_stack = np.zeros([32, self.chip_size,
                                       self.chip_size * self.num_chips])
            discbits_tmp *= (1 - roi_full_mask)
            discbits = -10*np.ones([self.chip_size,
                                    self.chip_size * self.num_chips]) * \
                       (1 - roi_full_mask)
            for scan in range(0, 32, 1):
                discbits_tmp = ((discbits_tmp + 1) % 32)*(1 - roi_full_mask)
                discbits_stack[scan, :, :] = discbits_tmp
                for chip in chips:
                    if disc_name == 'discH':
                        discLbits = self.open_discbits_file(chips, 'discLbits')
                        self.load_config_bits(range(chip, chip + 1),
                                              discLbits[:,
                                                  chip*256:chip*256 + 256],
                                              discbits_tmp[:,
                                                  chip*256:chip*256 + 256],
                                              roi_full_mask[:,
                                                  chip*256:chip*256 + 256])
                    if disc_name == 'discL':
                        discHbits = np.zeros([self.chip_size,
                                              self.chip_size * self.num_chips])
                        self.load_config_bits(range(chip, chip + 1),
                                              discbits_tmp[:,
                                                  chip*256:chip*256 + 256],
                                              discHbits[:,
                                                  chip*256:chip*256 + 256],
                                              roi_full_mask[:,
                                                  chip*256:chip*256 + 256])

                [dacscan_data, scan_range] = self.scan_dac(chips, threshold,
                                                           dac_range)
                edge_dacs = self.find_max(chips, dacscan_data, dac_range)
                edge_dacs_stack[scan, :, :] = edge_dacs
                scan_nb = np.argmin((abs(edge_dacs_stack - self.dac_target)),
                                    0)
                for chip in chips:
                    for x in range(256):
                        for y in range(chip*256, chip*256 + 256):
                            discbits[x, y] = \
                                discbits_stack[scan_nb[x, y], x, y]

                dnp.plot.image(discbits, name='discbits')
                dnp.plot.clear('Histogram of Final Discbits')

                for chip in chips:
                    roi_chip_mask = 1 - roi_full_mask[0:256,
                                                      chip*256:chip*256 + 256]
                    discbits_chip = discbits[0:256, chip*256:chip*256 + 256]
                    dnp.plot.addline(
                        np.histogram(discbits_chip[roi_chip_mask.astype(bool)],
                                     bins=range(32))[1][0:-1],
                        np.histogram(discbits_chip[roi_chip_mask.astype(bool)],
                                     bins=range(32))[0],
                        name='Histogram of Final Discbits')

        self.settings['disccsmspm'] = '0'
        self.settings['equalization'] = '0'

        print("Pixel threshold equalization complete")

        self.load_config(chips)
        self.scan_dac(chips, threshold, (40, 10, 2))

        return discbits

    def check_calib(self, chips, dac_range):
        """Check if dac scan looks OK after threshold calibration.

        NOT TESTED

        Args:
            chips: Chips to check
            dac_range: Range to scan DAC over (?)

        """
        pass  # TODO: Function doesn't work, what is `roi`?

        self.load_config(chips)
        equ_pix_tot = np.zeros(self.num_chips)
        self.settings['filename'] = 'dacscan'
        [dacscan_data, scan_range] = self.scan_dac(chips, "Threshold0",
                                                   dac_range)
        self.plot_name = self.settings['filename']
        edge_dacs = self.find_max(chips, dacscan_data, dac_range)

        # Display statistics on equalization and save discLbit files for each
        # chip
        for chip in chips:
            equ_pix_tot[chip] = ((edge_dacs[roi, chip*256:chip*256 + 256] >
                                  self.dac_target - self.allowed_delta) &
                                 (edge_dacs[roi, chip*256:chip*256 + 256] <
                                  self.dac_target + self.allowed_delta)).sum()
            print('####### ' + str(round(equ_pix_tot[chip], 0)) +
                  ' equalized pixels in chip ' + str(chip) +
                  ' (' + str(round(100*equ_pix_tot[chip]/(256**2), 4)) + '%)')

        # pixelsInTarget = (dacTarget - 5 < edge_dacs) & \
        #                      (edge_dacs < dacTarget + 5)

    def roi(self, chips, step, steps, roi_type):
        """Create a detector ROI to be used when equalizing thresholds.

        Using several ROIs during equalization was needed to avoid putting too
        many pixels in the noise at the same time. However the latest
        equalization scripts used for EXCALIBUR use the same technique as
        MERLIN to distribute equalization bits in diagonal across the matrix
        during equalization. Therefore the roi used by the latest scripts is
        always: roi = x.roi(range(8), 1, 1, 'rect')

        Args:
            chips: Chips to create ROI for
            step: Current step in the equalization process
            steps: Total number of steps
            roi_type: ROI type to create
                "rect": contiguous rectangles
                "spacing": arrays of equally-spaced pixels distributed across
                the chip

        """
        if roi_type == 'rect':
            roi_full_mask = np.zeros([self.chip_size,
                                      self.num_chips * self.chip_size])
            for chip in chips:
                roi_full_mask[step*256/steps:step*256/steps + 256/steps,
                              chip*256:chip*256 + 256] = 1
                # TODO: This doesn't set anything to 1
                # TODO: x range is always 256-512...
        if roi_type == 'spacing':
            spacing = steps**0.5
            roi_full_mask = np.zeros([self.chip_size,
                                      self.num_chips * self.chip_size])
            bin_repr = np.binary_repr(step, 2)
            for chip in chips:
                roi_full_mask[0 + int(bin_repr[0]):256 -
                                  int(bin_repr[0]):spacing,
                              chip*256 +
                                  int(bin_repr[1]):chip*256 +
                                  256 - int(bin_repr[1]):spacing] \
                    = 1

        self.dawn.plot_image(roi_full_mask, name="Roi Mask")

        return roi_full_mask

    def calibrate_disc(self, chips, disc_name, steps=1, roi_type='rect'):
        """Calibrate given discriminator for given chips.

        Args:
            chips: Chips to calibrate
            disc_name: Discriminator to equalize ('DiscL' or 'DiscH')
            steps: Total number of steps for equalization
            roi_type: ROI type to use (see roi)

        """
        self.optimize_dac_disc(chips, disc_name,
                               roi_full_mask=1 - self.roi(chips, 0, 1, 'rect'))

        # Run threshold_equalization over each roi
        for step in range(steps):
            roi_full_mask = self.roi(chips, step, steps, roi_type)
            discbits = self.equalise_discbits(chips, disc_name,
                                              1 - roi_full_mask, 'stripes')
            self.save_discbits(chips, discbits,
                               disc_name + 'bits_roi_' + str(step))
        discbits = self.combine_rois(chips, disc_name, steps, roi_type)

        self.save_discbits(chips, discbits, disc_name + 'bits')
        self.load_config(chips)  # Load threshold_equalization files created
        self.copy_slgm_into_other_gain_modes()  # Copy slgm threshold
        # equalization folder to other gain threshold_equalization folders

    def loop(self, ni):
        """Acquire images and plot the sum.

        Args:
            ni: Number of images to acquire

        """
        tmp = 0
        for i in range(ni):
            tmp = self.expose() + tmp
            self.dawn.plot_image(tmp, name='Sum')

            return  # TODO: This will always stop after first loop

    def csm(self, chips=range(8), gain='slgm'):
        """Set charge summing mode and associated default settings.

        Args:
            chips: Chips to load
            gain: Gain mode to set ('slgm', 'lgm', 'hgm', 'shgm')

        """
        # TODO: Why does it have to call expose to set these??
        self.settings['mode'] = 'csm'
        self.settings['gain'] = gain
        self.settings['counter'] = '1'  # '1'
        self.set_dac(range(8), 'Threshold0', 200)  # Make sure that chips not
        # used have also TH0 and Th1 well above the noise
        self.set_dac(range(8), 'Threshold1', 200)
        self.load_config(chips)
        self.set_dac(range(8), 'Threshold0', 45)
        self.set_dac(chips, 'Threshold1', 100)
        self.expose()
        self.expose()

    def set_gnd_fbk_cas_excalibur_rx001(self, chips, fem):
        """Set GND, FBK and CAS values.

        IMPORTANT NOTE: These values of GND, FBK and CAS Dacs were adjusted for
        the modules present in RX001 on 10 July 2015. If modules are replaced
        by new modules, these DACs need to be re-adjusted. If modules are
        swapped the DACs have to be swapped in the corresponding array. For
        example GND_DAC is an array of size 6 (fems) x 8 (chips)
        GND_DAC[x,:] will contain the GND DAC value of the 8 chips connected to
        fem/node x+1 where x=0 corresponds to the top half of the top module

        [NUMBERING STARTS AT 0 IN PYTHON SCRIPTS WHERAS NODES NUMBERING STARTS
        AT 1 IN EPICS]

        GND DAC needs to be adjusted manually to read back 0.65V
        FBK DAC needs to be adjusted manually to read back 0.9V
        CAS DAC needs to be adjusted manually to read back 0.85V
        (Values recommended by Rafa in May 2015)

        The procedure to adjust a DAC manually is as follows:
        For GND dac of chip 0:
        x.set_dac([0],'GND',150)
        x.read_dac([0],'GND')
        Once the DAC value correspond to the required analogue value, edit the
        GND_Dacs matrix:
        GND_Dacs[0,:]=[NEW DAC VALUE,x,x,x,x,x,x,x]

        For FBK dac of chip 0:
        x.set_dac([0],'FBK',190)
        x.read_dac([0],'FBK')
        Once the DAC value correspond to the required analogue value, edit the
        FBK_Dacs matrix:
        FBK_Dacs[0,:]=[NEW DAC VALUE,x,x,x,x,x,x,x]

        For Cas dac of chip 0:
        x.set_dac([0],'Cas',180)
        x.read_dac([0],'Cas')
        Once the DAC value correspond to the required analogue value, edit the
        CAS_Dacs matrix:
        CAS_Dacs[0,:]=[NEW DAC VALUE,x,x,x,x,x,x,x]

        This process could be automated if many modules have to be calibrated.

        Args:
            chips: Chips to set
            fem: FEM to set

        """

        GND_Dacs = 145*np.ones([6, 8]).astype('int')
        FBK_Dacs = 190*np.ones([6, 8]).astype('int')
        CAS_Dacs = 180*np.ones([6, 8]).astype('int')

        # TOP MODULE: AC-EXC-8

        gnd = 145
        fbk = 190
        cas = 180

        # #@ Moly temp: 35 degC on node 3

        GND_Dacs[0, :] = [141, 144, 154, 143, 161, 158, 144, 136]
        FBK_Dacs[0, :] = [190, 195, 201, 198, 220, 218, 198, 192]
        CAS_Dacs[0, :] = [178, 195, 196, 182, 213, 201, 199, 186]

        # GND_Dacs[1,:]=[145,141,142,142,141,141,143,150]
        # FBK_Dacs[1,:]=[205,190,197,200,190,190,200,210]
        # CAS_Dacs[1,:]=[187,187,183,187,177,181,189,194]
        GND_Dacs[1, :] = [154, 155, 147, 147, 147, 155, 158, 151]
        # Max current for fbk limited to 0.589 for chip 7
        FBK_Dacs[1, :] = [215, 202, 208, 200, 198, 211, 255, 209]
        CAS_Dacs[1, :] = [208, 197, 198, 194, 192, 207, 199, 188]

        # NOTE : chip 2 FBK cannot be set to target value

        # CENTRAL MODULE: AC-EXC-7

        # @ Moly temp: 27 degC on node 1
        GND_Dacs[2, :] = [143, 156, 139, 150, 144, 150, 149, 158]
        FBK_Dacs[2, :] = [189, 213, 185, 193, 204, 207, 198, 220]
        CAS_Dacs[2, :] = [181, 201, 177, 184, 194, 193, 193, 210]

        # @ Moly temp: 28 degC on node 2
        GND_Dacs[3, :] = [143, 156, 139, 150, 144, 150, 149, 158]
        FBK_Dacs[3, :] = [189, 213, 185, 193, 204, 207, 198, 220]
        CAS_Dacs[3, :] = [181, 201, 177, 184, 194, 193, 193, 210]

        # BOTTOM MODULE: AC-EXC-4

        # #@ Moly temp: 31 degC on node 5
        # GND_Dacs[4,:]=[136,146,136,160,142,135,140,148]
        # FBK_Dacs[4,:]=[190,201,189,207,189,189,191,208]
        # CAS_Dacs[4,:]=[180,188,180,197,175,172,185,200]

        # @ Moly temp: 31 degC on node 6

        # NOTE: DAC out read-back does not work for chip 2 (counting from 0) of
        # bottom 1/2 module
        # Using read_dac function on chip 2 will give FEM errors and the system
        # will need to be power-cycled
        # Got error on load pixel config command for chip 7: 2 Pixel
        # configuration loading failed
        # Exception caught during femCmd: Timeout on pixel configuration write
        # to chip7 acqState=3
        # Connecting to FEM at IP address 192.168.0.101 port 6969 ...
        # **************************************************************
        # Connecting to FEM at address 192.168.0.101
        # Configuring 10GigE data interface: host IP: 10.0.2.1 port: 61649 FEM
        # data IP: 10.0.2.2 port: 8 MAC: 62:00:00:00:00:01
        # Acquisition state at startup is 3 sending stop to reset
        # **** Loading pixel configuration ****
        # Last idx: 65536
        # Last idx: 65536
        # Last id

        gnd = 145
        fbk = 190
        cas = 180
        # GND_Dacs[5,:]=[158,140,gnd,145,158,145,138,153]
        # FBK_Dacs[5,:]=[215,190,fbk,205,221,196,196,210]
        # CAS_Dacs[5,:]=[205,178,cas,190,205,180,189,202]

        # 1st MODULE 1M:

        # up -fem5
        GND_Dacs[4, :] = [151, 135, 150, 162, 153, 144, 134, 145]
        FBK_Dacs[4, :] = [200, 195, 205, 218, 202, 194, 185, 197]
        CAS_Dacs[4, :] = [196, 180, 202, 214, 197, 193, 186, 187]
        # Bottom -fem6
        GND_Dacs[5, :] = [134, 145, 171, 146, 152, 142, 141, 141]
        FBK_Dacs[5, :] = [192, 203, 228, 197, 206, 191, 206, 189]
        CAS_Dacs[5, :] = [178, 191, 218, 184, 192, 186, 195, 185]

        for chip in chips:
            self.set_dac(range(chip, chip + 1), 'GND', GND_Dacs[fem - 1, chip])
            self.set_dac(range(chip, chip + 1), 'FBK', FBK_Dacs[fem - 1, chip])
            self.set_dac(range(chip, chip + 1), 'Cas', CAS_Dacs[fem - 1, chip])

        # self.read_dac(range(8), 'GND')
        # self.read_dac(range(8), 'FBK')
        # self.read_dac(range(8), 'Cas')

    @staticmethod
    def rotate_config(config_file):
        """Rotate array in given file 180 degrees

        Args:
            config_file: Config file to rotate

        """

        print(config_file)

        # shutil.copy(config_file, config_file + ".backup")
        config_bits = np.loadtxt(config_file)
        np.savetxt(config_file, np.rot90(config_bits, 2), fmt='%.18g',
                   delimiter=' ')

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

        print(EPICS_calib_path)

        template_path = posixpath.join(EPICS_calib_path,
                                       'fem{fem}',
                                       'spm',
                                       'slgm',
                                       '{disc}.chip{chip}'
                                       )

        for fem in [1, 3, 5]:

            print("Config files of node" + str(fem) +
                  " have to be rotated in EPICS calibration folder " +
                  str(self.fem))

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
                    self.rotate_config(discLbits_file)
                    print(discLbits_file + "rotated")
                if os.path.isfile(discHbits_file):
                    self.rotate_config(discHbits_file)
                    print(discHbits_file + "rotated")
                if os.path.isfile(pixel_mask_file):
                    self.rotate_config(pixel_mask_file)
                    print(pixel_mask_file + "rotated")
