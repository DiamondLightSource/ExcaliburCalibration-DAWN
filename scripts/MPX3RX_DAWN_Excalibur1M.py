"""
Python library for MPX3RX-based detectors calibration and test
/dls/detectors/support/silicon_pixels/excaliburRX/PyScripts/MPX3RX-DAWN.py
DIAMOND LIGHT SOURCE 30-07-2015

Initially developed for I13 EXCALIBUR-3M-RX001 detector
EXCALIBUR-specific functions need to be extracted and copied into a separate
library. This will allow for the scripts to be usable with any MPX3-based
system provided that a library of control functions is available for each
type of detector:
# EXCALIBUR
# LANCELOT/MERLIN
"""

import os
import shutil
import time
import math
import subprocess

import numpy as np
from scipy.optimize import curve_fit
import scisoftpy as dnp


def myerf(x, a, mu, sigma):
    """
    Function required to express S-curve

    Args:
        x:
        a:
        mu:
        sigma:

    Returns:

    """

    return a/2. * (1 + math.erf((x - mu) / (math.sqrt(2) * sigma)))


def lin_function(x, offset, gain):
    """
    Function definition for linear fits

    Args:
        x:
        offset:
        gain:

    Returns:

    """

    return offset + gain * x


def gauss_function(x, a, x0, sigma):
    """
    Function definition for Gaussian fits

    Args:
        x:
        a:
        x0:
        sigma:

    Returns:

    """

    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def s_curve_function(x, k, delta, e, sigma):
    """
    Function required to fit integral spectra during threshold calibration

    Args:
        x:
        k:
        delta:
        e:
        sigma:

    Returns:

    """

    return k * ((1 - 2*delta*(x/e - 0.5)) ** 2) * (1 - myerf(x, k, e, sigma))


class ExcaliburRX(object):
    """
    excaliburRX is a class defining methods required to calibrate each 1/2
    module (8 MPX3-RX chips) of an EXCALIBUR-RX detector.
    These calibration scripts will work only inside the Python interpreter of
    DAWN software running on the PC sever node connected to the FEM controlling
    the half-module which you wish to calibrate
    
    ===================== EXCALIBUR Test-Application =========================
    
    NOTE: excaliburRX detector class communicates with FEMs via excalibur
    Test-Application which is an executable file saved in:
    /dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp
    
    NOTE: excaliburRX detector class requires configuration files copied in:
    /dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/config/
    
    ========= To install libraries required by EXCALIBUR Test-Application
    
    excalibur Test-Application requires libboost and libhdf5 libraries to be
    installed locally. Use the following instructions to install the libraries:

    cd /dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburRxlib
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
    
    ================= EXCALIBUR Test-Application commands
    
    [ktf91651@p99-excalibur01 ~]$ /dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp -i 192.168.0.106 -p 6969 -m 0xff --help

    Usage: /dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp options
    
      -h --help                   Display this usage information.
      -i --ipaddress              IP address of FEM to connect to.
      -p --port                   Port of FEM to connect to.
      -m --mask                   Select MPX3 enable mask.
      -r --reset                  Issue front-end reset/init.
         --lvenable <mode>        Set power card LV enable: 0=off (default), 1=on.
         --hvenable <mode>        Set power card HV enable: 0=off (default), 1=on.
         --hvbias <volts>         Set power card HV bias in volts.
      -e --efuse                  Read and display MPX3 eFuse IDs.
      -d --dacs <filename>        Load MPX3 DAC values from filename if given, otherwise use default values
      -c --config                 Load MPX3 pixel configuration.
      -s --slow                   Display front-end slow control parameters.
      -a --acquire                Execute image acquisition loop.
         --burst                  Select burst mode for image acquisition.
         --matrixread             During acquisition, perform matrix read only (i.e. no shutter for config read or digital test).
      -n --frames <frames>        Number of frames to acquire.
      -t --acqtime <time>         Acquisition time (shutter duration) in milliseconds.
         --dacscan <params>       Execute DAC scan, params format must be comma separated dac,start,stop,step.
         --readmode <mode>        Readout mode: 0=sequential (default), 1=continuous.
         --trigmode <mode>        Trigger mode: 0=internal (default), 1=external shutter, 2=external sync.
         --colourmode <mode>      Select MPX3 colour mode: 0=fine pitch mode (default), 1=spectroscopic mode.
         --csmspm <mode>          Select MPX3 pixel mode: 0=single pixel mode (default), 1=charge summing mode.
         --disccsmspm <mode>      Select MPX3 discriminator output mode: 0=DiscL (default), 1=DiscH.
         --equalization <mode>    Select MPX3 equalization mode: 0=off (default), 1=on.
         --gainmode <mode>        Select MPX3 gain mode: 0=SHGM, 1=HGM, 2=LGM, 3=SLGM (default).
         --counter <counter>      Select MPX3 counter to read: 0 (default) or 1.
         --depth <depth>          Select MPX3 counter depth: 1, 6, 12 (default) or 24.
         --sensedac <id>          Set MPX3 sense DAC field to <id>. NB Requires DAC load to take effect
         --tpmask <filename>      Specify test pulse mask filename to load.
         --tpcount <count>        Set test pulse count to <count>, default is 0.
         --pixelmask <filename>   Specify pixel enable mask filename to load.
         --discl <filename>       Specify pixel DiscL configuration filename to load.
         --disch <filename>       Specify pixel DiscH configuration filename to load.
         --path <path>            Specify path to write data files to, default is /tmp.
         --hdffile <filename>     Write HDF file with optional filename, default is <path>/excalibur-YYMMDD-HHMMSS.hdf5
    
    
    ======================== MODULE CALIBRATION USING PYTHON SCRIPTS
    
    ================= FRONT-END POWER-ON
    
    To calibrate a 1/2 Module 
    
    ssh to the PC server node(standard DLS machine) connected to the MASTER FEM
    card (the one interfaced with the I2C bus of the Power card)
    On I13 EXCALIBUR-3M-RX001, this is the top FEM (192.168.0.106) connected to
    node 1
    ###########################
    >ssh i13-1-excalibur01
    ###########################
    Enable LV and make sure that HV is set to 120V during calibration:
    ##########################################################################
    >/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp -i 192.168.0.106 -p 6969 -m 0xff --lvenable 1
    >/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp -i 192.168.0.106 -p 6969 -m 0xff --lvenable 0
    >/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp -i 192.168.0.106 -p 6969 -m 0xff --lvenable 1
    >/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp -i 192.168.0.106 -p 6969 -m 0xff --hvbias 120
    >/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp -i 192.168.0.106 -p 6969 -m 0xff --hvenable 1
    ###########################################################################
    
    ssh to the PC server node (standard DLS machine) connected to the FEM card
    controlling the 1/2 module which you want to calibrate.
    ########################
    >ssh i13-1-excalibur0x with x in [1-6] and x=1 corresponds to the top FEM (Master FEM IP:192.168.0.106) connected to PC server node 1
    ########################
    
    ===================  DAWN START-UP
    
    On the PC server node, start DAWN by typing in a shell:
    ######################
    > module load dawn 
    > dawn &
    #######################
    
    ================== PYTHON SCRIPT
    
    Select the DExplore perspective 
    Open the file /dls/detectors/support/silicon_pixels/excaliburRX/PyScripts/MPX3RX-DAWN.py 
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
    > x=excaliburRX(node)
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
    [dacscanData,scanRange]=self.scan_dac([0],"Threshold0",(80,20,1))
    ###################################
    This will produce 2 plots:
    dacscan with intergal spectrum
    spectrum with differential spectrum
    
    Inspect the spectrum and evaluate the position of the energy peak in DAC
    units.
    Example: 
    E=6keV for energy peak DAC=60
    Since calibration is performed on noise peak, 0keV correspond to the
    selected DACtarget (10 by default)
    E=0keV for DAC=10
    Perform a linear fit of the DAC (in DAC units) as a function of energy
    (keV) and  edit threshold0 file in the calibration directory accordingly:
    
        Each Threshold calibration file consists of 2 rows of 8 floating point
        numbers:
        # g0     g1   g2   g3   g4   g5   g6   g7
        # Off0 Off1 Off2 Off3 Off4 Off5 Off6 Off7 
        # and the DAC value to apply to chip x for a requested threshold energy
        value E in keV is given by:
        # DACx= gx * E + Offx 
    
    Method 2:Using 1 energy and the noise peak dac
    
    
    
    Method 3: Using several energies. Script not written yet. 
    
    
    ============== ACQUIRE X_RAY TEST IMAGE WITH FE55 
    
    threshold_equalization data is then automatically loaded. And you can
    acquire a 60s image from Fe55 X-rays using the following command:
    ####################
    >x.fe55_image_rx001()
    ####################
    
    To change the exposure time used during image acquisition:
    ##########################################
    x.fe55_image_rx001(range(8),exp_time_in_ms)
    ##########################################
        
    ============== ACQUIRE IMAGES
    
    ##########
    x.expose()
    ##########
    
    To change acquisition time before exposure:
    #############################################
    x.settings['acqtime']=1000 (for 1s exposure)
    x.expose()
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
    x.set_dac(range(8),"Threshold0",40) to set threshold 0 at 40 DAC units
    ######################################################################
    This will allow you to put your threshold just above the noise to check the
     response of the 1/2 module to X-rays
    
    ================== LOAD DETECTOR CONFIGURATION
    To load discbits, maskbits and default thresholds from current calibration
    directory
    ###############
     x.load_config()
    ###############
    
    ================= TEST DETECTOR USING EXCALIBUR TEST PULSE LOGO
    ############
    x.testLogo()
    ############
    
    
    
    ==========================================================================
    NOTE: chip 0 in this program correspond to the left chip of the bottom
    half of a module or to the right chip of the top half of the module when
    facing the front-surface of sensor
    """

    # Test-application executable file path
    command = '/dls/detectors/support/silicon_pixels/excaliburRX/' \
              'TestApplication_15012015/excaliburTestApp'
    # Port of FEM to connect to
    port = '6969'

    # Threshold equalization will align pixel noise peaks at this DAC value
    dac_target = 10
    # Sigma value based on experimental data
    nb_of_sigma = 3.2

    # Count value used in function find_edge() searching for signal or noise
    # edges in dac scans
    edge_val = 10
    # Arbitrary value in DAC units used to check if the distance between the
    # optimum equalization target and the default equalization target
    # (dacTarget) is acceptable
    acc_dist = 4

    # Number of pixels per chip along each axis
    chip_size = 256
    # Number of chips in 1/2 module
    num_chips = 8

    # Settings used during threshold equalization and calibration
    calib_settings = {'calibDir': '/dls/detectors/support/silicon_pixels/'
                                 'excaliburRX/3M-RX001/calib/',
                     'configDir': '/dls/detectors/support/silicon_pixels/'
                                  'excaliburRX/TestApplication_15012015/'
                                  'config/',
                     'dacfilename': 'dacs',
                     'dacfile': '',
                     'noiseEdge': '10'}

    # Detector default Settings 
    settings = {'mode': 'spm',  # 'spm' or 'csm'
                'gain': 'shgm',  # 'slgm', 'lgm', 'hgm' or 'shgm'
                'bitdepth': '12',  # '1', '8', '12' or '24'; 24 bits needs
                # disccsmspm set at 1 to use discL
                'readmode': '0',  # '0' or '1' (cf Test-application commands)
                'counter': '0',  # '0' or '1' (cf Test-application commands)
                'disccsmspm': '0',  # '0' or '1' (cf Test-application commands)
                'equalization': '0',  # '0' or '1' (cf Test-application
                # commands)
                'trigmode': '0',  # (cf Test-application commands)
                'acqtime': '100',  # in ms (cf Test-application commands)
                'frames': '1',  # Number of frames to acquires cf
                # Test-application commands (cf Test-application commands)
                'imagepath': '/tmp/',  # Image path
                'filename': 'image',  # Image filename
                'Threshold': 'Not set',  # Threshold in keV
                'filenameIndex': ''}  # Image file index (used to avoid
    # overwriting files)
    
    # Conversion of human-readable settings into parameter to be passed to the
    # Test Application
    mode_code = {'spm': '0', 'csm': '1'}
    gain_code = {'shgm': '0', 'hgm': '1', 'lgm': '2', 'slgm': '3'}
    dac_code = {'Threshold0': '1',
                'Threshold1': '2',
                'Threshold2': '3',
                'Threshold3': '4',
                'Threshold4': '5',
                'Threshold5': '6',
                'Threshold6': '7',
                'Threshold7': '8',
                'Preamp': '9',  # To check
                'Ikrum': '10',
                'Shaper': '11',
                'Disc': '12',
                'DiscLS': '13',
                'ShaperTest': '14',
                'DACDiscL': '15',
                'DACTest': '30',
                'DACDiscH': '31',
                'Delay': '16',
                'TPBuffIn': '17',
                'TPBuffOut': '18',
                'RPZ': '19',
                'GND': '20',
                'TPREF': '21',
                'FBK': '22',
                'Cas': '23',
                'TPREFA': '24',
                'TPREFB': '25'}

    # Line number used when editing dac file with new dac values
    dac_number = {'Threshold0': '1',
                  'Threshold1': '2',
                  'Threshold2': '3',
                  'Threshold3': '4',
                  'Threshold4': '5',
                  'Threshold5': '6',
                  'Threshold6': '7',
                  'Threshold7': '8',
                  'Preamp': '9',
                  'Ikrum': '10',
                  'Shaper': '11',
                  'Disc': '12',
                  'DiscLS': '13',
                  'ShaperTest': '14',
                  'DACDiscL': '15',
                  'DACTest': '16',
                  'DACDiscH': '17',
                  'Delay': '18',
                  'TPBuffIn': '19',
                  'TPBuffOut': '20',
                  'RPZ': '21',
                  'GND': '22',
                  'TPREF': '23',
                  'FBK': '24',
                  'Cas': '25',
                  'TPREFA': '26',
                  'TPREFB': '27'}

    chip_range = range(0, num_chips)
    plot_name = ''
    
    def __init__(self, node=0):
        """
        Initialises EXCALIBUR detector object
        Usage: x=excaliburRX(node) where node is the EXCALIBUR PC node number
        (between 1 and 6 for EXCALIBUR-3M) of the 1/2 module under test
        For example: on I13 the top FEM of EXCALIBUR-3M-RX001 is connected to
        node 1 (i13-1-excalibur01) and the bottom fem to node
        (i13-1-excalibur06).
        When running Python calibration scripts on node i13-1-excalibur0X
        (with X in [1:6]), you should use: x = excaliburRX(X)
        """

        self.fem = node
        self.ipaddress = "192.168.0.10" + str(7 - self.fem)
        if self.fem == 0:
            self.ipaddress = "192.168.0.106"

        # self.read_chip_id()

    def threshold_equalization(self, chips=range(8)):
        """
        Set the

        You need to edit this function to define which mode (SPM or CSM) and
        which gains you want to calibrate during the threshold_equalization
        sequence

        Args:
            chips(list(int)): Chips to calibrate
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
        # for the modules present in RX001 on 20 june 2015. If modules are
        # replaced, these DACs need to be re-adjusted and the FBK_DAC, GND_DAC
        # and Cas_DAC arrays in GND_FBK_CAS_ExcaliburRX001 have to be edited

        self.calibrate_disc(chips, 'discL')  # Calibrates DiscL Discriminator
        # connected to Threshold 0 using a rectangular ROI in 1

        # NOTE: Always equalize DiscL before DiscH since Threshold1 is set at 0
        # when equalizing DiscL. So if DiscH was equalized first, this would
        # induce noisy counts interfering with DiscL equalization

        # self.calibrate_disc(chips,'discH',1,'rect')
        # self.settings['mode']='csm'
        # self.settings['gain']='slgm'
        # self.calibrate_disc(chips,'discL',1,'rect')
        # self.calibrate_disc(chips,'discH',1,'rect')

        # EG (13/06/2016) creates mask for horizontal noise
        # badPixels = self.mask_row_block(range(4), 256-20, 256)

    def threshold_calibration_all_gains(self, threshold="0"):
        """
        Usage: x.threshold_calibration_all_gains()
        This will save a threshold calibration file called threshold0 or
        threshold1 in the calibration directory under each gain setting
        subfolder. Each Threshold calibration file consists of 2 rows of 8
        floating point numbers:
        # g0     g1   g2   g3   g4   g5   g6   g7
        # Off0 Off1 Off2 Off3 Off4 Off5 Off6 Off7 
        # and the DAC value to apply to chip x for a requested threshold energy
        # value E in keV is given by:
        # DACx= gx * E + Offx
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
        """
        This function produces threshold calibration data required to convert
        an X-ray energy detection threshold in keV into threshold DAC units
        It uses a first-approximation calibration data assuming that 6keV X-ray
        energy corresponds to dac code = 62 in SHGM.
        Dac scans showed that this was true +/- 2 DAC units in 98% of the chips
        tested when using threshold equalization function with default
        parameters (dacTarget=10 and nbOfSigma=3.2).
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
        """
        Each Threshold calibration file consists of 2 rows of 8 floating point
        numbers:
        # g0     g1   g2   g3   g4   g5   g6   g7
        # Off0 Off1 Off2 Off3 Off4 Off5 Off6 Off7 
        # and the DAC value to apply to chip x for a requested threshold energy
        # value E in keV is given by:
        # DACx= gx * E + Offx 
        """

        thresh_coeff = np.zeros([2, 8])
        thresh_coeff[0, :] = gain

        thresh_coeff[1, :] = offset
        thresh_filename = self.calib_settings['calibDir'] + 'fem' + \
            str(self.fem) + '/' + self.settings['mode'] + '/' + \
            self.settings['gain'] + '/' + 'threshold' + str(threshold)

        print(thresh_filename)

        if os.path.isfile(thresh_filename):
            np.savetxt(thresh_filename, thresh_coeff, fmt='%.2f')
        else:
            np.savetxt(thresh_filename, thresh_coeff, fmt='%.2f')
            os.chmod(thresh_filename, 0777)  # First time the file is created,
            # permissions need to be changed to allow anyone to overwrite
            # calibration data

        return gain, offset

    def find_xray_energy_dac(self, chips=range(8), threshold="0", energy=5.9):
        """
        ############## NOT TESTED
        Performs a DAC scan and fits monochromatic spectra in order to find the
        DAC value corresponding to the X-ray energy
        Usage: x.find_xray_energy_dac(chips, Threshold, energy) where:
        chips is a list of chips (e.g. [0,1,2,3])
        Threshold is "0" or "1" for Threshold0 or Threshold1
        energy is the energy of the monochromatic X-rays used in the spectrum
        """

        self.settings['acqtime'] = 100
        if self.settings['gain'] == 'shgm':
            dac_range = (self.dac_target + 100, self.dac_target + 20, 2)
        
        self.load_config(chips)
        self.settings['filename'] = 'Threshold' + str(threshold) + 'Scan_' + \
                                    str(energy) + 'keV'
        [dac_scan_data, scan_range] = self.scan_dac(chips, "Threshold" +
                                                    str(threshold), dac_range)
        dac_scan_data[dac_scan_data > 200] = 0
        [chip_dac_scan, dac_axis] = self.plot_dac_scan(chips, dac_scan_data,
                                                       scan_range)
        self.fit_dac_scan(chips, chip_dac_scan, dac_axis)
        
        # edge_dacs = self.find_edge(chips, dac_scan_data, dac_range, 2)
        # chip_edge_dacs = np.zeros(range(8))
        # for chip in chips:
        #     chip_edge_dacs[chip] = edge_dacs[0:256,
        #                            chip*256:chip*256 + 256].mean()

        return chip_dac_scan, dac_axis

    @staticmethod
    def fit_dac_scan(chips, chip_dac_scan, dac_axis):
        """
        ############## NOT TESTED
        """

        p0 = [100, 0.8, 3]
        for chip in chips:
            # dnp.plot.addline(dacAxis,chipDacScan[chip,:])
            popt, pcov = curve_fit(myerf, dac_axis, chip_dac_scan[chip, :], p0)
            # popt, pcov = curve_fit(s_curve_function, dacAxis,
            #                        chipDacScan[chip, :], p0)
            dnp.plot.addline(dac_axis, myerf(dac_axis,
                                             popt[0], popt[1], popt[2]))

        return chip_dac_scan, dac_axis

    def mask_row_block(self, chips, start, stop):
        """
        Mask a block of rows of pixels on the given chips

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
            pixel_mask_file = self.calib_settings['calibDir'] + 'fem' + \
                str(self.fem) + '/' + self.settings['mode'] + \
                '/' + self.settings['gain'] + '/' + \
                'pixelmask.chip' + str(chip_idx)
            np.savetxt(pixel_mask_file,
                       bad_pixels[0:chip_size,
                                  chip_idx*chip_size:(chip_idx + 1)*chip_size],
                       fmt='%.18g', delimiter=' ')

        dnp.plot.image(bad_pixels)
        self.load_config(chips)

        return bad_pixels

    def one_energy_thresh_calib(self, threshold="0"):
        """
        This functions produces threshold calibration files using DAC scans
        performed with a monochromatic X-ray source. Dac scans and spectrum are
        performed and plotted in dacscan and spectrum plots. You need to
        inspect the spectra and edit manually the array oneE_DAC with the DAC
        value corresponding to teh X-ray energy for each chip
        
        This method will use 2 points for calculating Energy to DAC conversion
        coefficients:
            # The energy used in the dac scan
            # The noise peak at dacTarget (10 DAC units by default)
    
        Possible improvement: Fit dacs scans to populate automatically
        oneE_DAC array (cf fit dac scan)
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
        
        slope = (OneE_Dac[self.fem-1, :] - self.dac_target) / OneE_E
        offset = [self.dac_target, self.dac_target, self.dac_target,
                  self.dac_target, self.dac_target, self.dac_target,
                  self.dac_target, self.dac_target]
        self.save_kev2dac_calib(threshold, slope, offset)
        
        print(str(slope) + str(offset))

        self.settings['filename'] = 'image'

    def multiple_energy_thresh_calib(self, chips=range(8), threshold="0"):
        """
        This functions produces threshold calibration files using DAC scans
        performed with several monochromatic X-ray spectra. Dac scans and
        spectra are performed and plotted in dacscan and spectrum plots. You
        need to inspect the spectra and edit manually the DAC arrays with the
        DAC value corresponding to the X-ray energy for each chip
        
        This method will use several points for calculating Energy to DAC
        conversion coefficients:
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
        dnp.plot.clear('DAC vs energy')
        dnp.plot.clear('DAC vs energy fits')
        for chip in chips:
            x = np.array([E1_E, E2_E, E3_E])
            y = np.array([E1_Dac[self.fem-1, chip], E2_Dac[self.fem-1, chip],
                          E3_Dac[self.fem-1, chip]])
            dnp.plot.addline(x, y, name='DAC vs energy')
        
            popt, pcov = curve_fit(lin_function, x, y, [0, 1])
            offset[chip] = popt[0]
            gain[chip] = popt[1]
            dnp.plot.addline(x, lin_function(x, offset[chip], gain[chip]),
                             name='DAC vs energy fits')
            
        self.save_kev2dac_calib(threshold, gain, offset)
        
        print(str(gain) + str(offset))
        print(self.settings['gain'])

    def set_threshold0(self, thresh_energy='5'):
        """
        Sets Threshold in keV 
        Threshold0 dacs are calculated using threshold0 calibration file
        located in the calibration directory corresponding to the current mode
        and gain setting
        Usage: x.set_threshold0(E) with E in keV
        """

        self.settings['Threshold'] = thresh_energy
        self.set_thresh_energy('0', float(self.settings['Threshold']))
        
    def set_thresh_energy(self, threshold="0", thresh_energy=5.0):
        """
        Sets Threshold0 or Threshold1 in keV 
        ThresholdX dacs are calculated using thresholdX calibration file
        located in the calibration directory corresponding to the current mode
        and gain setting
        Usage: x.set_thresh_energy(Threshold, threshEnergy) where
        Threshold is "0" or "1"
        threshEnergy is in keV
        """

        thresh_coeff = np.genfromtxt(self.calib_settings['calibDir'] + 'fem' +
                                     str(self.fem) + '/' +
                                     self.settings['mode'] + '/' +
                                     self.settings['gain'] + '/' +
                                     'threshold' + str(threshold))

        print(thresh_coeff[0, :].astype(np.int))

        thresh_DACs = (thresh_energy * thresh_coeff[0, :] +
                       thresh_coeff[1, :]).astype(np.int)

        for chip in range(8):
            self.set_dac(range(chip, chip + 1),
                         "Threshold" + str(threshold), thresh_DACs[chip])

        time.sleep(0.2)
        self.settings['acqtime'] = '100'
        self.expose()
        time.sleep(0.2)
        self.expose()

        print("A Threshold" + str(threshold) + " of " + str(thresh_energy) +
              "keV corresponds to " + str(thresh_DACs) +
              " DAC units for each chip")

        return thresh_DACs
        
    def set_dacs(self, chips):
        """
        Set dacs to values recommended by MEDIPIX3-RX designers
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

    def mask(self, chips):
        """
        Creates hexadecimal chip mask corresponding to chips to be enabled when
        sending commands to the front-end
        Usage: mask_hex=x.mask([0]) to  get the mask value to enable chip 0
        only ('0x80') mask_hex=x.mask([0,1,2,3,4,5,6,7]) or
        mask_hex=x.mask(range(8)) to get the mask value to enable all
        chips ('0xff')
        """

        mask_hex = 0
        for chip in chips:
            mask_hex += 2**(self.num_chips - chip - 1)

        return str(hex(mask_hex))

    def read_chip_id(self):
        """
        Reads chip IDs
        Usage: x.read_chip_id(chips) where chips is a list of chips
        i.e. x.read_chip_id(range(8))
        """

        subprocess.call([self.command, "-i", self.ipaddress, "-p", self.port,
                         "-m", self.mask(self.chip_range), "-r", "-e"])
        print(str(self.chip_range))
    
    def log_chip_id(self):
        """
        Reads chip IDs and logs chipIDs in calibration directory
        Usage: x.log_chip_id(chips) where chips is a list of chips
        i.e. x.read_chip_id(range(8))
        """

        log_filename = self.calib_settings['calibDir'] + 'fem' + \
            str(self.fem) + '/efuseIDs'
        with open(log_filename, "w") as outfile:
            subprocess.call([self.command, "-i", self.ipaddress,
                             "-p", self.port, "-m", self.mask(self.chip_range),
                             "-r", "-e"], stdout=outfile)
        print(str(self.chip_range))
    
    def monitor(self):
        """
        Monitors Temperature, Humidity, front-end module voltage regulator
        status and DAC out
        """
        subprocess.call([self.command, "-i", self.ipaddress, "-p", self.port,
                         "-m", self.mask(self.chip_range), "--slow"])
    
    def set_threshold0_dac(self, chips=range(8), dac_value=40):
        """
        This function sets Threshold 0 DAC to a selected value for all chips  
        Usage: x.set_threshold0_dac(30)
        """

        self.set_dac(chips, 'Threshold0', dac_value)
        self.expose()

    def fe55_image_rx001(self, chips=range(8), exp_time=60000):
        """
        THIS WILL SAVE FE55 IMAGE IN
        /dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/ DIRECTORY
        """

        img_path = self.settings['imagepath']
        self.settings['gain'] = 'shgm'
        self.load_config(chips)
        self.set_threshold0_dac(chips, 40)
        self.settings['acqtime'] = str(exp_time)
        self.settings['filename'] = 'Fe55_image_node_' + str(self.fem) + \
                                    '_' + str(self.settings['acqtime']) + 's'
        self.settings['imagepath'] = '/dls/detectors/support/silicon_pixels/' \
                                     'excaliburRX/3M-RX001/Fe55_images/'

        print(self.settings['acqtime'])

        self.settings['acqtime'] = str(exp_time)
        time.sleep(0.5)
        self.expose()
        self.settings['imagepath'] = img_path
        self.settings['filename'] = 'image'
        self.settings['acqtime'] = '100'
    
    def set_dac(self, chips, dac_name="Threshold0", dac_value=40):
        """
        Sets any chip DAC at a given value 
        Usage: x.set_dac([0],'Threshold0', 30)
               x.set_dac(range(8),'Threshold1', 100)
        """

        for chip in chips:
            dac_file = self.calib_settings['calibDir'] + 'fem' + \
                str(self.fem) + '/' + self.settings['mode'] + '/' + \
                self.settings['gain'] + '/' + \
                self.calib_settings['dacfilename']
            f = open(dac_file, 'r+b')
            f_content = f.readlines()
            line_nb = chip*29 + np.int(self.dac_number[dac_name])
            new_line = dac_name + " = " + str(dac_value) + "\r\n"
            f_content[line_nb] = new_line
            f.seek(0)
            f.writelines(f_content)
            f.close()
            subprocess.call([self.command, "-i", self.ipaddress,
                             "-p", self.port,
                             "-m", self.mask(range(chip, chip + 1)),
                             "--dacs=" + dac_file])

    def read_dac(self, chips, dac_name):
        """
        Reads back DAC analogue voltage using chip sense DAC (DAC out)  
        Usage: x.read_dac([0],'Threshold0')
        x.read_dac(range(8),'Threshold1')
        """

        dac_file = self.calib_settings['calibDir'] + 'fem' + str(self.fem) + \
            '/' + self.settings['mode'] + '/' + self.settings['gain'] + \
            '/' + self.calib_settings['dacfilename']

        subprocess.call([self.command, "-i", self.ipaddress, "-p", self.port,
                         "-m", self.mask(chips),
                         "--sensedac=" + str(np.int(self.dac_code[dac_name])),
                         "--dacs=" + dac_file])
        time.sleep(1)
        subprocess.call([self.command, "-i", self.ipaddress, "-p", self.port,
                         "-m", self.mask(self.chip_range),
                         "--sensedac=" + str(np.int(self.dac_code[dac_name])),
                         "--slow"])

    @staticmethod
    def plot_dac_scan(chips, dac_scan_data, dac_range):
        """
        Plots the results of threshold dac scan in an integrated spectrum plot
        window (dacscan) and a differential spectrum (spectrum)
        """

        dnp.plot.clear("dacScan")
        dnp.plot.clear("Spectrum")
        chip_dac_scan = np.zeros([8])

        print(str())

        if dac_range[0] > dac_range[1]:
            for chip in chips:
                dac_axis = (np.array(range(dac_range[0],
                                           dac_range[1] - dac_range[2],
                                           -dac_range[2])))
                chip_dac_scan = np.zeros([8, dac_axis.size])

                chip_dac_scan[chip, :] = \
                    (dac_scan_data[:, 0:256,
                                   chip*256:chip*256 + 256].mean(2).mean(1))

                dnp.plot.addline(
                    np.array(range(dac_range[0],
                                   dac_range[1] - dac_range[2],
                                   -dac_range[2])),
                    np.squeeze(dac_scan_data[:, 0:256,
                               chip*256:chip*256 + 256].mean(2).mean(1)),
                    name="dacScan")

                spectrum = np.diff(np.squeeze(
                    dac_scan_data[:, 0:256,
                                  chip*256:chip*256 + 256].mean(2).mean(1)))

                dnp.plot.addline(
                    np.array(
                        range(dac_range[0], dac_range[1], -dac_range[2]))[1:],
                    spectrum[1:], name="Spectrum")
        else:
            for chip in chips:
                dac_axis = (np.array(range(dac_range[0],
                                           dac_range[1] + dac_range[2],
                                           dac_range[2])))
                chip_dac_scan = np.zeros([8, dac_axis.size])

                chip_dac_scan[chip, :] = \
                    (dac_scan_data[:, 0:256,
                     chip*256:chip*256 + 256].mean(2).mean(1))

                dnp.plot.addline(
                    np.array(range(dac_range[0],
                                   dac_range[1] + dac_range[2], dac_range[2])),
                    np.squeeze(dac_scan_data[:, 0:256,
                               chip*256:chip*256 + 256].mean(2).mean(1)),
                    name="dacScan")

                spectrum = -np.diff(
                    np.squeeze(dac_scan_data[:, 0:256,
                               chip*256:chip*256 + 256].mean(2).mean(1)))

                dnp.plot.addline(
                    np.array(
                        range(dac_range[0], dac_range[1], dac_range[2]))[1:],
                    spectrum[1:], name="Spectrum")

        return chip_dac_scan, dac_axis

    def scan_dac(self, chips, dac_name, dac_range):  # ONLY FOR THRESHOLD DACS
        """
        Performs a dac scan and plot the result (mean counts as a function of
        DAC values)
        Usage: x.scan_dac(chips, dacName, dacRange) where
        chips is a list of chips [0, 1, 2, 3]
        dacRange = (DAC_start_value, DAC_stop_value, DAC_step_value)
        DACs can be scanned in both directions
        dacName is one of the following dacs 
            'Threshold0'
            'Threshold1'
            'Threshold2'
            'Threshold3'
            'Threshold4'
            'Threshold5'
            'Threshold6'
            'Threshold7'
            'Preamp'
            'Ikrum'
            'Shaper'
            'Disc'
            'DiscLS'
            'ShaperTest'
            'DACDiscL'
            'DACTest'
            'DACDiscH'
            'Delay'
            'TPBuffIn'
            'TPBuffOut'
            'RPZ'
            'GND'
            'TPREF'
            'FBK'
            'Cas'
            'TPREFA'
            'TPREFB'
        """

        self.update_filename_index()

        dac_scan_file = self.settings['filename'] + "_" + "dacscan" + \
            self.settings['filenameIndex'] + ".hdf5"
        dac_file = self.calib_settings['calibDir'] + 'fem' + str(self.fem) + \
            '/' + self.settings['mode'] + '/' + self.settings['gain'] + \
            '/' + self.calib_settings['dacfilename']
        string = str(np.int(self.dac_code[dac_name]) - 1) + ',' + \
            str(dac_range[0]) + ',' + str(dac_range[1]) + ',' + \
            str(dac_range[2])

        subprocess.call([self.command, "-i", self.ipaddress, "-p", self.port,
                         "-m", self.mask(chips),
                         "--dacs=" + dac_file,
                         "--csmspm=" + self.mode_code[self.settings['mode']],
                         "--disccsmspm=" + self.settings['disccsmspm'],
                         "--depth=12",
                         "--equalization=" + self.settings['equalization'],
                         "--counter=" + self.settings['counter'],
                         "--acqtime=" + str(self.settings['acqtime']),
                         "--gainmode=" + self.gain_code[self.settings['gain']],
                         "--dacscan", string,
                         "--path=" + self.settings['imagepath'],
                         "--hdffile=" + dac_scan_file])
        time.sleep(1)

        dh = dnp.io.load(self.settings['imagepath'] + dac_scan_file)
        dac_scan_data = dh.image[...]
        dnp.plot.clear("dacScan")
        dnp.plot.clear("Spectrum")
        if dac_range[0] > dac_range[1]:
            for chip in chips:
                dnp.plot.addline(
                    np.array(
                        range(dac_range[0],
                              dac_range[1] - dac_range[2], -dac_range[2])),
                    np.squeeze(dac_scan_data[:, 0:256,
                               chip*256:chip*256 + 256].mean(2).mean(1)),
                    name="dacScan")

                spectrum = np.diff(
                    np.squeeze(dac_scan_data[:, 0:256,
                               chip*256:chip*256 + 256].mean(2).mean(1)))

                dnp.plot.addline(
                    np.array(
                        range(dac_range[0], dac_range[1], -dac_range[2]))[1:],
                    spectrum[1:], name="Spectrum")
        else:
            for chip in chips:
                dnp.plot.addline(
                    np.array(
                        range(dac_range[0],
                              dac_range[1] + dac_range[2], dac_range[2])),
                    np.squeeze(dac_scan_data[:, 0:256,
                               chip*256:chip*256 + 256].mean(2).mean(1)),
                    name="dacScan")

                spectrum = -np.diff(
                    np.squeeze(dac_scan_data[:, 0:256,
                               chip*256:chip*256 + 256].mean(2).mean(1)))

                dnp.plot.addline(
                    np.array(
                        range(dac_range[0], dac_range[1], dac_range[2]))[1:],
                    spectrum[1:], name="Spectrum")

        return [dac_scan_data, dac_range]

    @staticmethod
    def show_pixel(dac_scan_data, dac_range, pixel):
        """
        Plots individual pixel dac scan
        Example: x.show_pixel(dacScanData,[0,30,1],[20,20])
        """

        dnp.plot.addline(
            np.array(
                range(dac_range[0], dac_range[1] + dac_range[2],
                      dac_range[2])),
            (dac_scan_data[:, pixel[0], pixel[1]]),
            name='Pixel S curve')

        spectrum = -np.diff(np.squeeze(dac_scan_data[:, pixel[0], pixel[1]]))

        dnp.plot.addline(
            np.array(range(dac_range[0], dac_range[1], dac_range[2])),
            spectrum, name="Pixel Spectrum")

    def load_config_bits(self, chips, discLbits, discHbits, mask_bits):
        """
        Load specific detector configuration files (discbits, masbits) 
        Usage: x.load_config(chips,discLbits, discHbits, maskbits)
        where chips is a list of chips [0,1,2,3,4]
        discLbits and discHbits are 256 x 256 array with integers between
        0 and 31 maskbits if a 256x256 array of 0 and 1
        """

        discL_bits_file = self.calib_settings['calibDir'] + 'fem' + \
            str(self.fem) + '/' + self.settings['mode'] + '/' + \
            self.settings['gain'] + '/' + 'discLbits.tmp'
        discH_bits_file = self.calib_settings['calibDir'] + 'fem' + \
            str(self.fem) + '/' + self.settings['mode'] + '/' + \
            self.settings['gain'] + '/' + 'discHbits.tmp'
        mask_bits_file = self.calib_settings['calibDir'] + 'fem' + \
            str(self.fem) + '/' + self.settings['mode'] + '/' + \
            self.settings['gain'] + '/' + 'maskbits.tmp'

        np.savetxt(discL_bits_file, discLbits, fmt='%.18g', delimiter=' ')
        np.savetxt(discH_bits_file, discHbits, fmt='%.18g', delimiter=' ')
        np.savetxt(mask_bits_file, mask_bits, fmt='%.18g', delimiter=' ')

        for chip in chips:
                subprocess.call([self.command, "-i", self.ipaddress,
                                 "-p", self.port,
                                 "-m", self.mask(range(chip, chip+1)),
                                 "--config",
                                 "--discl=" + discL_bits_file,
                                 "--disch=" + discH_bits_file,
                                 "--pixelmask=" + mask_bits_file])

    def load_config(self, chips=range(8)):
        """
        Load detector configuration files (discbits, masbits) and default
        thresholds from calibration directory corresponding to selected mode
        and gain
        Usage x.load_config()
        """

        for chip in chips:
            discHbits_file = self.calib_settings['calibDir'] + 'fem' + \
                str(self.fem) + '/' + self.settings['mode'] + \
                '/' + self.settings['gain'] + '/' + 'discHbits.chip' + \
                str(chip)
            discLbits_file = self.calib_settings['calibDir'] + 'fem' + \
                str(self.fem) + '/' + self.settings['mode'] + \
                '/' + self.settings['gain'] + '/' + 'discLbits.chip' + \
                str(chip)
            pixel_mask_file = self.calib_settings['calibDir'] + 'fem' + \
                str(self.fem) + '/' + self.settings['mode'] + \
                '/' + self.settings['gain'] + '/' + 'pixelmask.chip' + \
                str(chip)
            
            if os.path.isfile(discLbits_file):
                if os.path.isfile(discHbits_file):
                    if os.path.isfile(pixel_mask_file):
                        subprocess.call([self.command, "-i", self.ipaddress,
                                         "-p", self.port,
                                         "-m", self.mask(
                                                   range(chip, chip + 1)),
                                         "--config",
                                         "--discl=" + discLbits_file,
                                         "--disch=" + discHbits_file,
                                         "--pixelmask=" + pixel_mask_file])
                    else:
                        subprocess.call([self.command, "-i", self.ipaddress,
                                         "-p", self.port,
                                         "-m", self.mask(
                                                   range(chip, chip + 1)),
                                         "--config",
                                         "--discl=" + discLbits_file,
                                         "--disch=" + discHbits_file])
                else:
                    if os.path.isfile(pixel_mask_file):
                        subprocess.call([self.command, "-i", self.ipaddress,
                                         "-p", self.port,
                                         "-m", self.mask(
                                                   range(chip, chip + 1)),
                                         "--config",
                                         "--discl=" + discLbits_file,
                                         "--pixelmask=" + pixel_mask_file])
                    else:
                        subprocess.call([self.command, "-i", self.ipaddress,
                                         "-p", self.port,
                                         "-m", self.mask(
                                                   range(chip, chip + 1)),
                                         "--config",
                                         "--discl=" + discLbits_file])
            else: 
                print(str(discLbits_file) + " does not exist !")

        self.set_dac(range(8), "Threshold1", 100)
        self.set_dac(range(8), "Threshold0", 40)
        self.expose()
        
    def update_filename_index(self):
        """
        Automatically increments filename index in filename.idx file in
        image path before writing image data file
        """

        idx_filename = self.settings['imagepath'] + \
            self.settings['filename'] + '.idx'

        print(idx_filename)

        if os.path.isfile(idx_filename):
            print(os.path.isfile(idx_filename))
            idx_file = open(idx_filename, 'rw')
            new_idx = int(idx_file.read()) + 1
            idx_file.close()
            idx_file = open(idx_filename, 'w')
            idx_file.write(str(new_idx))
            idx_file.close()
            self.settings['filenameIndex'] = str(new_idx)
        else:
            idx_file = open(idx_filename, 'a')
            new_idx = 0
            idx_file.write(str(new_idx))
            idx_file.close()
            self.settings['filenameIndex'] = str(new_idx)
            os.chmod(idx_filename, 0777)

        return new_idx

    def acquire_ff(self, ni, acquire_time):
        """
        NOT TESTED
        Acquires and sum flat-field images.
        Produces flat-field correction coefficients 
        """

        self.settings['fullFilename'] = "FlatField.hdf5"
        self.settings['acqtime'] = acquire_time

        ff_image = 0
        for n in range(ni):
            image = self.expose()
            ff_image += image

        chip = 3
        ff_image[ff_image == 0] = \
            ff_image[0:256, chip*256:chip*256 + 256].mean()
        ff_coeff = np.ones([256, 256*8]) * \
            ff_image[0:256, chip*256:chip*256 + 256].mean()
        ff_coeff = ff_coeff/ff_image

        dnp.plot.image(ff_coeff[0:256, chip*256:chip*256 + 256],
                       name='Flat Field coefficients')
        ff_coeff[ff_coeff > 2] = 1
        ff_coeff[ff_coeff < 0] = 1

        return ff_coeff

    def burst(self, frames, acquire_time):
        """
        Acquires images in burst mode
        """

        self.settings['acqtime'] = acquire_time
        self.update_filename_index()
        self.settings['frames'] = frames
        self.settings['fullFilename'] = self.settings['filename'] + "_" + \
            self.settings['filenameIndex'] + ".hdf5"

        command = [self.command, "-i", self.ipaddress, "-p", self.port,
                   "-m", self.mask(self.chip_range),
                   "--depth=" + self.settings['bitdepth'],
                   "--csmspm=" + self.mode_code[self.settings['mode']],
                   "--disccsmspm=" + self.settings['disccsmspm'],
                   "--equalization=" + self.settings['equalization'],
                   "--gainmode=" + self.gain_code[self.settings['gain']],
                   "--burst",
                   "--frames=" + str(self.settings['frames']),
                   "--acqtime=" + str(self.settings['acqtime']),
                   "--trigmode=" + self.settings['trigmode'],
                   "--path=" + self.settings['imagepath'],
                   "--hdffile=" + self.settings['fullFilename']]

        print(command)

        subprocess.call(command)
        time.sleep(0.5)

    def expose(self):
        """
        Acquires single images using current detector settings (x.settings)
        Usage: x.expose()
        """

        print(self.settings)

        self.update_filename_index()
        self.settings['frames'] = '1'
        self.settings['fullFilename'] = self.settings['filename'] + "_" + \
            self.settings['filenameIndex'] + ".hdf5"

        subprocess.call([self.command, "-i", self.ipaddress, "-p", self.port,
                         "-m", self.mask(self.chip_range),
                         "--depth=" + self.settings['bitdepth'],
                         "--csmspm=" + self.mode_code[self.settings['mode']],
                         "--disccsmspm=" + self.settings['disccsmspm'],
                         "--equalization=" + self.settings['equalization'],
                         "--gainmode=" + self.gain_code[self.settings['gain']],
                         "--acquire",
                         "--frames=" + str(self.settings['frames']),
                         "--acqtime=" + str(self.settings['acqtime']),
                         "--trigmode=" + self.settings['trigmode'],
                         "--path=" + self.settings['imagepath'],
                         "--hdffile=" + self.settings['fullFilename']])

        print(self.settings['filename'])

        time.sleep(0.5)
        dh = dnp.io.load(self.settings['imagepath'] +
                         self.settings['fullFilename'])
        image_raw = dh.image[...]
        image = dnp.squeeze(image_raw.astype(np.int))
        dnp.plot.clear()
        dnp.plot.image(image, name='Image data')

        return image

    def shoot(self, acquire_time):
        """
        Acquires an image at a given exposure time
        Usage: x.shoot(acquire_time) where acquire_time is in ms
        """

        self.settings['acqtime'] = acquire_time
        self.settings['frames'] = '1'
        self.update_filename_index()
        self.settings['fullFilename'] = self.settings['filename'] + "_" + \
            self.settings['filenameIndex']+".hdf5"

        subprocess.call([self.command, "-i", self.ipaddress, "-p", self.port,
                         "-m", self.mask(self.chip_range),
                         "--depth=" + self.settings['bitdepth'],
                         "--gainmode=" + self.gain_code[self.settings['gain']],
                         "--acquire",
                         "--frames=" + str(self.settings['frames']),
                         "--acqtime=" + str(self.settings['acqtime']),
                         "--path=" + self.settings['imagepath'],
                         "--hdffile=" + self.settings['fullFilename']])

        print(self.settings['filename'])

        time.sleep(0.2)
        dh = dnp.io.load(self.settings['imagepath'] +
                         self.settings['fullFilename'])
        image_raw = dh.image[...]
        image = dnp.squeeze(image_raw.astype(np.int))
        dnp.plot.clear()
        dnp.plot.image(image, name='Image data')

        return self.settings['fullFilename']
    
    def cont(self, frames, acquire_time):
        """
        Acquires images in continuous mode
        Usage:x.cont(frames, acquire_time) where frames is the number of frames
        and acquire_time the acquisition time in ms
        """

        self.settings['acqtime'] = acquire_time
        self.update_filename_index()
        self.settings['frames'] = frames
        self.settings['readmode'] = '1'
        self.settings['fullFilename'] = self.settings['filename'] + "_" + \
            self.settings['filenameIndex'] + ".hdf5"

        subprocess.call([self.command, "-i", self.ipaddress, "-p", self.port,
                         "-m", self.mask(self.chip_range),
                         "--depth=" + self.settings['bitdepth'],
                         "--gainmode=" + self.gain_code[self.settings['gain']],
                         "--acquire",
                         "--readmode=" + str(self.settings['readmode']),
                         "--counter=" + str(self.settings['counter']),
                         "--frames=" + str(self.settings['frames']),
                         "--acqtime=" + str(self.settings['acqtime']),
                         "--trigmode=" + self.settings['trigmode'],
                         "--path=" + self.settings['imagepath'],
                         "--hdffile=" + self.settings['fullFilename']])

        print(self.settings['filename'])

        time.sleep(0.2)
        dh = dnp.io.load(self.settings['imagepath'] +
                         self.settings['fullFilename'])
        image_raw = dh.image[...]
        image = dnp.squeeze(image_raw.astype(np.int))

        plots = 5
        if frames < 5:
            plots = frames

        for p in range(plots):
            # dnp.plot.clear()
            dnp.plot.image(image[p, :, :], name='Image data ' + str(p))
    
    def cont_burst(self, frames, acquire_time):
        """
        Acquires images in continuous burst mode
        Usage:x.cont(frames,acqtime) where frames is the number of frames and
        acqtime the acquisition time in ms
        """
        
        self.settings['acqtime'] = acquire_time
        self.update_filename_index()
        self.settings['frames'] = frames
        self.settings['fullFilename'] = self.settings['filename'] + "_" + \
            self.settings['filenameIndex'] + ".hdf5"

        subprocess.call([self.command, "-i", self.ipaddress, "-p", self.port,
                         "-m", self.mask(self.chip_range),
                         "--depth=" + self.settings['bitdepth'],
                         "--gainmode=" + self.gain_code[self.settings['gain']],
                         "--burst",
                         "--readmode=" + str(self.settings['readmode']),
                         "--counter=" + str(self.settings['counter']),
                         "--frames=" + str(self.settings['frames']),
                         "--acqtime=" + str(self.settings['acqtime']),
                         "--trigmode=" + self.settings['trigmode'],
                         "--path=" + self.settings['imagepath'],
                         "--hdffile=" + self.settings['fullFilename']])
    
    def apply_ff_correction(self, ni, ff_coeff):
        """
        NOT TESTED
        Applies flat-field correction
        """

        self.settings['fullFilename'] = self.settings['filename'] + "_" + \
            self.settings['filenameIndex'] + ".hdf5"
        dh = dnp.io.load(self.settings['imagepath'] +
                         self.settings['fullFilename'])
        image_raw = dh.image[...]
        image = dnp.squeeze(image_raw.astype(np.int))

        for p in range(ni):
            dnp.plot.clear()
            ff = image[p, :, :] * ff_coeff
            ff[ff > 3000] = 0
            chip = 3
            dnp.plot.image(ff[0:256, chip*256:chip*256 + 256],
                           name='Image data Cor')
            time.sleep(1)

    def logo_test(self):
        """
        Test the detector using test pulses representing excalibur logo 
        Usage: x.logo_test()
        """

        chips = range(self.num_chips)
        self.set_dac(chips, "Threshold0", 40)
        self.shoot(10)
        logo_tp = np.ones([256, 8*256])
        logo_small = np.loadtxt(self.calib_settings['configDir'] + 'logo.txt')
        logo_tp[7:250, 225:1823] = logo_small
        logo_tp[logo_tp > 0] = 1
        logo_tp = 1 - logo_tp
        
        for chip in chips:
            test_bits_file = self.calib_settings['calibDir'] + 'Logo_chip' + \
                str(chip) + '_mask'
            np.savetxt(test_bits_file, logo_tp[0:256, chip*256:chip*256 + 256],
                       fmt='%.18g', delimiter=' ')
            
            discHbits_file = self.calib_settings['calibDir'] + 'fem' + \
                str(self.fem) + '/' + self.settings['mode'] + \
                '/' + self.settings['gain'] + \
                '/' + 'discHbits.chip' + str(chip)
            discLbits_file = self.calib_settings['calibDir'] + 'fem' + \
                str(self.fem) + '/' + self.settings['mode'] + \
                '/' + self.settings['gain'] + \
                '/' + 'discLbits.chip' + str(chip)
            pixel_mask_file = self.calib_settings['calibDir'] + 'fem' + \
                str(self.fem) + '/' + self.settings['mode'] + \
                '/' + self.settings['gain'] + \
                '/' + 'pixelmask.chip' + str(chip)
            
            if os.path.isfile(discLbits_file) \
                    and os.path.isfile(discHbits_file) \
                    and os.path.isfile(pixel_mask_file):
                subprocess.call([self.command, "-i", self.ipaddress,
                                 "-p", self.port,
                                 "-m", self.mask(range(chip, chip + 1)),
                                 "--dacs", self.calib_settings['calibDir'] +
                                 self.calib_settings['dacfilename'],
                                 "--config",
                                 "--discl=" + discLbits_file,
                                 "--disch=" + discHbits_file,
                                 "--pixelmask=" + pixel_mask_file,
                                 "--tpmask=" + test_bits_file])
            else:
                subprocess.call([self.command, "-i", self.ipaddress,
                                 "-p", self.port,
                                 "-m", self.mask(range(chip, chip + 1)),
                                 "--dacs", self.calib_settings['calibDir'] +
                                 self.calib_settings['dacfilename'],
                                 "--config",
                                 "--tpmask=" + test_bits_file])

        self.settings['fullFilename'] = \
            self.settings['filename'] + "_" + \
            self.settings['filenameIndex'] + ".hdf5"
        time.sleep(0.2)
        subprocess.call([self.command, "-i", self.ipaddress,
                         "-p", self.port,
                         "-m", self.mask(chips),
                         "--depth=" + self.settings['bitdepth'],
                         "--acquire",
                         "--readmode=" + str(self.settings['readmode']),
                         "--frames=" + str(self.settings['frames']),
                         "--acqtime=" + str(self.settings['acqtime']),
                         "--counter=0",
                         "--path=" + self.settings['imagepath'],
                         "--hdffile=" + self.settings['fullFilename'],
                         "--tpcount=" + str(100)])

        dh = dnp.io.load(self.settings['imagepath'] +
                         self.settings['fullFilename'])
        image_raw = dh.image[...]
        image = dnp.squeeze(image_raw.astype(np.int))
        dnp.plot.image(image, name='Image data')

    def test_pulse(self, chips, test_bits, pulses):
        """
        Usage: x.test_pulse([0],'excaliburRx/config/triangle.mask',1000)
        """

        if type(test_bits) == str:
            test_bits_file = test_bits
        else:
            discLbits_file = self.calib_settings['calibDir'] + 'fem' + \
                str(self.fem) + '/' + self.settings['mode'] + \
                '/' + self.settings['gain'] + '/' + 'testbits.tmp'
            np.savetxt(test_bits_file, test_bits, fmt='%.18g', delimiter=' ')

        dnp.plot.clear()
        # self.update_filename_index()
        self.settings['fullFilename'] = self.settings['filename'] + "_" + \
            self.settings['filenameIndex'] + ".hdf5"

        for chip in chips:
            subprocess.call([self.command, "-i", self.ipaddress,
                             "-p", self.port,
                             "-m", self.mask(range(chip, chip + 1)),
                             "--dacs", self.calib_settings['calibDir'] +
                             self.calib_settings['dacfilename'],
                             "--config",
                             "--tpmask=" + test_bits_file])
        subprocess.call([self.command, "-i", self.ipaddress,
                         "-p", self.port,
                         "-m", self.mask(chips),
                         "--depth=" + self.settings['bitdepth'],
                         "--acquire",
                         "--readmode=" + str(self.settings['readmode']),
                         "--frames=" + str(self.settings['frames']),
                         "--acqtime=" + str(self.settings['acqtime']),
                         "--counter=0",
                         "--path=" + self.settings['imagepath'],
                         "--hdffile=" + self.settings['fullFilename'],
                         "--tpcount=" + str(pulses)])

        print(self.settings['fullFilename'])

        dh = dnp.io.load(self.settings['imagepath'] +
                         self.settings['fullFilename'])
        image_raw = dh.image[...]
        image = dnp.squeeze(image_raw.astype(np.int))
         
        dnp.plot.image(image, name='Image data')

        return image

    def save_discbits(self, chips, discbits, discbitsFilename):
        """
        saves discbit array into discbit file in current calibration directory
        """

        for chip in chips:
            discbits_file = self.calib_settings['calibDir'] + 'fem' + \
                str(self.fem) + '/' + self.settings['mode'] + \
                '/' + self.settings['gain'] + '/' + discbitsFilename + \
                '.chip' + str(chip)

            np.savetxt(discbits_file, discbits[0:256, chip*256:chip*256 + 256],
                       fmt='%.18g', delimiter=' ')

    def mask_sup_col(self, chip, super_column):
        """
        masks a noisy super column in a chip (bit=1 to mask a pixel) and
        updates the corresponding  maskfile
        Usage: x.mask_sup_col(chips,supCol) where
        chips is a list of chips ([0,1,2,3])
        supCol in an integer between 0 and 7
        """

        bad_pixels = np.zeros([self.chip_size,
                               self.chip_size * self.num_chips])
        bad_pixels[:, chip*256 + super_column * 32:chip * 256 +
                      super_column * 32 + 64] = 1

        discLbits_file = self.calib_settings['calibDir'] + 'fem' + \
            str(self.fem) + '/' + self.settings['mode'] + \
            '/' + self.settings['gain'] + '/' + 'discLbits.chip' + str(chip)
        pixel_mask_file = self.calib_settings['calibDir'] + 'fem' + \
            str(self.fem) + '/' + self.settings['mode'] + '/' + \
            self.settings['gain'] + '/' + 'pixelmask.chip' + str(chip)

        np.savetxt(pixel_mask_file, bad_pixels[0:256, chip*256:chip*256 + 256],
                   fmt='%.18g', delimiter=' ')
        subprocess.call([self.command, "-i", self.ipaddress, "-p", self.port,
                         "-m", self.mask(range(chip, chip + 1)),
                         "--config",
                         "--pixelmask=" + pixel_mask_file,
                         "--config",
                         "--discl=" + discLbits_file])
        dnp.plot.image(bad_pixels)

    def mask_col(self, chip, col):
        """
        masks a noisy column in a chip (bit=1 to mask a pixel) and updates the
        corresponding  maskfile
        Usage: x.mask_col(chips,Col) where
        chips is a list of chips ([0,1,2,3])
        supCol in an integer between 0 and 255
        """

        bad_pixels = np.zeros([self.chip_size,
                               self.chip_size * self.num_chips])
        bad_pixels[:, col] = 1

        discLbits_file = self.calib_settings['calibDir'] + 'fem' + \
            str(self.fem) + '/' + self.settings['mode'] + '/' + \
            self.settings['gain'] + '/' + 'discLbits.chip' + str(chip)
        pixel_mask_file = self.calib_settings['calibDir'] + 'fem' + \
            str(self.fem) + '/' + self.settings['mode'] + '/' + \
            self.settings['gain'] + '/' + 'pixelmask.chip' + str(chip)

        np.savetxt(pixel_mask_file, bad_pixels[0:256, chip*256:chip*256 + 256],
                   fmt='%.18g', delimiter=' ')
        subprocess.call([self.command, "-i", self.ipaddress, "-p", self.port,
                         "-m", self.mask(range(chip, chip + 1)),
                         "--config",
                         "--pixelmask=" + pixel_mask_file,
                         "--config",
                         "--discl=" + discLbits_file])
        dnp.plot.image(bad_pixels, name='Bad pixels')
        
    def mask_pixels(self, chips, image_data, max_counts):
        """
        masks pixels in imageData with counnts above maxCounts
        updates the corresponding maskfile in the calibration directory
        """

        bad_pix_tot = np.zeros(8)
        bad_pixels = image_data > max_counts
        dnp.plot.image(bad_pixels, name='Bad pixels')
        for chip in chips:
            discLbits_file = self.calib_settings['calibDir'] + 'fem' + \
                str(self.fem) + '/' + self.settings['mode'] + \
                '/' + self.settings['gain'] + '/' + 'discLbits.chip' + \
                str(chip)
            bad_pix_tot[chip] = bad_pixels[0:256,
                                           chip*256:chip*256 + 256].sum()

            print('####### ' + str(bad_pix_tot[chip]) +
                  ' noisy pixels in chip ' + str(chip) +
                  ' (' + str(100*bad_pix_tot[chip]/(256**2)) + '%)')

            pixel_mask_file = self.calib_settings['calibDir'] + 'fem' + \
                str(self.fem) + '/' + self.settings['mode'] + '/' + \
                self.settings['gain'] + '/' + 'pixelmask.chip' + str(chip)

            np.savetxt(pixel_mask_file,
                       bad_pixels[0:256, chip*256:chip*256 + 256],
                       fmt='%.18g', delimiter=' ')
            subprocess.call([self.command, "-i", self.ipaddress,
                             "-p", self.port,
                             "-m", self.mask(range(chip, chip + 1)),
                             "--config",
                             "--pixelmask=" + pixel_mask_file,
                             "--config",
                             "--discl=" + discLbits_file])

        print('####### ' + str(bad_pix_tot.sum()) +
              ' noisy pixels in half module ' +
              ' (' + str(100*bad_pix_tot.sum()/(8*256**2)) + '%)')

    def mask_pixels_using_dac_scan(self, chips=range(8),
                                   threshold="Threshold0",
                                   dac_range=(20, 120, 2)):
        """
        Performs threshold dac scan and masks pixels in with counnts above a
        max_counts value in any of the dac scan images
        updates the corresponding maskfile in the calibration directory
        Usage: x.mask_pixels_using_dac_scan(chips,Threshold,dacRange) where
        chips is a list of chips [0,1,2,3]
        Threshold = "Threshold0" or "Threshold1"
        dacRange = (DAC_start_value,DAC_stop_value,DAC_step_value) DACs can be
        scanned in both directions
        # dacRange=(20,120,2)
        """

        max_counts = 1
        bad_pix_tot = np.zeros(8)
        self.settings['acqtime'] = 100
        [dac_scan_data, scan_range] = self.scan_dac(chips, threshold,
                                                    dac_range)
        bad_pixels = dac_scan_data.sum(0) > max_counts
        dnp.plot.image(bad_pixels, name='Bad pixels')
    
        for chip in chips:
            bad_pix_tot[chip] = \
                bad_pixels[0:256, chip*256:chip*256 + 256].sum()

            print('####### ' + str(bad_pix_tot[chip]) +
                  ' noisy pixels in chip ' + str(chip) +
                  ' (' + str(100*bad_pix_tot[chip]/(256**2)) + '%)')

            pixel_mask_file = self.calib_settings['calibDir'] + 'fem' + \
                str(self.fem) + '/' + self.settings['mode'] + '/' + \
                self.settings['gain'] + '/' + 'pixelmask.chip' + str(chip)

            np.savetxt(pixel_mask_file,
                       bad_pixels[0:256, chip*256:chip*256 + 256],
                       fmt='%.18g', delimiter=' ')
            # subprocess.call([self.command, "-i", self.ipaddress,
            #                  "-p", self.port,
            #                  "-m", self.mask(range(chip, chip + 1)),
            #                  "--config",
            #                  "--pixelmask=" + pixel_mask_file,
            #                  "--config",
            #                  "--discl=" + discLbitsFile])

        print('####### ' + str(bad_pix_tot.sum()) +
              ' noisy pixels in half module ' +
              ' (' + str(100*bad_pix_tot.sum() / (8*256**2)) + '%)')

    def unmask_all_pixels(self, chips):
        """
        Unmasks all pixels and updates maskfile in calibration directory 
        """

        bad_pixels = np.zeros([self.chip_size,
                               self.chip_size * self.num_chips])

        for chip in chips:
            discLbits_file = self.calib_settings['calibDir'] + 'fem' + \
                str(self.fem) + '/' + self.settings['mode'] + '/' + \
                self.settings['gain'] + '/' + 'discLbits.chip' + str(chip)
            pixel_mask_file = self.calib_settings['calibDir'] + 'fem' + \
                str(self.fem) + '/' + self.settings['mode'] + '/' + \
                self.settings['gain'] + '/' + 'pixelmask.chip' + str(chip)

            np.savetxt(pixel_mask_file,
                       bad_pixels[0:256, chip*256:chip*256 + 256],
                       fmt='%.18g', delimiter=' ')
            subprocess.call([self.command, "-i", self.ipaddress,
                             "-p", self.port,
                             "-m", self.mask(range(chip, chip + 1)),
                             "--config",
                             "--pixelmask=" + pixel_mask_file,
                             "--config",
                             "--discl=" + discLbits_file])

    def unequalize_all_pixels(self, chips):
        """
        Resets discL_bits and pixel_mask bits to 0
        """

        discL_bits = 31*np.zeros([self.chip_size,
                                  self.chip_size * self.num_chips])
        for chip in chips:
            discLbits_file = self.calib_settings['calibDir'] + 'fem' + \
                str(self.fem) + '/' + self.settings['mode'] + '/' + \
                self.settings['gain'] + '/' + 'discL_bits.chip' + str(chip)
            pixel_mask_file = self.calib_settings['calibDir'] + 'fem' + \
                str(self.fem) + '/' + self.settings['mode'] + '/' + \
                self.settings['gain'] + '/' + 'pixelmask.chip' + str(chip)

            np.savetxt(discLbits_file,
                       discL_bits[0:256, chip*256:chip*256 + 256],
                       fmt='%.18g', delimiter=' ')
            subprocess.call([self.command, "-i", self.ipaddress,
                             "-p", self.port,
                             "-m", self.mask(range(chip, chip + 1)),
                             "--config",
                             "--pixelmask=" + pixel_mask_file,
                             "--config",
                             "--discl=" + discLbits_file])

    def check_calib_dir(self):
        """
        Checks if calibration directory exists, if YES backs-up the directory
        otherwise creates it
        """

        calib_dir = (self.calib_settings['calibDir'] + 'fem' + str(self.fem) +
                     '/' + self.settings['mode'] +
                     '/' + self.settings['gain'] + '/')

        if (os.path.isdir(calib_dir)) == 0:
            os.makedirs(calib_dir)
        else:
            backup_dir = self.calib_settings['calibDir'][:-1] + '_backup_' + \
                         time.asctime()
            shutil.copytree(self.calib_settings['calibDir'], backup_dir)

            print(backup_dir)

        dac_file = calib_dir + self.calib_settings['dacfilename']
        if os.path.isfile(dac_file) == 0:
            shutil.copy(self.calib_settings['configDir'] +
                        self.calib_settings['dacfilename'], calib_dir)

        # if os.path.isfile(dac_file) == 0:
        #     shutil.copy(self.calibSettings['configDir'] + 'zeros.mask',
        #                 calib_dir)

        return dac_file

    def copy_slgm_into_other_gain_modes(self):
        """
        The functions simply copies /femx/slgm calibration folder into
        /femx/lgm, /femx/hgm and /femx/shgm calibration folders. This function
        is used at the end of threshold equalization because threshold
        equalization is performed in the more favorable gain mode slgm and
        threshold equalization data is independent of the gain mode
        """

        lgm_dir = (self.calib_settings['calibDir'] + 'fem' + str(self.fem) +
                   '/' + self.settings['mode'] + '/' + 'lgm' + '/')
        hgm_dir = (self.calib_settings['calibDir'] + 'fem' + str(self.fem) +
                   '/' + self.settings['mode'] + '/' + 'hgm' + '/')
        slgm_dir = (self.calib_settings['calibDir'] + 'fem' + str(self.fem) +
                    '/' + self.settings['mode'] + '/' + 'slgm' + '/')
        shgm_dir = (self.calib_settings['calibDir'] + 'fem' + str(self.fem) +
                    '/' + self.settings['mode'] + '/' + 'shgm' + '/')

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
        """
        returns discbit array stored in current calibration folder
        """

        discbits = np.zeros([self.chip_size, self.chip_size * self.num_chips])
        for chip in chips:
            discbits_file = self.calib_settings['calibDir'] + 'fem' + \
                str(self.fem) + '/' + self.settings['mode'] + '/' + \
                self.settings['gain'] + '/' + \
                discbits_filename + '.chip' + str(chip)
            discbits[0:256, chip*256:chip*256 + 256] = \
                np.loadtxt(discbits_file)

        return discbits

    def combine_rois(self, chips, disc_name, steps, roi_type):
        """
        Combines intermediate discbits_roi files produced when equalizing
        various ROIs into one discbits file
        Usage: x.combine_rois(chips,discName,steps,roiType)
        chips is a list of chips [1,2,3,4]
        discName is 'discL' or 'discH' 
        steps is the number of ROIs to merge (number of steps used during
        equalization)
        roiType is "rect" or "spacing" defined in the roi function)
        """

        discbits = np.zeros([self.chip_size, self.chip_size * self.num_chips])
        for step in range(steps):
            roi_full_mask = self.roi(chips, step, steps, roi_type)
            discbits_roi = self.open_discbits_file(chips, disc_name +
                                                   'bits_roi_' + str(step))
            discbits[roi_full_mask.astype(bool)] = \
                discbits_roi[roi_full_mask.astype(bool)]
            dnp.plot.image(discbits_roi, name='discbits')

        self.save_discbits(chips, discbits, disc_name + 'bits')
        dnp.plot.image(discbits, name='discbits')

        return discbits

    @staticmethod
    def find_edge(chips, dac_scan_data, dac_range, edge_val):
        """
        Find noise or X-ray edge in threshold DAC scan
        """

        dnp.plot.clear("noise edges histogram")
        dnp.plot.clear("noise edges")
        if dac_range[1] > dac_range[0]:
            edge_dacs = dac_range[1] - dac_range[2] * np.argmax(
                (dac_scan_data[::-1, :, :] > edge_val), 0)
        else:
            edge_dacs = dac_range[0] - dac_range[2] * np.argmax(
                (dac_scan_data[:, :, :] > edge_val), 0)
            
        dnp.plot.image(edge_dacs, name="noise edges")
        for chip in chips:
            dnp.plot.addline(np.histogram(edge_dacs[0:256,
                                          chip*256:chip*256 + 256])[1][0:-1],
                             np.histogram(edge_dacs[0:256,
                                          chip*256:chip*256 + 256])[0],
                             name="noise edges histogram")
        return edge_dacs

    @staticmethod
    def find_max(chips, dac_scan_data, dac_range):
        """
        Find noise max in threshold dac scan
        """

        dnp.plot.clear("noise edges histogram")
        dnp.plot.clear("noise edges")
        edge_dacs = dac_range[1] - dac_range[2] * np.argmax(
            (dac_scan_data[::-1, :, :]), 0)
        dnp.plot.image(edge_dacs, name="noise edges")

        for chip in chips:
            dnp.plot.addline(np.histogram(edge_dacs[0:256,
                                          chip*256:chip*256 + 256])[1][0:-1],
                             np.histogram(edge_dacs[0:256,
                                          chip*256:chip*256 + 256])[0],
                             name="noise edges histogram")
        return edge_dacs

    def optimize_dac_disc(self, chips, disc_name, roi_full_mask):
        """
        Usage:
        [opt_dac_disc] = x.optimize_dac_disc(chips, discName, roiFullMask)
        where "OptDacdisc" is an array of 8 integers corresponding to the
        optimum DAC disc value of each chip, "chips" is a list containing the
        number of the chips to calibrate, "discName" is either "discL" or
        "discH", roiFullMask is 256x256 logical array used to select the pixels
        masked during threshold_equalization
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
        dnp.plot.clear(plot_name)
        dnp.plot.clear(fit_plot_name)
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
            dnp.plot.clear(calib_plot_name)
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
        dnp.plot.clear(plot_name)
        dnp.plot.clear(fit_plot_name)
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

        if abs(x0-self.dac_target).any() > self.acc_dist:  # To be checked
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
        """
        Equalizes pixel discriminator
        Usage: x.equalise_discbits(range(8),chips,discName.roiFullMask,method)
        where chips is a list of chips [1,2,3,4], discName is 'discL' or
        'discH', roiFullMask is a mask file produced with the roi function
        method is 'dacscan' or 'bitscan' or 'stripes'
        Use stripes method as default (trimbits distributed across the matrix
        during each dacscan to avoid saturation the chips when pixels are in
        the noise at the same time)
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
        """
        NOT TESTED
        Checks if dac scan looks ok after threshold calibration
        """

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
                                  self.dac_target - self.acc_dist) &
                                 (edge_dacs[roi, chip*256:chip*256 + 256] <
                                  self.dac_target + self.acc_dist)).sum()
            print('####### ' + str(round(equ_pix_tot[chip], 0)) +
                  ' equalized pixels in chip ' + str(chip) +
                  ' (' + str(round(100*equ_pix_tot[chip]/(256**2), 4)) + '%)')

        # pixelsInTarget = (dacTarget - 5 < edge_dacs) & \
        #                      (edge_dacs < dacTarget + 5)

    def roi(self, chips, step, steps, roi_type):
        """
        Creates a detector ROI to be used when equalizing threshold in several
        steps
        usage: x.roi(chips, step, steps,roiType)
        chips is a list of chips (e.g. [0,1,2,3])
        step is the current step in the equalization process
        steps is the total number of steps during the equalization process
        roiType is 
            "rect": contiguous rectangles
            "spacing": arrays of equally-spaced pixels distributed across the
                chip
            
        Using several ROIs during equalization was needed to avoid putting too
        many pixels in the noise at the same time. However the latest
        equalization scripts used for EXCALIBUR use the same technique as
        MERLIN to distribute equalization bits in diagonal across the matrix
        during equalization. Therefore the roi used by the latest scripts is
        always:
        roi = x.roi(range(8), 1, 1, 'rect')
        """

        if roi_type == 'rect':
            roi_full_mask = np.zeros([self.chip_size,
                                      self.num_chips * self.chip_size])
            for chip in chips:
                roi_full_mask[step*256/steps:step*256/steps + 256/steps,
                              chip*256:chip*256 + 256] = 1
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

        dnp.plot.image(roi_full_mask)

        return roi_full_mask

    def calibrate_disc(self, chips, disc_name, steps=1, roi_type='rect'):
        """
        Usage x.calibrate_disc([0],"threshold0") to calibrate threshold 0 of
        chip 0 using the full matrix as a ROI during threshold_equalization
        x.calibrate_disc(range(8),"threshold1") to calibrate threshold 1 of all
        chips using the full chip matrix as a ROI during threshold_equalization
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
        """
        Acquires several images in a loop and display the sum of images in sum
        plot name
        """

        tmp = 0
        for i in range(ni):
            tmp = self.expose() + tmp
            dnp.plot.image(tmp, name='sum')

            return
        
    def csm(self, chips=range(8), gain='slgm'):
        """
        Loads CSM mode and associated default settings
        """

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
        """
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
        
        This process could be automated if many modules have to be calibrated
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
        #  bottom 1/2 module
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

        print(config_file)

        # shutil.copy(config_file, config_file + ".backup")
        config_bits = np.loadtxt(config_file)
        np.savetxt(config_file, np.rot90(config_bits, 2), fmt='%.18g',
                   delimiter=' ')

    def rotate_all_configs(self):
        """
        x.rotateAllConfig("/dls/tmp/excalibur/calib_spm_shgm/")
        Calibration files of node 1, 3 and 5 have to be rotated in order to be
        loaded correctly in EPICS. This routine copies calib into calib_epics
        and rotate discLbits, discHbits and maskbits files when they exist for
        node 1, 3, and 5
        """

        chips = [0, 1, 2, 3, 4, 5, 6, 7]
        EPICS_calib_path = self.calib_settings['calibDir'][:-1] + '_epics'
        shutil.copytree(self.calib_settings['calibDir'], EPICS_calib_path)

        print(EPICS_calib_path)
        
        for fem in range(6):
            if fem % 2 == 1:

                print("Config files of node" + str(fem) +
                      " have to be rotated in EPICS calibration folder " +
                      str(self.fem))

                for chip in chips:
                    discLbits_file = EPICS_calib_path + "/fem" + str(fem) + \
                        "/spm/slgm/discLbits.chip" + str(chip)
                    if os.path.isfile(discLbits_file):
                        self.rotate_config(discLbits_file)
                        print(discLbits_file + "rotated")
                    discHbits_file = EPICS_calib_path + "/fem" + str(fem) + \
                        "/spm/slgm/discHbits.chip" + str(chip)
                    if os.path.isfile(discHbits_file):
                        print(discHbits_file + "rotated")
                        self.rotate_config(discHbits_file)
                    pixel_mask_file = EPICS_calib_path + "/fem" + str(fem) + \
                        "/spm/slgm/pixelmask.chip" + str(chip)
                    if os.path.isfile(pixel_mask_file):
                        self.rotate_config(pixel_mask_file)
                        print(pixel_mask_file + "rotated")
