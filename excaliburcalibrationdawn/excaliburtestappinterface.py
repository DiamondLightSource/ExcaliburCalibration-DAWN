"""A Python interface to the ExcaliburTestApplication command line tool

Command Line Options:

  -h --help                   Display this usage information.
  -i --ipaddress              IP address of FEM to connect to.
  -p --port                   Port of FEM to connect to.
  -m --mask                   Select MPX3 enable mask.
  -r --reset                  Issue front-end reset/init.
     --lvenable <mode>        Set power card LV enable: 0=off (default), 1=on.
     --hvenable <mode>        Set power card HV enable: 0=off (default), 1=on.
     --hvbias <volts>         Set power card HV bias in volts.
  -e --efuse                  Read and display MPX3 eFuse IDs.
  -d --dacs <filename>        Load MPX3 DAC values from filename if given,
                              otherwise use default values
  -c --config                 Load MPX3 pixel configuration.
  -s --slow                   Display front-end slow control parameters.
  -a --acquire                Execute image acquisition loop.
     --burst                  Select burst mode for image acquisition.
     --matrixread             During acquisition, perform matrix read only i.e.
                              no shutter for config read or digital test.
  -n --frames <frames>        Number of frames to acquire.
  -t --acqtime <time>         Acquisition time (shutter duration) in
                              milliseconds.
     --dacscan <params>       Execute DAC scan, params format must be comma
                              separated dac,start,stop,step.
     --readmode <mode>        Readout mode: 0=sequential (default),
                              1=continuous.
     --trigmode <mode>        Trigger mode: 0=internal (default),
                              1=external shutter, 2=external sync.
     --colourmode <mode>      Select MPX3 colour mode: 0=fine pitch mode
                              (default), 1=spectroscopic mode.
     --csmspm <mode>          Select MPX3 pixel mode: 0=single pixel mode
                              (default), 1=charge summing mode.
     --disccsmspm <mode>      Select MPX3 discriminator output mode: 0=DiscL
                              (default), 1=DiscH.
     --equalization <mode>    Select MPX3 equalization mode: 0=off (default),
                              1=on.
     --gainmode <mode>        Select MPX3 gain mode: 0=SHGM, 1=HGM, 2=LGM,
                              3=SLGM (default).
     --counter <counter>      Select MPX3 counter to read: 0 (default) or 1.
     --depth <depth>          Select MPX3 counter depth: 1, 6, 12 (default)
                              or 24.
     --sensedac <id>          Set MPX3 sense DAC field to <id>. NB Requires
                              DAC load to take effect
     --tpmask <filename>      Specify test pulse mask filename to load.
     --tpcount <count>        Set test pulse count to <count>, default is 0.
     --pixelmask <filename>   Specify pixel enable mask filename to load.
     --discl <filename>       Specify pixel DiscL configuration filename to
                              load.
     --disch <filename>       Specify pixel DiscH configuration filename to
                              load.
     --path <path>            Specify path to write data files to, default is
                              /tmp.
     --hdffile <filename>     Write HDF file with optional filename, default is
                               <path>/excalibur-YYMMDD-HHMMSS.hdf5

"""
import os
import posixpath
import time
import subprocess
import logging

logging.basicConfig(level=logging.DEBUG)


class ExcaliburTestAppInterface(object):

    """A class to make subprocess calls to the excaliburTestApp tool."""

    # ExcaliburTestApp flags & example usage
    IP_ADDRESS = "-i"  # -i 192.168.0.10
    PORT = "-p"  # -p 6969
    MASK = "-m"  # -m 0xff
    RESET = "-r"
    LV = "--lvenable"  # --lvenable 1
    HV = "--hvenable"  # --hvenable 1
    HV_BIAS = "--hvbias"  # --hvbias 120
    READ_EFUSE = "-e"
    DAC_FILE = "--dacs"  # --dacs default_dac_values.txt
    READ_SLOW_PARAMS = "-s"
    SENSE = "--sensedac"  # --sensedac 5
    SCAN = "--dacscan"  # --dacscan 0,10,1,5
    ACQUIRE = "-a"
    NUM_FRAMES = "-n"  # -n 10
    ACQ_TIME = "-t"  # -t 100
    TRIG_MODE = "--trigmode"  # --trigmode 1
    BURST = "--burst"
    DEPTH = "--depth"  # --depth 1
    PIXEL_MODE = "--csmspm"  # --csmspm 1
    DISC_MODE = "--disccsmspm"  # --disccsmspm 1
    COUNTER = "--counter"  # --counter 1
    EQUALIZATION = "--equalization"  # --equalization 1
    READ_MODE = "--readmode"  # --readmode 1
    GAIN_MODE = "--gainmode"  # --gainmode 3
    PATH = "--path"  # --path /scratch/excalibur_images
    HDF_FILE = "--hdffile"  # --hdffile image_1
    TP_COUNT = "--tpcount"  # --tpcount 2
    CONFIG = "--config"
    PIXEL_MASK = "--pixelmask"  # --pixelmask mask.txt
    DISC_L = "--discl"  # --discl default_disc_L.txt
    DISC_H = "--disch"  # --disch default_disc_H.txt
    TP_MASK = "--tpmask"  # --tpmask logo_mask.txt

    # Parameters representing detector specification
    num_chips = 8
    chip_range = range(num_chips)

    mode_code = dict(spm='0', csm='1')
    disc_code = dict(discL='0', discH='1')
    read_code = dict(sequential='0', continuous='1')
    gain_code = dict(shgm='0', hgm='1', lgm='2', slgm='3')
    dac_code = dict(Threshold0='1', Threshold1='2', Threshold2='3',
                    Threshold3='4', Threshold4='5', Threshold5='6',
                    Threshold6='7', Threshold7='8',
                    Preamp='9', Ikrum='10', Shaper='11', Disc='12',
                    DiscLS='13', ShaperTest='14', DACDiscL='15', DACTest='30',
                    DACDiscH='31', Delay='16', TPBuffIn='17', TPBuffOut='18',
                    RPZ='19', GND='20', TPREF='21', FBK='22', Cas='23',
                    TPREFA='24', TPREFB='25')

    def __init__(self, ip_address, port):
        self.path = "/dls/detectors/support/silicon_pixels/excaliburRX/" \
                    "TestApplication_15012015/excaliburTestApp"
        self.ip_address = ip_address
        self.port = str(port)

        self.base_cmd = [self.path,
                         self.IP_ADDRESS, self.ip_address,
                         self.PORT, self.port]

    def _construct_command(self, chips, *cmd_args):
        """Construct a command from the base_cmd, given chips and any cmd_args.

        Args:
            chips(list(int): Chips to enable for command process
            *cmd_args(list(str)): Arguments defining the process to be called

        Returns:
            list(str)): Full command to send to subprocess call

        """
        return self.base_cmd + [self.MASK, self._mask(chips)] + list(cmd_args)

    def _mask(self, chips):
        """Create a hexadecimal mask to activate the given chips.

        Args:
            chips(list(int)): List of chips to be enabled

        Returns:
            str: Hexadecimal mask representing list of chips

        """
        if len(chips) != len(set(chips)):
            raise ValueError("Given list must not contain duplicate values")

        valid_index_range = self.chip_range
        max_chip_index = valid_index_range[-1]

        mask_hex = 0
        for chip_index in chips:
            if chip_index not in valid_index_range:
                raise ValueError("Invalid index given, must be in " +
                                 str(valid_index_range))
            else:
                mask_hex += 2**(max_chip_index - chip_index)

        return str(hex(mask_hex))

    @staticmethod
    def _send_command(command, **cmd_kwargs):
        """Send a command line call and handle any subprocess.CallProcessError.

        Will catch any exception and log the error message. If successful, just
        returns.

        Args:
            command(list(str)): List of arguments to send to command line call

        """
        logging.debug("Sending command: " + "'{}'".format(" ".join(command)) +
                      " with kwargs " + str(cmd_kwargs))

        try:
            output = subprocess.check_output(command, **cmd_kwargs)
        except subprocess.CalledProcessError as error:
            logging.debug(error.output)
        else:
            logging.debug(output)

    def set_lv_state(self, lv_state):
        """Set LV to given state; 0 - Off, 1 - On."""
        if lv_state not in [0, 1]:
            raise ValueError("LV can only be on (0) or off (1), got " +
                             lv_state)

        chips = range(8)
        command = self._construct_command(chips, self.LV, str(lv_state))
        self._send_command(command)

    def set_hv_state(self, hv_state):
        """Set HV to given state; 0 - Off, 1 - On."""
        if hv_state not in [0, 1]:
            raise ValueError("HV can only be on (0) or off (1), got " +
                             hv_state)
        chips = range(8)
        command = self._construct_command(chips, self.HV, str(hv_state))
        self._send_command(command)

    def set_hv_bias(self, hv_bias):
        """Set HV bias to given value."""
        if hv_bias < 0 or hv_bias > 120:
            raise ValueError("HV bias must be between 0 and 120 volts, got " +
                             hv_bias)
        chips = range(8)
        command = self._construct_command(chips, self.HV_BIAS, str(hv_bias))
        self._send_command(command)

    def acquire(self, chips, frames, acq_time, burst=None, pixel_mode=None,
                disc_mode=None, depth=None, counter=None, equalization=None,
                gain_mode=None, read_mode=None, trig_mode=None, tp_count=None,
                path=None, hdf_file=None):
        """Construct and send an acquire command.

        excaliburTestApp default values are marked with *

        Args:
            chips(list(int)): Chips to enable for command process
            frames: Number of frames to acquire
            acq_time: Exposure time for each frame
            burst: Enable burst mode capture
            pixel_mode: Pixel mode (SPM*, CSM = 0*, 1)
            disc_mode: Discriminator mode (DiscL*, DiscH = 0*, 1)
            depth: Counter depth (1, 6, 12*, 24)
            counter: Counter to read (0* or 1)
            equalization: Enable equalization (0*, 1 = off*, on)
            gain_mode: Gain mode (SHGM*, HGM, LGM, SLGM = 0*, 1, 2, 3)
            read_mode: Readout mode (0*, 1 = sequential, continuous)
            trig_mode: Trigger mode (internal*, shutter, sync = 0*, 1, 2
            tp_count: Set test pulse count (0*)
            path: Path to image folder (/tmp*)
            hdf_file: Name of file to save (excalibur-YYMMDD-HHMMSS*)

        Returns:
            list(str): Full acquire command to send to subprocess call

        """
        if burst is not None and burst:
            extra_params = [self.BURST]
        else:
            extra_params = [self.ACQUIRE]
        extra_params.extend([self.NUM_FRAMES, str(frames),
                             self.ACQ_TIME, str(acq_time)])

        # Add any optional parameters if they are provided
        # TODO: Are any combinations invalid?
        if self._check_argument_valid("Pixel mode", pixel_mode,
                                      self.mode_code.keys()):
            extra_params.extend([self.PIXEL_MODE, self.mode_code[pixel_mode]])
        if self._check_argument_valid("Discriminator mode", disc_mode,
                                      self.disc_code.keys()):
            extra_params.extend([self.DISC_MODE, self.disc_code[disc_mode]])
        if self._check_argument_valid("Depth", depth, [1, 6, 12, 24]):
            extra_params.extend([self.DEPTH, str(depth)])
        if self._check_argument_valid("Counter", counter, [0, 1]):
            extra_params.extend([self.COUNTER, str(counter)])
        if self._check_argument_valid("Equalization", equalization, [0, 1]):
            extra_params.extend([self.EQUALIZATION, str(equalization)])
        if self._check_argument_valid("Gain Mode", gain_mode,
                                      self.gain_code.keys()):
            extra_params.extend([self.GAIN_MODE, self.gain_code[gain_mode]])
        if self._check_argument_valid("Readout mode", read_mode,
                                      self.read_code.keys()):
            extra_params.extend([self.READ_MODE, self.read_code[read_mode]])
        if self._check_argument_valid("Trigger mode", trig_mode, [0, 1, 2]):
            extra_params.extend([self.TRIG_MODE, str(trig_mode)])

        if tp_count is not None:
            # TODO: What are the valid values for this?
            extra_params.extend([self.TP_COUNT, str(tp_count)])
        if path is not None:
            extra_params.extend([self.PATH, str(path)])
        if hdf_file is not None:
            if path is None:
                path = "/tmp"
            # TODO: Using join() assumes ETA is smart also?
            full_path = posixpath.join(path, hdf_file)
            if os.path.isfile(full_path):
                raise IOError("File already exists")
            else:
                extra_params.extend([self.HDF_FILE, str(hdf_file)])

        command = self._construct_command(chips, *extra_params)
        self._send_command(command)

    @staticmethod
    def _check_argument_valid(name, value, valid_values):
        if value is None:
            return False
        elif value not in valid_values:
            raise ValueError("{argument} can only be {valid_values}, "
                             "got {value}".format(argument=name,
                                                  valid_values=valid_values,
                                                  value=value))
        else:
            return True

    def sense(self, chips, dac, dac_file):
        """Read the given DAC analogue voltage.

        Args:
            chips: Chips to read for
            dac: Name of DAC to read
            dac_file: File to load DAC values from

        """
        # TODO: Check is command_2 'Requires DAC LOAD to take effect'?
        # Set up DAC for sensing
        command_1 = self._construct_command(chips,
                                            self.SENSE, self.dac_code[dac],
                                            self.DAC_FILE, dac_file)
        self._send_command(command_1)

        time.sleep(1)  # TODO: Test and see if this is really necessary

        # Read back params
        command_2 = self._construct_command(chips,
                                            self.SENSE, self.dac_code[dac],
                                            self.READ_SLOW_PARAMS)
        self._send_command(command_2)

    def perform_dac_scan(self, chips, dac, scan_range, dac_file,
                         hdf_file):
        """Execute a DAC scan and save the results to the given file.

        Args:
            chips: Chips to scan
            dac: Name of DAC to scan
            scan_range(Range): Start, stop and step of scan
            dac_file: File to load config from
            hdf_file: File to save to

        """
        scan_command = "{dac},{start},{stop},{step}".format(
            dac=self.dac_code[dac],
            start=scan_range.start, stop=scan_range.stop, step=scan_range.step)

        command = self._construct_command(chips,
                                          self.DAC_FILE, dac_file,
                                          self.SCAN, scan_command,
                                          self.HDF_FILE, hdf_file)
        self._send_command(command)

    def read_chip_ids(self, chips=range(8), **cmd_kwargs):
        """Read and display MPX3 eFuse IDs for the given chips.

        Args:
            chips(list(int): Chips to read

        """
        command = self._construct_command(chips,
                                          self.RESET,
                                          self.READ_EFUSE)
        self._send_command(command, **cmd_kwargs)

    def read_slow_control_parameters(self, **cmd_kwargs):
        """Read and display slow control parameters for the given chips.

        These are Temperature, Humidity, FEM Voltage Regulator Status
        and DAC Out

        """
        command = self._construct_command(self.chip_range,
                                          self.READ_SLOW_PARAMS)
        self._send_command(command, **cmd_kwargs)

    def load_dacs(self, chips, dac_file):
        """Read DAC values from the given file and set them on the given chips.

        Args:
            chips(list(int): List of chips to assign DAC for
            dac_file(str): Path to file containing DAC values

        """
        command = self._construct_command(chips,
                                          self.DAC_FILE, dac_file)
        self._send_command(command)

    def configure_test_pulse(self, chips, dac_file, tp_mask,
                             config_files=None):
        """Load DAC file and test mask ready acquire test.

        Args:
            chips: List of chips to configure
            dac_file: Path to file containing DAC values
            tp_mask: Test pulse mask to load for test pulses
            config_files(dict): Config files for discL, discH and pixel mask

        """
        # TODO: Check if this really needs to be coupled to loading DACs
        extra_params = []
        if config_files is not None:
            extra_params.extend([self.DISC_L, config_files['discl'],
                                 self.DISC_H, config_files['disch'],
                                 self.PIXEL_MASK, config_files['pixelmask']])
        command = self._construct_command(chips,
                                          self.DAC_FILE, dac_file,
                                          self.TP_MASK, tp_mask,
                                          *extra_params)
        self._send_command(command)

    def load_config(self, chips, discl, disch=None, pixelmask=None):
        """Read the given config files and load them onto the given chips.

        Args:
            chips(list(int)): Chips to load config for
            discl(str): File path for discl config
            disch(str): File path for disch config
            pixelmask(str): File path for pixelmask config

        """
        extra_parameters = [self.CONFIG]

        if os.path.isfile(discl):
            extra_parameters.extend([self.DISC_L, discl])

            if disch is not None and os.path.isfile(disch):
                extra_parameters.extend([self.DISC_H, disch])

            if pixelmask is not None and os.path.isfile(pixelmask):
                extra_parameters.extend([self.PIXEL_MASK, pixelmask])

            command = self._construct_command(chips, *extra_parameters)
            self._send_command(command)
        else:
            print(str(discl) + " does not exist !")
