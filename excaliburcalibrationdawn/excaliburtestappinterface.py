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

import util

import logging
logging.basicConfig(level=logging.DEBUG)


class ExcaliburTestAppInterface(object):

    """A class to make subprocess calls to the excaliburTestApp tool."""

    # ExcaliburTestApp flags & example usage
    IP_ADDRESS = "-i"  # -i 192.168.0.101
    PORT = "-p"  # -p 6969
    MASK = "-m"  # -m 0xff
    RESET = "-r"
    LV = "--lvenable"  # --lvenable 1
    HV = "--hvenable"  # --hvenable 1
    HV_BIAS = "--hvbias"  # --hvbias 120
    READ_EFUSE = "-e"
    DAC_FILE = "--dacs="  # --dacs=default_dac_values.txt
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
    PATH = "--path="  # --path=/scratch/excalibur_images
    HDF_FILE = "--hdffile="  # --hdffile=image_1.hdf5
    TP_COUNT = "--tpcount"  # --tpcount 100
    CONFIG = "--config"
    PIXEL_MASK = "--pixelmask="  # --pixelmask mask.txt
    DISC_L = "--discl="  # --discl=default_disc_L.txt
    DISC_H = "--disch="  # --disch=default_disc_H.txt
    TP_MASK = "--tpmask="  # --tpmask=logo_mask.txt

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

    def __init__(self, node, ip_address, port, server_name=None):
        self.node = node
        self.server_path = "{}.diamond.ac.uk".format(server_name)
        self.ip_address = ip_address
        self.port = str(port)

        self.path = "/dls/detectors/support/silicon_pixels/excaliburRX/" \
                    "TestApplication_15012015/excaliburTestApp"

        self.base_cmd = []
        if server_name is not None:
            self.base_cmd.extend(["ssh", self.server_path])
        self.base_cmd.extend([self.path,
                              self.IP_ADDRESS, self.ip_address,
                              self.PORT, self.port])

        self.lv = 0
        self.hv = 0
        self.hv_bias = 0
        self.dacs_loaded = None
        self.initialised = False

        self.quiet = True  # Flag to stop printing of terminal output
        logging.info("Set self.quiet to False to display terminal output.")

    def _construct_command(self, chips, *cmd_args):
        """Construct a command from the base_cmd, given chips and any cmd_args.

        Args:
            chips(list(int): Chip(s) to enable for command process
            *cmd_args(list(str)): Arguments defining the process to be called

        Returns:
            list(str)): Full command to send to subprocess call

        """
        return self.base_cmd + [self.MASK, self._mask(chips)] + list(cmd_args)

    def _mask(self, chips):
        """Create a hexadecimal mask to activate the given chip(s).

        Args:
            chips(list(int)): Chip(s) to be enabled

        Returns:
            str: Hexadecimal mask representing list of chips

        """
        chips = util.to_list(chips)
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

    def _send_command(self, command, loud_call=False, **cmd_kwargs):
        """Send a command line call and handle any subprocess.CallProcessError.

        Will catch any exception and log the error message. If successful, just
        returns.

        Args:
            command(list(str)): List of arguments to send to command line call

        Returns:
            bool: Whether command was successful

        """
        logging.debug("Sending Command:\n'%s' with kwargs %s",
                      " ".join(command), str(cmd_kwargs))

        try:
            if self.quiet and not loud_call:
                subprocess.check_output(command, **cmd_kwargs)
            else:
                subprocess.check_call(command, **cmd_kwargs)
        except subprocess.CalledProcessError as error:
            logging.debug("Error Output:\n%s", error.output)
            if self.quiet:
                logging.info("Set self.quiet to False to display terminal "
                             "output.")
            return False

        return True

    def set_lv_state(self, lv_state):
        """Set LV to given state.

        Args:
            lv_state(int): State to set (0 - Off, 1 - On)

        """
        logging.debug("Setting LV to %s", lv_state)
        if lv_state not in [0, 1]:
            raise ValueError("LV can only be on (0) or off (1), got "
                             "{value}".format(value=lv_state))

        chips = range(8)
        command = self._construct_command(chips, self.LV, str(lv_state))
        success = self._send_command(command)
        if success:
            self.lv = lv_state

    def set_hv_state(self, hv_state):
        """Set HV to given state; 0 - Off, 1 - On.

        Args:
            hv_state(int): State to set (0 - Off, 1 - On)

        """
        logging.debug("Setting HV to %s", hv_state)
        if hv_state not in [0, 1]:
            raise ValueError("HV can only be on (0) or off (1), got "
                             "{value}".format(value=hv_state))
        chips = range(8)
        command = self._construct_command(chips, self.HV, str(hv_state))
        success = self._send_command(command)
        if success:
            self.hv = hv_state

    def set_hv_bias(self, hv_bias):
        """Set HV bias to given value.

        Args:
            hv_bias(int): Voltage to set

        """
        logging.debug("Setting HV bias to %s", hv_bias)
        if hv_bias < 0 or hv_bias > 120:
            raise ValueError("HV bias must be between 0 and 120 volts, got "
                             "{value}".format(value=hv_bias))
        chips = range(8)
        command = self._construct_command(chips, self.HV_BIAS, str(hv_bias))
        success = self._send_command(command)
        if success:
            self.hv_bias = hv_bias

    def acquire(self, chips, frames, acq_time, burst=None, pixel_mode=None,
                disc_mode=None, depth=None, counter=None, equalization=None,
                gain_mode=None, read_mode=None, trig_mode=None, tp_count=None,
                path=None, hdf_file=None):
        """Construct and send an acquire command.

        excaliburTestApp default values are marked with *

        Args:
            chips(list(int)): Chips to enable for command process
            frames(int): Number of frames to acquire
            acq_time(int): Exposure time for each frame
            burst(bool): Enable burst mode capture
            pixel_mode(str): Pixel mode (SPM*, CSM = 0*, 1)
            disc_mode(int): Discriminator mode (DiscL*, DiscH = 0*, 1)
            depth(int): Counter depth (1, 6, 12*, 24)
            counter(int): Counter to read (0* or 1)
            equalization(int): Enable equalization (0*, 1 = off*, on)
            gain_mode(str): Gain mode (SHGM*, HGM, LGM, SLGM = 0*, 1, 2, 3)
            read_mode(str): Readout mode (0*, 1 = sequential, continuous)
            trig_mode(int): Trigger mode (internal*, shutter, sync = 0*, 1, 2)
            tp_count(int): Set test pulse count (0*)
            path(str): Path to image folder (/tmp*)
            hdf_file(str): Name of file to save (excalibur-YYMMDD-HHMMSS*)

        Returns:
            list(str): Full acquire command to send to subprocess call

        """
        logging.debug("Sending acquire command for chips %s", chips)
        # Check detector has been initialised correctly
        if self.dacs_loaded is None:
            raise ValueError("No DAC file loaded to FEM. Call setup().")
        if not self.initialised:
            raise ValueError("FEM has not been initialised. Call setup().")

        extra_params = [self.ACQUIRE,
                        self.NUM_FRAMES, str(frames),
                        self.ACQ_TIME, str(acq_time)]

        # Add any optional parameters if they are provided
        # TODO: Are any combinations invalid?
        if burst is not None and burst:
            extra_params.append(self.BURST)
        if self._arg_valid("Pixel mode", pixel_mode, self.mode_code.keys()):
            extra_params.extend([self.PIXEL_MODE, self.mode_code[pixel_mode]])
        if self._arg_valid("Discriminator mode", disc_mode,
                           self.disc_code.keys()):
            extra_params.extend([self.DISC_MODE, self.disc_code[disc_mode]])
        if self._arg_valid("Depth", depth, [1, 6, 12, 24]):
            extra_params.extend([self.DEPTH, str(depth)])
        if self._arg_valid("Counter", counter, [0, 1]):
            extra_params.extend([self.COUNTER, str(counter)])
        if self._arg_valid("Equalization", equalization, [0, 1]):
            extra_params.extend([self.EQUALIZATION, str(equalization)])
        if self._arg_valid("Gain Mode", gain_mode, self.gain_code.keys()):
            extra_params.extend([self.GAIN_MODE, self.gain_code[gain_mode]])
        if self._arg_valid("Readout mode", read_mode, self.read_code.keys()):
            extra_params.extend([self.READ_MODE, self.read_code[read_mode]])
        if self._arg_valid("Trigger mode", trig_mode, [0, 1, 2]):
            extra_params.extend([self.TRIG_MODE, str(trig_mode)])

        if tp_count is not None:
            # TODO: What are the valid values for this?
            extra_params.extend([self.TP_COUNT, str(tp_count)])
        if path is not None:
            extra_params.extend([self.PATH + str(path)])
        if hdf_file is not None:
            if path is None:
                path = "/tmp"
            full_path = posixpath.join(path, hdf_file)
            if os.path.isfile(full_path):
                raise IOError("File already exists")
            else:
                extra_params.extend([self.HDF_FILE + str(hdf_file)])

        command = self._construct_command(chips, *extra_params)
        self._send_command(command)

    @staticmethod
    def _arg_valid(name, value, valid_values):
        """Check if given argument is not None and is a valid value.

        Args:
            name(str): Name of argument (for error message)
            value(int/str): Value to check
            valid_values(list(int)): Allowed values

        Returns:
            bool: True if valid, False if None
        Raises:
            ValueError: Argument not None, but not valid

        """
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
            chips(list(int)): Chips to read for
            dac(str): Name of DAC to read
            dac_file(str): File to load DAC values from

        """
        logging.debug("Sending sense command for %s on chips %s", dac, chips)

        # Set up DAC for sensing
        command_1 = self._construct_command(chips,
                                            self.SENSE, self.dac_code[dac],
                                            self.DAC_FILE + dac_file)
        self._send_command(command_1)

        # Read back params
        command_2 = self._construct_command(chips,
                                            self.SENSE, self.dac_code[dac],
                                            self.READ_SLOW_PARAMS)
        self._send_command(command_2, loud_call=True)

    def perform_dac_scan(self, chips, threshold, scan_range, dac_file,
                         path, hdf_file):
        """Execute a DAC scan and save the results to the given file.

        Args:
            chips(list(int)): Chips to scan
            threshold(str): Threshold to scan
            scan_range(Range): Start, stop and step of scan
            dac_file(str): File to load config from
            path(str): Folder to save into
            hdf_file(str): File to save to

        """
        logging.debug("Sending DAC scan command")
        scan_command = "{dac},{start},{stop},{step}".format(
            dac=int(self.dac_code[threshold]) - 1,
            start=scan_range.start, stop=scan_range.stop, step=scan_range.step)

        command = self._construct_command(chips,
                                          self.DAC_FILE + dac_file,
                                          self.ACQ_TIME, "5",
                                          self.SCAN, scan_command,
                                          self.PATH + path,
                                          self.HDF_FILE + hdf_file)
        self._send_command(command)

    def read_chip_ids(self, chips=range(8), **cmd_kwargs):
        """Read and display MPX3 eFuse IDs for the given chips.

        Args:
            chips(list(int): Chips to read

        """
        logging.debug("Sending read chip IDs command")
        command = self._construct_command(chips,
                                          self.RESET,
                                          self.READ_EFUSE)
        success = self._send_command(command, loud_call=True, **cmd_kwargs)
        if success and not self.initialised:
            self.initialised = True

    def read_slow_control_parameters(self, **cmd_kwargs):
        """Read and display slow control parameters for the given chips.

        These are Temperature, Humidity, FEM Voltage Regulator Status
        and DAC Out

        """
        logging.debug("Sending read slow command")
        command = self._construct_command(self.chip_range,
                                          self.READ_SLOW_PARAMS)
        self._send_command(command, loud_call=True, **cmd_kwargs)

    def load_dacs(self, chips, dac_file):
        """Read DAC values from the given file and set them on the given chips.

        Args:
            chips(list(int): List of chips to assign DAC for
            dac_file(str): Path to file containing DAC values

        """
        logging.debug("Sending load DACs command for chips %s", chips)
        command = self._construct_command(chips, self.DAC_FILE + dac_file)
        success = self._send_command(command)
        if success:
            self.dacs_loaded = dac_file.split('/')[-1]

    def configure_test_pulse(self, chips, tp_mask, dac_file,
                             config_files=None):
        """Load DAC file and test mask ready acquire test.

        Args:
            chips(list(int)): List of chips to configure
            tp_mask(numpy.array): Test pulse mask to load for test pulses
            dac_file(str): Path to file containing DAC values
            config_files(dict): Config files for discL, discH and pixel mask

        """
        logging.debug("Sending configure test pulse command for chips %s",
                      chips)
        # TODO: Check if this really needs to be coupled to loading DACs
        extra_params = []
        if config_files is not None:
            extra_params.extend([self.DISC_L + config_files['discL'],
                                 self.DISC_H + config_files['discH'],
                                 self.PIXEL_MASK + config_files['pixel_mask']])
        command = self._construct_command(chips,
                                          self.DAC_FILE + dac_file,
                                          self.TP_MASK + tp_mask,
                                          *extra_params)
        self._send_command(command)

    def load_tp_mask(self, chips, tp_mask):
        """Load the given mask onto the given chips.

        Args:
            chips(numpy.array): Chips to load for
            tp_mask(str): Path to tp_mask file

        """
        logging.debug("Sending load TP mask command for chips %s", chips)
        command = self._construct_command(chips,
                                          self.CONFIG,
                                          self.TP_MASK + tp_mask)
        self._send_command(command)

    def acquire_tp_image(self, chips=range(8), exposure=1000, tp_count=1000,
                         hdf_file="triangle.hdf5", **kwargs):
        """Acquire and plot a test pulse image.

        Args:
            chips(list(int)): Chips to capture for
            tp_count(int): Test pulse count
            exposure(int): Exposure time
            hdf_file(str): Name of image file to save

        """
        self.acquire(chips, 1, exposure, tp_count=tp_count, hdf_file=hdf_file,
                     **kwargs)

    def load_config(self, chips, discl, disch=None, pixelmask=None):
        """Read the given config files and load them onto the given chips.

        Args:
            chips(list(int)): Chip(s) to load config for
            discl(str): File path for discl config
            disch(str): File path for disch config
            pixelmask(str): File path for pixelmask config

        """
        logging.info("Loading config for chip(s) %s", chips)

        extra_parameters = [self.CONFIG]

        # TODO: Why does it always need to load discL, not just pixelmask?

        if os.path.isfile(discl):
            extra_parameters.extend([self.DISC_L + discl])

            if disch is not None and os.path.isfile(disch):
                extra_parameters.extend([self.DISC_H + disch])

            if pixelmask is not None and os.path.isfile(pixelmask):
                extra_parameters.extend([self.PIXEL_MASK + pixelmask])

            command = self._construct_command(chips, *extra_parameters)
            self._send_command(command)
        else:
            print(str(discl) + " does not exist !")

    def grab_remote_file(self, server_source):
        """Use scp to copy the given file from the server to the local host.

        Args:
            server_source(str): File path on server

        Returns:
            str: File path to local copied file

        """
        logging.info("Fetching remote file")
        file_name, extension = posixpath.splitext(server_source)
        full_source = "{server}:{source}".format(server=self.server_path,
                                                 source=server_source)
        new_file = "{base}_fem{node}{ext}".format(base=file_name,
                                                  node=self.node,
                                                  ext=extension)

        command = ["scp", full_source, new_file]
        self._send_command(command)

        return new_file
