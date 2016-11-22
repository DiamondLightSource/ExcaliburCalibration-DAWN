"""Utility functions for excaliburcalibrationdawn."""
import os
import time
import filecmp
from datetime import datetime
from threading import Thread

import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG)

CHIP_SIZE = 256  # The width and height of a detector chip


def grab_slice(array, start, stop):
    """Grab a section of a 2D numpy array.

    Args:
        array: Array to grab from
        start: Start coordinate of sub array
        stop: Stop coordinate of sub array

    Returns:
        numpy.array: Sub array

    """
    return array[start[0]:stop[0] + 1, start[1]:stop[1] + 1]


def set_slice(array, start, stop, value):
    """Set a section in a 2D numpy array.

    Args:
        array: Array to grab from
        start: Start coordinate of sub array
        stop: Stop coordinate of sub array
        value: Value to set slice to (array of same shape or single value)

    Returns:
        numpy.array: Sub array

    """
    array[start[0]:stop[0] + 1, start[1]:stop[1] + 1] = value


def grab_chip_slice(array, chip_idx):
    """Grab a chip from a full array.

    Args:
        array(numpy.array): Array to grab from
        chip_idx(int): Index of section of array to grab

    Returns:
        numpy.array: Sub array

    """
    start, stop = generate_chip_range(chip_idx)
    return grab_slice(array, start, stop)


def set_chip_slice(array, chip_idx, value):
    """Grab a section of a 2D numpy array.

    Args:
        array(numpy.array): Array to grab from
        chip_idx(int): Index of section of array to grab
        value(numpy.array/int/float): Value to set slice to

    """
    start, stop = generate_chip_range(chip_idx)
    set_slice(array, start, stop, value)


def generate_chip_range(chip_idx):
    """Calculate start and stop coordinates of given chip.

    Args:
        chip_idx(int): Chip to calculate range for

    """
    start = [0, chip_idx * CHIP_SIZE]
    stop = [CHIP_SIZE - 1, (chip_idx + 1) * CHIP_SIZE - 1]
    return start, stop


def rotate_array(config_file):
    """Rotate array in given file 180 degrees.

    Args:
        config_file: Config file to rotate

    """
    # shutil.copy(config_file, config_file + ".backup")
    config_bits = np.loadtxt(config_file)
    np.savetxt(config_file, np.rot90(config_bits, 2), fmt='%.18g',
               delimiter=' ')


def get_time_stamp():
    """Get a time stamp."""
    iso = datetime.now().isoformat(sep="~")  # Get ISO 8601 time stamp
    iso = iso.replace(":", "").replace("-", "")  # Remove date and time seps
    time_stamp = iso.split(".")[0]  # Remove milliseconds

    return time_stamp


def generate_file_name(base_name):
    """Generate file name with a time stamp from a given base_name.

    Args:
        base_name(str): Base file name - e.g. Image, DAC Scan

    Returns:
        str: New file name

    """
    return "{tag}_{base_name}.hdf5".format(base_name=base_name,
                                           tag=get_time_stamp())


def generate_plot_name(base_name):
    """Generate plot name with a time stamp from a given base_name.

    Args:
        base_name(str): Base plot name - e.g. Image, DAC Scan

    Returns:
        str: New plot name

    """
    return "{base_name} - {tag}".format(base_name=base_name,
                                        tag=get_time_stamp())


def to_list(value):
    """Return a list of value, or value if already a list.

    Args:
        value: List or variable to put in list

    """
    if isinstance(value, list):
        return value
    else:
        return [value]


def wait_for_file(file_path, wait_time):
    """
    Wait for a file to appear on the file system.

    Args:
        file_path: Path to file to check for
        wait_time: Time to wait before

    """
    logging.info("Waiting up to %s seconds for file %s", wait_time, file_path)
    loop_time = 0.1
    loops = wait_time / loop_time

    loop = 0
    while loop < loops:
        time.sleep(0.1)
        if os.path.isfile(file_path):
            logging.info("File appeared!")
            return True
        loop += 1
        if loop % 10 == 0 and loop > 0:
            logging.info("%s seconds", loop / 10)

    logging.info("File didn't appear within given time.")
    return False


def files_match(file1, file2):
    """Check if two files are identical.

    Args:
        file1(str): Path to first file
        file2(str): Path to second file

    Returns:
        bool: True if the same, else False

    """
    return filecmp.cmp(file1, file2)


def spawn_thread(function, *args, **kwargs):
    """Spawn a worker thread to call the given function.

    Args:
        function: Function to call
        *args: Arguments for function call
        **kwargs: Keyword arguments fro function call

    Returns:
        Thread: Worker thread calling function
    """
    thread = _ReturnThread(target=function, args=args, kwargs=kwargs)
    thread.start()
    return thread


class _ReturnThread(Thread):
    """A Thread with a return value."""

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(_ReturnThread, self).__init__(group, target, name, args, kwargs,
                                            verbose)
        self._return = None

    def run(self):
        """Override default run to store return value."""
        if self._Thread__target is not None:
            self._return = self._Thread__target(*self._Thread__args,
                                                **self._Thread__kwargs)

    def join(self, timeout=None):
        """Override join to return stored return value from run."""
        super(_ReturnThread, self).join(timeout)
        return self._return
