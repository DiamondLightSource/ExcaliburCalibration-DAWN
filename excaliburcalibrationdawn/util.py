"""Utility functions for excaliburcalibrationdawn"""
import os
import time
from datetime import datetime
import numpy as np


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


def rotate_config(config_file):
    """Rotate array in given file 180 degrees.

    Args:
        config_file: Config file to rotate

    """
    # shutil.copy(config_file, config_file + ".backup")
    config_bits = np.loadtxt(config_file)
    np.savetxt(config_file, np.rot90(config_bits, 2), fmt='%.18g',
               delimiter=' ')


def get_time_stamp():
    """Get a time stamp"""
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
    loop_time = 0.1
    loops = wait_time / loop_time

    loop = 0
    while loop < loops:
        time.sleep(0.1)
        if os.path.isfile(file_path):
            return True
        loop += 1

    return False
