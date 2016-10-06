"""Utility functions for slicing numpy arrays"""


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
