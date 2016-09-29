"""Python class to provide DAWN plotting, io and analysis to ExcaliburRX"""
import math as m
import logging

import numpy as np
from scipy.optimize import curve_fit

import scisoftpy

logging.basicConfig(level=logging.DEBUG)


class ExcaliburDAWN(object):

    """An interface to DAWN Scisoftpy utilities."""

    def __init__(self):
        self.plot = scisoftpy.plot
        self.io = scisoftpy.io

    def plot_image(self, data_set, name):
        """Plot the given data set as a 2D image.

        Args:
            data_set: 2D numpy array
            name: Name for plot

        """
        self.plot.image(data_set, name=name)

    def load_image(self, path):
        """Load image data in given file into a numpy array.

        Args:
            path: Path to file to load from

        Returns:
            Numpy array containing image data

        """
        return self.io.load(path).image[...]

    def load_image_data(self, path):
        """Load and squeeze an image or set of images from the given file.

        Args:
            path: Image to load

        """
        image_raw = self.load_image(path)
        image = scisoftpy.squeeze(image_raw.astype(np.int))

        return image

    def clear_plot(self, name):
        """Clear given plot.

        Args:
            name: Name of plot to clear

        """
        self.plot.clear(name)

    def plot_linear_fit(self, x_data, y_data, estimate, name="Linear", clear=False):
        """Plot the given 2D data with a linear least squares fit.

        Args:
            x_data: Independent variable data
            y_data: Dependent variable data
            estimate: Starting estimate for offset and gain
            name: Name of plot
            clear: Option to clear given plot before adding new one

        Returns:
            Optimal offset and gain values for least squares fit

        """
        fit_plot_name = '{} fits'.format(name)

        if clear:
            self.clear_plot(name)
            self.clear_plot(fit_plot_name)

        self.plot.addline(x_data, y_data, name=name)

        popt, _ = curve_fit(self.lin_function, x_data, y_data, estimate)
        offset = popt[0]
        gain = popt[1]

        self.plot.addline(x_data, self.lin_function(x_data, offset, gain),
                          name=fit_plot_name)

        return offset, gain

    def plot_histogram(self, chips, image_data, name="Histogram"):
        """Plot a histogram for each of the given chips.

        Args:
            chips: Chips to plot for
            image_data: Data for full array
            name: Name of plot

        """
        for chip_idx in chips:
            histogram = np.histogram(
                image_data[0:256, chip_idx*256:(chip_idx + 1)*256])
            self.plot.addline(histogram[1][0:-1],  histogram[0],
                              name=name)

    def fit_dac_scan(self, chips, chip_dac_scan, dac_axis):
        """############## NOT TESTED"""
        parameters_estimate = [100, 0.8, 3]
        for chip in chips:
            # dnp.plot.addline(dacAxis, chipDacScan[chip,:])
            popt, _ = curve_fit(self.myerf, dac_axis, chip_dac_scan[chip, :],
                                p0=parameters_estimate)
            # popt, pcov = curve_fit(s_curve_function, dacAxis,
            #                        chipDacScan[chip, :], p0)
            self.plot.addline(dac_axis, self.myerf(dac_axis,
                                                   popt[0], popt[1], popt[2]))

        return chip_dac_scan, dac_axis

    def plot_dac_scan(self, chips, dac_scan_data, dac_range):
        """Plot the results of threshold dac scan.

        Display in an integrated spectrum plot window (dac scan) and a
        differential spectrum (spectrum)

        Args:
            chips: Chips to plot for
            dac_scan_data: Data from dac scan to plot
            dac_range: Scan range used for dac scan

        Returns:
            np.array, list: Averaged scan data, DAC values of scan
        """
        self.clear_plot("DAC Scan")
        self.clear_plot("Spectrum")

        # TODO: Refactor to use Range()
        if dac_range[0] > dac_range[1]:
            # TODO: Remove brackets if unnecessary
            dac_axis = (np.array(range(dac_range[0],
                                       dac_range[1] - dac_range[2],
                                       -dac_range[2])))
        else:
            dac_axis = (np.array(range(dac_range[0],
                                       dac_range[1] + dac_range[2],
                                       dac_range[2])))

        chip_dac_scan = np.zeros([8])
        for chip in chips:
            # TODO: Should this be reset every loop?
            chip_dac_scan = np.zeros([8, dac_axis.size])

            # Store mean chip for each dac scan value
            chip_dac_scan[chip, :] = (dac_scan_data[:, 0:256,
                                      chip*256:chip*256 + 256].mean(2).mean(1))

            self.plot.addline(
                np.array(dac_axis),
                np.squeeze(dac_scan_data[:, 0:256,
                           chip*256:chip*256 + 256].mean(2).mean(1)),
                name="DAC Scan")

            spectrum = -np.diff(
                np.squeeze(dac_scan_data[:, 0:256,
                           chip*256:chip*256 + 256].mean(2).mean(1)))

            self.plot.addline(
                np.array(range(dac_range[0], dac_range[1], dac_range[2]))[1:],
                spectrum[1:],
                name="Spectrum")

        return chip_dac_scan, dac_axis

    def show_pixel(self, dac_scan_data, dac_range, pixel):
        """Plot dac scan for an individual pixel.

        Args:
            dac_scan_data: Data from dac scan to plot
            dac_range: Scan range used for dac scan
            pixel: X, Y coordinates for pixel

        """
        # TODO: Combine with plot_dac_scan if possible
        self.plot.addline(
            np.array(
                range(dac_range[0], dac_range[1] + dac_range[2],
                      dac_range[2])),
            (dac_scan_data[:, pixel[0], pixel[1]]),
            name='Pixel S curve')

        # Squeeze - Remove scalar dimensions: [[1], [2], [3]] -> [1, 2, 3]
        # Diff - Take difference of each pair along axis: [1, 5, 3] -> [4, -2]
        spectrum = -np.diff(np.squeeze(dac_scan_data[:, pixel[0], pixel[1]]))

        self.plot.addline(
            np.array(range(dac_range[0], dac_range[1], dac_range[2])),
            spectrum, name="Pixel Spectrum")

    @staticmethod
    def myerf(x_val, a, mu, sigma):
        """Function required to express S-curve."""

        return a/2. * (1 + m.erf((x_val - mu) / (m.sqrt(2) * sigma)))

    @staticmethod
    def lin_function(x_val, offset, gain):
        """Function definition for linear fits."""

        return offset + gain * x_val

    @staticmethod
    def gauss_function(x_val, a, x0, sigma):
        """Function definition for Gaussian fits."""

        return a * np.exp(-(x_val - x0) ** 2 / (2 * sigma ** 2))

    @classmethod
    def s_curve_function(cls, x_val, k, delta, e, sigma):
        """Function required to fit integral spectra.

        Used during threshold calibration
        """
        erf = cls.myerf(x_val, k, e, sigma)

        return k * ((1 - 2 * delta * (x_val / e - 0.5)) ** 2) * (1 - erf)
