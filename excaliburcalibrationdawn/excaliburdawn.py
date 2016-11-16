"""Python class to provide DAWN plotting, io and analysis to ExcaliburNode."""
import math as m
import logging

import numpy as np
from scipy.optimize import curve_fit
import scisoftpy

logging.basicConfig(level=logging.DEBUG)


class ExcaliburDAWN(object):

    """An interface to DAWN Scisoftpy utilities."""

    def __init__(self):
        """Set up plot and io APIs."""
        self.plot = scisoftpy.plot
        self.io = scisoftpy.io

    def plot_image(self, data_set, name):
        """Plot the given data set as a 2D image.

        Args:
            data_set(numpy.array): 2D numpy array
            name(str): Name for plot

        """
        self.plot.image(data_set, name=name)
        logging.info("Image plotted in DAWN as '%s'", name)

    def load_image(self, path):
        """Load image data in given file into a numpy array.

        Args:
            path(str): Path to file to load from

        Returns:
            numpy.array: Image data

        """
        logging.info("Loading HDF5 file; %s", path)
        return self.io.load(path).image[...]

    def load_image_data(self, path):
        """Load and squeeze an image or set of images from the given file.

        Args:
            path(str): Image to load

        """
        image_raw = self.load_image(path)
        image = image_raw.squeeze()
        return image

    def clear_plot(self, name):
        """Clear given plot.

        Args:
            name(str): Name of plot to clear

        """
        logging.debug("Clearing plot '%s'", name)
        self.plot.clear(name)

    def plot_linear_fit(self, x_data, y_data, estimate, x_name, y_name, label,
                        name=None, fit_name=None):
        """Fit the given 2D data with a linear least squares fit.

        Args:
            x_data(list/np.array): Independent variable data
            y_data(list/np.array): Dependent variable data
            x_name(str): Label for x-axis
            y_name(str): Label for y-axis
            estimate(list(int/float): Starting estimate for offset and gain
            label(str): Label for plot line added to any plots
            name(str): Name of plot
            fit_name(str): Name of fit plot

        Returns:
            Optimal offset and gain values for least squares fit

        """
        logging.info("Performing linear fit")

        if name is not None:
            self.clear_plot(name)
            self.add_plot_line(x_data, y_data, x_name, y_name, name, label)

        popt, _ = curve_fit(self.lin_function, x_data, y_data, estimate)
        offset = popt[0]
        gain = popt[1]

        if fit_name is not None:
            self.clear_plot(fit_name)
            self.add_plot_line(x_data, self.lin_function(x_data, offset, gain),
                               x_name, y_name, fit_name, label)

        return offset, gain

    def add_plot_line(self, x, y, x_name, y_name, plot_name, label):
        """Add a plot of x vs y to the given plot.

        Args:
            x(list/np.array): X axis data
            y(list/np.array): Y axis data
            x_name(str): Label for x-axis
            y_name(str): Label for y-axis
            plot_name(str): Name of plot to add to
            label(str): Label for plot line

        """
        self.plot.addline({x_name: x}, [{y_name: (y, label)}],
                          name=plot_name, title=plot_name)

    def plot_gaussian_fit(self, scan_data, plot_name, p0, bins):
        """Calculate the Gaussian least squares fit and plot the result.

        Args:
            scan_data(numpy.array): Data to fit
            plot_name(str): Name of resulting plot
            p0(list(int)): Initial guess for gaussian curve parameters
            bins(int): Bins to plot in histogram

        """
        logging.info("Performing Gaussian fit")
        fit_plot_name = plot_name + " (fitted)"
        a = np.zeros([8])
        x0 = np.zeros([8])
        sigma = np.zeros([8])

        self.clear_plot(plot_name)
        self.clear_plot(fit_plot_name)
        for chip_idx, chip_data in enumerate(scan_data):
            if chip_data is not None:
                bin_counts, bin_edges = np.histogram(chip_data, bins=bins)

                self.add_plot_line(bin_edges[0:-1], bin_counts,
                                   "Disc Value", "Bin Count", plot_name,
                                   label="Chip {}".format(chip_idx))
                popt, _ = curve_fit(self.gauss_function, bin_edges[0:-2],
                                    bin_counts[0:-1], p0)

                a[chip_idx] = popt[0]
                x0[chip_idx] = popt[1]
                sigma[chip_idx] = popt[2]
                self.add_plot_line(bin_edges[0:-1],
                                   self.gauss_function(bin_edges[0:-1],
                                                       a[chip_idx],
                                                       x0[chip_idx],
                                                       sigma[chip_idx]),
                                   "Disc Value", "Bin Count", fit_plot_name,
                                   label="Chip {}".format(chip_idx))

        return x0, sigma

    def plot_histogram(self, image_data, name, x_name):
        """Plot a histogram for each of the given chips.

        Args:
            image_data(numpy.array): Data for full array
            name(str): Name of plot
            x_name(str): Label for x-axis

        """
        self.clear_plot(name)
        for chip_idx, chip_data in enumerate(image_data):
            self._add_histogram(chip_data, name, x_name,
                                "Chip {}".format(chip_idx))

    def plot_histogram_with_mask(self, chips, image_data, mask, name, x_name):
        """Plot a histogram for each of the given chips, after applying a mask.

        Args:
            chips(list(int)): Chips to plot for
            image_data(numpy.array): Data for full array
            mask(numpy.array): Mask to apply before plotting data
            name(str): Name of plot
            x_name(str): Label for x-axis

        """
        self.clear_plot(name)
        for chip_idx in chips:
            chip_mask = mask[0:256, chip_idx * 256:(chip_idx + 1) * 256]
            chip_data = image_data[0:256, chip_idx * 256:(chip_idx + 1) * 256]
            masked_data = chip_data[chip_mask.astype(bool)]
            self._add_histogram(masked_data, name, x_name,
                                label="Chip {}".format(chip_idx))

    def _add_histogram(self, data, name, x_name, label, bins=10):
        """Add a histogram of data to the given plot name.

        Args:
            data(numpy.array): Data to plot
            name(str): Name of plot to add to
            x_name(str): Label for x-axis
            label(str): Label for plot line
            bins(int): Bins to plot over

        """
        histogram = np.histogram(data, bins=bins)
        self.add_plot_line(histogram[1][0:-1], histogram[0],
                           x_name, "Bin Counts", name, label)

    def fit_dac_scan(self, scan_data, dac_axis):
        """############## NOT TESTED."""
        parameters_estimate = [100, 0.8, 3]
        for chip_data in scan_data:
            # dnp.plot.addline(dacAxis, chipDacScan[chip,:])
            popt, _ = curve_fit(self.myerf, dac_axis, chip_data,
                                p0=parameters_estimate)
            # popt, pcov = curve_fit(s_curve_function, dacAxis,
            #                        chipDacScan[chip, :], p0)
            self.plot.addline(dac_axis, self.myerf(dac_axis,
                                                   popt[0], popt[1], popt[2]))

        return dac_axis

    def plot_dac_scan(self, scan_data, dac_axis):
        """Plot the results of threshold dac scan.

        Displays an integral plot (DAC Scan) and a differential plot
        (DAC Scan Differential)

        Args:
            scan_data(list(numpy.array)): Data from dac scan to plot
            dac_axis(list(int)): X-axis data for plots

        Returns:
            numpy.array: Averaged scan data

        """
        plot_name = "DAC Scan"
        diff_plot_name = "DAC Scan Differential"

        self.clear_plot(plot_name)
        self.clear_plot(diff_plot_name)

        x_axis = np.array(dac_axis)
        for chip_idx, chip_data in enumerate(scan_data):
            self.add_plot_line(x_axis, chip_data,
                               "DAC Value", "Mean Counts", plot_name,
                               label="Chip {}".format(chip_idx))
            spectrum = -np.diff(chip_data)
            self.add_plot_line(x_axis[1:-1], spectrum[1:],
                               "DAC Value", "Mean Counts", diff_plot_name,
                               label="Chip {}".format(chip_idx))

    def show_pixel(self, dac_scan_data, dac_range, pixel):
        """Plot dac scan for an individual pixel.

        Args:
            dac_scan_data(numpy.array): Data from dac scan to plot
            dac_range(list(int)): Scan range used for dac scan
            pixel(list(int)): X, Y coordinates for pixel

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
        return a / 2. * (1 + m.erf((x_val - mu) / (m.sqrt(2) * sigma)))

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
        """Function required to fit integral spectra."""
        erf = cls.myerf(x_val, k, e, sigma)
        return k * ((1 - 2 * delta * (x_val / e - 0.5))**2) * (1 - erf)
