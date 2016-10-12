import unittest

from pkg_resources import require
require("mock")
from mock import patch, MagicMock, ANY
DAWN_patch_path = "excaliburcalibrationdawn.excaliburdawn.ExcaliburDAWN"

import numpy as np

from excaliburcalibrationdawn.excaliburdawn import ExcaliburDAWN


class FunctionsTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburDAWN

    def test_myerf(self):
        self.assertEqual(1.8904014166008838, self.e.myerf(1, 2, 0.2, 0.5))

    def test_lin_function(self):
        self.assertEqual(110, self.e.lin_function(5, 100, 2))

    def test_gauss_function(self):
        self.assertEqual(0.55607460090638816, self.e.gauss_function(1, 2, 0.2, 0.5))

    def test_s_curve_function(self):
        self.assertEqual(242.0, self.e.s_curve_function(1, 2, 10, 20, 0.5))


class SimpleMethodsTest(unittest.TestCase):

    def setUp(self):
        self.e = ExcaliburDAWN()
        self.chips = [0]

    @patch('scisoftpy.plot.image')
    def test_plot_image(self, plot_mock):
        mock_data = MagicMock()

        self.e.plot_image(mock_data, name="Test Plot")

        plot_mock.assert_called_once_with(mock_data, name="Test Plot")

    @patch('scisoftpy.plot.addline')
    @patch('numpy.histogram')
    def test_plot_histogram(self, histo_mock, addline_mock):
        mock_data = [np.random.randint(10, size=[256, 256])]

        self.e.plot_histogram(mock_data, name="Test Histogram")

        np.testing.assert_array_equal(mock_data[0],
                                      histo_mock.call_args[0][0])
        addline_mock.assert_called_once_with(histo_mock.return_value[1][0:-1],
                                             histo_mock.return_value[0],
                                             name="Test Histogram")

    @patch('scisoftpy.io.load')
    def test_load_image(self, load_mock):

        value = self.e.load_image("test_path")

        self.assertEqual(load_mock.return_value.image[...], value)

    @patch(DAWN_patch_path + '.load_image')
    @patch('scisoftpy.squeeze')
    def test_load_image_data(self, squeeze_mock, load_mock):

        value = self.e.load_image_data("test_path")

        load_mock.assert_called_once_with("test_path")
        squeeze_mock.assert_called_once_with(load_mock.return_value.astype())
        self.assertEqual(squeeze_mock.return_value, value)

    @patch('scisoftpy.plot.clear')
    def test_clear_plot(self, clear_mock):

        self.e.clear_plot("Test Plot")

        clear_mock.assert_called_once_with("Test Plot")

    @patch('scisoftpy.plot.addline')
    def test_add_plot_line(self, add_mock):
        x = MagicMock()
        y = MagicMock()
        name = "Test"

        self.e.add_plot_line(x, y, name)

        add_mock.assert_called_once_with(x, y, name=name)

    @patch('scisoftpy.plot.addline')
    @patch('numpy.histogram')
    def test_add_histogram(self, histo_mock, add_mock):
        x = MagicMock()
        name = "Test"

        self.e._add_histogram(x, name, 5)

        histo_mock.assert_called_once_with(x, bins=5)
        add_mock.assert_called_once_with(histo_mock.return_value[1][0:-1],
                                         histo_mock.return_value[0],
                                         name=name)


@patch('scisoftpy.plot.addline')
@patch('numpy.squeeze')
@patch('numpy.diff')
class ShowPixelTest(unittest.TestCase):

    def test_correct_calls_made(self, diff_mock, squeeze_mock, addline_mock):
        e = ExcaliburDAWN()
        dac_scan_data = np.random.randint(10, size=(10, 256, 8*256))
        dac_range = [1, 10, 1]
        pixel = [20, 30]

        e.show_pixel(dac_scan_data, dac_range, pixel)

        # Check first addline call
        np.testing.assert_array_equal(range(dac_range[0], dac_range[1] + dac_range[2], dac_range[2]), addline_mock.call_args_list[0][0][0])
        np.testing.assert_array_equal(dac_scan_data[:, pixel[0], pixel[1]], addline_mock.call_args_list[0][0][1])
        # Check squeeze and diff call
        np.testing.assert_array_equal(dac_scan_data[:, pixel[0], pixel[1]], squeeze_mock.call_args[0][0])
        diff_mock.assert_called_once_with(squeeze_mock.return_value)
        # Check second addline call
        np.testing.assert_array_equal(range(dac_range[0], dac_range[1], dac_range[2]), addline_mock.call_args_list[1][0][0])
        self.assertEqual(diff_mock.return_value.__neg__(), addline_mock.call_args_list[1][0][1])


@patch(DAWN_patch_path + '.myerf')
@patch('excaliburcalibrationdawn.excaliburdawn.curve_fit',
       return_value=[[1, 2, 3], None])
@patch('scisoftpy.plot')
class FitDacScanTest(unittest.TestCase):

    mock_dac_scan = np.random.randint(10, size=(1, 20))
    mock_dac_axis = MagicMock()

    def test_correct_calls_made(self, plot_mock, fit_mock, myerf_mock):
        e = ExcaliburDAWN()

        values = e.fit_dac_scan(self.mock_dac_scan, self.mock_dac_axis)

        fit_mock.assert_called_once_with(myerf_mock, self.mock_dac_axis, ANY, p0=[100, 0.8, 3])
        np.testing.assert_array_equal(self.mock_dac_scan[0, :], fit_mock.call_args[0][2])
        plot_mock.addline.assert_called_once_with(self.mock_dac_axis, myerf_mock.return_value)
        self.assertEqual(self.mock_dac_axis, values)


@patch(DAWN_patch_path + '.lin_function')
@patch('excaliburcalibrationdawn.excaliburdawn.curve_fit',
       return_value=[[1, 2, 3], None])
@patch('scisoftpy.plot.addline')
@patch(DAWN_patch_path + '.clear_plot')
class PlotLinearFitTest(unittest.TestCase):

    def test_correct_calls_made(self, clear_mock, addline_mock, curve_fit,
                                lin_mock):
        e = ExcaliburDAWN()
        x = MagicMock()
        y = MagicMock()

        e.plot_linear_fit(x, y, [0, 1], name="Test", fit_name="Test fits", clear=True)

        # Check clear calls
        self.assertEqual("Test", clear_mock.call_args_list[0][0][0])
        self.assertEqual("Test fits", clear_mock.call_args_list[1][0][0])
        # Check first addline call
        self.assertEqual((x, y), addline_mock.call_args_list[0][0])
        self.assertEqual(dict(name="Test"), addline_mock.call_args_list[0][1])
        # Check fit calls
        curve_fit.assert_called_once_with(lin_mock, x, y, [0, 1])
        lin_mock.assert_called_once_with(x, curve_fit.return_value[0][0], curve_fit.return_value[0][1])
        # Check second addline call
        self.assertEqual((x, lin_mock.return_value), addline_mock.call_args_list[1][0])
        self.assertEqual(dict(name="Test fits"), addline_mock.call_args_list[1][1])


@patch('numpy.histogram')
@patch(DAWN_patch_path + '.gauss_function')
@patch('excaliburcalibrationdawn.excaliburdawn.curve_fit',
       return_value=[[1, 2, 3], None])
@patch('scisoftpy.plot.addline')
@patch(DAWN_patch_path + '.clear_plot')
class PlotGaussianFitTest(unittest.TestCase):

    def test_correct_calls_made(self, clear_mock, addline_mock, curve_fit,
                                gauss_mock, histo_mock):
        e = ExcaliburDAWN()
        x1 = MagicMock()
        scan_data_mock = [x1]
        bins = MagicMock()

        e.plot_gaussian_fit(scan_data_mock, "Test", [0, 1], bins)

        # Check clear calls
        self.assertEqual("Test", clear_mock.call_args_list[0][0][0])
        self.assertEqual("Test (fitted)", clear_mock.call_args_list[1][0][0])
        histo_mock.assert_called_once_with(x1, bins=bins)
        # Check first addline call
        self.assertEqual((histo_mock.return_value[1][0:-1],
                          histo_mock.return_value[0]),
                         addline_mock.call_args_list[0][0])
        self.assertEqual(dict(name="Test"), addline_mock.call_args_list[0][1])
        # Check fit calls
        curve_fit.assert_called_once_with(gauss_mock,
                                          histo_mock.return_value[1][0:-2],
                                          histo_mock.return_value[0][0:-1],
                                          [0, 1])
        gauss_mock.assert_called_once_with(histo_mock.return_value[1][0:-1],
                                           curve_fit.return_value[0][0],
                                           curve_fit.return_value[0][1],
                                           curve_fit.return_value[0][2])
        # Check second addline call
        self.assertEqual((histo_mock.return_value[1][0:-1],
                          gauss_mock.return_value),
                         addline_mock.call_args_list[1][0])
        self.assertEqual(dict(name="Test (fitted)"), addline_mock.call_args_list[1][1])


@patch('numpy.diff')
@patch('numpy.squeeze')
@patch('scisoftpy.plot.addline')
@patch(DAWN_patch_path + '.clear_plot')
class PlotDacScanTest(unittest.TestCase):

    def test_given_start_lower_than_stop(self, clear_mock, addline_mock,
                                         squeeze_mock, diff_mock):
        e = ExcaliburDAWN()
        chips = [0]
        dac_range = [1, 10, 1]
        expected_range = range(1, 11, 1)
        expected_subrange = range(1, 10, 1)[1:]
        dac_scan_data = np.random.randint(10, size=(10, 256, 8*256))
        expected_mean = dac_scan_data[:, 0:256, 0:256].mean(2).mean(1)

        scan, axis = e.plot_dac_scan(chips, dac_scan_data, dac_range)

        # Check clear calls
        self.assertEqual("DAC Scan", clear_mock.call_args_list[0][0][0])
        self.assertEqual("Spectrum", clear_mock.call_args_list[1][0][0])
        # Check first addline call
        np.testing.assert_array_equal(expected_range, addline_mock.call_args_list[0][0][0])
        np.testing.assert_array_equal(expected_mean, squeeze_mock.call_args_list[0][0][0])
        self.assertEqual(squeeze_mock.return_value, addline_mock.call_args_list[0][0][1])
        self.assertEqual(dict(name="DAC Scan"), addline_mock.call_args_list[0][1])
        # Check diff call
        diff_mock.assert_called_once_with(squeeze_mock.return_value)
        # Check second addline call
        np.testing.assert_array_equal(expected_subrange, addline_mock.call_args_list[1][0][0])
        np.testing.assert_array_equal(expected_mean, squeeze_mock.call_args_list[1][0][0])
        self.assertEqual(diff_mock.return_value.__neg__.return_value[1:], addline_mock.call_args_list[1][0][1])
        self.assertEqual(dict(name="Spectrum"), addline_mock.call_args_list[1][1])
        np.testing.assert_array_equal(expected_range, axis)

    def test_given_start_higher_than_stop(self, clear_mock, addline_mock,
                                          squeeze_mock, diff_mock):
        e = ExcaliburDAWN()
        chips = [0]
        dac_range = [10, 1, 1]
        expected_range = range(10, 0, -1)
        dac_scan_data = np.random.randint(10, size=(10, 256, 8*256))

        scan, axis = e.plot_dac_scan(chips, dac_scan_data, dac_range)

        np.testing.assert_array_equal(expected_range, axis)
