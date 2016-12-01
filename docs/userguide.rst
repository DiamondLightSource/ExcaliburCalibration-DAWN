User Guide
==========

The following is a guide for some of the most useful features of the ExcaliburCalibration-DAWN module.

Overview
--------

This module is built upon a script that was originally developed for I13 EXCALIBUR-3M-RX001 detector.

EXCALIBUR-specific functions need to be extracted and copied into a separate library. This will allow for the scripts to be usable with any MPX3-based system provided that a library of control functions is available for each type of detector:

* EXCALIBUR
* LANCELOT/MERLIN

These scripts communicate with FEMs via the ExcaliburTestApplicationInterface Python class and require configuration files in: /dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/config/

Chip 0 in this program corresponds to the left chip of the bottom half of a module or to the right chip of the top half of the module when facing the front-surface of sensor

Setting Up Excalibur Test App
-----------------------------

The excaliburTestApplication requires the libboost and libhdf5 locally and $LD_LIBRARY_PATH to be set to their location:

#. Choose an installation location (e.g. /home/<user>/Detectors).
#. Copy the lib folder from /dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburRxlib::

    $ cd /dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburRxlib
    $ mkdir /home/<user>/Detectors
    $ cp -r lib/ /home/<user>/Detectors/lib

#. Add the environment variable $LD_LIBRARY_PATH to show the location of the lib folder::

    $ export LD_LIBRARY_PATH=/home/<user>/Detectors/lib

#. You can also add this to your /home/<user>/.bashrc_local file::

    LIBDIR=/home/<user>/Detectors/lib
    if [ -d $LIBDIR ]; then
        export LD_LIBRARY_PATH=${LIBDIR}:${LD_LIBRARY_PATH}
    fi

Getting Started
---------------

.. include:: quickstart.rst

Setting up a new Detector
-------------------------

To interface with a new detector you must create a config module. Copy the detectorlab1 module in ExcaliburCalibration-DAWN/config into the same directory and name it after your detector. You the need to edit the new module to match the new detector; change the Detector name and master_node as well as entering the node, server and IP address of each FEM of the new detector. These three lists must correlate, i.e.::


    detector = Detector(name="newdetector", nodes=[1, 2], master_node=1,
                        servers=["node1-server", "node2-server"],
                        ip_addresses=["node1-ip", "node2-ip"])

Now you can instantiate an ExcaliburNode or ExcaliburDetector (1M or 3M) with this module as the detector_config and it will make the connections to each of the nodes given.

Calibration
-----------

DAC Values
~~~~~~~~~~

The very first time you calibrate a module, you need to manually adjust the FBK, CAS and GND DACs. The optimum read back values (recommended by Rafa in May 2015) are:

* GND: 0.65V
* FBK: 0.9V
* CAS: 0.85V

To calibrate these manually, you can use the set_dac and read_dac values to home in on the required DAC value to get the correct read back value. Once you have the optimum DAC value for a given chip, insert it into the corresponding array in the config folder. These values will be specific to each module and so each must have its own file in the config folder with the appropriate GND, FBK and CAS arrays.

### Add Scott's automated stuff here... ###

Threshold Equalisation
~~~~~~~~~~~~~~~~~~~~~~

To run threshold equalisation for all chips::

    >>> x.threshold_equalization()

By default, threshold equalization files will be created /tmp/femX of the server node X. You should copy this folder to the path were EPICS expects, for each of the FEMs.

Threshold Calibration
~~~~~~~~~~~~~~~~~~~~~

To calibrate thresholds using the default keV to DAC mapping::

    >>> x.threshold_calibration_all_gains()

To calibrate thresholds using X-rays:

Method 1: Brute-force:

    Produce monochromatic X-rays or use Fe55
    Perform a DAC scan using::

        >>> self.scan_dac(range(8), "Threshold0", Range(80, 20, 1))

    This will produce 2 plots, an integral plot and a differential spectrum. Inspect the spectrum and evaluate the position of the energy peak in DAC units. Energy = 6keV for energy peak DAC = 60; since calibration is performed on noise peak, 0keV corresponds to the selected DAC target (10 by default). Perform a linear fit of the DAC as a function of energy and edit the threshold0 file in the calibration directory accordingly. Each Threshold calibration file consists of 2 rows of 8 floating point numbers; 8 gain values and 8 offset values. The DAC value to apply for a requested threshold energy value E in keV is given by::

        DAC = Gain * Energy + Offset

Method 2: Using 1 energy and the noise peak dac
    ???

Method 3: Using several energies. Script not written yet.
    ???

Other Useful Functions
----------------------

To change a threshold::

    >>> x.set_dac(range(8), "Threshold0", 40)

This will allow you to put your threshold just above the noise to check the
response of the 1/2 module to X-rays

To capture an image::

    >>> x.expose(exposure=100)

The image is displayed in Image Data plot window which is opened with Window > Show Plot View > <Plot Name>.

Other useful functions::

    >>> x.acquire_ff()
    >>> x.read_chip_ids()
    >>> x.monitor()
    >>> x.load_config()
    >>> x.test_logo()

Full Detector Calibration
-------------------------

The Excalibur1M class allows some operations to be performed on two FEMs with a single command. To create an Excalibur1M instance::

    >>> from excaliburcalibrationdawn import Excalibur1M
    >>> from config import test1M
    >>> x = Excalibur1M(test1M)

Where test1M is the config module for a detector with two nodes. The class will then ssh into each node server, perform the relevant commands and return with any response.

You can capture images in the same way as with a single node using the expose() function. The images from both nodes will be loaded, combined and plotted in DAWN to give a full image.