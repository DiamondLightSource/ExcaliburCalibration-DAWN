User Guide
==========

The following is a guide for some of the most useful features of the ExcaliburCalibration-DAWN module.

Getting Started
---------------

.. include:: quickstart.rst


Further Usage
-------------

To capture an image::

    >>> x.expose(exposure=100)

As well as some other useful functions::

    >>> x.threshold_calibration()
    >>> x.acquire_ff()
    >>> x.read_chip_ids()
    >>> x.monitor()
    >>> x.set_gnd_fbk_cas_excalibur_rx001

Full Detector Calibration
-------------------------

The Excalibur1M class allows some operations to be performed on two FEMs with a single command. To create aN Excalibur1M instance::

    >>> from excaliburcalibrationdawn import Excalibur1M
    >>> x = Excalibur1M(server="p99-excalibur-0", master=6, node2=5)

Where server is the root of the server name, without the FEM specifier on the end, i.e. the server for FEM 6 will be p99-excalibur-06, which will be added to the server path on the ExcaliburNode instance for the master node. Master is the node set to control the power supply card and node2 is the second FEM making up the module. The class will then ssh into each node server, perform the relevant commands and return with any response.

You can then perform image capture::

    >>> x.expose(exposure=100)

The images from both nodes will be loaded, combined and plotted in DAWN to give a full image. Calibration functions can be used::

    >>> x.threshold_equalization()
    >>> x.logo_test()

where the class will simply ssh into the nodes and perform the given function.
