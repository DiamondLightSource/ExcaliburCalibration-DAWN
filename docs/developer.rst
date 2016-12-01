Developer Guide
===============

This is an overview of the structure of the module and how the classes work together.

Class Hierarchy
---------------

The class that does the bulk of the operations is ExcaliburNode. This is designed to represent a single node, FEM or half-module. It performs the logic behind the commands requested on the python terminal.

ExcaliburNode has two helper classes; ExcaliburTestAppInterface to construct and send commands to the CLI tool controlling each FEM, and ExcaliburDAWN as an interface to scisoftpy utilities and plotting in DAWN. These three classes, as well as the util module with a few general helper functions, can be used to control, test and calibrate a single node.

To control entire 1M or 3M detectors there are the Excalibur1M and Excalibur3M classes. These both inherit from the ExcaliburDetector class, which is where all their functionality is. These classes simply allow you to perform single operations on multiple nodes. It can calibrate each node of a detector in turn, or it can capture an image on each node and combine them into a full detector image, for example.

ExcaliburDetector
-----------------

To add functionality to the ExcaliburDetector class you can most likely just follow the template of an existing function. There are three types of function at present. Functions that simply run the same named function on each node (optimise_gnd_fbk_cas), functions that run a function on the MasterNode and then a different one on the reset (setup) and function that run only on the MasterNode (enable_hv). The first type can also run in threads or not, for example read_chip_ids does not as the console output would merge together. There are also a few special cases where there is a little bit of extra logic afterwards, for example stitching images together in expose and checking for errors in threshold_equalization.

Lets take an example:

.. autoclass:: ExcaliburDetector
   :members: optimise_gnd_fbk_cas

Here the same named function is called on each node, in a thread so that they all run simultaneously. This is a commonly used format, there are many functions that look like this, but with node.optimise_gnd_fbk_cas replaced with a different function call.

Detector Configuration Modules
------------------------------

A detector is defined by its module in the config folder. Within this is a detector specification and DAC values. The detector specification is the name, constituent nodes, master node, servers and IP addresses making up the detector. The module is then passed to the ExcaliburDetector (and ExcaliburNode) init to allow it to make connections to each node and grab the correct calibration data. With this, you can create an ExcaliburDetector instance, call setup() and each node will have its DACs and discriminator bits loaded. For first time setup you can create an ExcaliburDetector instance and call setup_new_detector to create a new folder for the detector on disk and set up the structure ready for calibration.

Excalibur Test Application
--------------------------

The excaliburTestApp CLI tool used in this module is a C program (using various C and C++ libraries) written by Tim Nicholls of STFC Application and Engineering Group. The source code can be found at /dls/detectors/support/silicon_pixels/excaliburRX/excalibur3rx/client.

Contributing
------------

This python package is hosted on GitHub at `ExcaliburCalibration-DAWN <https://github.com/dls-controls/ExcaliburCalibration-DAWN>`_ under continuous integration controls (`Travis <https://en.wikipedia.org/wiki/Travis_CI>`_, `Coverage <https://coverage.readthedocs.io/en/coverage-4.2/>`_ and `Landscape <https://docs.landscape.io/faq.html>`_).

Changes should be made on a branch, tested, and then merged into master via a pull request.

Travis will show if the build fails, i.e. if the package couldn't be installed or some tests fail. Coverage will show if added code is not tested. Landscape will show if the code follows `PEP8 <http://docs.python-guide.org/en/latest/writing/style/>`_ standards. Some style errors exist in this module, for example variables should usually be lowercase_with_underscores, however "discL" and "discH" are used as variables frequently, because that is the definition in the excaliburTestApp.

The important thing is to make the code as nice to read, easy to understand and simple to refactor. These tools generally help that goal but should be ignored if they do not.

Documentation
~~~~~~~~~~~~~

The docs are built automatically by ReadTheDocs from the docs folder on the master branch.
