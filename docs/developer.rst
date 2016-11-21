Developer Guide
===============

This is an overview of the structure of the module and how the classes work together.

Class Hierarchy
---------------

The class that does the bulk of the operations is ExcaliburNode. This is designed to represent a single node, FEM or half-module. It performs the logic behind the commands requested on the python terminal.

ExcaliburNode has two helper classes; ExcaliburTestAppInterface to construct and send commands to the CLI tool controlling each FEM, and ExcaliburDAWN as an interface to scisoftpy utilities and plotting in DAWN. These three classes, as well as the util module with a few general helper functions, can be used to control, test and calibrate a single node.

To control entire 1M or 3M detectors there are the Excalibur1M and Excalibur3M classes. These both inherit from the ExcaliburDetector class, which is where all their functionality is. These classes simply allow you to perform single operations on multiple nodes. It can calibrate each node of a detector in turn, or it can capture an image on each node and combine them into a full detector image, for example.

Detector Configuration Modules
------------------------------

A detector is defined by its module in the config folder. Within this is a detector specification and DAC values. The detector specification is the name, constituent nodes, master node, servers and IP addresses making up the detector. The module is then passed to the ExcaliburDetector (and ExcaliburNode) init to allow it to make connections to each node and grab the correct calibration data. With this, you can create an ExcaliburDetector instance, call setup() and each node will have its DACs and discriminator bits loaded. For first time setup you can create an ExcaliburDetector instance and call setup_new_detector to create a new folder for the detector on disk and set up the structure ready for calibration.

Excalibur Test Application
--------------------------

The excaliburTestApp CLI tool used in this module is a C program (using various C and C++ libraries) written by Tim Nicholls of STFC Application and Engineering Group. The source code can be found at /dls/detectors/support/silicon_pixels/excaliburRX/excalibur3rx/client.

Contributing
------------

This python package is hosted on GitHub at `ExcaliburCalibration-DAWN <https://github.com/dls-controls/ExcaliburCalibration-DAWN>`_ under continuous integration controls (`Travis <https://en.wikipedia.org/wiki/Travis_CI>`_, `Coverage <https://coverage.readthedocs.io/en/coverage-4.2/>`_ and `Landscape <https://docs.landscape.io/faq.html>`_).

Changes should be made on a branch, tested, and then merged into master via a pull request. New code should have tests to ensure it does as expected and to make it clear when you or someone else breaks what has been added.

`PEP8 <http://docs.python-guide.org/en/latest/writing/style/>`_ standards should be followed unless it is detrimental to the quality of the code in some other way. For example variables should be lowercase_with_underscores, however "discL" and "discH" are used as variables frequently, because that is the definition in the excaliburTestApp. Make the code as nice to read and easy to understand as possible.

Documentation
~~~~~~~~~~~~~

The docs are hosted on GitHub using GitHubPages. To update them simply add to the .rst files in the docs folder, or create new ones and add them to the index.rst. Before running make, you should set up a folder level with ExcaliburCalibration-DAWN called ECD-docs and within it create a html folder. So it will look like this::

    ├── ECD-docs
    │   ├── doctrees
    │   │   └── arch
    │   └── html
    │       ├── arch
    │       ├── _images
    │       ├── _modules
    │       ├── _sources
    │       └── _static
    └── ExcaliburCalibration-DAWN
        ├── config
        ├── docs
        │   └── arch
        ├── excaliburcalibrationdawn
        └── tests

In html folder we will set up a git repo and point it to the gh-pages branch of the repo.::

    $ git init
    $ git remote add origin git@github.com:dls-controls/ExcaliburCalibration-DAWN.git
    $ git fetch --all
    $ git checkout gh-pages

Then run 'make html' in the docs folder. This will write to the new git repo and overwrite the old built docs with the new ones. You can then cd to html folder and run the following git commands to push the built docs to the repo::

    $ git add .
    $ git commit -m "Rebuilt docs"
    $ git push origin gh-pages
