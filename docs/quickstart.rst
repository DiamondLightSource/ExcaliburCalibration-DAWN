Repo
~~~~

To checkout the module from GitHub run the following command in the directory you wish to set up the repo::

    $ git clone git@github.com:dls-controls/ExcaliburCalibration-DAWN.git

DAWN
~~~~

From the command line run the following to start up DAWN::

   $ module load dawn
   $ dawn &

In DAWN: File -> New -> Project; In the pop-up window: General -> Project. Then unselect default, browse and select the repo you just cloned. Add a name (ExcaliburCalibration-DAWN) and Finish. If it says the directory is already a DAWN project, use File -> Import... -> General -> Existing Projects instead. If asked to set up your interpreter: Advanced Auto-Config -> '/dls_sw/apps/python/anaconda/1.7.0/64/bin/python2.7', or if that isn't there 'Anaconda - select to install'.

Create a python console (Window -> Show View -> Other -> General -> Console and then in the console view Open Console (Icon) -> PyDev Console). That should be DAWN ready to go.

Using the Script
~~~~~~~~~~~~~~~~

Create a python console and create an ExcaliburNode instance for the master FEM; the one interfaced with the I2C bus of the power card (e.g. FEM 1 on detectorlab1)::

   >>> from excaliburcalibrationdawn import ExcaliburNode
   >>> from config import detectorlab1
   >>> master = ExcaliburNode(1, detectorlab1)

ExcaliburNode provides some helper functions to perform initialisation for the
detector. To enable LV, set the HV bias to 120 and enable HV::

   >>> master.initialise_lv()
   >>> master.set_hv_bias(120)
   >>> master.enable_hv()

To create an ExcaliburNode instance to calibrate a 1/2 module (e.g. FEM 2) and then initialize (enable LV, set HV bias to 120) and load default DAC values::

   >>> x = ExcaliburNode(2, detectorlab1)
   >>> x.setup()

Check the status of the FEM and enable the HV bias once the humidity and temperature are OK::

    >>> x.monitor()
    >>> x.enable_hv_bias()

You can then perform calibration processes, to perform threshold equalisation on chip 1 (index 0)::

   >>> x.threshold_equalization(0)

Or to run for all chips::

   >>> x.threshold_equalization()
