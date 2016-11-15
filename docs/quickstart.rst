From the command line run the following to start up DAWN::

   $ module load dawn
   $ dawn &

Create a python console and create an ExcaliburNode instance for the master FEM; the one interfaced with the I2C bus of the power card (e.g. FEM 1 on server p99-excalibur06)::

   >>> from excaliburcalibrationdawn import ExcaliburNode
   >>> master = ExcaliburNode(1, "p99-excalibur0")

ExcaliburNode provides some helper functions to perform initialisation for the
detector. To enable LV, set the HV bias to 120 and enable HV::

   >>> master.initialise_lv()
   >>> master.set_hv_bias(120)
   >>> master.enable_hv()

To create an ExcaliburNode instance to calibrate a 1/2 module (e.g. FEM 2) and then initialize and load default DAC values::

   >>> x = ExcaliburNode(2, "p99-excalibur0")
   >>> x.setup()

You can then perform calibration processes, to perform threshold equalisation on chip 1 (index 0)::

   >>> x.threshold_equalization(0)

Or to run for all chips::

   >>> x.threshold_equalization([0, 1, 2, 3, 4, 5, 6, 7])
