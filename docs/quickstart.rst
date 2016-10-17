From the command line run the following to start up DAWN::

   $ module load dawn
   $ dawn &

Create a python console and create an ExcaliburNode instance for the master FEM (e.g. FEM 1)::

   >>> from excaliburcalibrationdawn import ExcaliburNode
   >>> master = ExcaliburNode(1)

ExcaliburNode provides some helper functions to perform initialisation for the
detector. To enable LV, set the HV bias to 120 and enable HV::

   >>> master.enable_lv()
   >>> master.set_hv_bias(120)
   >>> master.enable_hv()

To create an ExcaliburNode instance to calibrate a 1/2 module (e.g. FEM2)::

   >>> x = ExcaliburNode(2)

You can then perform calibration processes, to perform threshold equilisation
on chip 1 (index 0)::

   >>> x.threshold_equalization(0)

Or to run for all chips::

   >>> x.threshold_equalization([0, 1, 2, 3, 4, 5, 6, 7])
