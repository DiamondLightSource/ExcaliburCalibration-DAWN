"""Detector Lab 1M.

To create a new detector module, copy this file into the same directory with a
name specifying the new detector and fill in the correct data. Also replace
the top line description of the detector.
"""
from collections import OrderedDict, namedtuple

Detector = namedtuple("Detector", ["name", "nodes", "master_node",
                                   "servers", "ip_addresses", "root_path",
                                   "calib"])

# Detector Specification:
detector = Detector(name="detectorlab1", nodes=[6], master_node=6,
                    servers=["p99-excalibur01"],
                    ip_addresses=["192.168.0.101"],
                    root_path="/dls/detectors/support/silicon_pixels/"
                              "excaliburRX/3M-RX001/detectorlab1/",
                    calib="calib")

# Default DAC Values:
DACS = OrderedDict([('Threshold1', 0),
                    ('Threshold2', 0),
                    ('Threshold3', 0),
                    ('Threshold4', 0),
                    ('Threshold5', 0),
                    ('Threshold6', 0),
                    ('Threshold7', 0),
                    ('Preamp', 175),  # Could use 200
                    ('Ikrum', 10),  # Low Ikrum for better low energy X-ray
                                    # sensitivity
                    ('Shaper', 150),
                    ('Disc', 125),
                    ('DiscLS', 100),
                    ('ShaperTest', 0),
                    ('DACDiscL', 90),
                    ('DACTest', 0),
                    ('DACDiscH', 90),
                    ('Delay', 30),
                    ('TPBuffIn', 128),
                    ('TPBuffOut', 4),
                    ('RPZ', 255),  # RPZ is disabled at 255
                    ('TPREF', 128),
                    ('TPREFA', 500),
                    ('TPREFB', 500)])
