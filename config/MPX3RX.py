"""Configuration data for MPX3RX Excalibur detector."""
from collections import OrderedDict

import numpy as np

E1_DAC = dict(shgm=np.array([[20, 62, 62, 62, 62, 62, 62, 62],
                             [64, 64, 64, 64, 64, 64, 64, 64],
                             [62, 62, 62, 62, 62, 62, 62, 62],
                             [60, 35, 64, 64, 64, 64, 64, 64],
                             [64, 64, 64, 64, 64, 64, 64, 64],
                             [64, 64, 64, 64, 64, 64, 64, 64]]
                            ).astype('float'),
              hgm=np.array([[62, 62, 62, 62, 62, 62, 62, 62],
                            [64, 64, 64, 64, 64, 64, 64, 64],
                            [62, 62, 62, 62, 62, 62, 62, 62],
                            [60, 35, 64, 64, 64, 64, 64, 64],
                            [64, 64, 64, 64, 64, 64, 64, 64],
                            [64, 64, 64, 64, 64, 64, 64, 64]]
                           ).astype('float'),
              lgm=np.array([[20, 62, 62, 62, 62, 62, 62, 62],
                            [64, 64, 64, 64, 64, 64, 64, 64],
                            [62, 62, 62, 62, 62, 62, 62, 62],
                            [60, 35, 64, 64, 64, 64, 64, 64],
                            [64, 64, 64, 64, 64, 64, 64, 64],
                            [64, 64, 64, 64, 64, 64, 64, 64]]
                           ).astype('float'),
              slgm=np.array([[62, 62, 62, 62, 62, 62, 62, 62],
                             [64, 64, 64, 64, 64, 64, 64, 64],
                             [62, 62, 62, 62, 62, 62, 62, 62],
                             [60, 35, 64, 64, 64, 64, 64, 64],
                             [64, 64, 64, 64, 64, 64, 64, 64],
                             [64, 64, 64, 64, 64, 64, 64, 64]]
                            ).astype('float'))

E2_DAC = dict(shgm=np.array([[120, 110, 100, 90, 80, 70, 60, 50],
                             [64, 64, 64, 64, 64, 64, 64, 64],
                             [62, 62, 62, 62, 62, 62, 62, 62],
                             [60, 35, 64, 64, 64, 64, 64, 64],
                             [64, 64, 64, 64, 64, 64, 64, 64],
                             [64, 64, 64, 64, 64, 64, 64, 64]]
                            ).astype('float'),
              hgm=np.array([[62, 62, 62, 62, 62, 62, 62, 62],
                            [64, 64, 64, 64, 64, 64, 64, 64],
                            [62, 62, 62, 62, 62, 62, 62, 62],
                            [60, 35, 64, 64, 64, 64, 64, 64],
                            [64, 64, 64, 64, 64, 64, 64, 64],
                            [64, 64, 64, 64, 64, 64, 64, 64]]
                           ).astype('float'),
              lgm=np.array([[20, 62, 62, 62, 62, 62, 62, 62],
                            [64, 64, 64, 64, 64, 64, 64, 64],
                            [62, 62, 62, 62, 62, 62, 62, 62],
                            [60, 35, 64, 64, 64, 64, 64, 64],
                            [64, 64, 64, 64, 64, 64, 64, 64],
                            [64, 64, 64, 64, 64, 64, 64, 64]]
                           ).astype('float'),
              slgm=np.array([[62, 62, 62, 62, 62, 62, 62, 62],
                             [64, 64, 64, 64, 64, 64, 64, 64],
                             [62, 62, 62, 62, 62, 62, 62, 62],
                             [60, 35, 64, 64, 64, 64, 64, 64],
                             [64, 64, 64, 64, 64, 64, 64, 64],
                             [64, 64, 64, 64, 64, 64, 64, 64]]
                            ).astype('float'))

E3_DAC = dict(shgm=np.array([[250, 110, 100, 90, 80, 70, 60, 50],
                             [64, 64, 64, 64, 64, 64, 64, 64],
                             [62, 62, 62, 62, 62, 62, 62, 62],
                             [60, 35, 64, 64, 64, 64, 64, 64],
                             [64, 64, 64, 64, 64, 64, 64, 64],
                             [64, 64, 64, 64, 64, 64, 64, 64]]
                            ).astype('float'),
              hgm=np.array([[62, 62, 62, 62, 62, 62, 62, 62],
                            [64, 64, 64, 64, 64, 64, 64, 64],
                            [62, 62, 62, 62, 62, 62, 62, 62],
                            [60, 35, 64, 64, 64, 64, 64, 64],
                            [64, 64, 64, 64, 64, 64, 64, 64],
                            [64, 64, 64, 64, 64, 64, 64, 64]]
                           ).astype('float'),
              lgm=np.array([[20, 62, 62, 62, 62, 62, 62, 62],
                            [64, 64, 64, 64, 64, 64, 64, 64],
                            [62, 62, 62, 62, 62, 62, 62, 62],
                            [60, 35, 64, 64, 64, 64, 64, 64],
                            [64, 64, 64, 64, 64, 64, 64, 64],
                            [64, 64, 64, 64, 64, 64, 64, 64]]
                           ).astype('float'),
              slgm=np.array([[62, 62, 62, 62, 62, 62, 62, 62],
                             [64, 64, 64, 64, 64, 64, 64, 64],
                             [62, 62, 62, 62, 62, 62, 62, 62],
                             [60, 35, 64, 64, 64, 64, 64, 64],
                             [64, 64, 64, 64, 64, 64, 64, 64],
                             [64, 64, 64, 64, 64, 64, 64, 64]]
                            ).astype('float'))

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
