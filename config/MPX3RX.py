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

GND_DAC = np.array([[141, 144, 154, 143, 161, 158, 144, 136],
                    [154, 155, 147, 147, 147, 155, 158, 151],
                    [143, 156, 139, 150, 144, 150, 149, 158],
                    [143, 156, 139, 150, 144, 150, 149, 158],
                    [151, 135, 150, 162, 153, 144, 134, 145],
                    [134, 145, 171, 146, 152, 142, 141, 141]]
                   ).astype('int')

# Max current for fbk limited to 0.589 for chip 7
FBK_DAC = np.array([[190, 195, 201, 198, 220, 218, 198, 192],
                    [215, 202, 208, 200, 198, 211, 255, 209],
                    [189, 213, 185, 193, 204, 207, 198, 220],
                    [189, 213, 185, 193, 204, 207, 198, 220],
                    [200, 195, 205, 218, 202, 194, 185, 197],
                    [192, 203, 228, 197, 206, 191, 206, 189]]
                   ).astype('int')

CAS_DAC = np.array([[178, 195, 196, 182, 213, 201, 199, 186],
                    [208, 197, 198, 194, 192, 207, 199, 188],
                    [181, 201, 177, 184, 194, 193, 193, 210],
                    [181, 201, 177, 184, 194, 193, 193, 210],
                    [196, 180, 202, 214, 197, 193, 186, 187],
                    [178, 191, 218, 184, 192, 186, 195, 185]]
                   ).astype('int')

# TOP MODULE: AC-EXC-8
# #@ Moly temp: 35 degC on node 3
# NOTE : chip 2 FBK cannot be set to target value
# CENTRAL MODULE: AC-EXC-7
# @ Moly temp: 27 degC on node 1
# @ Moly temp: 28 degC on node 2
# BOTTOM MODULE: AC-EXC-4
# #@ Moly temp: 31 degC on node 5
# @ Moly temp: 31 degC on node 6

# NOTE: DAC out read-back does not work for chip 2 (counting from 0) of
# bottom 1/2 module
# Using read_dac function on chip 2 will give FEM errors and the system
# will need to be power-cycled
# Got error on load pixel config command for chip 7: 2 Pixel
# configuration loading failed
# Exception caught during femCmd: Timeout on pixel configuration write
# to chip7 acqState=3
# Connecting to FEM at IP address 192.168.0.101 port 6969 ...
# **************************************************************
# Connecting to FEM at address 192.168.0.101
# Configuring 10GigE data interface: host IP: 10.0.2.1 port: 61649 FEM
# data IP: 10.0.2.2 port: 8 MAC: 62:00:00:00:00:01
# Acquisition state at startup is 3 sending stop to reset
# **** Loading pixel configuration ****
# Last idx: 65536
# Last idx: 65536
# Last id

# 1st MODULE 1M:
# up -fem5
# Bottom -fem6

# GND_Dacs[1,:]=[145,141,142,142,141,141,143,150]
# FBK_Dacs[1,:]=[205,190,197,200,190,190,200,210]
# CAS_Dacs[1,:]=[187,187,183,187,177,181,189,194]
# GND_Dacs[4,:]=[136,146,136,160,142,135,140,148]
# FBK_Dacs[4,:]=[190,201,189,207,189,189,191,208]
# CAS_Dacs[4,:]=[180,188,180,197,175,172,185,200]
# GND_Dacs[5,:]=[158,140,gnd,145,158,145,138,153]
# FBK_Dacs[5,:]=[215,190,fbk,205,221,196,196,210]
# CAS_Dacs[5,:]=[205,178,cas,190,205,180,189,202]

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
