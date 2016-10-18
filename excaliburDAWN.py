"""
EXCALIBUR calibration Python scripts
/dls/detectors/support/silicon_pixels/excaliburRX/PyScripts/excaliburDAWN.py
15-07-2015
"""
import math
import subprocess
import numpy as np
import time
import os
import shutil
from scipy.optimize import curve_fit


def myerf(x,A, mu, sigma):
    return A/2. * (1+math.erf((x-mu)/(math.sqrt(2)*sigma)))

def lin_function(x,offset,gain):
    return offset+gain*x

def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def S_curve_function(x,k,delta,E,sigma):
    return k*((1-2*delta*(x/E-0.5))**2) * (1-myerf(x,k,E,sigma))


class excaliburRX(object):
    """
    excaliburRX is a class defining methods required to calibrate each 1/2 module (8 MPX3-RX chips) of an EXCALIBUR-RX detector.
    These calibration scripts will work only inside the Python interpreter of DAWN software running on the PC sever node connected to the FEM controlling the half-module which you wish to calibrate
    
    =========================== EXCALIBUR Test-Application========================================================
    
    NOTE: excaliburRX detector class communicates with FEMs via excalibur Test-Application which is an executable file saved in: 
    /dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp
    
    NOTE: excaliburRX detector class requires configuration files copied in:
    /dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/config/
    
    ========= To install libraries required by EXCALIBUR Test-Application
    
    excalibur Test-Application requires libboost and libhdf5 libraries to be installed locally. Use the following instructions to install the libraries:
    cd /dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburRxlib
    mkdir /home/fedID/lib/
    cp lib* /home/fedID/lib/
    This will copy the required libraries:
    [ktf91651@pc0066 /]$ ll /home/ktf91651/lib
    total 17880
    -rwxr-xr-x. 1 ktf91651 ktf91651   17974 Mar  7  2014 libboost_system.so
    -rwxr-xr-x. 1 ktf91651 ktf91651   17974 Mar  7  2014 libboost_system.so.1.47.0
    -rwxr-xr-x. 1 ktf91651 ktf91651  138719 Mar  7  2014 libboost_thread.so
    -rwxr-xr-x. 1 ktf91651 ktf91651  138719 Mar  7  2014 libboost_thread.so.1.47.0
    -rwxr-xr-x. 1 ktf91651 ktf91651 8946608 Mar  7  2014 libhdf5.so
    -rwxr-xr-x. 1 ktf91651 ktf91651 8946608 Mar  7  2014 libhdf5.so.7

    edit
    /home/fedID/.bashrc_local
    to add path to excalibur libraries

    [ktf91651@p99-excalibur01 ~]$ more .bashrc_local 
    LIBDIR=$HOME/lib
    if [ -d $LIBDIR ]; then
        export LD_LIBRARY_PATH=${LIBDIR}:${LD_LIBRARY_PATH}
    fi

    check path using
    [ktf91651@p99-excalibur01 ~]$ echo $LD_LIBRARY_PATH
    /home/ktf91651/lib:
    
    ================= EXCALIBUR Test-Application commands
    
    [ktf91651@p99-excalibur01 ~]$ /dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp -i 192.168.0.106 -p 6969 -m 0xff --help

    Usage: /dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp options
    
      -h --help                   Display this usage information.
      -i --ipaddress              IP address of FEM to connect to.
      -p --port                   Port of FEM to connect to.
      -m --mask                   Select MPX3 enable mask.
      -r --reset                  Issue front-end reset/init.
         --lvenable <mode>        Set power card LV enable: 0=off (default), 1=on.
         --hvenable <mode>        Set power card HV enable: 0=off (default), 1=on.
         --hvbias <volts>         Set power card HV bias in volts.
      -e --efuse                  Read and display MPX3 eFuse IDs.
      -d --dacs <filename>        Load MPX3 DAC values from filename if given, otherwise use default values
      -c --config                 Load MPX3 pixel configuration.
      -s --slow                   Display front-end slow control parameters.
      -a --acquire                Execute image acquisition loop.
         --burst                  Select burst mode for image acquisition.
         --matrixread             During acquisition, perform matrix read only (i.e. no shutter for config read or digital test).
      -n --frames <frames>        Number of frames to acquire.
      -t --acqtime <time>         Acquisition time (shutter duration) in milliseconds.
         --dacscan <params>       Execute DAC scan, params format must be comma separated dac,start,stop,step.
         --readmode <mode>        Readout mode: 0=sequential (default), 1=continuous.
         --trigmode <mode>        Trigger mode: 0=internal (default), 1=external shutter, 2=external sync.
         --colourmode <mode>      Select MPX3 colour mode: 0=fine pitch mode (default), 1=spectroscopic mode.
         --csmspm <mode>          Select MPX3 pixel mode: 0=single pixel mode (default), 1=charge summing mode.
         --disccsmspm <mode>      Select MPX3 discriminator output mode: 0=DiscL (default), 1=DiscH.
         --equalization <mode>    Select MPX3 equalization mode: 0=off (default), 1=on.
         --gainmode <mode>        Select MPX3 gain mode: 0=SHGM, 1=HGM, 2=LGM, 3=SLGM (default).
         --counter <counter>      Select MPX3 counter to read: 0 (default) or 1.
         --depth <depth>          Select MPX3 counter depth: 1, 6, 12 (default) or 24.
         --sensedac <id>          Set MPX3 sense DAC field to <id>. NB Requires DAC load to take effect
         --tpmask <filename>      Specify test pulse mask filename to load.
         --tpcount <count>        Set test pulse count to <count>, default is 0.
         --pixelmask <filename>   Specify pixel enable mask filename to load.
         --discl <filename>       Specify pixel DiscL configuration filename to load.
         --disch <filename>       Specify pixel DiscH configuration filename to load.
         --path <path>            Specify path to write data files to, default is /tmp.
         --hdffile <filename>     Write HDF file with optional filename, default is <path>/excalibur-YYMMDD-HHMMSS.hdf5
    
    
    ===================================== MODULE CALIBRATION USING PYTHON SCRIPTS 
    
    ================= FRONT-END POWER-ON
    
    To calibrate a 1/2 Module 
    
    ssh to the PC server node(standard DLS machine) connected to the MASTER FEM card (the one interfaced with the I2C bus of the Power card)
    On I13 EXCALIBUR-3M-RX001, this is the top FEM (192.168.0.106) connected to node 1
    ###########################
    >ssh i13-1-excalibur01
    ###########################
    Enable LV and make sure that HV is set to 120V during calibration:
    ##########################################################################################################################################
    >/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp -i 192.168.0.106 -p 6969 -m 0xff --lvenable 1
    >/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp -i 192.168.0.106 -p 6969 -m 0xff --lvenable 0
    >/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp -i 192.168.0.106 -p 6969 -m 0xff --lvenable 1
    >/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp -i 192.168.0.106 -p 6969 -m 0xff --hvbias 120
    >/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp -i 192.168.0.106 -p 6969 -m 0xff --hvenable 1
    ##########################################################################################################################################
    
    ssh to the PC server node (standard DLS machine) connected to the FEM card controlling the 1/2 module which you want to calibrate.
    ########################
    >ssh i13-1-excalibur0x with x in [1-6] and x=1 corresponds to the top FEM (Master FEM IP:192.168.0.106) connected to PC server node 1
    ########################
    
    ===================  DAWN START-UP
    
    On the PC server node, start DAWN by typing in a shell:
    ######################
    > module load dawn 
    > dawn &
    #######################
    
    
    ================== PYTHON SCRIPT
    
    Select the DExplore perspective 
    Open the file /dls/detectors/support/silicon_pixels/excaliburRX/PyScripts/excaliburDAWN.py 
    Run excaliburDAWN.py in the interactive console by clicking on the python icon "activates the interactive console" (CTL Alt ENTER)
    Select "Python Console" when the interactive interpreter console opens
    (You might have to run the script twice)
    
    In the Interactive console you need to create an excaliburRx object:
    ########################
    > x=excaliburRX(node)
    ########################
    were node is the PC server node number you are connected to.
    For example when running the Python calibration scripts on node i13-1-excalibur0X (with X in [1:6]), you should use: x=excaliburRX(X)
    For I13 installation top FEM is connected to node 1 and bottom fem to node 6
    
    ================ FBK GND and CAS DACs adjustment
    
    The very first time you calibrate a module, you need to manually adjust 3 DACs: FBK, CAS and GND
    The procedure is described in set_GND_FBK_CAS_ExcaliburRX001
    If you swap modules you also need to edit set_GND_FBK_CAS_ExcaliburRX001 accordingly since set_GND_FBK_CAS_ExcaliburRX001 contains DAC parameters specific each 3 modules based on the position of the module
    
    ================= THRESHOLD EQUALIZATION
    
    To run threshold_equalization scripts, just type in the interactive Python console:
    ########################
    > x.threshold_equalization()  
    ########################
    By default, threshold_equalization files will be created locally in a temporary folder : /tmp/femX of the PC server node X.
    You should copy the folder /femX in the path were EPICS expects threshold_equalization files for all the fems/nodes 
    
        
    At the end of the threshold_equalization you should get the following message in the interactive console: 
    Pixel threshold equalization complete
    
    ================= THRESHOLD CALIBRATION
    
    To calibrate thresholds using default keV to DAC mapping 
    ########################
    > x.threshold_calibration_allGains()  
    ########################
    

    
    ============== ACQUIRE X_RAY IMAGE WITH FE55 
    
    
    threshold_equalization data is then automatically loaded. And you can acquire a 60s image from Fe55 X-rays using the following command:
    ############
    >x.Fe55imageRX001()
    ############
    To change the exposure time used during image acquisition:
    ############
    >x.Fe55imageRX001(range(8),exp_time_in_ms)
    ############
        
    To change acquisition time:
    ############################
    >x.settings['acqtime']=1000 (for 1s exposure)
    ############################
    where the time is in ms
    The image will be automatically saved in /dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/Fe55_images
    The image is displayed in Image Data plot window which is opened by selected Show Plot View in the Window tab

    =====================PIXEL MASKING=======================================================


    ===================== SET THRESHOLD DAC
    To change threshold:
    ###################################
    >x.setDac(range(8),"Threshold0",40) to set threshold 0 at 40 DAC units
    ###################################
    This will allow you to put your threshold just above the noise to check the response of the 1/2 module to X-rays
    
    
    
    ===========================================================================================
    NOTE: chip 0 in this program correspond to the left chip of the bottom half of a module
    or to the right chip of the top half of the module when facing the front-surface of sensor 
    """
    
    def __init__(self,node):
        self.fem=node
        self.ipaddress="192.168.0.10"+str(7-self.fem)
        self.chipId()

    def threshold_equalization(self,chips=range(8)):
        """
        To calibrate all chips: x.threshold_equalization() or x.threshold_equalization(range(8)) or x.threshold_equalization([0, 1, 2, 3, 4, 5, 6, 7])
        To calibrate chip 0: x.threshold_equalization([0])
        To calibrate chip 0 , 2 and 5 : x.threshold_equalization([0,2,5])
        You need to edit this function to define which mode (SPM or CSM) and which gains you want to calibrate during the threshold_equalization sequence
        """
        self.settings['mode']='spm'
        self.settings['gain']='slgm'
        self.checkCalibDir()# Checks whether a threshold_equalization directory exists, if not it creates one with default dacs
        self.logChipId() # Log chip IDs in threshold_equalization folder
        self.setdacs(chips) # Set DACs recommended by Rafa in May 2015
        self.set_GND_FBK_CAS_ExcaliburRX001(chips,x.fem) # This will load the DAC values specific to each chip to have FBK, CAS and GND reading back the recommended analogue value 
        """
        IMPORTANT NOTE: These values of GND, FBK and CAS Dacs were adjusted for the modules present in RX001 on 20 june 2015
        If modules are replaced, these DACs need to be re-adjusted and the FBK_DAC, GND_DAC and Cas_DAC arrays in GND_FBK_CAS_ExcaliburRX001 have to be edited
        """
        self.calibrateDisc(chips,'discL')# Calibrates DiscL Discriminator connected to Threshold 0 using a rectangular ROI in 1  
        """
        NOTE: Always equalize DiscL before DiscH since Threshold1 is set at 0 when equalizing DiscL. So if DiscH was equalized first, this would induce noisy counts interfering with DiscL equalization 
        """
        #self.calibrateDisc(chips,'discH',1,'rect')

        #self.settings['mode']='csm'
        #self.settings['gain']='slgm'
        #self.calibrateDisc(chips,'discL',1,'rect')
        #self.calibrateDisc(chips,'discH',1,'rect')
        
        badPixels=self.maskRowBlock(range(4),256-20,256)
        return 

    def findXrayEnergyDac(self,chips=range(8),Threshold="0",energy=5.9):
        self.settings['acqtime']=100
        if x.settings['gain']=='shgm':
                dacRange=(self.dacTarget+100,self.dacTarget+20,2)
        
        self.loadConfig(chips)
        self.settings['filename']='Threshold'+str(Threshold)+'Scan_'+str(energy)+'keV'
        [dacscanData,scanRange]=self.scanDac(chips,"Threshold"+str(Threshold),dacRange)
        dacscanData[dacscanData>200]=0
        [chipDacScan,dacAxis]=x.plotDacScan(chips,dacscanData, scanRange)
        self.fitDacScan(chips,chipDacScan,dacAxis)
        
#        edgeDacs=self.findEdge(chips,dacscanData,dacRange,2)        
#        chipEdgeDacs=np.zeros(range(8))
#        for chip in chips:
#            chipEdgeDacs[chip]=edgeDacs[0:256,chip*256:chip*256+256].mean()
        return chipDacScan,dacAxis

    def fitDacScan(self,chips,chipDacScan,dacAxis):
#         p0=[100,0.8,3]
#         for chip in chips:
#             
#             #dnp.plot.addline(dacAxis,chipDacScan[chip,:])
#             popt, pcov = curve_fit(gauss_function,dacAxis,chipDacScan[chip,:],p0)
#             #popt, pcov = curve_fit(S_curve_function,dacAxis,chipDacScan[chip,:],p0)
#             dnp.plot.addline(dacAxis,gauss_function(dacAxis,popt[0],popt[1],popt[2]))
#         
        p0=[100,0.8,3]
        for chip in chips:
            
            #dnp.plot.addline(dacAxis,chipDacScan[chip,:])
            popt, pcov = curve_fit(myerf,dacAxis,chipDacScan[chip,:],p0)
            #popt, pcov = curve_fit(S_curve_function,dacAxis,chipDacScan[chip,:],p0)
            dnp.plot.addline(dacAxis,myerf(dacAxis,popt[0],popt[1],popt[2]))
        
        return chipDacScan,dacAxis


    def maskRowBlock(self,chips,RowMin,RowMax):
        """
        x.maskRowBlock([0],0,2) to mask the first 3 rows of chip 0
        badPixels=x.maskRowBlock(range(4),256-20,256))
        """
        badPixels=np.zeros([x.chipSize,x.chipSize*x.nbOfChips])
        for chip in chips:
            badPixels[RowMin:RowMax,chip*256:chip*256+256]=1
        for chip in chips:
            pixelmaskFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/' +self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'pixelmask.chip'+str(chip)
            np.savetxt(pixelmaskFile,badPixels[0:256,chip*256:chip*256+256],fmt='%.18g', delimiter=' ' )
        dnp.plot.image(badPixels)
        x.loadConfig(chips)
        return badPixels

    def threshold_calibration_allGains(self,chips=range(8),Threshold="0"):
        """
        Usage: x.threshold_calibration_allGains()
        This will save a threshold calibration file called threshold0 or threshold1 the calibration directory under each gain setting subfolder 
        Each Threshold calibration file consists of 2 rows of 8 floating point numbers:
        # g0     g1   g2   g3   g4   g5   g6   g7
        # Off0 Off1 Off2 Off3 Off4 Off5 Off6 Off7 
        # and the DAC value to apply to chip x for a requested threshold energy value E in keV is given by:
        # DACx= gx * E + Offx         
        """
        self.settings['gain']='shgm'
        self.threshold_calibration(chips,Threshold="0")
        self.settings['gain']='hgm'
        self.threshold_calibration(chips,Threshold="0")
        self.settings['gain']='lgm'
        self.threshold_calibration(chips,Threshold="0")
        self.settings['gain']='slgm'
        self.threshold_calibration(chips,Threshold="0")


    def threshold_calibration(self,chips=range(8),Threshold="0",):
        """
        This functions produces threshold calibration data required to convert an X-ray energy detection threshold in keV into threshold DAC units
        """
        x.checkCalibDir()
        
        NbofEnergyPoints=1
        default6keVDAC=62
        
        Energy=np.ones(2)
        E0=0
        Dac0=self.dacTarget*np.ones([6,8]).astype('float')
        
        E1=5.9#keV
        if self.settings['gain']=='shgm':
            Dac1=Dac0+1*(default6keVDAC-Dac0)*np.ones([6,8]).astype('float')
        if self.settings['gain']=='hgm':
            Dac1=Dac0+0.75*(default6keVDAC-Dac0)*np.ones([6,8]).astype('float')
        if self.settings['gain']=='lgm':
            Dac1=Dac0+0.5*(default6keVDAC-Dac0)*np.ones([6,8]).astype('float')
        if self.settings['gain']=='slgm':
            Dac1=Dac0+0.25*(default6keVDAC-Dac0)*np.ones([6,8]).astype('float')
        
        print str(E0)
        print str(Dac0)
        print str(E1)
        print str(Dac1)
        
        slope=(Dac1[self.fem-1,:]-Dac0[self.fem-1,:])/(E1-E0)
        offset=Dac0[self.fem-1,:]
        self.save_keV2dac_calib(Threshold,slope,offset)
        print str(slope) + str(offset)
        
        return 
        
    def save_keV2dac_calib(self,Threshold,gain,offset):
        """
        Each Threshold calibration file consists of 2 rows of 8 floating point numbers:
        # g0     g1   g2   g3   g4   g5   g6   g7
        # Off0 Off1 Off2 Off3 Off4 Off5 Off6 Off7 
        # and the DAC value to apply to chip x for a requested threshold energy value E in keV is given by:
        # DACx= gx * E + Offx 
        """
        threshCoeff=np.zeros([2,8])
        threshCoeff[0,:]=gain
        threshCoeff[1,:]=offset
        threshFilename=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/'  + 'threshold'+ str(Threshold)
        if os.path.isfile(threshFilename):
            np.savetxt(threshFilename,threshCoeff,fmt='%.2f')
        else:
            np.savetxt(threshFilename,threshCoeff,fmt='%.2f')
            os.chmod(threshFilename,0777)#First time the file is created. permissions need to be changed to allow anyone to overwrite calibration data
        return gain,offset


    def Fe55ThreshCalib(self,chips=range(8),Threshold="Threshold0"):
        
        self.settings['acqtime']=1000
        dacRange=(self.dacTarget+80,self.dacTarget+20,2)
        self.loadConfig(range(8))
        self.settings['filename']='Fe55_Threshold_scan'
        [dacscanData,scanRange]=self.scanDac(chips,"Threshold"+str(Threshold),dacRange)
        edgeDacs=self.findEdge(chips,dacscanData,dacRange,2)        
        chipEdgeDacs=np.zeros([len(chips)])
        for chip in chips:
            chipEdgeDacs[chip]=edgeDacs[0:256,chip*256:chip*256+256].mean()
        
        Fe55_Dac=np.ones([6,8]).astype('float')
        Fe55_E=6#keV
        
        Fe55_Dac[0,:]=[62,62,62,62,62,62,62,62]
        Fe55_Dac[1,:]=[64,64,64,64,64,64,64,64]
        Fe55_Dac[2,:]=[62,62,62,62,62,62,62,62]
        Fe55_Dac[3,:]=[60,35,64,64,64,64,64,64]
        Fe55_Dac[4,:]=[64,64,64,64,64,64,64,64]
        Fe55_Dac[5,:]=[64,64,64,64,64,64,64,64]
        
        slope=(Fe55_Dac[self.fem-1,:]-self.dacTarget)/Fe55_E
        offset=[self.dacTarget,self.dacTarget,self.dacTarget,self.dacTarget,self.dacTarget,self.dacTarget,self.dacTarget,self.dacTarget]
        self.save_keV2dac_calib(Threshold,slope,offset)
        
        print str(slope) + str(offset)
        
        self.settings['filename']='image'
        return 


    def arbThreshCalib(self,chips=range(8),Threshold="Threshold0"):
        """ 
        Simple function wich performs a DAC scan and align the noise edge of all th echips using offset parameter and put edges at an arbitrary value using slope 
        chipEdgeDac=np.array([10,20,20,20,20,20,20,20])
        offset= [0,0,0,0,0,0,0,0]
        """
        dacRange=(self.dacTarget+10,self.dacTarget,1)
        x.loadConfig(range(8))
        [dacscanData,scanRange]=self.scanDac(chips,"Threshold"+str(Threshold),dacRange)
        edgeDacs=self.findEdge(chips,dacscanData,dacRange,2)        
        chipEdgeDacs=np.zeros([len(chips)])
        for chip in chips:
            chipEdgeDacs[chip]=edgeDacs[0:256,chip*256:chip*256+256].mean()
        offset= chipEdgeDacs.mean()-chipEdgeDacs
        E=float(self.arbNoiseEdgeEnergySPM[self.settings['gain']])
        gain = chipEdgeDacs/E
        self.save_keV2dac_calib(Threshold,gain,offset)
        return chipEdgeDacs
        
    def setTh0(self,threshEnergy=5):
        x.setThreshEnergy(0,threshEnergy)
        
    def setThreshEnergy(self,Threshold="0",threshEnergy=5):
        threshCoeff= np.genfromtxt(self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/'  + 'threshold'+ str(Threshold))
        threshDACs=(threshEnergy*threshCoeff[0,:]+threshCoeff[1,:]).astype(np.int)
        for chip in range(8):
            self.setDac(range(chip,chip+1),"Threshold"+str(Threshold), threshDACs[chip])
        time.sleep(0.2)
        self.settings['acqtime']='100'
        self.expose()
        time.sleep(0.2)
        self.expose()
        print  "A Threshold" + str(Threshold) + " of " + str(threshEnergy) + "keV corresponds to " + str(threshDACs) + " DAC units for each chip"
        return threshDACs
        



    chipSize=256
    nbOfChips=8
    command="/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/excaliburTestApp"

    port='6969'
    chipRange=range(0,8)
    plotname=''
    dacTarget=10
    edgeVal=10
    accDist=4
    nbOfSigma=3.2 # based on experimental data


    calibSettings={'calibDir':'/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/calib/',#'/tmp/'
                   'configDir':'/dls/detectors/support/silicon_pixels/excaliburRX/TestApplication_15012015/config/',
                   'dacfilename':'dacs',
                   'dacfile':'',
                   'noiseEdge':'10'} #Default .dacs file. This will be overwritten by setThreshold function


    settings={'mode':'spm',#'spm','csm'
              'gain':'shgm',#'slgm','lgm','hgm',shgm'
              'bitdepth':'12',# 24 bits needs disccsmspm at 1 to use discL 
              'readmode':'0',
              'counter':'0',
              'disccsmspm':'0',
              'equalization':'0',
              'trigmode':'0',
              'acqtime':'100',
              'frames':'1',
              'imagepath':'/tmp/',#'/tmp/rob/'############## WHEN CREATING IMAGE.IDX FILE FIRST TIME USE CHMOD TO CHANGE PERMISSION
              'filename':'image',
              'Threshold':'Not set',
              'filenameIndex':''}
    
    modeCode={'spm':'0',
            'csm':'1'}
    
    gainCode={'shgm':'0',
              'hgm':'1',
              'lgm':'2',
              'slgm':'3'}
    
    dacNumber={'Threshold0':'1',
            'Threshold1':'2',
            'Threshold2':'3',
            'Threshold3':'4',
            'Threshold4':'5',
            'Threshold5':'6',
            'Threshold6':'7',
            'Threshold7':'8',
            'Preamp':'9',
            'Ikrum':'10',
            'Shaper':'11',
            'Disc':'12',
            'DiscLS':'13',
            'ShaperTest':'14',
            'DACDiscL':'15',
            'DACTest':'16',
            'DACDiscH':'17',
            'Delay':'18',
            'TPBuffIn':'19',
            'TPBuffOut':'20',
            'RPZ':'21',
            'GND':'22',
            'TPREF':'23',
            'FBK':'24',
            'Cas':'25',
            'TPREFA':'26',
            'TPREFB':'27'}

    dacCode={'Threshold0':'1',
            'Threshold1':'2',
            'Threshold2':'3',
            'Threshold3':'4',
            'Threshold4':'5',
            'Threshold5':'6',
            'Threshold6':'7',
            'Threshold7':'8',
            'Preamp':'9', # To check
            'Ikrum':'10',
            'Shaper':'11',
            'Disc':'12',
            'DiscLS':'13',
            'ShaperTest':'14',
            'DACDiscL':'15',
            'DACTest':'30',
            'DACDiscH':'31',
            'Delay':'16',
            'TPBuffIn':'17',
            'TPBuffOut':'18',
            'RPZ':'19',
            'GND':'20',
            'TPREF':'21',
            'FBK':'22',
            'Cas':'23',
            'TPREFA':'24',
            'TPREFB':'25'}

    arbNoiseEdgeEnergySPM={'shgm':'2.5',
                           'hgm':'3',
                          'lgm':'3.5',
                        'slgm':'4'}
    
    def setdacs(self,chips):
        self.setDac(chips,'Threshold1', 0)
        self.setDac(chips,'Threshold2', 0)
        self.setDac(chips,'Threshold3', 0)
        self.setDac(chips,'Threshold4', 0)
        self.setDac(chips,'Threshold5', 0)
        self.setDac(chips,'Threshold6', 0)
        self.setDac(chips,'Threshold7', 0)
        self.setDac(chips,'Preamp', 175)# Could use 200  
        self.setDac(chips,'Ikrum', 10)
        self.setDac(chips,'Shaper', 150)
        self.setDac(chips,'Disc', 125)
        self.setDac(chips,'DiscLS', 100)
        self.setDac(chips,'ShaperTest', 0)
        self.setDac(chips,'DACDiscL', 90)
        self.setDac(chips,'DACTest', 0)
        self.setDac(chips,'DACDiscH', 90)
        self.setDac(chips,'Delay', 30)
        self.setDac(chips,'TPBuffIn', 128)
        self.setDac(chips,'TPBuffOut', 4)
        self.setDac(chips,'RPZ', 255) # RPZ is disabled at 255
        self.setDac(chips,'TPREF', 128)
        self.setDac(chips,'TPREFA', 500)
        self.setDac(chips,'TPREFB', 500)

    def mask(self,chips):
        """
        Creates hexadecimal chip mask corresponding to chips to be enabled when sending commands to the front-end
        Usage: maskHex=x.mask([0]) to  get the mask value to enable chip 0 only ('0x80')
        maskHex=x.mask([0,1,2,3,4,5,6,7]) or maskHex=x.mask(range(8)) to  get the mask value to enable all chips ('0xff')
        """
        maskHex=0
        for chip in chips:
            maskHex=2**(self.nbOfChips-chip-1)+maskHex
        return str(hex(maskHex))

    def chipId(self):
        """
        Reads chip IDs
        Usage: x.chipId(chips) where chips is a list of chips i.e. x.chipId(range(8))
        """
        subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(self.chipRange),"-r","-e"])
        print(str(self.chipRange))
    
    def logChipId(self):
        logfilename=self.calibSettings['calibDir']+'fem'+str(self.fem)+'/efuseIDs'
        with open(logfilename, "w") as outfile:
            subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(self.chipRange),"-r","-e"],stdout=outfile)
        print(str(self.chipRange))
    
    def monitor(self):
        subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(self.chipRange),"--slow"])
    
    def setTh0Dac(self,chips=range(8),dacValue=40):
        """
        This function sets Threshold 0 DAC to a selected value for all chips  
        Usage: x.setTh0Dac(30)
        """
        self.setDac(chips,'Threshold0', dacValue)
        x.expose()
        
    
    def Fe55imageRX001(self,chips=range(8),exptime=60000):
        """
        THIS WILL SAVE FE55 IMAGE IN /dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/ DIRECTORY
        """
        imgpath=self.settings['imagepath']
        self.settings['gain']='shgm'
        self.loadConfig(chips)
        self.setTh0Dac(chips,40)
        self.settings['acqtime']=str(exptime)
        self.settings['filename']='Fe55_image_node_'+str(self.fem)+'_'+str(self.settings['acqtime'])+'s'
        self.settings['imagepath']='/dls/detectors/support/silicon_pixels/excaliburRX/3M-RX001/Fe55_images/'
        print (self.settings['acqtime'])
        self.settings['acqtime']=str(exptime)
        time.sleep(0.5)
        self.expose()
        self.settings['imagepath']=imgpath
        self.settings['filename']='image'
        self.settings['acqtime']='100'
        return
    
    def setDac(self,chips,dacName="Threshold0",dacValue=40):
        """
        Usage: x.setDac([0],'Threshold0', 30)
               x.setDac(range(8),'Threshold1', 100)
        """
        for chip in chips:
            dacFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + self.calibSettings['dacfilename']
            f = open(dacFile , 'r+b')
            f_content = f.readlines()
            lineNb=chip*29+np.int(self.dacNumber[dacName])
            oldLine=f_content[lineNb]
            newLine=dacName + " = " + str(dacValue) + "\r\n"
            f_content[lineNb]=newLine
            f.seek(0)
            f.writelines(f_content)
            f.close()
            subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--dacs="+dacFile])
        return

    def readDac(self,chips,dacName):
            dacFile=self.calibSettings['calibDir']+ 'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + self.calibSettings['dacfilename']
            subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(chips),"--sensedac="+str(np.int(self.dacCode[dacName])),"--dacs="+dacFile])
            time.sleep(1)
            subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(self.chipRange),"--sensedac="+str(np.int(self.dacCode[dacName])),"--slow"])
            return

    def plotDacScan(self,chips,dacScanData,dacRange):
        dnp.plot.clear("dacScan")
        dnp.plot.clear("Spectrum")
        chipDacScan=np.zeros([8])
        print str()
        if dacRange[0]>dacRange[1]:
            for chip in chips:
                dacAxis=(np.array(range(dacRange[0],dacRange[1]-dacRange[2],-dacRange[2])))
                chipDacScan=np.zeros([8,dacAxis.size])
                chipDacScan[chip,:]=(dacScanData[:,0:256,chip*256:chip*256+256].mean(2).mean(1))
                dnp.plot.addline(np.array(range(dacRange[0],dacRange[1]-dacRange[2],-dacRange[2])),np.squeeze(dacScanData[:,0:256,chip*256:chip*256+256].mean(2).mean(1)),name="dacScan")
                spectrum=np.diff(np.squeeze(dacScanData[:,0:256,chip*256:chip*256+256].mean(2).mean(1)))
            #for chip in chips:
                dnp.plot.addline(np.array(range(dacRange[0],dacRange[1],-dacRange[2]))[1:],spectrum[1:],name="Spectrum")
        else:
            for chip in chips:
                dacAxis=(np.array(range(dacRange[0],dacRange[1]+dacRange[2],dacRange[2])))
                chipDacScan=np.zeros([8,dacAxis.size])
                chipDacScan[chip,:]=(dacScanData[:,0:256,chip*256:chip*256+256].mean(2).mean(1))
                dnp.plot.addline(np.array(range(dacRange[0],dacRange[1]+dacRange[2],dacRange[2])),np.squeeze(dacScanData[:,0:256,chip*256:chip*256+256].mean(2).mean(1)),name="dacScan")
                spectrum=-np.diff(np.squeeze(dacScanData[:,0:256,chip*256:chip*256+256].mean(2).mean(1)))
            #for chip in chips:
                dnp.plot.addline(np.array(range(dacRange[0],dacRange[1],dacRange[2]))[1:],spectrum[1:],name="Spectrum")
        return chipDacScan,dacAxis


    def scanDac(self,chips,dacName,dacRange):# ONLY FOR THRESHOLD DACS
        self.updateFilenameIndex()
        dacScanFile=self.settings['filename']+"_"+ "dacscan" +self.settings['filenameIndex']+".hdf5"
        dacFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + self.calibSettings['dacfilename']
        string=str(np.int(self.dacCode[dacName])-1)+','+str(dacRange[0])+','+str(dacRange[1])+','+str(dacRange[2])
        subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(chips),"--dacs="+dacFile,"--csmspm="+self.modeCode[self.settings['mode']],"--disccsmspm="+self.settings['disccsmspm'],"--depth=12","--equalization="+self.settings['equalization'],"--counter="+self.settings['counter'],"--acqtime="+str(self.settings['acqtime']),"--gainmode="+self.gainCode[self.settings['gain']],"--dacscan",string,"--path="+self.settings['imagepath'],"--hdffile="+dacScanFile])
        time.sleep(1)
        dh = dnp.io.load(self.settings['imagepath']+dacScanFile)
        dacScanData=dh.image[...]
        dnp.plot.clear("dacScan")
        dnp.plot.clear("Spectrum")
        if dacRange[0]>dacRange[1]:
            for chip in chips:
                dnp.plot.addline(np.array(range(dacRange[0],dacRange[1]-dacRange[2],-dacRange[2])),np.squeeze(dacScanData[:,0:256,chip*256:chip*256+256].mean(2).mean(1)),name="dacScan")
                spectrum=np.diff(np.squeeze(dacScanData[:,0:256,chip*256:chip*256+256].mean(2).mean(1)))
            #for chip in chips:
                dnp.plot.addline(np.array(range(dacRange[0],dacRange[1],-dacRange[2]))[1:],spectrum[1:],name="Spectrum")
        else:
            for chip in chips:
                dnp.plot.addline(np.array(range(dacRange[0],dacRange[1]+dacRange[2],dacRange[2])),np.squeeze(dacScanData[:,0:256,chip*256:chip*256+256].mean(2).mean(1)),name="dacScan")
                spectrum=-np.diff(np.squeeze(dacScanData[:,0:256,chip*256:chip*256+256].mean(2).mean(1)))
            #for chip in chips:
                dnp.plot.addline(np.array(range(dacRange[0],dacRange[1],dacRange[2]))[1:],spectrum[1:],name="Spectrum")
        return [dacScanData,dacRange]
    
    def showPixel(self,dacscanData,dacRange,pixel):#xclbr.showPixel(dacScanData,[0,30,1],[20,20])
        dnp.plot.addline(np.array(range(dacRange[0],dacRange[1]+dacRange[2],dacRange[2])),(dacscanData[:,pixel[0],pixel[1]]),name='Pixel S curve')
        spectrum=-np.diff(np.squeeze(dacscanData[:,pixel[0],pixel[1]]))
        dnp.plot.addline(np.array(range(dacRange[0],dacRange[1],dacRange[2])),spectrum,name="Pixel Spectrum")

    def loadDiscbits(self,chips,discName,discbits):
        if type(discbits)==str:
            discbitsFile=discbits
        else:
            discbitsFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + discName +'bits.tmp'
            np.savetxt(discbitsFile,discbits,fmt='%.18g', delimiter=' ' )
        for chip in chips:
            if discName=="discL":
                subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--config","--discl="+discbitsFile])
            if discName=="discH":
                subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--config","--disch="+discbitsFile])

    def loadConfigbits(self,chips,discLbits,discHbits,maskbits):
        discLbitsFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' +  'discLbits.tmp'
        discHbitsFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' +  'discHbits.tmp'
        maskbitsFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' +  'maskbits.tmp'
        np.savetxt(discLbitsFile,discLbits,fmt='%.18g', delimiter=' ' )
        np.savetxt(discHbitsFile,discHbits,fmt='%.18g', delimiter=' ' )
        np.savetxt(maskbitsFile,maskbits,fmt='%.18g', delimiter=' ' )
        for chip in chips:
                subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--config","--discl="+discLbitsFile,"--disch="+discHbitsFile,"--pixelmask="+maskbitsFile])

    def loadConfig(self,chips=range(8)):
        for chip in chips:
            discHbitsFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'discHbits.chip'+str(chip)
            discLbitsFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'discLbits.chip'+str(chip)
            pixelmaskFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'pixelmask.chip'+str(chip)
            
            if os.path.isfile(discLbitsFile):
                if os.path.isfile(discHbitsFile):
                    if os.path.isfile(pixelmaskFile):
                        subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--config","--discl="+discLbitsFile,"--disch="+discHbitsFile,"--pixelmask="+pixelmaskFile])
                    else:
                        subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--config","--discl="+discLbitsFile,"--disch="+discHbitsFile])
                else:
                    if os.path.isfile(pixelmaskFile):
                        subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--config","--discl="+discLbitsFile,"--pixelmask="+pixelmaskFile])
                    else:
                        subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--config","--discl="+discLbitsFile])
            else: 
                print str(discLbitFile) + " does not exist !"

        self.setDac(range(8),"Threshold1", 100)
        self.setDac(range(8),"Threshold0", 40)
        self.expose()
        return
        
    def updateFilenameIndex(self):
        idxFilename=self.settings['imagepath']+self.settings['filename']+'.idx'
        print(idxFilename)
        if os.path.isfile(idxFilename):
            print os.path.isfile(idxFilename)
            file=open(idxFilename, 'rw')
            newIdx=int(file.read())+1
            file.close()
            file=open(idxFilename, 'w')
            file.write(str(newIdx))
            file.close()
            self.settings['filenameIndex']=str(newIdx)
        else: 
            #print os.path.isfile(idxFilename)
            file=open(idxFilename, 'a')
            newIdx=0
            file.write(str(newIdx))
            file.close()
            self.settings['filenameIndex']=str(newIdx)
            os.chmod(idxFilename,0777)
        return newIdx

    def acquireFF(self,ni,acqtime):
        self.settings['fullFilename']="FlatField.hdf5"
        self.settings['acqtime']=acqtime
        FFimage=0
        for n in range(ni):
            image=self.expose()
            FFimage=FFimage+image
        chip=3
        FFimage[FFimage==0]=FFimage[0:256,chip*256:chip*256+256].mean()
        FFcoeff=np.ones([256,256*8])*FFimage[0:256,chip*256:chip*256+256].mean()
        FFcoeff=FFcoeff/FFimage
        dnp.plot.image(FFcoeff[0:256,chip*256:chip*256+256],name='Flat Field coefficients')
        FFcoeff[FFcoeff>2]=1
        FFcoeff[FFcoeff<0]=1
        return FFcoeff

    def burst(self,frames,acqtime):
        self.settings['acqtime']=acqtime
        self.updateFilenameIndex()
        self.settings['frames']=frames
        self.settings['fullFilename']=self.settings['filename']+"_"+self.settings['filenameIndex']+".hdf5"
        string=[self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(self.chipRange),"--depth="+self.settings['bitdepth'],"--csmspm="+self.modeCode[self.settings['mode']],"--disccsmspm="+self.settings['disccsmspm'],"--equalization="+self.settings['equalization'],"--gainmode="+self.gainCode[self.settings['gain']],"--burst","--frames="+str(self.settings['frames']),"--acqtime="+str(self.settings['acqtime']),"--trigmode="+self.settings['trigmode'],"--path="+self.settings['imagepath'],"--hdffile="+self.settings['fullFilename']]
        print(string)
        subprocess.call(string)
        time.sleep(0.5)

    def expose(self):
        print(self.settings)
        self.updateFilenameIndex()
        self.settings['frames']='1'
        self.settings['fullFilename']=self.settings['filename']+"_"+self.settings['filenameIndex']+".hdf5"
        subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(self.chipRange),"--depth="+self.settings['bitdepth'],"--csmspm="+self.modeCode[self.settings['mode']],"--disccsmspm="+self.settings['disccsmspm'],"--equalization="+self.settings['equalization'],"--gainmode="+self.gainCode[self.settings['gain']],"--acquire","--frames="+str(self.settings['frames']),"--acqtime="+str(self.settings['acqtime']),"--trigmode="+self.settings['trigmode'],"--path="+self.settings['imagepath'],"--hdffile="+self.settings['fullFilename']])
        print self.settings['filename']
        time.sleep(0.5)
        dh = dnp.io.load(self.settings['imagepath'] + self.settings['fullFilename'])
        imageRaw=dh.image[...]
        image=dnp.squeeze(imageRaw.astype(np.int))
        dnp.plot.clear()
        dnp.plot.image(image,name='Image data')
        return image

    def shoot(self,acqtime):
        self.settings['acqtime']=acqtime
        self.settings['frames']='1'
        self.updateFilenameIndex()
        self.settings['fullFilename']=self.settings['filename']+"_"+self.settings['filenameIndex']+".hdf5"
        subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(self.chipRange),"--depth="+self.settings['bitdepth'],"--gainmode="+self.gainCode[self.settings['gain']],"--acquire","--frames="+str(self.settings['frames']),"--acqtime="+str(self.settings['acqtime']),"--path="+self.settings['imagepath'],"--hdffile="+self.settings['fullFilename']])
        print self.settings['filename']
        time.sleep(0.2)
        dh = dnp.io.load(self.settings['imagepath'] + self.settings['fullFilename'])
        imageRaw=dh.image[...]
        image=dnp.squeeze(imageRaw.astype(np.int))
        dnp.plot.clear()
        dnp.plot.image(image,name='Image data')
        return self.settings['fullFilename']
    
    def cont(self,frames,acqtime):        
        self.settings['acqtime']=acqtime
        self.updateFilenameIndex()
        self.settings['frames']=frames
        self.settings['readmode']='1'
        self.settings['fullFilename']=self.settings['filename']+"_"+self.settings['filenameIndex']+".hdf5"
        subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(self.chipRange),"--depth="+self.settings['bitdepth'],"--gainmode="+self.gainCode[self.settings['gain']],"--acquire","--readmode="+str(self.settings['readmode']),"--counter="+str(self.settings['counter']),"--frames="+str(self.settings['frames']),"--acqtime="+str(self.settings['acqtime']),"--trigmode="+self.settings['trigmode'],"--path="+self.settings['imagepath'],"--hdffile="+self.settings['fullFilename']])
        #subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(self.chipRange),"--depth="+self.settings['bitdepth'],"--gainmode="+self.gainmode[self.settings['gain']],"--burst","--readmode="+str(self.settings['readmode']),"--counter="+str(self.settings['counter']),"--frames="+str(self.settings['frames']),"--acqtime="+str(self.settings['acqtime']),"--path="+self.settings['imagepath'],"--hdffile="+self.settings['fullFilename']])
        print self.settings['filename']
        time.sleep(0.2)
        dh = dnp.io.load(self.settings['imagepath'] + self.settings['fullFilename'])
        imageRaw=dh.image[...]
        image=dnp.squeeze(imageRaw.astype(np.int))
        plots=5
        if frames<5:
            plots=frames
        for p in range(plots):
            #dnp.plot.clear()
            dnp.plot.image(image[p,:,:],name='Image data '+ str(p))
        return 
    
    def contBurst(self,frames,acqtime):        
        self.settings['acqtime']=acqtime
        self.updateFilenameIndex()
        self.settings['frames']=frames
        self.settings['fullFilename']=self.settings['filename']+"_"+self.settings['filenameIndex']+".hdf5"
        subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(self.chipRange),"--depth="+self.settings['bitdepth'],"--gainmode="+self.gainCode[self.settings['gain']],"--burst","--readmode="+str(self.settings['readmode']),"--counter="+str(self.settings['counter']),"--frames="+str(self.settings['frames']),"--acqtime="+str(self.settings['acqtime']),"--trigmode="+self.settings['trigmode'],"--path="+self.settings['imagepath'],"--hdffile="+self.settings['fullFilename']])
    #subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(self.chipRange),"--depth="+self.settings['bitdepth'],"--gainmode="+self.gainmode[self.settings['gain']],"--burst","--readmode="+str(self.settings['readmode']),"--counter="+str(self.settings['counter']),"--frames="+str(self.settings['frames']),"--acqtime="+str(self.settings['acqtime']),"--path="+self.settings['imagepath'],"--hdffile="+self.settings['fullFilename']])
#         print self.settings['filename']
#         time.sleep(0.2)
#         dh = dnp.io.load(self.settings['imagepath'] + self.settings['fullFilename'])
#         imageRaw=dh.image[...]
#         image=dnp.squeeze(imageRaw.astype(np.int))
#         plots=5
#         if frames<5:
#             plots=frames
#             for p in range(plots):
#                 #dnp.plot.clear()
#                 dnp.plot.image(image[p,:,:],name='Image data '+ str(p))
#             return 
    
    def FFcorrect(self,ni,FFcoeff):
        self.settings['fullFilename']=self.settings['filename']+"_"+self.settings['filenameIndex']+".hdf5"    
        dh = dnp.io.load(self.settings['imagepath'] + self.settings['fullFilename'])
        imageRaw=dh.image[...]
        image=dnp.squeeze(imageRaw.astype(np.int))
        for p in range(ni):
            dnp.plot.clear()
            FF=image[p,:,:]*FFcoeff
            FF=image[p,:,:]
            FF[FF>3000]=0
            chip=3
            dnp.plot.image(FF[0:256,chip*256:chip*256+256],name='Image data Cor')
            time.sleep(1)
        return 

    def init(self):
        chips=range(x.nbOfChips)
        x.setDac(chips,"Threshold0", 40)
        x.shoot(10)
        LogoTP=np.ones([256,8*256])
        logoSmall=np.loadtxt(self.calibSettings['configDir']+'logo.txt')
        LogoTP[7:250,225:1823]=logoSmall
        LogoTP[LogoTP>0]=1
        LogoTP=1-LogoTP
        
        for chip in chips:
            testbitsFile=self.calibSettings['calibDir']+'Logo_chip'+str(chip)+'_mask'
            np.savetxt(testbitsFile,LogoTP[0:256,chip*256:chip*256+256],fmt='%.18g', delimiter=' ' )
            
            discHbitsFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'discHbits.chip'+str(chip)
            discLbitsFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'discLbits.chip'+str(chip)
            pixelmaskFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'pixelmask.chip'+str(chip)
            
            if os.path.isfile(discLbitsFile) and os.path.isfile(discHbitsFile) and os.path.isfile(pixelmaskFile):
                subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--dacs",self.calibSettings['calibDir']+self.calibSettings['dacfilename'],"--config","--discl="+discLbitsFile,"--disch="+discHbitsFile,"--pixelmask="+pixelmaskFile,"--tpmask="+testbitsFile])
            else :
                subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--dacs",self.calibSettings['calibDir']+self.calibSettings['dacfilename'],"--config","--tpmask="+testbitsFile])

        self.settings['fullFilename']=self.settings['filename']+"_"+self.settings['filenameIndex']+".hdf5"
        time.sleep(0.2)
        subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(chips),"--depth="+self.settings['bitdepth'],"--acquire","--readmode="+str(self.settings['readmode']),"--frames="+str(self.settings['frames']),"--acqtime="+str(self.settings['acqtime']),"--counter=0","--path="+self.settings['imagepath'],"--hdffile="+self.settings['fullFilename'],"--tpcount="+str(100)])
        dh = dnp.io.load(self.settings['imagepath'] + self.settings['fullFilename'])
        imageRaw=dh.image[...]
        image=dnp.squeeze(imageRaw.astype(np.int))
        dnp.plot.image(image,name='Image data')

    def testPulse(self,chips,testbits,pulses):#xclbr.testPulse([0],'excaliburRx/config/triangle.mask',1000)
        if type(testbits)==str:
            testbitsFile=testbits
        else:
            discLbitsFile=self.calibSettings['calibDir']+  'fem' + str(xclbr.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'testbits.tmp'
            np.savetxt(testbitsFile,testbits,fmt='%.18g', delimiter=' ' )
        dnp.plot.clear()
        #self.updateFilenameIndex()
        self.settings['fullFilename']=self.settings['filename']+"_"+self.settings['filenameIndex']+".hdf5"
        for chip in chips:
            subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--dacs",self.calibSettings['calibDir']+self.calibSettings['dacfilename'],"--config","--tpmask="+testbitsFile])
        subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(chips),"--depth="+self.settings['bitdepth'],"--acquire","--readmode="+str(self.settings['readmode']),"--frames="+str(self.settings['frames']),"--acqtime="+str(self.settings['acqtime']),"--counter=0","--path="+self.settings['imagepath'],"--hdffile="+self.settings['fullFilename'],"--tpcount="+str(pulses)])
        print self.settings['fullFilename']
        dh = dnp.io.load(self.settings['imagepath'] + self.settings['fullFilename'])
        imageRaw=dh.image[...]
        image=dnp.squeeze(imageRaw.astype(np.int))
         
        dnp.plot.image(image,name='Image data')
        return image
# 
    def saveDiscbits(self,chips,discbits,discbitsFilename):
        for chip in chips:
            discbitsFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/' +self.settings['mode'] + '/' + self.settings['gain']+ '/' + discbitsFilename + '.chip' + str(chip)
            
            #if self.fem%2==1:
            #    discbits=np.rot90(discbits,2)# Mirror discbits array for odd node numbers because EPICS mirrors the array before loading the discbits for these nodes
            
            
            np.savetxt(discbitsFile,discbits[0:256,chip*256:chip*256+256],fmt='%.18g', delimiter=' ' )

    def maskSupCol(self,chip,supCol):# bit=1 to mask a pixel
        badPixels=np.zeros([xclbr.chipSize,xclbr.chipSize*xclbr.nbOfChips])
        badPixels[:,chip*256+supCol*32:chip*256+supCol*32+64]=1
        discLbitsFile=self.calibSettings['calibDir']+  'fem' + str(xclbr.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'discLbits.chip'+str(chip)
        pixelmaskFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/' +self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'pixelmask.chip'+str(chip)
        np.savetxt(pixelmaskFile,badPixels[0:256,chip*256:chip*256+256],fmt='%.18g', delimiter=' ' )
        subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--config","--pixelmask="+pixelmaskFile,"--config","--discl="+discLbitsFile])
        dnp.plot.image(badPixels)

    def maskCol(self,chip,Col):# bit=1 to mask a pixel
        badPixels=np.zeros([self.chipSize,self.chipSize*self.nbOfChips])
        badPixels[:,Col]=1
        discLbitsFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'discLbits.chip'+str(chip)
        pixelmaskFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/' +self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'pixelmask.chip'+str(chip)
        np.savetxt(pixelmaskFile,badPixels[0:256,chip*256:chip*256+256],fmt='%.18g', delimiter=' ' )
        subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--config","--pixelmask="+pixelmaskFile,"--config","--discl="+discLbitsFile])
        dnp.plot.image(badPixels,name='Bad pixels')
        
    def maskPixels(self,chips,imageData,maxCounts): 
        badPixTot=np.zeros(8)
        badPixels=imageData>maxCounts
        dnp.plot.image(badPixels,name='Bad pixels')
        for chip in chips:
            discLbitsFile=self.calibSettings['calibDir']+  'fem' + str(xclbr.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'discLbits.chip'+str(chip)
            badPixTot[chip]=badPixels[0:256,chip*256:chip*256+256].sum()
            print ('####### ' + str(badPixTot[chip]) + ' noisy pixels in chip ' + str(chip) + ' (' + str(100*badPixTot[chip]/(256**2)) + '%)')
            pixelmaskFile=xclbr.calibSettings['calibDir']+  'fem' + str(xclbr.fem) +'/' +xclbr.settings['mode'] + '/' + xclbr.settings['gain']+ '/' + 'pixelmask.chip'+str(chip)
            np.savetxt(pixelmaskFile,badPixels[0:256,chip*256:chip*256+256],fmt='%.18g', delimiter=' ' )
            subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--config","--pixelmask="+pixelmaskFile,"--config","--discl="+discLbitsFile])
        print ('####### ' + str(badPixTot.sum()) + ' noisy pixels in half module ' + ' (' + str(100*badPixTot.sum()/(8*256**2)) + '%)')

    def findNoiseMeanDAC(self,chips=range(8),Threshold="Threshold0",dacRange=(80,20,2)):
        self.settings['acqtime']=100
        [dacScanData,scanRange]=self.scanDac(chips,Threshold,dacRange)
        edgeDACs=self.findEdge(chips,dacScanData,dacRange,self.edgeVal)
        return edgeDACs.mean()

    def maskPixelsUsingDACscan(self,chips=range(8),Threshold="Threshold0",dacRange=(20,120,2)):# dacRange=(20,120,2)
        badPixTot=np.zeros(8)
        self.settings['acqtime']=100
        [dacScanData,scanRange]=self.scanDac(chips,Threshold,dacRange)
        badPixels=dacScanData.sum(0)>1
        dnp.plot.image(badPixels,name='Bad pixels')
    
        for chip in chips:
            badPixTot[chip]=badPixels[0:256,chip*256:chip*256+256].sum()
            print ('####### ' + str(badPixTot[chip]) + ' noisy pixels in chip ' + str(chip) + ' (' + str(100*badPixTot[chip]/(256**2)) + '%)')
            pixelmaskFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/' +self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'pixelmask.chip'+str(chip)
            np.savetxt(pixelmaskFile,badPixels[0:256,chip*256:chip*256+256],fmt='%.18g', delimiter=' ' )
            #subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--config","--pixelmask="+pixelmaskFile,"--config","--discl="+discLbitsFile])
        print ('####### ' + str(badPixTot.sum()) + ' noisy pixels in half module ' + ' (' + str(100*badPixTot.sum()/(8*256**2)) + '%)')

    def unmaskAllPixels(self,chips): 
        badPixels=np.zeros([xclbr.chipSize,xclbr.chipSize*xclbr.nbOfChips])
        for chip in chips:
            discLbitsFile=self.calibSettings['calibDir']+  'fem' + str(xclbr.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'discLbits.chip'+str(chip)
            pixelmaskFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/' +self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'pixelmask.chip'+str(chip)
            np.savetxt(pixelmaskFile,badPixels[0:256,chip*256:chip*256+256],fmt='%.18g', delimiter=' ' )
            subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--config","--pixelmask="+pixelmaskFile,"--config","--discl="+discLbitsFile])

    def unEqualizeAllPixels(self,chips): 
        discLbits=31*np.zeros([xclbr.chipSize,xclbr.chipSize*xclbr.nbOfChips])
        for chip in chips:
            discLbitsFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'discLbits.chip'+str(chip)
            pixelmaskFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/' +self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'pixelmask.chip'+str(chip)
            np.savetxt(discLbitsFile,discLbits[0:256,chip*256:chip*256+256],fmt='%.18g', delimiter=' ' )
            subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--config","--pixelmask="+pixelmaskFile,"--config","--discl="+discLbitsFile])

    def checkCalibDir(self):
        dir=(self.calibSettings['calibDir']+ 'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/')
        if (os.path.isdir(dir))==0:
            os.makedirs(dir)
        else:
            backupDir=self.calibSettings['calibDir'][:-1]+'_backup_'+time.asctime()
            shutil.copytree(self.calibSettings['calibDir'],backupDir)
            print backupDir
        dacFile=dir+self.calibSettings['dacfilename']
        if os.path.isfile(dacFile)==0:
            shutil.copy(self.calibSettings['configDir']+self.calibSettings['dacfilename'],dir)
            
        discfile=dir+'discLbits.tmp'
        #if os.path.isfile(dacFile)==0:
        #    shutil.copy(self.calibSettings['configDir']+'zeros.mask',dir)
        return dacFile

    def copy_SLGM_into_other_gain_modes(self):
        """
        The functions simply copies /femx/slgm calibration folder into /femx/lgm, /femx/hgm and /femx/shgm calibration folders
        This function is used at the end of threshold equalization because threshold equalization is performed in the more favorable gain mode slgm
        and  threshold equalization data is independent of the gain mode
        """
        slgmDir=(self.calibSettings['calibDir']+ 'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + 'slgm'+ '/')
        lgmDir=(self.calibSettings['calibDir']+ 'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + 'lgm'+ '/')
        hgmDir=(self.calibSettings['calibDir']+ 'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + 'hgm'+ '/')
        shgmDir=(self.calibSettings['calibDir']+ 'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + 'shgm'+ '/')
        if os.path.exists(lgmDir):
            shutil.rmtree(lgmDir)
        if os.path.exists(hgmDir):
            shutil.rmtree(hgmDir)
        if os.path.exists(shgmDir):
            shutil.rmtree(shgmDir)
        shutil.copytree(slgmDir,lgmDir)
        shutil.copytree(slgmDir,hgmDir)
        shutil.copytree(slgmDir,shgmDir)


    def openDiscLbitsFile(self,chips,discLbitsFilename):
        discLbits=np.zeros([self.chipSize,self.chipSize*self.nbOfChips])
        for chip in chips:
            #discLbitsFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'discLbits.chip'+str(chip)
            discLbitsFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/' +self.settings['mode'] + '/' + self.settings['gain']+ '/' + discLbitsFilename + '.chip' + str(chip)
            discLbits[0:256,chip*256:chip*256+256]=np.loadtxt(discLbitsFile)
        return discLbits

    def combineRois(self,chips,discName,steps,roiType):
        discbits=np.zeros([self.chipSize,self.chipSize*self.nbOfChips])
        for step in range(steps):
            roiFullMask=self.Roi(chips,step, steps,roiType)
            discbitsRoi=self.openDiscLbitsFile(chips,discName+'bits_roi_'+str(step))
            discbits[roiFullMask.astype(bool)]=discbitsRoi[roiFullMask.astype(bool)]
            dnp.plot.image(discbitsRoi,name='discbits')
        self.saveDiscbits(chips,discbits,discName+'bits')
        dnp.plot.image(discbits,name='discbits')
        return discbits

    def mergeRoi(self,chips,discName,discbitsRoi,roiFullMask):
        #discbits=np.zeros([self.chipSize,self.chipSize*self.nbOfChips])
        discbits=self.openDiscLbitsFile(chips,discName+'bits')
        discbits[roiFullMask.astype(bool)]=discbitsRoi[roiFullMask.astype(bool)]
        #dnp.plot.image(discbitsRoi,name='discbits')
        self.saveDiscbits(chips,discbits,discName+'bits')
        dnp.plot.image(discbits,name='discbits')
        return discbits

    def findEdge(self,chips,dacscanData,dacRange,edgeVal):
        dnp.plot.clear("noise edges histogram")
        dnp.plot.clear("noise edges")
        if dacRange[1]>dacRange[0]:
            edgeDacs=dacRange[1]-dacRange[2]*np.argmax((dacscanData[::-1,:,:]>edgeVal),0)
        else:
            edgeDacs=dacRange[0]-dacRange[2]*np.argmax((dacscanData[:,:,:]>edgeVal),0)
            
        dnp.plot.image(edgeDacs,name="noise edges")
        for chip in chips:
            dnp.plot.addline(np.histogram(edgeDacs[0:256,chip*256:chip*256+256])[1][0:-1],np.histogram(edgeDacs[0:256,chip*256:chip*256+256])[0],name="noise edges histogram")
        return edgeDacs

    def findMax(self,chips,dacscanData,dacRange,edgeVal):# edgeVal aprameter not needed anymore
        dnp.plot.clear("noise edges histogram")
        dnp.plot.clear("noise edges")
        edgeDacs=dacRange[1]-dacRange[2]*np.argmax((dacscanData[::-1,:,:]),0)
        dnp.plot.image(edgeDacs,name="noise edges")
        for chip in chips:
            dnp.plot.addline(np.histogram(edgeDacs[0:256,chip*256:chip*256+256])[1][0:-1],np.histogram(edgeDacs[0:256,chip*256:chip*256+256])[0],name="noise edges histogram")
        return edgeDacs

    def optimize_DACDisc(self,chips,discName,roiFullMask):
        """
        Usage [OptDACdisc]=x.optimize_DACDisc([chips,discName,roiFullMask)
        where 
        "OptDacdisc" is an array of 8 integers corresponding to the optimum DAC disc value of each chip 
        "chips" is a list containing the number of the chips to calibrate
        "discName" is either "discL" or "discH"
        roiFullMask is 256x256 logical array used to select the pixels masked during threshold_equalization 
        """
        # Definition of parameters to be used for theshold scans
        self.settings['acqtime']='5'
        self.settings['counter']='0'
        self.settings['equalization']='1' # Migth not be necessary when optimizing DAC Disc
        dacRange=(0,150,5)
        #Setting variables depending on whether discL or discH is equalized (Note: equalize discL before discH)
        if discName=='discL':
             Threshold='Threshold0'
             self.setDac(chips,'Threshold1', 0)
             self.settings['disccsmspm']='0'
             DACdiscName='DACDiscL'
        if discName=='discH':
             Threshold='Threshold1'
             self.setDac(chips,'Threshold0', 60) # To be above the noise since DiscL is equalized before DiscH
             self.settings['disccsmspm']='1'
             DACdiscName='DACDiscH'
        """
        ###########################################################################################
        STEP 1
        Perform threshold DAC scans for all discbits set at 0 and various DACdisc values
        discbits set at 0 shift DAC scans to the right
        Plot noise edge position as a function of DACdisc
        Calculate noise edge shift in threshold DAC units per DACdisc DAC unit  
        ##########################################################################################
        """
        discbit=0
        DACDiscRange=range(0,150,50)#range(50,150,50)
        Bins=(dacRange[1]-dacRange[0])/dacRange[2]
        # Initialization of fit parameters and plots
        OptDACdisc=np.zeros(self.nbOfChips)
        sigma=np.zeros([8,len(DACDiscRange)])
        x0=np.zeros([8,len(DACDiscRange)])
        a=np.zeros([8,len(DACDiscRange)])
        p0 = [5000, 50, 30]
        offset=np.zeros(8)
        gain=np.zeros(8)
        name="Histogram of edges when scanning DACDisc"
        plotname=name+" for discbit =" + str(discbit)
        fitplotname=plotname + " (fitted)"
        calibplotname="Mean edge shift in Threshold DACs as a function of DACDisc for discbit =" + str(discbit)
        dnp.plot.clear(plotname)
        dnp.plot.clear(fitplotname)
        dnp.plot.clear(calibplotname)
        # Set discbits at 0
        discbits=discbit*np.ones([self.chipSize,self.chipSize*self.nbOfChips])
        for chip in chips:
            if discName=='discH':
                discLbits=self.openDiscLbitsFile(chips, 'discLbits')
                self.loadConfigbits(range(chip,chip+1),discLbits[:,chip*256:chip*256+256],discbits[:,chip*256:chip*256+256],roiFullMask[:,chip*256:chip*256+256])
            if discName=='discL':
                discHbits=np.zeros([self.chipSize,self.chipSize*self.nbOfChips])
                self.loadConfigbits(range(chip,chip+1),discbits[:,chip*256:chip*256+256],discHbits[:,chip*256:chip*256+256],roiFullMask[:,chip*256:chip*256+256])# Set discL and DiscH bits at 0 and unmask the whole matrix
        # Threshold DAC scans, fitting and plotting
        idx=0
        for DACDisc in DACDiscRange:
            self.setDac(chips,DACdiscName,DACDisc)
            [dacscanData,scanRange]=self.scanDac(chips,Threshold,dacRange)# Scan threshold
            edgeDacs=self.findMax(chips,dacscanData,dacRange,self.edgeVal)# Find noise edges
            for chip in chips:
                edgeHisto=np.histogram(edgeDacs[0:256,chip*256:chip*256+256],bins=Bins)
                dnp.plot.addline(edgeHisto[1][0:-1],edgeHisto[0],name=plotname)
                popt, pcov = curve_fit(gauss_function, edgeHisto[1][0:-2], edgeHisto[0][0:-1],p0)
                x=edgeHisto[1][0:-1]
                a[chip,idx]=popt[0]
                sigma[chip,idx]=popt[2]
                x0[chip,idx]=popt[1]
                dnp.plot.addline(x,gauss_function(x,a[chip,idx],x0[chip,idx],sigma[chip,idx]),name=fitplotname)
            idx=idx+1
            dnp.plot.clear(calibplotname)
            for chip in chips:
                dnp.plot.addline(np.asarray(DACDiscRange[0:idx]),x0[chip,0:idx],name=calibplotname)
        # Plots mean noise edge as a function of DAC Disc for all discbits set at 0
        for chip in chips:
            popt, pcov = curve_fit(lin_function,np.asarray(DACDiscRange),x0[chip,:],[0,-1])
            offset[chip]=popt[0]
            gain[chip]=popt[1]# Noise edge shift in DAC units per DACdisc DAC unit 
            dnp.plot.addline(np.asarray(DACDiscRange),lin_function(np.asarray(DACDiscRange),offset[chip],gain[chip]),name=calibplotname)
        
        print "Edge shift (in Threshold DAC units) produced by 1 DACdisc DAC unit for discbits=15:"
        for chip in chips:
            print "Chip" + str(chip) + ' : ' + str(round(gain[chip],2)) + ' Threshold DAC units'
#################################
# Fit range should be adjusted to remove outliers at 0 and max DAC 150 
        """
        ###########################################################################################
        STEP 2
        Perform threshold DAC scan for all discbits set at 15 (no correction) 
        Fit threshold scan and calculate width of noise edge distribution
        ##########################################################################################
        """
        discbit=15
        DACDisc=80 # Value does no tmatter since no correction is applied when discbits=15
        Bins=(dacRange[1]-dacRange[0])/dacRange[2]
        # Initialization of fit parameters and plots
        sigma=np.zeros([8])
        x0=np.zeros([8])
        a=np.zeros([8])
        p0 = [5000, 0, 30]
        plotname=name+" for discbit =" + str(discbit)
        fitplotname=plotname + " (fitted)"
        dnp.plot.clear(plotname)
        dnp.plot.clear(fitplotname)
        dnp.plot.clear(calibplotname)
        # Set discbits at 15
        discbits=discbit*np.ones([self.chipSize,self.chipSize*self.nbOfChips])
        for chip in chips:
            #self.loadDiscbits(range(chip,chip+1),discName,discbit*roiFullMask[:,chip*256:chip*256+256])
            if discName=='discH':
                #discLbits=np.zeros([self.chipSize,self.chipSize*self.nbOfChips])
                discLbits=self.openDiscLbitsFile(chips, 'discLbits')
                self.loadConfigbits(range(chip,chip+1),discLbits[:,chip*256:chip*256+256],discbits[:,chip*256:chip*256+256],roiFullMask[:,chip*256:chip*256+256])
            if discName=='discL':
                discHbits=np.zeros([self.chipSize,self.chipSize*self.nbOfChips])
                self.loadConfigbits(range(chip,chip+1),discbits[:,chip*256:chip*256+256],discHbits[:,chip*256:chip*256+256],roiFullMask[:,chip*256:chip*256+256])

        self.setDac(chips,DACdiscName,DACDisc)
        [dacscanData,scanRange]=self.scanDac(chips,Threshold,dacRange)
        edgeDacs=self.findMax(chips,dacscanData,dacRange,self.edgeVal)
        for chip in chips:
            edgeHisto=np.histogram(edgeDacs[0:256,chip*256:chip*256+256],bins=Bins)
            dnp.plot.addline(edgeHisto[1][0:-1],edgeHisto[0],name=plotname)
            popt, pcov = curve_fit(gauss_function, edgeHisto[1][0:-2], edgeHisto[0][0:-1],p0)
            x=edgeHisto[1][0:-1]
            a[chip]=popt[0]
            sigma[chip]=popt[2]
            x0[chip]=popt[1]
            dnp.plot.addline(x,gauss_function(x,a[chip],x0[chip],sigma[chip]),name=fitplotname)
        
        
        print "Mean noise edge for discbits =15 :"
        for chip in chips:
            print "Chip" + str(chip) + ' : ' + str(round(x0[chip])) + ' DAC units'
        
        print "sigma of Noise edge distribution for discbits =15 :"
        for chip in chips:
            print "Chip" + str(chip) + ' : ' + str(round(sigma[chip])) + ' DAC units rms'
        
        """
        ###########################################################################################
        STEP 3
        Calculate DAC disc required to bring all noise edges within X sigma of the mean noise edge 
        X is defined by self.nbOfsigma 
        ###########################################################################################
        """
        
        print "Optimum equalization target :"
        for chip in chips:
            print "Chip" + str(chip) + ' : ' + str(round(x0[chip])) + ' DAC units'
        if abs(x0-self.dacTarget).any()>self.accDist:
            print "########################### ONE OR MORE CHIPS NEED A DIFFERENT EQUALIZATION TARGET"
        else:
            print "Default equalization target of " + str(self.dacTarget) + " DAC units can be used."
        
        print "DAC shift required to bring all pixels with an edge within +/- " + "sigma of the target, at the target of " + str(self.dacTarget) + " DAC units : "
        for chip in chips:
            print "Chip" + str(chip) + ' : ' + str(int(self.nbOfSigma*sigma[chip])) + ' Threshold DAC units'
        
        print "Edge shift (in Threshold DAC units) produced by 1 DACdisc DAC unit for discbits=15 :"
        for chip in chips:
            print "Chip" + str(chip) + ' : ' + str(round(gain[chip],2)) + ' Threshold DAC units'
            
        for chip in chips:
            OptDACdisc[chip]=int(self.nbOfSigma*sigma[chip]/gain[chip])
        
        print "#######################################################################################"
        print "Optimum DACdisc value required to bring all pixels with an edge within +/- " + str(self.nbOfSigma)+ " sigma of the target, at the target of " + str(self.dacTarget) + " DAC units : "
        for chip in chips:
            print "Chip" + str(chip) + ' : ' + str(OptDACdisc[chip]) + ' Threshold DAC units'
        print "#######################################################################################"
        
        
        print "Edge shift (in Threshold DAC Units) produced by 1 step of the 32 discbit correction steps :"
        for chip in chips:
            print "Chip" + str(chip) + ' : ' + str(OptDACdisc[chip]/16) + ' Threshold DAC units'
        for chip in chips:
            self.setDac(range(chip,chip+1),DACdiscName,int(OptDACdisc[chip]))
        return OptDACdisc
# 
    def equalise_Discbits(self,chips,discName,roiFullMask,method):
        # method = 'dacscan' or 'bitscan
        self.settings['acqtime']='5'
        self.settings['counter']='0'
        self.settings['equalization']='1'
        if discName=='discL':
             Threshold='Threshold0'
             self.setDac(chips,'Threshold1', 0)
             self.settings['disccsmspm']='0'
        if discName=='discH':
             Threshold='Threshold1'
             self.setDac(chips,'Threshold0', 60) # Well above noise since discL bits are loaded
             self.settings['disccsmspm']='1'
        dnp.plot.image(roiFullMask,name='roi')
        eqPixels=0*np.ones([self.chipSize,self.chipSize*self.nbOfChips])*(1-roiFullMask)
        equPixTot=np.zeros(self.nbOfChips)
        discbitsTmpSum=0
        if method == 'stripes':
            dacRange=(0,20,2)
            discbitsTmp=np.zeros([self.chipSize,self.chipSize*self.nbOfChips])*(1-roiFullMask)
            for idx in range(self.chipSize):
                discbitsTmp[idx,:]=idx%32
            for idx in range(self.chipSize*self.nbOfChips):
                discbitsTmp[:,idx]=(idx%32+discbitsTmp[:,idx])%32
            edgeDacsStack=np.zeros([32,self.chipSize,self.chipSize*self.nbOfChips])
            discbitsStack=np.zeros([32,self.chipSize,self.chipSize*self.nbOfChips])
            discbitsTmp=discbitsTmp*(1-roiFullMask)
            discbits=-10*np.ones([self.chipSize,self.chipSize*self.nbOfChips])*(1-roiFullMask)
            for scan in range(0,32,1):
                discbitsTmp=((discbitsTmp+1)%32)*(1-roiFullMask)
                discbitsStack[scan,:,:]=discbitsTmp
                for chip in chips:
                    if discName=='discH':
                        discLbits=self.openDiscLbitsFile(chips, 'discLbits')
                        self.loadConfigbits(range(chip,chip+1),discLbits[:,chip*256:chip*256+256],discbitsTmp[:,chip*256:chip*256+256],roiFullMask[:,chip*256:chip*256+256])
                    if discName=='discL':
                        discHbits=np.zeros([self.chipSize,self.chipSize*self.nbOfChips])
                        self.loadConfigbits(range(chip,chip+1),discbitsTmp[:,chip*256:chip*256+256],discHbits[:,chip*256:chip*256+256],roiFullMask[:,chip*256:chip*256+256])
                [dacscanData,scanRange]=self.scanDac(chips,Threshold,dacRange)
                edgeDacs=self.findMax(chips,dacscanData,dacRange,self.edgeVal)
                edgeDacsStack[scan,:,:]=edgeDacs
                scanNb=np.argmin((abs(edgeDacsStack-self.dacTarget)),0)
                for chip in chips:
                    for x in range(256):
                        for y in range(chip*256,chip*256+256):
                            discbits[x,y]=discbitsStack[scanNb[x,y],x,y]
                dnp.plot.image(discbits,name='discbits')
                dnp.plot.clear('Histogram of Final Discbits')
                for chip in chips:
                    roiChipMask=1-roiFullMask[0:256,chip*256:chip*256+256]
                    discbitsChip=discbits[0:256,chip*256:chip*256+256]
                    dnp.plot.addline(np.histogram(discbitsChip[roiChipMask.astype(bool)],bins=range(32))[1][0:-1],np.histogram(discbitsChip[roiChipMask.astype(bool)],bins=range(32))[0],name='Histogram of Final Discbits')
#            for chip in chips:
#                equPixTot[chip]=256**2-(discbits[0:256,chip*256:chip*256+256]==-10).sum()
#                print ('####### ' + str(round(equPixTot[chip],0)) + ' equalized pixels in chip ' + str(chip) + ' (' + str(round(100*equPixTot[chip]/(256**2),4)) + '%)')
        self.settings['disccsmspm']='0'
        self.settings['equalization']='0'
        print "Pixel threshold equalization complete"
        return discbits

    def checkCalib(self,chips,DiscName,dacRange):
        ########## SHOULD RELOAD DISCBITS
        self.loadConfig(chips) # MIGHT BE A PROBLEM IF MASK FILE IS MISSING
        equPixTot=np.zeros(xclbr.nbOfChips)
        self.settings['filename']='dacscan'
        [dacscanData,scanRange]=self.scanDac(chips,"Threshold0",dacRange)
        self.plotname=self.settings['filename']
        edgeDacs=self.findMax(chips,dacscanData,dacRange,self.edgeVal)
        
                # Display statistics on equalization and save discLbit files for each chip
        for chip in chips:
            equPixTot[chip]=((edgeDacs[roi,chip*256:chip*256+256]>self.dacTarget-self.accDist)&(edgeDacs[roi,chip*256:chip*256+256]<self.dacTarget+self.accDist)).sum()
            print ('####### ' + str(round(equPixTot[chip],0)) + ' equalized pixels in chip ' + str(chip) + ' (' + str(round(100*equPixTot[chip]/(256**2),4)) + '%)')
            
#         #pixelsInTarget=(dacTarget-5<edgeDacs)&(edgeDacs<dacTarget+5)

    def Roi(self,chips,step,steps,roiType):
        if roiType == 'rect':
            roiFullMask=np.zeros([self.chipSize,self.nbOfChips*self.chipSize])
            for chip in chips:
                roiFullMask[step*256/steps:step*256/steps+256/steps,chip*256:chip*256+256]=1
        if roiType == 'spacing':
            spacing=steps**0.5
            roiFullMask=np.zeros([self.chipSize,self.nbOfChips*self.chipSize])
            for chip in chips:
                roiFullMask[0+int(np.binary_repr(step,2)[0]):256-int(np.binary_repr(step,2)[0]):spacing,chip*256+int(np.binary_repr(step,2)[1]):chip*256+256-int(np.binary_repr(step,2)[1]):spacing]=1
        dnp.plot.image(roiFullMask)
        return roiFullMask

    def calibrateDisc(self,chips,discName,steps=1,roiType='rect'):
        """
        Usage x.calibrateDisc([0],"threshold0") to calibrate threshold 0 of chip 0 using the full matrix as a ROI during threshold_equalization 
        x.calibrateDisc(range(8),"threshold1") to calibrate threshold 1 of all chips using the full chip matrix as a ROI during threshold_equalization 
        """
        self.optimize_DACDisc(chips,discName,roiFullMask=1-self.Roi(chips,0,1,'rect'))
        for step in range(steps):# Run threshold_equalization over each roi
             roiFullMask=self.Roi(chips,step,steps,roiType)
             discbits=self.equalise_Discbits(chips,discName,1-roiFullMask,'stripes')
             self.saveDiscbits(chips,discbits,discName+'bits_roi_'+str(step))
        discbits=self.combineRois(chips,discName,steps,roiType)
        
        self.saveDiscbits(chips,discbits,discName+'bits')
        self.loadConfig(chips)# Load threshold_equalization files created
        self.copy_SLGM_into_other_gain_modes() # Copy slgm threshold_equalization folder to other gain threshold_equalization folders 
        return 
    
    def loop(self,ni):
        tmp=0
        for i in range(ni):
            tmp=self.expose()+tmp
            dnp.plot.image(tmp,name='sum')
        return
        
    def csm(self,chips=range(8),gain='slgm'):
        self.settings['mode']='csm'
        self.settings['gain']=gain
        self.settings['counter']='1'#'1'
        self.setDac(range(8),'Threshold0', 200)### Make sure that chips not used have also TH0 and Th1 well abobe the noise
        self.setDac(range(8),'Threshold1', 200)###
        self.loadConfig(chips)
        self.setDac(range(8),'Threshold0', 45)
        self.setDac(chips,'Threshold1', 100)
        self.expose()
        self.expose()
        
    
    def set_GND_FBK_CAS_ExcaliburRX001(self,chips,fem):
        """
        IMPORTANT NOTE: These values of GND, FBK and CAS Dacs were adjusted for the modules present in RX001 on 10 July 2015
        If modules are replaced by new modules, these DACs need to be re-adjusted 
        If modules are swapped the DACs have to be swapped in the corresponding array :
        For example GND_DAC is an array of size 6 (fems) x 8 (chips)
        GND_DAC[x,:] will contain the GND DAC value of the 8 chips connected to fem/node x+1 where x=0 corresponds to the top half of the top module
        [NUMBERING STARTS AT 0 IN PYTHON SCRIPTS WHERAS NODES NUMBERING STARTS AT 1 IN EPICS]
         
        GND DAC needs to be adjusted manually to read back 0.65V 
        FBK DAC needs to be adjusted manually to read back 0.9V
        CAS DAC needs to be adjusted manually to read back 0.85V
        (Values recommended by Rafa in May 2015)
        
        The procedure to adjust a DAC manually is as follows:
        For GND dac of chip 0:
        x.setDac([0],'GND',150)
        x.readDac([0],'GND')
        Once the DAC value correspond to the required analogue value, edit the GND_Dacs matrix:
        GND_Dacs[0,:]=[NEW DAC VALUE,x,x,x,x,x,x,x]
        
        For FBK dac of chip 0:
        x.setDac([0],'FBK',190)
        x.readDac([0],'FBK')
        Once the DAC value correspond to the required analogue value, edit the FBK_Dacs matrix:
        FBK_Dacs[0,:]=[NEW DAC VALUE,x,x,x,x,x,x,x]
        
        For Cas dac of chip 0:
        x.setDac([0],'Cas',180)
        x.readDac([0],'Cas')
        Once the DAC value correspond to the required analogue value, edit the CAS_Dacs matrix:
        CAS_Dacs[0,:]=[NEW DAC VALUE,x,x,x,x,x,x,x]
        
        This process could be automated if many modules have to be calibrated
        
        """
        GND_Dacs=145*np.ones([6,8]).astype('int')
        FBK_Dacs=190*np.ones([6,8]).astype('int')
        CAS_Dacs=180*np.ones([6,8]).astype('int')
        """
        TOP MODULE: AC-EXC-8
        """
        
        gnd=145
        fbk=190
        cas=180
#         #@ Moly temp: 35 degC on node 3
        GND_Dacs[0,:]=[141,144,154,143,161,158,144,136]
        FBK_Dacs[0,:]=[190,195,201,198,220,218,198,192]
        CAS_Dacs[0,:]=[178,195,196,182,213,201,199,186]
#         
#         GND_Dacs[1,:]=[145,141,142,142,141,141,143,150]
#         FBK_Dacs[1,:]=[205,190,197,200,190,190,200,210]
#         CAS_Dacs[1,:]=[187,187,183,187,177,181,189,194]
        GND_Dacs[1,:]=[154,155,147,147,147,155,158,151]
        FBK_Dacs[1,:]=[215,202,208,200,198,211,255,209] # Max current for fbk limited to 0.589 for chip 7
        CAS_Dacs[1,:]=[208,197,198,194,192,207,199,188]
        
        # NOTE : chip 2 FBK cannot be set to target value
        """
        CENTRAL MODULE: AC-EXC-7
        """
        # @ Moly temp: 27 degC on node 1
        GND_Dacs[2,:]=[154,145,140,170,158,145,145,140]
        FBK_Dacs[2,:]=[212,196,192,230,212,205,201,195]
        CAS_Dacs[2,:]=[201,185,186,226,208,190,198,187]
        
        # @ Moly temp: 28 degC on node 2
        GND_Dacs[3,:]=[138,145,146,156,162,157,155,145]
        FBK_Dacs[3,:]=[190,201,200,221,221,212,220,204]
        CAS_Dacs[3,:]=[181,188,192,208,210,201,207,190]
        """
        BOTTOM MODULE: AC-EXC-4
        """
        #@ Moly temp: 31 degC on node 5
        GND_Dacs[4,:]=[136,146,136,160,142,135,140,148]
        FBK_Dacs[4,:]=[190,201,189,207,189,189,191,208]
        CAS_Dacs[4,:]=[180,188,180,197,175,172,185,200]
        
        #@ Moly temp: 31 degC on node 6
        """
        NOTE: DAC out read-back does not work for chip 2 (counting from 0) of bottom 1/2 module 
        Using readDac function on chip 2 will give FEM errors and the system will need to be power-cycled
        Got error on load pixel config command for chip 7: 2 Pixel configuration loading failed
        Exception caught during femCmd: Timeout on pixel configuration write to chip7 acqState=3
        Connecting to FEM at IP address 192.168.0.101 port 6969 ...
        **************************************************************
        Connecting to FEM at address 192.168.0.101
        Configuring 10GigE data interface: host IP: 10.0.2.1 port: 61649 FEM data IP: 10.0.2.2 port: 8 MAC: 62:00:00:00:00:01
        Acquisition state at startup is 3 sending stop to reset
        **** Loading pixel configuration ****
        Last idx: 65536
        Last idx: 65536
        Last id
        """
        
        gnd=145
        fbk=190
        cas=180
        GND_Dacs[5,:]=[158,140,gnd,145,158,145,138,153]
        FBK_Dacs[5,:]=[215,190,fbk,205,221,196,196,210]
        CAS_Dacs[5,:]=[205,178,cas,190,205,180,189,202]
        
        for chip in chips:
            self.setDac(range(chip,chip+1),'GND',GND_Dacs[fem-1,chip])
            self.setDac(range(chip,chip+1),'FBK',FBK_Dacs[fem-1,chip])
            self.setDac(range(chip,chip+1),'Cas',CAS_Dacs[fem-1,chip])
            
        #self.readDac(range(8), 'GND')
        #self.readDac(range(8),'FBK')
        #self.readDac(range(8), 'Cas')
        return


# 
#     def set_GND_FBK_CAS_ExcaliburRX001_until_10July2015(self,chips,fem):
#         """
#         These DACs were valid before 10/07/2015 when the following modules were installed in EXCALIBUR:
#         TOP    : AC-EXC-7
#         CENTRAL: AC-EXC-6 (half-unbonded chip)
#         BOTTOM : AC-EXC-4
#
#         IMPORTANT NOTE: These values of GND, FBK and CAS Dacs were adjusted for the modules present in RX001 on 20 june 2015
#         If modules are replaced, these DACs need to be re-adjusted 
#         
#         
#         GND DAC needs to be adjusted manually to read back 0.65V
#         FBK DAC needs to be adjusted manually to read back 0.9V
#         CAS DAC needs to be adjusted manually to read back 0.85V
#         
#         The procedure to adjust a DAC manually is as follows:
#         For GND dac of chip 0:
#         x.setDac([0],'GND',150)
#         x.readDac([0],'GND')
#         Once the DAC value correspond to the required analogue value, edit the GND_Dacs matrix:
#         GND_Dacs[0,:]=[NEW DAC VALUE,x,x,x,x,x,x,x]
#         
#         For FBK dac of chip 0:
#         x.setDac([0],'FBK',150)
#         x.readDac([0],'FBK')
#         Once the DAC value correspond to the required analogue value, edit the FBK_Dacs matrix:
#         FBK_Dacs[0,:]=[NEW DAC VALUE,x,x,x,x,x,x,x]
#         
#         For Cas dac of chip 0:
#         x.setDac([0],'Cas',150)
#         x.readDac([0],'Cas')
#         Once the DAC value correspond to the required analogue value, edit the CAS_Dacs matrix:
#         CAS_Dacs[0,:]=[NEW DAC VALUE,x,x,x,x,x,x,x]
#         
#         This process could be automated if many modules have to be calibrated
#         
#         """
#         GND_Dacs=145*np.ones([6,8]).astype('int')
#         FBK_Dacs=190*np.ones([6,8]).astype('int')
#         CAS_Dacs=180*np.ones([6,8]).astype('int')
#         
#         # DACs specific to FEM1 (Top FEM)
#         # @ Moly temp: 27 degC on node 1
#         GND_Dacs[0,:]=[154,145,140,170,158,145,145,140]
#         FBK_Dacs[0,:]=[212,196,192,230,212,205,201,195]
#         CAS_Dacs[0,:]=[201,185,186,226,208,190,198,187]
#         
#         # @ Moly temp: 28 degC on node 2
#         GND_Dacs[1,:]=[138,145,146,156,162,157,155,145]
#         FBK_Dacs[1,:]=[190,201,200,221,221,212,220,204]
#         CAS_Dacs[1,:]=[181,188,192,208,210,201,207,190]
#         
#         #@ Moly temp: 35 degC on node 3
#         GND_Dacs[2,:]=[155,150,145,146,160,145,136,152]
#         FBK_Dacs[2,:]=[218,205,198,200,210,196,190,200]
#         CAS_Dacs[2,:]=[201,193,186,183,200,190,180,187]
#         
#         #@ Moly temp: 34 degC on node 4
#         GND_Dacs[3,:]=[145,141,142,142,141,141,143,150]
#         FBK_Dacs[3,:]=[205,190,197,200,190,190,200,210]
#         CAS_Dacs[3,:]=[187,187,183,187,177,181,189,194]
#         # NOTE : chip 2 FBK cannot be set to target value
#         
#         #@ Moly temp: 31 degC on node 5
#         GND_Dacs[4,:]=[136,146,136,160,142,135,140,148]
#         FBK_Dacs[4,:]=[190,201,189,207,189,189,191,208]
#         CAS_Dacs[4,:]=[180,188,180,197,175,172,185,200]
#         
#         #@ Moly temp: 31 degC on node 6
#         """
#         NOTE: DAC out read-back selection crashes one of the bottom 1/2 module 
#         """
#         gnd=145
#         fbk=190
#         cas=180
#         GND_Dacs[5,:]=[gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd]
#         FBK_Dacs[5,:]=[fbk,fbk,fbk,fbk,fbk,fbk,fbk,fbk]
#         CAS_Dacs[5,:]=[cas,cas,cas,cas,cas,cas,cas,cas]
#         
#         for chip in chips:
#             self.setDac(range(chip,chip+1),'GND',GND_Dacs[fem-1,chip])
#             self.setDac(range(chip,chip+1),'FBK',FBK_Dacs[fem-1,chip])
#             self.setDac(range(chip,chip+1),'Cas',CAS_Dacs[fem-1,chip])
#         
#         return



#    def keV2dac(self,XrayEnergy1,XrayEnergy2,XrayEdge1,XrayEdge2):
#        slope=(XrayEdge2-XrayEdge1)/(XrayEnergy2-XrayEnergy1)
#        offset=XrayEdge2-XrayEnergy2*slope
#        print slope,offset
#        return slope,offset
     
#    def save_keV2dac(self,thresholdNb,mode,gain,slope,offset):
#        np.savetxt(self.calibSettings['calibDir']+ mode + '/' + gain + '/' + 'th'+ str(thresholdNb) + '.slope',slope)
#        np.savetxt(self.calibSettings['calibDir']+ mode + '/' + gain + '/' + 'th'+ str(thresholdNb) +'.offset',offset)
#        return slope,offset

#    def findXrayEdge(self,XrayEnergy,thresholdDacNb,chipNb,scanStart,scanStep,scanStop,acqtime,dacfilename):
#        XrayEdge=self.threshold_dacScan(thresholdDacNb,chipNb,scanStart,scanStep,scanStop,acqtime,dacfilename)
#        return XrayEdge
 
#    def threshold_calibration(self,thresholdNb,mode,gain,XrayEnergy1,XrayEnergy2,chipNb,scanStart,scanStep,scanStop,acqtime,dacfilename):
#        self.save_keV2dac(0,mode,gain,[1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0])
#        self.setThreshold(0,mode,gain,scanStart)
#        var = raw_input(str(XrayEnergy1)+" keV X-rays Ready?: ")
#        print "Ok", var
#        XrayEdge1=self.findXrayEdge(XrayEnergy1,thresholdNb,chipNb,scanStart,scanStep,scanStop,acqtime,dacfilename)
#        var = raw_input(str(XrayEnergy2)+" keV X-rays Ready?: ")
#        print "Ok", var
#        XrayEdge2=self.findXrayEdge(XrayEnergy2,thresholdNb,chipNb,scanStart,scanStep,scanStop,acqtime,dacfilename)
#        [slope,offset]=self.keV2dac(XrayEnergy1,XrayEnergy2,XrayEdge1,XrayEdge2)
#        self.save_keV2dac(thresholdNb,mode,gain,slope,offset)
#        return [slope,offset]


    def rotateConfig(self,configFile):
        print configFile
        shutil.copy(configFile,configFile+".backup")
        configBits=np.loadtxt(configFile)
        np.savetxt(configFile,np.rot90(configBits,2),fmt='%.18g', delimiter=' ' )
        return 
    
#     def rotateAllConfig(self,calibPath):
#         #x.rotateAllConfig("/dls/tmp/excalibur/calib_spm_shgm/")
#         chips=[0,1,2,3,4,5,6,7]
#         print calibPath
#         if self.fem % 2 == 1 :
#             print "Config files will be backed-up and rotated before saving new config file for fem " + str(self.fem)
#             for chip in chips:
#                 discLbitsFile=calibPath+"/fem"+str(self.fem)+"/"+ self.settings['mode'] + "/" + self.settings['gain']+       "/discLbits.chip"+str(chip)
#                 self.rotateConfig(discLbitsFile)
#                 discHbitsFile=calibPath+"/fem"+str(self.fem)+"/"+ self.settings['mode'] + "/" + self.settings['gain']+       "/discHbits.chip"+str(chip)
#                 self.rotateConfig(discHbitsFile)
#                 pixelmaskFile=calibPath+"/fem"+str(self.fem)+"/"+ self.settings['mode'] + "/" + self.settings['gain']+       "/pixelmask.chip"+str(chip)
#                 self.rotateConfig(pixelmaskFile)
#         return 
             




#x=excaliburRX(1)
#x.threshold_calibration()


#x.threshold_equalization(range(8))
#x.loadConfig(range(8))
#x.maskPixelsUsingDACscan(range(8),"Threshold0",(40,120,2))
#x.Fe55ThreshCalib(range(8),0)



#     def setThreshold(self,thresholdNb,mode,gain,thresholdValue):
#         subDir=self.calibSettings['calibDir']+ mode + '/' + gain + '/'
#         for chipNb in self.chipRange:
#             self.loadDiscLbits(range(chipNb,chipNb+1),subDir + 'discLbits.chip'+str(chipNb))
#             print(self.calibSettings['calibDir']+ mode + '/' + gain + '/''discLbits.chip'+str(chipNb))
#         #dacTh0 = np.genfromtxt(subDir+ '/keV2dac',delimiter=",")*threshold
#         dacTh0 = np.genfromtxt(subDir+ '/' + 'th'+ str(thresholdNb) +'.slope',delimiter=",")*thresholdValue+np.genfromtxt(subDir+ '/' +'th'+ str(thresholdNb) +'.offset',delimiter=",")
#         #dacTh0 = ([0,0,0,0,0,0,1,1]*dacTh0+[120,120,120,120,120,120,0,0]).astype(np.int)
#         for chipNb in self.chipRange:
#             self.setDacTh0(chipNb,dacTh0[chipNb].astype(np.int),subDir+'dacs')
#         self.calibSettings['dacfile']=subDir+'dacs'
#         self.settings['gainmode']=gain
#         return dacTh0
# 
#     def energyEdgeScan(self,energy,scanRange):
#         #scanRange=range(50,30,-2)
#         meanCounts=np.zeros((np.array(scanRange).shape[0],8),np.int)
#         exca.settings['acqtime']='1000'
#         scanIdx=0
#         threshVal=np.zeros(10,np.int)
#         for thresh in scanRange:
#             self.setThreshold('spm','slgm',thresh)
#             self.settings['filechipNb,scanStart,scanStep,scanStop,acqtime,dacfilenamename']='energy' + str(energy) + 'thresh' + str(thresh) +'.hdf5'
#             image=self.expose()
#             time.sleep(1)
#             subprocess.call(["bin/excaliburTestApp","-i","192.168.0.101","-p","6969","-m","0x3f","--slow"])
#             
#             #dh = dnp.io.load(self.settings['imagepath'] + self.settings['filename'])
#             #imageRaw=dh.image[...]
#             #image=dnp.squeeze(imageRaw.astype(np.int))
#             #dnp.plot.image(image)
# #             for chip in range(8):
# #                 meanCoun)ts[scanIdx][chip]=image[0:256,chip*256:chip*256+256].mean()
# #                 threshVal[scanIdx]=thresh
# #             scanIdx=scanIdx+1
#         return
 
#     def threshCalib(self,energy1,energy2,scanRange):
#         scanIdx=0
#         meanCounts=np.zeros((np.array(scanRange).shape[0],8),np.float)
#         threshVal=np.zeros(np.array(scanRange).shape[0],np.float)
#         for thresh in scanRange:
#             self.settings['filename']='energy' + str(energy1) + 'thresh' + str(thresh) +'.hdf5'
#             dh = dnp.io.load(self.settings['imagepath'] + self.settings['filename'])
#             imageRaw=dh.image[...]
#             image=dnp.squeeze(imageRaw.astype(np.int))
#             dnp.plot.image(image)
#             for chip in range(8):
#                 meanCounts[scanIdx][chip]=image[0:256,chip*256:chip*256+256].mean()
#                 threshVal[scanIdx]=thresh
#             scanIdx=scanIdx+1
#             energyEdge1=(threshVal[np.argmax((meanCounts>10),0)])
#         
#         scanIdx=0
#         for thresh in scanRange:
#             self.settings['filename']='energy' + str(energy2) + 'thresh' + str(thresh) +'.hdf5'
#             dh = dnp.io.load(self.settings['imagepath'] + self.settings['filename'])
#             imageRaw=dh.image[...]
#             image=dnp.squeeze(imageRaw.astype(np.int))
#             dnp.plot.image(image)
#             for chip in range(8):
#                 meanCounts[scanIdx][chip]=image[0:256,chip*256:chip*256+256].mean()
#                 threshVal[scanIdx]=thresh
#             scanIdx=scanIdx+1
#         energyEdge2=(threshVal[np.argmax((meanCounts>10),0)])
#         
#         slope=(energyEdge2-energyEdge1)/(energy2-energy1)
#         offset=energyEdge2-energy2*slope
#         exca.keV2dac(slope,offset)
#         return (slope,offset,energyEdge1,energyEdge2)





#x.threshold_equalization(range(8))
#[dacScanData,scanRange]=x.scanDac(range(8),"Threshold0",(60,15,1))


#x.maskPixelsUsingDACscan([7],'Threshold0',(150,50,2))
#x.settings['mode']='csm'
#chips=[5]
# 
# # 

# plot='CSM_30kV_4gains'

#x.csm(range(8),'shgm')


#chips=range(8)
#x.settings['acqtime']='1000'
#x.settings['gain']='hgm'
#dacRange=(60,34,1)
#dacRange=(60,15,1)
#x.settings['filename']='test'
#x.chipId()
#[dacScanData,scanRange]=x.scanDac(chips,"Threshold0",dacRange)



# x.settings['mode']='csm'
# x.settings['gain']='slgm'
# x.setDac(chips,'Threshold0', 25)
# x.setDac(chips,'Threshold1', 20)
# x.expose()
# 
# 
# loops=2
#   
# x.settings['gain']='slgm'
# dacRange=(17,150,1)
# x.setDac(chips,'Threshold1', 20)
# dacScanDataSum=0
# for i in range(loops):
#     [dacScanData,scanRange]=x.scanDac(chips,"Threshold0",dacRange)
#     dacScanDataSum=dacScanData+dacScanDataSum
# for chip in chips:
#     spectrum=-np.diff(np.squeeze(dacScanDataSum[:,0:256,chip*256:chip*256+256].mean(2).mean(1)))
#     dnp.plot.addline(np.array(range(dacRange[0],dacRange[1],dacRange[2]))[1:],spectrum[1:],name=plot)
# # 
# x.settings['gain']='lgm'
# x.setDac(chips,'Threshold1', 20)
# dacRange=(17,200,1)
# dacScanDataSum=0
# for i in range(loops):
#     [dacScanData,scanRange]=x.scanDac(chips,"Threshold0",dacRange)
#     dacScanDataSum=dacScanData+dacScanDataSum
# for chip in chips:
#     spectrum=-np.diff(np.squeeze(dacScanDataSum[:,0:256,chip*256:chip*256+256].mean(2).mean(1)))
#     dnp.plot.addline(np.array(range(dacRange[0],dacRange[1],dacRange[2]))[1:],spectrum[1:],name=plot)
#  
# x.settings['gain']='hgm'
# x.setDac(chips,'Threshold1', 30)
# dacRange=(30,250,1)
# dacScanDataSum=0
# for i in range(loops):
#     [dacScanData,scanRange]=x.scanDac(chips,"Threshold0",dacRange)
#     dacScanDataSum x.maskPixelsUsingDACscan([7],'Threshold0',(50,120,2))=dacScanData+dacScanDataSum
# for chip in chips:
#     spectrum=-np.diff(np.squeeze(dacScanDataSum[:,0:256,chip*256:chip*256+256].mean(2).mean(1)))
#     dnp.plot.addline(np.array(range(dacRange[0],dacRange[1],dacRange[2]))[1:],spectrum[1:],name=plot)
#  
# x.settings['gain']='shgm'
# x.setDac(chips,'Threshold1', 30)
# dacRange=(54,250,1)
# dacScanDataSum=0
# for i in range(loops):
#     [dacScanData,scanRange]=x.scanDac(chips,"Threshold0",dacRange)
#     dacScanDataSum=dacScanData+dacScanDataSum
# for chip in chips:
#     spectrum=-np.diff(np.squeeze(dacScanDataSum[:,0:256,chip*256:chip*256+256].mean(2).mean(1)))
#     dnp.plot.addline(np.array(range(dacRange[0],dacRange[1],dacRange[2]))[1:],spectrum[1:],name=plot)
#  
#   
#x.maskPixelsUsingDACscan([5],'Threshold0',(20,120,2))


# #x.threshold_equalization(chips)
# plot='SPM_30kV_4gains'
# dacRange=(16,250,1)
# x.settings['mode']='spm'
# 
# x.settings['gain']='slgm'
# dacScanDataSum=0
# for i in range(5):
#     [dacScanData,scanRange]=x.scanDac(chips,"Threshold0",dacRange)
#     dacScanDataSum=dacScanData+dacScanDataSum
# spectrum=-np.diff(np.squeeze(dacScanDataSum[:,0:256,chip*256:chip*256+256].mean(2).mean(1)))
# for chip in chips:
#     dnp.plot.addline(np.array(range(dacRange[0],dacRange[1],dacRange[2]))[1:],spectrum[1:],name=plot)
# 
# x.settings['gain']='lgm'
# dacScanDataSum=0
# for i in range(5):
#     [dacScanData,scanRange]=x.scanDac(chips,"Threshold0",dacRange)
#     dacScanDataSum=dacScanData+dacScanDataSum
# spectrum=-np.diff(np.squeeze(dacScanDataSum[:,0:256,chip*256:chip*256+256].mean(2).mean(1)))
# for chip in chips:
#     dnp.plot.addline(np.array(range(dacRange[0],dacRange[1],dacRange[2]))[1:],spectrum[1:],name=plot)
#  
# x.settings['gain']='hgm'
# dacScanDataSum=0
# for i in range(5):
#     [dacScanData,scanRange]=x.scanDac(chips,"Threshold0",dacRange)
#     dacScanDataSum=dacScanData+dacScanDataSum
# spectrum=-np.diff(np.squeeze(dacScanDataSum[:,0:256,chip*256:chip*256+256].mean(2).mean(1)))
# for chip in chips:
#     dnp.plot.addline(np.array(range(dacRange[0],dacRange[1],dacRange[2]))[1:],spectrum[1:],name=plot)
#  
# x.settings['gain']='shgm'
# dacScanDataSum=0
# for i in range(5):
#     [dacScanData,scanRange]=x.scanDac(chips,"Threshold0",dacRange)
#     dacScanDataSum=dacScanData+dacScanDataSum
# spectrum=-np.diff(np.squeeze(dacScanDataSum[:,0:256,chip*256:chip*256+256].mean(2).mean(1)))
# for chip in chips:
#     dnp.plot.addline(np.array(range(dacRange[0],dacRange[1],dacRange[2]))[1:],spectrum[1:],name=plot)
#  

#x.maskPixelsUsingDACscan(range(8),'Threshold0',(20,120,2))



#x.setDac(range(8),"Threshold0",20)
#x.expose()

#dacRange=(20,120,2)
#[dacScanData,scanRange]=x.scanDac(chips,"Threshold0",dacRange)




#HGMdiscLbits=x.openDiscLbitsFile(range(8),'discLbits')
#x.settings['gain']='slgm'
#SLGMdiscLbits=x.openDiscLbitsFile(range(8),'discLbits')
#diff=SLGMdiscLbits-HGMdiscLbits
#dnp.plot.image(diff)
#dnp.plot.clear()
#dnp.plot.addline(np.histogram(diff,bins=32)[1][0:-1],np.histogram(diff,bins=32)[0])

