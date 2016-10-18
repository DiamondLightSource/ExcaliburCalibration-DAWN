import subprocess
import numpy as np
import time
import os
import shutil
from scipy.optimize import curve_fit


def lin_function(x,offset,gain):
    return offset+gain*x

def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

class excaliburRX(object):
    fem=32 #start numbering at 1
    chipSize=256
    nbOfChips=8
    command='/home/ktf91651/bin/excaliburTestApp'
    ipaddress="192.168.0.100"
    port='6969'
    chipRange=range(0,8)
    plotname=''
    dacTarget=10
    edgeVal=10
    accDist=4
    nbOfSigma=3.2 # based on experimental data

    settings={'mode':'spm',#'spm','csm'
              'gain':'slgm',#'slgm','lgm','hgm',shgm'
              'bitdepth':'12',# 24 bits needs disccsmspm at 1 to use discL 
              'readmode':'0',
              'counter':'0',
              'disccsmspm':'0',
              'equalization':'0',
              'trigmode':'0',
              'acqtime':'100',
              'frames':'1',
              'imagepath':'/tmp/',
              'filename':'cont',
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


    
    calibSettings={'calibDir':'/tmp/',
                   'configDir':'/home/ktf91651/excaliburRx/config/',
                   'dacfilename':'dacs',
                   'dacfile':'',
                   'noiseEdge':'10'} #Default .dacs file. This will be overwritten by setThreshold function
#
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

    def mask(self,chips):# Create hexadecimal chip mask from selected chip range
        maskBin=0
        for chip in chips:
            maskBin=2**(self.nbOfChips-chip-1)+maskBin
        return str(hex(maskBin))

    def chipId(self):
        #subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(self.chipRange),"-r","-e"])
        subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(self.chipRange),"-r"])
        print(str(self.chipRange))
    
    def logChipId(self):
        logfilename=self.calibSettings['calibDir']+'fem'+str(self.fem)+'/efuseIDs'
        with open(logfilename, "w") as outfile:
            subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(self.chipRange),"-r","-e"],stdout=outfile)
        print(str(self.chipRange))
    
    def monitor(self):
        subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(self.chipRange),"--slow"])
    
    def setDac(self,chips,dacName,dacValue):
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

    def scanDac(self,chips,dacName,dacRange):# ONLY FOR THRESHOLD DACS
        self.updateFilenameIndex
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
            for chip in chips:
                dnp.plot.addline(np.array(range(dacRange[0],dacRange[1],-dacRange[2]))[1:],spectrum[1:],name="Spectrum")
        else:
            for chip in chips:
                dnp.plot.addline(np.array(range(dacRange[0],dacRange[1]+dacRange[2],dacRange[2])),np.squeeze(dacScanData[:,0:256,chip*256:chip*256+256].mean(2).mean(1)),name="dacScan")
            spectrum=-np.diff(np.squeeze(dacScanData[:,0:256,chip*256:chip*256+256].mean(2).mean(1)))
            for chip in chips:
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

    def loadConfig(self,chips):
        for chip in chips:
            discHbitsFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'discHbits.chip'+str(chip)
            discLbitsFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'discLbits.chip'+str(chip)
            pixelmaskFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'pixelmask.chip'+str(chip)
            subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--config","--discl="+discLbitsFile,"--disch="+discHbitsFile,"--pixelmask="+pixelmaskFile])
            discbits=self.openDiscLbitsFile(chips,'discLbits')
        return discbits
        
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
            print os.path.isfile(idxFilename)
            file=open(idxFilename, 'a')
            newIdx=0
            file.write(str(newIdx))
            file.close()
            self.settings['acqtime']='100'
            self.settings['filenameIndex']=str(newIdx)
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
        self.updateFilenameIndex()
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
        x.loadConfig(chips)
        x.setDac(chips,"Threshold0", 20)
        x.shoot(10)
        LogoTP=np.ones([256,8*256])
        logoSmall=np.loadtxt(self.calibSettings['configDir']+'logo.txt')
        LogoTP[7:250,225:1823]=logoSmall
        LogoTP[LogoTP>0]=1
        LogoTP=1-LogoTP
        for chip in chips:
            testbitsFile=self.calibSettings['calibDir']+'Logo_chip'+str(chip)+'_mask'
            np.savetxt(testbitsFile,LogoTP[0:256,chip*256:chip*256+256],fmt='%.18g', delimiter=' ' )
            subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--dacs",self.calibSettings['calibDir']+self.calibSettings['dacfilename'],"--config","--tpmask="+testbitsFile])
        self.settings['fullFilename']=self.settings['filename']+"_"+self.settings['filenameIndex']+".hdf5"
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

    def maskPixelsUsingDACscan(self,chips,Threshold,dacRange):# dacRange=(20,120,2)
        badPixTot=np.zeros(8)
        [dacScanData,scanRange]=x.scanDac(chips,Threshold,dacRange)
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
        dacFile=dir+self.calibSettings['dacfilename']
        if os.path.isfile(dacFile)==0:
            shutil.copy(self.calibSettings['configDir']+self.calibSettings['dacfilename'],dir)
        discfile=dir+'discLbits.tmp'
        #if os.path.isfile(dacFile)==0:
        #    shutil.copy(self.calibSettings['configDir']+'zeros.mask',dir)
        return dacFile

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
        edgeDacs=dacRange[1]-dacRange[2]*np.argmax((dacscanData[::-1,:,:]>edgeVal),0)
        dnp.plot.image(edgeDacs,name="noise edges")
        for chip in chips:
            dnp.plot.addline(np.histogram(edgeDacs[0:256,chip*256:chip*256+256])[1][0:-1],np.histogram(edgeDacs[0:256,chip*256:chip*256+256])[0],name="noise edges histogram")
        return edgeDacs

    def findMax(self,chips,dacscanData,dacRange,edgeVal):
        dnp.plot.clear("noise edges histogram")
        dnp.plot.clear("noise edges")
        edgeDacs=dacRange[1]-dacRange[2]*np.argmax((dacscanData[::-1,:,:]),0)
        dnp.plot.image(edgeDacs,name="noise edges")
        for chip in chips:
            dnp.plot.addline(np.histogram(edgeDacs[0:256,chip*256:chip*256+256])[1][0:-1],np.histogram(edgeDacs[0:256,chip*256:chip*256+256])[0],name="noise edges histogram")
        return edgeDacs

    def optimize_DACDisc(self,chips,discName,roiFullMask):
        self.settings['acqtime']='5'
        self.settings['counter']='0'
        self.settings['equalization']='1'
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
        dacRange=(0,150,5)     
        OptDACdisc=np.zeros(self.nbOfChips)
        name="Histogram of edges when scanning DACDisc"
        DACDiscRange=range(50,150,50)
        Bins=(dacRange[1]-dacRange[0])/dacRange[2]
        sigma=np.zeros([8,len(DACDiscRange)])
        x0=np.zeros([8,len(DACDiscRange)])
        a=np.zeros([8,len(DACDiscRange)])
        discbit=0
        discbits=discbit*np.ones([self.chipSize,self.chipSize*self.nbOfChips])
        p0 = [5000, 50, 30]
        plotname=name+" for discbit =" + str(discbit)
        fitplotname=plotname + " (fitted)"
        calibplotname="Mean edge shift in Threshold DACs as a function of DACDisc for discbit =" + str(discbit)
        dnp.plot.clear(plotname)
        dnp.plot.clear(fitplotname)
        dnp.plot.clear(calibplotname)
        for chip in chips:
            if discName=='discH':
                discLbits=self.openDiscLbitsFile(chips, 'discLbits')
                self.loadConfigbits(range(chip,chip+1),discLbits[:,chip*256:chip*256+256],discbits[:,chip*256:chip*256+256],roiFullMask[:,chip*256:chip*256+256])
            if discName=='discL':
                discHbits=np.zeros([self.chipSize,self.chipSize*self.nbOfChips])
                self.loadConfigbits(range(chip,chip+1),discbits[:,chip*256:chip*256+256],discHbits[:,chip*256:chip*256+256],roiFullMask[:,chip*256:chip*256+256])
        idx=0
        for DACDisc in DACDiscRange:
            self.setDac(chips,DACdiscName,DACDisc)
            [dacscanData,scanRange]=self.scanDac(chips,Threshold,dacRange)
            edgeDacs=self.findMax(chips,dacscanData,dacRange,self.edgeVal)
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
        offset=np.zeros(8)
        gain=np.zeros(8)
        for chip in chips:
            popt, pcov = curve_fit(lin_function,np.asarray(DACDiscRange),x0[chip,:],[0,-1])
            offset[chip]=popt[0]
            gain[chip]=popt[1]
            dnp.plot.addline(np.asarray(DACDiscRange),lin_function(np.asarray(DACDiscRange),offset[chip],gain[chip]),name=calibplotname)

################################
#####################################
#################################
# Adjust Fit range to remove outliers at 0 and max DAC 150 



        #name="Histogram of edges for discbit=15 (no correction)"
        DACDiscRange=range(80,85,5)
        Bins=(dacRange[1]-dacRange[0])/dacRange[2]
        
        sigma=np.zeros([8,len(DACDiscRange)])
        x0=np.zeros([8,len(DACDiscRange)])
        a=np.zeros([8,len(DACDiscRange)])
        
        discbit=15
        discbits=discbit*np.ones([self.chipSize,self.chipSize*self.nbOfChips])
        p0 = [5000, 0, 30]
        plotname=name+" for discbit =" + str(discbit)
        fitplotname=plotname + " (fitted)"
        calibplotname="Mean edge shift in Threshold DACs as a function of DACDisc for discbit =" + str(discbit)
        dnp.plot.clear(plotname)
        dnp.plot.clear(fitplotname)
        dnp.plot.clear(calibplotname)
        for chip in chips:
            #self.loadDiscbits(range(chip,chip+1),discName,discbit*roiFullMask[:,chip*256:chip*256+256])
            if discName=='discH':
                #discLbits=np.zeros([self.chipSize,self.chipSize*self.nbOfChips])
                discLbits=self.openDiscLbitsFile(chips, 'discLbits')
                self.loadConfigbits(range(chip,chip+1),discLbits[:,chip*256:chip*256+256],discbits[:,chip*256:chip*256+256],roiFullMask[:,chip*256:chip*256+256])
            if discName=='discL':
                discHbits=np.zeros([self.chipSize,self.chipSize*self.nbOfChips])
                self.loadConfigbits(range(chip,chip+1),discbits[:,chip*256:chip*256+256],discHbits[:,chip*256:chip*256+256],roiFullMask[:,chip*256:chip*256+256])
        idx=-1
        for DACDisc in DACDiscRange:
            idx=idx+1
            self.setDac(chips,DACdiscName,DACDisc)
            [dacscanData,scanRange]=self.scanDac(chips,Threshold,dacRange)
            edgeDacs=self.findMax(chips,dacscanData,dacRange,self.edgeVal)
            for chip in chips:
                edgeHisto=np.histogram(edgeDacs[0:256,chip*256:chip*256+256],bins=Bins)
                dnp.plot.addline(edgeHisto[1][0:-1],edgeHisto[0],name=plotname)
                popt, pcov = curve_fit(gauss_function, edgeHisto[1][0:-2], edgeHisto[0][0:-1],p0)
                x=edgeHisto[1][0:-1]
                a[chip,idx]=popt[0]
                sigma[chip,idx]=popt[2]
                x0[chip,idx]=popt[1]
                dnp.plot.addline(x,gauss_function(x,a[chip,idx],x0[chip,idx],sigma[chip,idx]),name=fitplotname)
            dnp.plot.clear(calibplotname)
            for chip in chips:
                dnp.plot.addline(np.asarray(DACDiscRange[0:idx]),x0[chip,0:idx],name=calibplotname)
        print "#######################################################################################"
        print "Optimum equalization target :"
        for chip in chips:
            print "Chip" + str(chip) + ' : ' + str(round(x0[chip,idx])) + ' DAC units'
        if abs(x0-self.dacTarget).any()>self.accDist:
            print "########################### ONE OR MORE CHIPS NEED A DIFFERENT EQUALIZATION TARGET"
        else:
            print "Default equalization target of " + str(self.dacTarget) + " DAC units can be used."
        print "DAC shift required to bring all pixels with an edge within +/- " + "sigma of the target, at the target of " + str(self.dacTarget) + " DAC units : "
        for chip in chips:
            print "Chip" + str(chip) + ' : ' + str(int(self.nbOfSigma*sigma[chip,idx])) + ' Threshold DAC units'
        print "Edge shift (in Threshold DAC units) produced by 1 DACdisc DAC unit for discbits=31 (maximum correction) :"
        for chip in chips:
            print "Chip" + str(chip) + ' : ' + str(round(gain[chip],2)) + ' Threshold DAC units'
        for chip in chips:
            #OptDACdisc[chip]=-int(self.nbOfSigma*sigma[chip,idx]/gain[chip])
            OptDACdisc[chip]=int(self.nbOfSigma*sigma[chip,idx]/gain[chip])
        print "DACdisc value required to bring all pixels with an edge within +/- " + str(self.nbOfSigma)+ " sigma of the target, at the target of " + str(self.dacTarget) + " DAC units : "
        for chip in chips:
            print "Chip" + str(chip) + ' : ' + str(OptDACdisc[chip]) + ' Threshold DAC units'
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
            for chip in chips:
                equPixTot[chip]=256**2-(discbits[0:256,chip*256:chip*256+256]==-10).sum()
                print ('####### ' + str(round(equPixTot[chip],0)) + ' equalized pixels in chip ' + str(chip) + ' (' + str(round(100*equPixTot[chip]/(256**2),4)) + '%)')
        self.settings['disccsmspm']='0'
        self.settings['equalization']='0'
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

    def calibrateDisc(self,chips,discName,steps,roiType):
        self.checkCalibDir() # Create calibration directory structure if necessary
        self.setdacs(chips)
        self.logChipId() # Log chip IDs in calibration folder
        self.optimize_DACDisc(chips,discName,roiFullMask=1-self.Roi(chips,0,1,'rect'))
        for step in range(steps):# Run calibration over each roi
             roiFullMask=self.Roi(chips,step,steps,roiType)
             discbits=self.equalise_Discbits(chips,discName,1-roiFullMask,'stripes')
             self.saveDiscbits(chips,discbits,discName+'bits_roi_'+str(step))
        discbits=self.combineRois(chips,discName,steps,roiType)
        self.saveDiscbits(chips,discbits,discName+'bits')
        return 
    

    
    def calibration(self,chips):
        # Function defining the automatic calibration sequence
        self.settings['mode']='spm'
        self.settings['gain']='slgm'
        self.calibrateDisc(chips,'discL',1,'rect')
        #self.calibrateDisc(chips,'discH',1,'rect')
        #self.settings['mode']='csm'
        #self.settings['gain']='slgm'
        #self.calibrateDisc(chips,'discL',1,'rect')
        #self.calibrateDisc(chips,'discH',1,'rect')
        return 

    def loop(self,ni):
        tmp=0
        for i in range(ni):
            tmp=self.expose()+tmp
            dnp.plot.image(tmp,name='sum')
        return
        
    def csm(self,chips,gain):
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
        
        
x=excaliburRX()
#x.calibration(range(8))



#x.maskPixelsUsingDACscan([7],'Threshold0',(150,50,2))
#x.settings['mode']='csm'
#chips=[5]
# 
# # 

# plot='CSM_30kV_4gains'

#x.csm(range(8),'shgm')


#chips=[7]
#x.settings['acqtime']='1000'
#x.settings['gain']='hgm'
#dacRange=(60,34,1)
#dacRange=(60,25,1)
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


# #x.calibration(chips)
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

