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
    fem=16 #start numbering at 1
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
    nbOfSigma=3

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

    def mask(self,chips):# Create hexadecimal chip mask from selected chip range
        maskBin=0
        for chip in chips:
            maskBin=2**(self.nbOfChips-chip-1)+maskBin
        return str(hex(maskBin))

    def chipId(self):
        subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(self.chipRange),"-r","-e"])
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
        for chip in chips:
            dnp.plot.addline(np.array(range(dacRange[0],dacRange[1]+dacRange[2],dacRange[2])),np.squeeze(dacScanData[:,0:256,chip*256:chip*256+256].mean(2).mean(1)),name="dacScan")
        dnp.plot.clear("Spectrum")
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

    def loadConfig(self,chips):
        for chip in chips:
            discHbitsFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'discHbits.chip'+str(chip)
            discLbitsFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'discLbits.chip'+str(chip)
            pixelmaskFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'pixelmask.chip'+str(chip)
            #subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--config","--pixelmask="+pixelmaskFile,"--config","--discl="+discLbitsFile])
            #subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--config","--discl="+discLbitsFile])
            subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--config","--discl="+discLbitsFile,"--disch="+discHbitsFile,"--pixelmask="+pixelmaskFile])
            discbits=self.openDiscLbitsFile(chips,'discLbits')
        return discbits
        
    def updateFilenameIndex(self):
        idxFilename=self.settings['imagepath']+self.settings['filename']+'.idx'
        print(idxFilename)
        if os.path.isfile(idxFilename):
            #print os.path.isfile(idxFilename)
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
            self.settings['acqtime']='100'
            self.settings['filenameIndex']=str(newIdx)
        return newIdx

    def acquireFF(self,ni,acqtime):
        #self.settings['fullFilename']="FlatField.hdf5"
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
        #self.settings['acqtime']='1000'
        #self.settings['frames']='1'
        self.settings['fullFilename']=self.settings['filename']+"_"+self.settings['filenameIndex']+".hdf5"
        #self.settings['fullFilename']="truc.hdf5"
        
        
        string=[self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(self.chipRange),"--depth="+self.settings['bitdepth'],"--csmspm="+self.modeCode[self.settings['mode']],"--disccsmspm="+self.settings['disccsmspm'],"--equalization="+self.settings['equalization'],"--gainmode="+self.gainCode[self.settings['gain']],"--burst","--frames="+str(self.settings['frames']),"--acqtime="+str(self.settings['acqtime']),"--trigmode="+self.settings['trigmode'],"--path="+self.settings['imagepath'],"--hdffile="+self.settings['fullFilename']]
        print(string)
        subprocess.call(string)
        #subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(self.chipRange),"--depth="+self.settings['bitdepth'],"--csmspm="+self.modeCode[self.settings['mode']],"--disccsmspm="+self.settings['disccsmspm'],"--equalization="+self.settings['equalization'],"--gainmode="+self.gainCode[self.settings['gain']],"--burst","--frames="+str(self.settings['frames']),"--acqtime="+str(self.settings['acqtime']),"--trigmode="+self.settings['trigmode'],"--path="+self.settings['imagepath'],"--hdffile="+self.settings['fullFilename']])
        
        #subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(self.chipRange),"--depth="+self.settings['bitdepth'],"--disccsmspm="+self.settings['disccsmspm'],"--gainmode="+self.gainmode[self.settings['gain']],"--burst","--frames="+str(self.settings['frames']),"--acqtime="+str(self.settings['acqtime']),"--path="+self.settings['imagepath'],"--hdffile="+self.settings['fullFilename']])
        #print self.settings['filename']
        time.sleep(0.5)
        #dh = dnp.io.load(self.settings['imagepath'] + self.settings['fullFilename'])
        #imageRaw=dh.image[...]
        #image=dnp.squeeze(imageRaw.astype(np.int))
        #dnp.plot.clear()
        #dnp.plot.image(image,name='Image data')
        #return image

    def expose(self):
        #self.settings['acqtime']='1000'
        #self.settings['frames']='1'
        self.updateFilenameIndex()
        self.settings['fullFilename']=self.settings['filename']+"_"+self.settings['filenameIndex']+".hdf5"
        #self.settings['fullFilename']="truc.hdf5"
        subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(self.chipRange),"--depth="+self.settings['bitdepth'],"--csmspm="+self.modeCode[self.settings['mode']],"--disccsmspm="+self.settings['disccsmspm'],"--equalization="+self.settings['equalization'],"--gainmode="+self.gainCode[self.settings['gain']],"--acquire","--frames="+str(self.settings['frames']),"--acqtime="+str(self.settings['acqtime']),"--trigmode="+self.settings['trigmode'],"--path="+self.settings['imagepath'],"--hdffile="+self.settings['fullFilename']])
        #subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(self.chipRange),"--depth="+self.settings['bitdepth'],"--disccsmspm="+self.settings['disccsmspm'],"--gainmode="+self.gainmode[self.settings['gain']],"--burst","--frames="+str(self.settings['frames']),"--acqtime="+str(self.settings['acqtime']),"--path="+self.settings['imagepath'],"--hdffile="+self.settings['fullFilename']])
        #print self.settings['filename']
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
            #dnp.plot.clear()
            FF=image[p,:,:]*FFcoeff
            #FF=image[p,:,:]
            #FF[FF>3000]=0
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
        subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(chips),"--depth="+self.settings['bitdepth'],"--acquire","--readmode="+str(self.settings['readmode']),"--frames="+str(self.settings['frames']),"--acqtime="+str(self.settings['acqtime']),"--counter=0","--path="+self.settings['imagepath'],"--hdffile="+self.settings['fullFilename'],"--tpcount="+str(10)])
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
        
        #subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(chips),"--depth="+self.settings['bitdepth'],"--acquire","--readmode="+str(self.settings['readmode']),"--frames="+str(self.settings['frames']),"--acqtime="+str(self.settings['acqtime']),"--counter=0","--path="+self.settings['imagepath'],"--hdffile="+self.settings['fullFilename']])
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
        
        #for chip in chips:
        badPixels[:,chip*256+supCol*32:chip*256+supCol*32+64]=1
        discLbitsFile=self.calibSettings['calibDir']+  'fem' + str(xclbr.fem) +'/'+self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'discLbits.chip'+str(chip)
        pixelmaskFile=self.calibSettings['calibDir']+  'fem' + str(self.fem) +'/' +self.settings['mode'] + '/' + self.settings['gain']+ '/' + 'pixelmask.chip'+str(chip)
        np.savetxt(pixelmaskFile,badPixels[0:256,chip*256:chip*256+256],fmt='%.18g', delimiter=' ' )
        subprocess.call([self.command,"-i",self.ipaddress,"-p",self.port,"-m",self.mask(range(chip,chip+1)),"--config","--pixelmask="+pixelmaskFile,"--config","--discl="+discLbitsFile])
        dnp.plot.image(badPixels)

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
            #roiChipMask=np.zeros([self.chipSize,self.nbOfChips*self.chipSize])
            #roiChipMask[s*256/steps:s*256/steps+256/steps,:]=1
            #roi=range(s*256/steps,(s+1)*256/steps)
            discbitsRoi=self.openDiscLbitsFile(chips,discName+'bits_roi_'+str(step))
            #discLbits[roi,:]=discLbitsRoi[roi,:]
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
        #dnp.plot.clear('Histogram of edge Dacs')
        #dnp.plot.clear(self.plotname+"_edges_histo")
        #dnp.plot.clear(self.plotname+"_edges")
        dnp.plot.clear("noise edges histogram")
        dnp.plot.clear("noise edges")
        edgeDacs=dacRange[1]-dacRange[2]*np.argmax((dacscanData[::-1,:,:]>edgeVal),0)
        dnp.plot.image(edgeDacs,name="noise edges")
        for chip in chips:
 #            dnp.plot.addline(np.histogram(edegDacs[0:256,chip*256:chip*256+256],bins=31)[1][0:-1],np.histogram(discLbits[0:256,chip*256:chip*256+256],bins=31)[0],name='Histogram of Final DiscLbits')
            dnp.plot.addline(np.histogram(edgeDacs[0:256,chip*256:chip*256+256])[1][0:-1],np.histogram(edgeDacs[0:256,chip*256:chip*256+256])[0],name="noise edges histogram")
        return edgeDacs

    def optimize_DACDisc(self,chip,discName,roiFullMask):
        
        
        self.settings['counter']='0'
        self.settings['equalization']='1'
        if discName=='discL':
             Threshold='Threshold0'
             self.settings['disccsmspm']='0'
             DACdiscName='DACDiscL'
        if discName=='discH':
             Threshold='Threshold1'
             self.settings['disccsmspm']='1'
             DACdiscName='DACDiscH'
        dacRange=(0,150,5)     
        discbits=np.zeros([self.chipSize,self.chipSize*self.nbOfChips])

#        name="Histogram of edges when scanning DACDisc"
        OptDACdisc=np.zeros(self.nbOfChips)
#        gain=1
 
        name="Histogram of edges when scanning DACDisc"
 
        DACDiscRange=range(0,40,5)
        Bins=(dacRange[1]-dacRange[0])/dacRange[2]
         
        sigma=np.zeros([8,len(DACDiscRange)])
        x0=np.zeros([8,len(DACDiscRange)])
        a=np.zeros([8,len(DACDiscRange)])
         
        discbit=31
        p0 = [5000, 0, 30]
        plotname=name+" for discbit =" + str(discbit)
        fitplotname=plotname + " (fitted)"
        calibplotname="Mean edge shift in Threshold DACs as a function of DACDisc for discbit =" + str(discbit)
        dnp.plot.clear(plotname)
        dnp.plot.clear(fitplotname)
        dnp.plot.clear(calibplotname)
        for chip in chips:
            self.loadDiscbits(range(chip,chip+1),discName,discbit*roiFullMask[:,chip*256:chip*256+256])
        idx=0
        for DACDisc in DACDiscRange:
            self.setDac(chips,DACdiscName,DACDisc)
            [dacscanData,scanRange]=self.scanDac(chips,Threshold,dacRange)
            edgeDacs=self.findEdge(chips,dacscanData,dacRange,self.edgeVal)
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

        #name="Histogram of edges for discbit=15 (no correction)"
        DACDiscRange=range(80,85,5)
        Bins=(dacRange[1]-dacRange[0])/dacRange[2]
        
        sigma=np.zeros([8,len(DACDiscRange)])
        x0=np.zeros([8,len(DACDiscRange)])
        a=np.zeros([8,len(DACDiscRange)])
        
        discbit=15
        p0 = [5000, 0, 30]
        plotname=name+" for discbit =" + str(discbit)
        fitplotname=plotname + " (fitted)"
        calibplotname="Mean edge shift in Threshold DACs as a function of DACDisc for discbit =" + str(discbit)
        dnp.plot.clear(plotname)
        dnp.plot.clear(fitplotname)
        dnp.plot.clear(calibplotname)
        for chip in chips:
            self.loadDiscbits(range(chip,chip+1),discName,discbit*roiFullMask[:,chip*256:chip*256+256])
        idx=-1
        for DACDisc in DACDiscRange:
            idx=idx+1
            self.setDac(chips,DACdiscName,DACDisc)
            [dacscanData,scanRange]=self.scanDac(chips,Threshold,dacRange)
            edgeDacs=self.findEdge(chips,dacscanData,dacRange,self.edgeVal)
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
            OptDACdisc[chip]=-int(self.nbOfSigma*sigma[chip,idx]/gain[chip])

        print "DACdisc value required to bring all pixels with an edge within +/- " + str(self.nbOfSigma)+ " sigma of the target, at the target of " + str(self.dacTarget) + " DAC units : "
        for chip in chips:
            print "Chip" + str(chip) + ' : ' + str(OptDACdisc[chip]) + ' Threshold DAC units'
            
        print "Edge shift (in Threshold DAC Units) produced by 1 step of the 32 discbit correction steps :"
        for chip in chips:
            print "Chip" + str(chip) + ' : ' + str(OptDACdisc[chip]/16) + ' Threshold DAC units'
#         offset=np.zeros(8)
#         gain=np.zeros(8)
#         for chip in chips:
#             popt, pcov = curve_fit(lin_function,np.asarray(DACDiscRange),x0[chip,:],[0,-1])
#             offset[chip]=popt[0]
#             gain[chip]=popt[1]
#             dnp.plot.addline(np.asarray(DACDiscRange),lin_function(np.asarray(DACDiscRange),offset[chip],gain[chip]),name=calibplotname)

        for chip in chips:
            self.setDac(range(chip,chip+1),DACdiscName,int(OptDACdisc[chip]))
        
        return OptDACdisc
# 
    def equalise_Discbits(self,chips,discName,roiFullMask,method):
        # method = 'dacscan' or 'bitscan
        self.settings['counter']='0'
        self.settings['equalization']='1'
        if discName=='discL':
             Threshold='Threshold0'
             self.settings['disccsmspm']='0'
        if discName=='discH':
             Threshold='Threshold1'
             self.settings['disccsmspm']='1'

        dnp.plot.image(roiFullMask,name='roi')
        eqPixels=0*np.ones([self.chipSize,self.chipSize*self.nbOfChips])*roiFullMask
        
        equPixTot=np.zeros(self.nbOfChips)
        
        discbitsTmpSum=0
        
        if method == 'stripes':
            #dacRange=(self.dacTarget-self.accDist,self.dacTarget+2*self.accDist,1)
            dacRange=(0,20,2)
            discbitsTmp=np.zeros([self.chipSize,self.chipSize*self.nbOfChips])*roiFullMask
#            dnp.plot.image(discbits,name='discbits')
# SRIPES
#            for idx in range(self.chipSize):
#                discbitsTmp[idx,:]=idx%32
# DIAGONAL PATTERN
            for idx in range(self.chipSize):
                discbitsTmp[idx,:]=idx%32
            for idx in range(self.chipSize*self.nbOfChips):
                discbitsTmp[:,idx]=(idx%32+discbitsTmp[:,idx])%32
            
            edgeDacsStack=np.zeros([32,self.chipSize,self.chipSize*self.nbOfChips])
            discbitsStack=np.zeros([32,self.chipSize,self.chipSize*self.nbOfChips])
            discbitsTmp=discbitsTmp*roiFullMask
            
            discbits=-10*np.ones([self.chipSize,self.chipSize*self.nbOfChips])*roiFullMask
            for scan in range(0,32,1):
                discbitsTmp=((discbitsTmp+1)%32)*roiFullMask
                
                discbitsTmpSum=discbitsTmpSum+discbitsTmp
                #dnp.plot.image(discbitsTmp,name='discbitsTmp')
                #dnp.plot.image(discbitsTmpSum,name='discbitsTmpSum')
                
                discbitsStack[scan,:,:]=discbitsTmp
                
#                dnp.plot.image(discbits,name='discbits')
                for chip in chips:
                    self.loadDiscbits(range(chip,chip+1),discName,discbitsTmp[0:256,chip*256:chip*256+256])
                [dacscanData,scanRange]=self.scanDac(chips,Threshold,dacRange)
                edgeDacs=self.findEdge(chips,dacscanData,dacRange,self.edgeVal)                
                
                edgeDacsStack[scan,:,:]=edgeDacs
                #dnp.plot.image(abs(edgeDacsStack[scan,:,:]-self.dacTarget),name=str(scan))
                #discbits[(self.dacTarget-self.accDist<edgeDacs)&(edgeDacs<self.dacTarget+self.accDist)]=discbitsTmp[(self.dacTarget-self.accDist<edgeDacs)&(edgeDacs<self.dacTarget+self.accDist)]
                scanNb=np.argmin((abs(edgeDacsStack-self.dacTarget)),0)
                for chip in chips:
                    for x in range(256):
                        for y in range(chip*256,chip*256+256):
                            discbits[x,y]=discbitsStack[scanNb[x,y],x,y]
                
                dnp.plot.image(discbits,name='discbits')                
                #time.sleep(0.5)
                
                dnp.plot.clear('Histogram of Final Discbits')
                for chip in chips:
                    roiChipMask=roiFullMask[0:256,chip*256:chip*256+256]
                    discbitsChip=discbits[0:256,chip*256:chip*256+256]
                    dnp.plot.addline(np.histogram(discbitsChip[roiChipMask.astype(bool)],bins=range(32))[1][0:-1],np.histogram(discbitsChip[roiChipMask.astype(bool)],bins=range(32))[0],name='Histogram of Final Discbits')
            
            for chip in chips:
                equPixTot[chip]=256**2-(discbits[0:256,chip*256:chip*256+256]==-10).sum()
                print ('####### ' + str(round(equPixTot[chip],0)) + ' equalized pixels in chip ' + str(chip) + ' (' + str(round(100*equPixTot[chip]/(256**2),4)) + '%)')
            
            #discbits[discbits<0]=0
        
        return discbits
    #return#discbits

    def checkCalib(self,chips,DiscName,dacRange):
        ########## SHOULD RELOAD DISCBITS
        self.loadConfig(chips) # MIGHT BE A PROBLEM IF MASK FILE IS MISSING
        equPixTot=np.zeros(xclbr.nbOfChips)
        self.settings['filename']='dacscan'
        [dacscanData,scanRange]=self.scanDac(chips,"Threshold0",dacRange)
        self.plotname=self.settings['filename']
        edgeDacs=self.findEdge(chips,dacscanData,dacRange,self.edgeVal)
        
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
        return roiFullMask

    def calibrateDisc(self,chips,discName,steps,roiType):
        self.checkCalibDir() # Create calibration directory structure if necessary
        self.logChipId() # Log chip IDs in calibration folder
        self.optimize_DACDisc(chips,discName,roiFullMask=self.Roi(chips,0,1,'rect'))
        for step in range(steps):# Run calibration over each roi
            roiFullMask=self.Roi(chips,step,steps,roiType)
            discbits=self.equalise_Discbits(chips,discName,roiFullMask,'stripes')
            self.saveDiscbits(chips,discbits,discName+'bits_roi_'+str(step))
        discbits=self.combineRois(chips,discName,steps,roiType)
        self.saveDiscbits(chips,discbits,discName+'bits')
        return 
    
    def calibration(self,chips):
        # Function defining the automatic calibration sequence
        self.settings['mode']='spm'
        self.settings['gain']='slgm'
        #self.calibrateDisc(chips,'discL',1,'rect')
        self.calibrateDisc(chips,'discH',1,'rect')
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
        self.setDac(chips,'Threshold0', 25)
        self.setDac(chips,'Threshold1', 15)
        self.expose()
        self.expose()
        
        
x=excaliburRX()

chips=[5]

# 
#x.calibration(chips)
plot='CSM_30kV_4gains'
dacRange=(17,150,1)
x.settings['mode']='csm'
x.settings['gain']='slgm'
x.setDac(chips,'Threshold0', 25)
x.setDac(chips,'Threshold1', 20)
x.expose()


loops=2
  
x.settings['gain']='slgm'
dacRange=(17,150,1)
x.setDac(chips,'Threshold1', 20)
dacScanDataSum=0
for i in range(loops):
    [dacScanData,scanRange]=x.scanDac(chips,"Threshold0",dacRange)
    dacScanDataSum=dacScanData+dacScanDataSum
for chip in chips:
    spectrum=-np.diff(np.squeeze(dacScanDataSum[:,0:256,chip*256:chip*256+256].mean(2).mean(1)))
    dnp.plot.addline(np.array(range(dacRange[0],dacRange[1],dacRange[2]))[1:],spectrum[1:],name=plot)
# 
x.settings['gain']='lgm'
x.setDac(chips,'Threshold1', 20)
dacRange=(17,200,1)
dacScanDataSum=0
for i in range(loops):
    [dacScanData,scanRange]=x.scanDac(chips,"Threshold0",dacRange)
    dacScanDataSum=dacScanData+dacScanDataSum
for chip in chips:
    spectrum=-np.diff(np.squeeze(dacScanDataSum[:,0:256,chip*256:chip*256+256].mean(2).mean(1)))
    dnp.plot.addline(np.array(range(dacRange[0],dacRange[1],dacRange[2]))[1:],spectrum[1:],name=plot)
 
x.settings['gain']='hgm'
x.setDac(chips,'Threshold1', 30)
dacRange=(30,250,1)
dacScanDataSum=0
for i in range(loops):
    [dacScanData,scanRange]=x.scanDac(chips,"Threshold0",dacRange)
    dacScanDataSum=dacScanData+dacScanDataSum
for chip in chips:
    spectrum=-np.diff(np.squeeze(dacScanDataSum[:,0:256,chip*256:chip*256+256].mean(2).mean(1)))
    dnp.plot.addline(np.array(range(dacRange[0],dacRange[1],dacRange[2]))[1:],spectrum[1:],name=plot)
 
x.settings['gain']='shgm'
x.setDac(chips,'Threshold1', 30)
dacRange=(54,250,1)
dacScanDataSum=0
for i in range(loops):
    [dacScanData,scanRange]=x.scanDac(chips,"Threshold0",dacRange)
    dacScanDataSum=dacScanData+dacScanDataSum
for chip in chips:
    spectrum=-np.diff(np.squeeze(dacScanDataSum[:,0:256,chip*256:chip*256+256].mean(2).mean(1)))
    dnp.plot.addline(np.array(range(dacRange[0],dacRange[1],dacRange[2]))[1:],spectrum[1:],name=plot)
 
  
#x.maskPixelsUsingDACscan(range(8),'Threshold0',(20,120,2))


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

