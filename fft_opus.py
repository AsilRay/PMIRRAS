import numpy as np
#from scanf import scanf
import os.path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import correlate
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from scipy import fftpack 
from scipy.signal import get_window
class fft_opus:
    def __init__(self,filepath, filename):
        self.filepath = filepath
        self.filename = filename
    def readInterfData(self):
        self.f = open(self.filename,'rb')
        data_uint8 = np.frombuffer(self.f.read(), dtype = 'uint8')
        self.f = open(self.filename,'rb')
        data_sgl = np.frombuffer(self.f.read(), dtype = 'float32')
    
        data_size_uint8 = data_uint8.size
        data_uint8_4 = np.reshape(data_uint8[0:data_size_uint8-(data_size_uint8%4)],(int(data_size_uint8/4),4))
    
        startPts, NbrPts = self.getPts(data_uint8_4,data_uint8)
        print(startPts, NbrPts)
        interf_1 = data_sgl[startPts[3]:startPts[3]+NbrPts[0]]
        interf_2 = data_sgl[startPts[4]+2:startPts[4]+2+NbrPts[0]]
    
        return interf_1, interf_2
    def readfData(self):
        self.f = open(self.filename,'rb')
        data_uint8 = np.frombuffer(self.f.read(), dtype = 'uint8')
        self.f = open(self.filename,'rb')
        data_sgl = np.frombuffer(self.f.read(), dtype = 'float32')
        
        data_size_uint8 = data_uint8.size
        data_uint8_4 = np.reshape(data_uint8[0:data_size_uint8-(data_size_uint8%4)],(int(data_size_uint8/4),4))
       
        startPts, NbrPts = self.getPts(data_uint8_4, data_uint8)
        startPts = startPts[3:7]
        NbrPts   = NbrPts[0:4]
        len_spec, len_interf = NbrPts[1], NbrPts[0]
        interf_1 = data_sgl[startPts[0]:startPts[0]+NbrPts[0]]
        spec_1 = data_sgl[startPts[1]:startPts[1]+NbrPts[1]]
        interf_2 = data_sgl[startPts[2]+2:startPts[2]+2+NbrPts[2]]
        spec_2 = data_sgl[startPts[3]:startPts[3]+NbrPts[3]]
        N = int(len_interf/2)
        #interfs=[ interf_1, interf_2 ]
        specs = [spec_1, spec_2]
        self.f.close()
        return interf_1, interf_2, specs
    def getPts(self,data_4, data_uint8):
        startPts = []
        NbrPts   = []
    
        for i in range(data_4.shape[0]):
            data_str_4 = "".join([chr(item) for item in data_4[i,:]])
            if 'END' in data_str_4:
                startPts.append(i)
            elif 'NPT' in data_str_4:
                NbrPts.append(data_uint8[i*4 + 8:i*4 + 12].view(dtype='uint32')[0])
        return startPts, NbrPts
    def normalize(self,interf):
        if np.array(interf).shape[0]==1:
            interf_n = interf-np.mean(interf)
        return interf_n
    def apodize(self,interf_in, windowfuncname):
        N = len(interf_in)        
        mid_wdw=np.argmin(interf_in)
        window = get_window(windowfuncname,2*mid_wdw)
        interf_out = interf_in*window[0:N]
    
        return window,interf_out

    def zeroFill(self,fillF, interf):
        interf_zf = []
        N = len(interf)
        print("N="+str(N))
        interf_zf = np.zeros([16384*fillF])
        interf_zf[0:N] = np.array(interf)[0:N]
        return interf_zf
    def corrPhaseShift(self,interf):
        N = int(len(interf)/2)
        PhaseShift = np.argmax(np.abs(np.fft.ifft(fftpack.fft(interf[0:N]) *np.conj(fftpack.fft(np.flip(interf[N:]))))))
        print(PhaseShift)
        corr1 =interf[0:N]-np.mean(interf[0:N])
        corr2 =np.roll(np.flip(interf[N:]),PhaseShift)-np.mean(interf[N:])
        return corr1, corr2
    def FFT_simple(self, interfs):
        spectrum = fftpack.fft(interfs)
        return spectrum
    def FFT_like_OPUS(self,interf):
        _, interf_a = self.apodize(interf, 'blackmanharris')
        interf_zf = self.zeroFill( 2, interf_a)
        spectrum = self.FFT_simple( interf_zf)
        spectrum_half = spectrum[0:int(len(spectrum)/2)]
        spectrum_out = np.roll(spectrum_half,-2)
        return spectrum_out
