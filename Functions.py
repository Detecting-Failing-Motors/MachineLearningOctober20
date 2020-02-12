
# coding: utf-8

# In[2]:


import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np 
from scipy.signal import welch
from detect_peaks import detect_peaks
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import entropy #seems non beneficial
from scipy.signal import hilbert


# In[3]:


#Import Machine Learning Libraries
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# In[4]:


def BearingInfomation(n,N,Bd,Pd,phi):
    xx = Bd/Pd*np.cos(phi)
    BPFI = (N/2)*(1 + xx)*n
    BPFO = (N/2)*(1 - xx)*n
    BSF = (Pd/(2*Bd))*(1-(xx)**2)*n
    FTF= (1/2)*(1 - xx)*n
    x = {
        "BPFI": BPFI,
        "BPFO": BPFO,
        "BSF":  BSF,
        "FTF":  FTF
    }
    return x


# In[5]:


def RemoveDCOffset(sig):
    m = sig - np.mean(sig)
    return m


# In[6]:


def FourierTransform(comb_sig, T, N, f_s):
    #Fast Fourier Transform
    #number_of_time_samples = len(t)
    number_of_time_samples = N
    
    frq = np.arange(number_of_time_samples)/(T)# two sides frequency range
    frq = frq[range(int(number_of_time_samples/(2)))] # one side frequency range
    Y = abs(np.fft.fft(comb_sig))/number_of_time_samples # fft computing and normalization
    Y = Y[range(int(number_of_time_samples/2))]
    #End fft
    x = {
        "Frequency":frq,
        "Freq. Amp.": Y
        }
    return x


# In[7]:


def get_psd_values(comb_sig, T, N, fs):
    frq, psd_values = welch(comb_sig, fs=fs)
    x = {
        "Frequency":frq,
        "PSD": psd_values
        }
    return x


# In[8]:


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]
 
def get_autocorr_values(comb_sig, T, N, f_s):
    autocorr_values = autocorr(comb_sig)
    x_values = np.array([T * jj for jj in range(0, N)])
    x = {
        "X Values":x_values,
        "Autocorr Values": autocorr_values
        }
    return x


# In[9]:


def TimeDomainInformation(comb_sig, T, N, f_s):
    x = {
        "RMS": np.mean(comb_sig**2),
        "STD": np.std(comb_sig),
        "Mean": np.mean(comb_sig),
        "Max": np.max(comb_sig),
        "Min": np.min(comb_sig),
        "Peak-to-Peak": (np.max(comb_sig) - np.min(comb_sig)),
        "Max ABS": np.max(abs(comb_sig)),
        "Kurtosis": kurtosis(comb_sig),
        "Skew": skew(comb_sig),
    }

    return x


# In[10]:


def GetSortedPeak(frq,comb_sig, T, N, f_s):
    max_peak_height = 0.1 * np.nanmax(comb_sig)
    threshold = 0.05 * np.nanmax(comb_sig)
    #Get indices of peak
    peak = detect_peaks(comb_sig,edge = 'rising',mph = max_peak_height, mpd = 2, threshold = threshold )
    
    m = []
    mm = []
    for i in peak:
        m.append(comb_sig[i]) 
        mm.append(frq[i])

    mmm = np.argsort(m)
    n = []
    nn = []
    for i in mmm:
        n.append(m[i])
        nn.append(mm[i])

    n  = n[::-1]
    nn = nn[::-1]

    return n, nn


# In[11]:


def FrequencyDomainInformation(comb_sig, T, N, f_s):
    x1 = FourierTransform(comb_sig, T, N, f_s)
    x2 = get_psd_values(comb_sig, T, N, f_s)
    x3 = get_autocorr_values(comb_sig, T, N, f_s)
    FTamp,FTfreq = GetSortedPeak(x1['Frequency'],x1['Freq. Amp.'], T, N, f_s)
    PSDamp,PSDfreq = GetSortedPeak(x2['Frequency'],x2['PSD'], T, N, f_s)
    print(x3['X Values'])
    print(x3['Autocorr Values'])
    Cor,CorTime = GetSortedPeak(x3['X Values'],x3['Autocorr Values'], T, N, f_s)
    
    while len(FTamp) <= 5:
        FTamp.append(['-999'])
    while len(FTfreq) <= 5:
        FTfreq.append(['-999'])
    while len(PSDamp) <= 5:
        PSDamp.append(['-999'])
    while len(PSDfreq) <= 5:
        PSDfreq.append(['-999'])
    while len(Cor) <= 5:
        Cor.append(['-999'])
    while len(CorTime) <= 5:
        CorTime.append(['-999'])
    
    x = {
        "FFT Frq @ Peak 1": FTfreq[0],
        "FFT Frq @ Peak 2": FTfreq[1],
        "FFT Frq @ Peak 3": FTfreq[2],
        "FFT Frq @ Peak 4": FTfreq[3],
        "FFT Frq @ Peak 5": FTfreq[4],
        "FFT Amp @ Peak 1": FTamp[0],
        "FFT Amp @ Peak 2": FTamp[1],
        "FFT Amp @ Peak 3": FTamp[2],
        "FFT Amp @ Peak 4": FTamp[3],
        "FFT Amp @ Peak 5": FTamp[4],
        "PSD Frq @ Peak 1": PSDfreq[0],
        "PSD Frq @ Peak 2": PSDfreq[1],
        "PSD Frq @ Peak 3": PSDfreq[2],
        "PSD Frq @ Peak 4": PSDfreq[3],
        "PSD Frq @ Peak 5": PSDfreq[4],
        "PSD Amp @ Peak 1": PSDamp[0],
        "PSD Amp @ Peak 2": PSDamp[1],
        "PSD Amp @ Peak 3": PSDamp[2],
        "PSD Amp @ Peak 4": PSDamp[3],
        "PSD Amp @ Peak 5": PSDamp[4],
        "Autocorrelate Time @ Peak 1": CorTime[0],
        "Autocorrelate Time @ Peak 2": CorTime[1],
        "Autocorrelate Time @ Peak 3": CorTime[2],
        "Autocorrelate Time @ Peak 4": CorTime[3],
        "Autocorrelate Time @ Peak 5": CorTime[4],
        "Autocorrelate @ Peak 1": Cor[0],
        "Autocorrelate @ Peak 2": Cor[1],
        "Autocorrelate @ Peak 3": Cor[2],
        "Autocorrelate @ Peak 4": Cor[3],
        "Autocorrelate @ Peak 5": Cor[4]
    }
    return x


# In[12]:


def getAbsoluteTime(file):
    month  = int(file[5:7])
    day    = int(file[8:10])
    hour   = int(file[11:13])
    minute = int(file[14:16])
    second = int(file[17:19])
    x = second + 60*minute + 60*60*hour + 24*60*60*day + 31*24*60*60*(month - 10)
    return x


# In[13]:


"""
http://mkalikatzarakis.eu/wp-content/uploads/2018/12/IMS_dset.html
Previous work done on this dataset states that seven different states of health were observed:

Early (initial run-in of the bearings)
Normal
Suspect (the health seems to be deteriorating)
Imminent failure (for bearings 1 and 2, which didn’t actually fail, but were severely worn out)
Inner race failure (bearing 3)
Rolling element failure (bearing 4)
Stage 2 failure (bearing 4)
For the first test (the one we are working on), the following labels have been proposed per file:

Bearing 1
early: 2003.10.22.12.06.24 - 2013.10.23.09.14.13
suspect: 2013.10.23.09.24.13 - 2003.11.08.12.11.44 (bearing 1 was in suspicious health from the beginning, but showed some self-healing effects)
normal: 2003.11.08.12.21.44 - 2003.11.19.21.06.07
suspect: 2003.11.19.21.16.07 - 2003.11.24.20.47.32
imminent failure: 2003.11.24.20.57.32 - 2003.11.25.23.39.56

Bearing 2
early: 2003.10.22.12.06.24 - 2003.11.01.21.41.44
normal: 2003.11.01.21.51.44 - 2003.11.24.01.01.24
suspect: 2003.11.24.01.11.24 - 2003.11.25.10.47.32
imminent failure: 2003.11.25.10.57.32 - 2003.11.25.23.39.56

Bearing 3
early: 2003.10.22.12.06.24 - 2003.11.01.21.41.44
normal: 2003.11.01.21.51.44 - 2003.11.22.09.16.56
suspect: 2003.11.22.09.26.56 - 2003.11.25.10.47.32
Inner race failure: 2003.11.25.10.57.32 - 2003.11.25.23.39.56

Bearing 4
early: 2003.10.22.12.06.24 - 2003.10.29.21.39.46
normal: 2003.10.29.21.49.46 - 2003.11.15.05.08.46
suspect: 2003.11.15.05.18.46 - 2003.11.18.19.12.30
Rolling element failure: 2003.11.19.09.06.09 - 2003.11.22.17.36.56
Stage 2 failure: 2003.11.22.17.46.56 - 2003.11.25.23.39.56
"""

def StateInformation(comb_sig, T, N, f_s,file,BearingNum):
    
    absolutetime = getAbsoluteTime(file)
    #in seconds don't include years taking 10 as the start month
    
    #Bearing 1 transitions
    b1e2s  = getAbsoluteTime("2013.10.23.09.14.13")
    b1s2n  = getAbsoluteTime("2003.11.08.12.11.44")
    b1n2s  = getAbsoluteTime("2003.11.19.21.06.07")
    b1s2i  = getAbsoluteTime("2003.11.24.20.47.32")
    
    #Bearing 2 transitions
    b2e2n  = getAbsoluteTime("2003.11.01.21.41.44")
    b2n2s  = getAbsoluteTime("2003.11.24.01.01.24")
    b2s2i  = getAbsoluteTime("2003.11.25.10.47.32")
    
    #Bearing 3 transitions
    b3e2n  = getAbsoluteTime("2003.11.01.21.41.44")
    b3n2s  = getAbsoluteTime("2003.11.22.09.16.56")
    b3s2irf  = getAbsoluteTime("2003.11.25.10.47.32")
    
    #Bearing 4 transitions
    b4e2n  = getAbsoluteTime("2003.10.29.21.39.46")
    b4n2s  = getAbsoluteTime("2003.11.15.05.08.46")
    b4s2r  = getAbsoluteTime("2003.11.18.19.12.30")
    b4r2f  = getAbsoluteTime("2003.11.22.17.36.56")
    
    m = "ERROR"
    if BearingNum == 1:
        if absolutetime   <= b1e2s:
            m = "Early"
        elif absolutetime <= b1s2n:
            m = "Suspect"
        elif absolutetime <= b1n2s:
            m = "Normal"
        elif absolutetime <= b1s2i:
            m = "Suspect"
        elif absolutetime > b1s2i:
            m = "Imminent Failure"
    elif BearingNum == 2:
        if absolutetime   <= b2e2n:
            m = "Early"
        elif absolutetime <= b2n2s:
            m = "Normal"
        elif absolutetime <= b2s2i:
            m = "Suspect"
        elif absolutetime > b2s2i:
            m = "Imminent Failure" 
    elif BearingNum == 3:
        if absolutetime   <= b3e2n:
            m = "Early"
        elif absolutetime <= b3n2s:
            m = "Normal"
        elif absolutetime <= b3s2irf:
            m = "Suspect"
        elif absolutetime >= b3s2irf:
            m = "Inner Race Failure"   
    elif BearingNum == 4:
        if absolutetime   <= b4e2n:
            m = "Early"
        elif absolutetime <= b4n2s:
            m = "Normal"
        elif absolutetime <= b4s2r:
            m = "Suspect"
        elif absolutetime <= b4r2f:
            m = "Rolling Element Failure"
        elif absolutetime > b4r2f:
            m = "Stage 2 Failure"
    else:
        m = "ERROR"
        
    x = {
        "State": m
    }
    return x


# In[14]:


def MotorInformation(comb_sig, T, N, f_s):
    x = {
        "Motor Type AC(1)-DC(0)": 1,
        "Shaft Speed [Hz]": 2000/60
    }
    return x


# In[15]:


def getCompleteDataFrame(sig,Tmax,NumberOfSamples,SampleFrequency,n,N,Bd,Pd,phi,file,BearingNum):
    sig = RemoveDCOffset(sig)
    BearingInfo = BearingInfomation(n,N,Bd,Pd,phi)
    TimeDomainInfo = TimeDomainInformation(sig,Tmax,NumberOfSamples,SampleFrequency)
    FrequecyDomainInfo = FrequencyDomainInformation(sig,Tmax,NumberOfSamples,SampleFrequency)
    StateInfo = StateInformation(sig,Tmax,NumberOfSamples,SampleFrequency,file,BearingNum)
    MotorInfo = MotorInformation(sig,Tmax,NumberOfSamples,SampleFrequency)
    Features = {**StateInfo,**MotorInfo,**BearingInfo,**TimeDomainInfo,**FrequecyDomainInfo}
    Features = pd.DataFrame(Features, index=[0])
    return Features 


# In[16]:
def getTESTDataFrame(sig,Tmax,NumberOfSamples,SampleFrequency,n,N,Bd,Pd,phi):
    sig = RemoveDCOffset(sig)
    BearingInfo = BearingInfomation(n,N,Bd,Pd,phi)
    TimeDomainInfo = TimeDomainInformation(sig,Tmax,NumberOfSamples,SampleFrequency)
    FrequecyDomainInfo = FrequencyDomainInformation(sig,Tmax,NumberOfSamples,SampleFrequency)
    MotorInfo = MotorInformation(sig,Tmax,NumberOfSamples,SampleFrequency)
    Features = {**MotorInfo,**BearingInfo,**TimeDomainInfo,**FrequecyDomainInfo}
    Features = pd.DataFrame(Features, index=[0])
    return Features 

def getPlot(X,Y,xlabel,ylabel,Title):
    fig = plt.figure()
    plt.plot(X,Y,'r')
    plt.xlabel(ylabel)
    plt.ylabel(ylabel)
    plt.title(Title)
    plt.grid(True)
    return fig

def getGraphs(sig,Tmax,NumberOfSamples,SampleFrequency,n,N,Bd,Pd,phi,file,BearingNum):
    t = np.arange(0,Tmax,1/SampleFrequency) #same as x*dt
    figs = []
    x1 = FourierTransform(sig,Tmax,NumberOfSamples,SampleFrequency)
    x2 = get_psd_values(sig,Tmax,NumberOfSamples,SampleFrequency)
    x3 = get_autocorr_values(sig,Tmax,NumberOfSamples,SampleFrequency)
    figs.append(getPlot(t,sig,"time (s)","Amplitude","Raw Data"))
    figs.append(getPlot(x1['Frequency'],x1['Freq. Amp.'],'Frequency [Hz]',"time (s)","FFT"))
    figs.append(getPlot(x2['Frequency'],x2['PSD'],'Frequency [Hz]','PSD [V**2 / Hz]',"PSD"))
    figs.append(getPlot(x3['X Values'],x3['Autocorr Values'],'time delay [s]',"Autocorrelation amplitude","Autocorrelation"))

    return figs
