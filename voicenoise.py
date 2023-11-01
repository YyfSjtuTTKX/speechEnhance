import numpy as np
import matplotlib.pyplot as plt
import math
import os
import re
from scipy import signal
from scipy.signal import butter, lfilter, find_peaks_cwt
from scipy.io import wavfile
# from statsmodels.tsa.seasonal import seasonal_decompose
# from python_speech_features import logfbank
import scipy.io.wavfile as wav
import random

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def Add_noise(x, d, SNR):
     P_signal = np.sum(abs(x)**2)
     P_d = np.sum(abs(d)**2)
     P_noise = P_signal/10**(SNR/10)
     noise = np.sqrt(P_noise/P_d)*d
     # noise = d
     noise_signal = x
     if len(noise)<len(noise_signal):
         noise_signal[0:len(noise)] = noise_signal[0:len(noise)] + noise
     else:
         noise_signal = noise_signal + noise[0:len(noise_signal)]
     return noise_signal

def mic_adjust(rawdata):
    original_data_mics = np.transpose(rawdata)
    en_mics = []
    for i in range(original_data_mics.shape[0]):
        original_data_mic = original_data_mics[i]
        b, a = signal.butter(3, 7000 / (48000 / 2), 'low')
        original_data_mic = signal.filtfilt(b, a, original_data_mic)
        en_mics.append(np.mean(np.abs(original_data_mic)))
    # plt.figure()
    # plt.plot(en_mics)
    en_mics_new = sorted(en_mics)
    index_ret1 = en_mics.index(en_mics_new[0])
    index_ret2 = en_mics.index(en_mics_new[1])
    return_data = original_data_mics[index_ret1]
    print(index_ret1,index_ret2)
    if np.abs(index_ret1-index_ret2) == 1:
        if np.min([index_ret1,index_ret2]) == 0:
            original_data_mics_new = original_data_mics[2:]
        elif np.max([index_ret1,index_ret2]) == original_data_mics.shape[0]-1:
            original_data_mics_new = original_data_mics[:-2]
        else:
            original_data_mics_new = original_data_mics[np.max([index_ret1,index_ret2])+1:]
            original_data_mics_new = np.vstack([original_data_mics_new, original_data_mics[:np.min([index_ret1,index_ret2])]])
    elif np.abs(index_ret1-index_ret2) == original_data_mics.shape[0]-1:
        original_data_mics_new = original_data_mics[1:original_data_mics.shape[0]-1]
    else:
        print('回采通道估算可能出错')
    original_data_mics_new = np.transpose(original_data_mics_new)
    return original_data_mics_new


fs = 48000
filepath = 'G:/experiments/2023/prp/se/data/'
files = os.listdir(filepath)
N_filepath = 'G:/experiments/2023/prp/se/data/'
N_files = os.listdir(N_filepath)
kk = 0
list1 = []
list2 = []
list3 = []
for file in N_files:
    pattern = re.compile(r'\d+')
    res = re.findall(pattern, file)
    if len(res) == 3 and int(res[1]) >= 0:
        list1.append(int(res[0]))
        list2.append(int(res[1]))
        list3.append(int(res[2]))

for file in files:
    pattern = re.compile(r'\d+')
    res = re.findall(pattern, file)
    if len(res) == 2 and (int(res[1]) == 0) and (int(res[0]) >= 200):
        filename = filepath +file
        ind1 = np.random.randint(0, 200)
        ind2 = np.random.randint(1, 3)
        print(ind1)
        N_file = str(ind1)+'_'+str(ind2) + '.wav'
        # ind = np.random.randint(1, 101)
        # N_file = 'n%d.wav' % (ind)
        N_file = N_filepath+N_file
        # (Nsr, Nsignal) = wav.read(N_file)
        # new_length = len(Nsignal)*fs/Nsr
        # Nsignal = Nsignal.astype(np.float32)
        # Nsignal = Utils.resampling(np.arange(0, len(Nsignal), 1), Nsignal, [0, len(Nsignal)], new_length)[1]
        # Nsignal = np.hstack([Nsignal, Nsignal, Nsignal])
        _, Nsignal = wav.read(N_file)
        Nsignal = mic_adjust(Nsignal)
        Nsignal = Nsignal[24000:]/32768
        noise_signal = []
        for i in range(Nsignal.shape[1]):
            noise_signal.append(butter_lowpass_filter(Nsignal[:,i], 10000, fs, 5))
        noise_signal = np.array(noise_signal)
        # tmp = [0]*12000
        # tmp = np.array(tmp)
        # Nsignal = np.hstack([tmp, Nsignal])
        # plt.figure()
        # plt.plot(noise_signal[0])
        _, rawdata = wav.read(filename)
        rawdata = mic_adjust(rawdata)
        rawdata = rawdata[24000:]/32768
        # rawdata = np.load(filename)
        # signal = rawdata['datapre']
        data_signal = []
        for i in range(rawdata.shape[1]):
            data_signal.append(butter_lowpass_filter(rawdata[:, i], 10000, fs, 5))
        data_signal = np.array(data_signal)
        # plt.figure()
        # plt.plot(signal[0])
        # plt.plot(signal[5])

        SNR = random.randint(0, 100)/10 - 5
        new_signal = []
        for i in range(len(data_signal)):
            new_signal.append(Add_noise(data_signal[i], noise_signal[i], SNR))
        new_signal = np.transpose(np.array(new_signal))
        # plt.figure()
        # plt.plot(new_signal[0])
        new_file = 'G:/experiments/2023/prp/se/data_noise/'
        wavfile.write(new_file+file, 48000, new_signal.astype(np.float32))
        print(kk)
        kk = kk + 1