import scipy
# import matplotlib.pyplot as plt
import math
import re
import numpy as np
import os
import scipy.signal as signal
# import scipy.io as scio
from scipy.signal import butter, lfilter
import scipy.io.wavfile as wav


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
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

if __name__ == '__main__':
    # filepath = 'D:/zq/OneDrive/experiments/2019/20191013/lipcontrol/sentence2/'
    filepath = 'G:/experiments/2023/prp/se/data/'
    files = os.listdir(filepath)
    kk = 0
    for file in files:
        pattern = re.compile(r'\d+')
        res = re.findall(pattern, file)
        if len(res) == 2 and int(res[1]) == 0 and int(res[0]) >= 200 and file[-3:] == 'wav':
            filename = filepath + file
            print(filename)
            # rawdata = np.memmap(filename, dtype=np.float32, mode='r')
            # rawdata = rawdata[48000:]
            _, rawdata = wav.read(filename)
            #如果是原始音频加上下面两行
            rawdata = mic_adjust(rawdata)
            rawdata = rawdata[24000:, ]/32768
            # plt.figure()
            # plt.plot(rawdata)
            data_mics = []
            for i in range(rawdata.shape[1]):
                signaldata = butter_lowpass_filter(rawdata[:,i], 10000, 48000, 5)
                # Create test signal and STFT.
                x = signaldata[0:len(signaldata):3]
                fs = 16000
                # X = stft(x, fs, framesz, hop)
                # freq, t, X = signal.stft(x, fs, nperseg=fs*0.05, noverlap=fs*0.025, nfft=1024)
                #  noise_stft3 clean_stft3
                freq, t, X = signal.stft(x, fs, nperseg=fs * 0.032, noverlap=fs * 0.016, nfft=512)
                # noise_stft2 clean_stft
                X = np.transpose(X)
                X_real = np.real(X)
                X_imag = np.imag(X)
                X_data = np.hstack([X_real, X_imag])
                data_mics.append(X_data)
            data_mics = np.array(data_mics[0])
            data_mics = data_mics.astype(np.float32)
            np.savez_compressed(
                'G:/experiments/2023/prp/se/data_stft/datapre%d-%d' % (int(res[0]), int(res[1])),
                datapre=data_mics)
            print(kk)
            kk = kk + 1
