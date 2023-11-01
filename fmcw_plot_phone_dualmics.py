import numpy as np
import scipy.fftpack as fftp
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.fftpack import fft, ifft
import os
import re
import time
import scipy.io.wavfile as wav
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import stft
from scipy.ndimage import gaussian_filter1d

# params
C = 343.00 #声速
Fc = 18000 # 开始频率
Tw = 0.01 #chirp时长
Tf = 0.0
PRT1 = 0.0
PRT2 = 0.0
B = 4000  #带宽
Fs = 48000 #采样频率
Ts = 1 / Fs
k = B / Tw
len_flag = round(Fs * Tf)
len_cycle = round(Fs * (Tw + PRT1))
len_chirp = round(Fs * Tw)
len_blank = round(6 * PRT2 * Fs)
Nfft = 1*round(Tw * Fs)  # fft值，根据需要确定
dist_min = 0.01  # 最小距离 m
dist_max = 0.8  # 最大距离 m
bi_flag = 1

#生成单边信号的发射信号
def generate_chirp(sample_rate, chirp_duration, start_freq, band_width):
    amp = 1
    B = band_width
    Tw = chirp_duration
    init_phase = 0.0
    Fc = start_freq
    step = 1.0 / sample_rate
    t = np.arange(0.0, Tw, step, dtype='float')
    trans_sw_sin = amp * np.sin(init_phase + 2 * np.pi * (Fc * t + B / Tw / 2 * t ** 2))
    trans_sw_cos = amp * np.cos(init_phase + 2 * np.pi * (Fc * t + B / Tw / 2 * t ** 2))

    # plt.figure()
    # plt.plot(trans_sw_sin)
    return (trans_sw_sin, trans_sw_cos, t)

#生成双边信号的发射信号
def generate_chirp_bilateral(sample_rate, chirp_duration, start_freq, band_width):
    amp = 1
    B = band_width
    Tw = chirp_duration
    init_phase = 0.0
    Fc = start_freq
    step = 1.0 / sample_rate
    t = np.arange(0.0, Tw / 2, step, dtype='float')
    t = t[:len_cycle//2]
    chirp_sin_up = amp * np.sin(init_phase + 2 * np.pi * (Fc * t + B / Tw * t ** 2))
    chirp_cos_up = amp * np.cos(init_phase + 2 * np.pi * (Fc * t + B / Tw * t ** 2))
    chirp_sin_down = amp * np.sin(init_phase + 2 * np.pi * ((Fc + B) * t - B / Tw * t ** 2))
    chirp_cos_down = amp * np.cos(init_phase + 2 * np.pi * ((Fc + B) * t - B / Tw * t ** 2))
    trans_sw_sin = np.hstack((chirp_sin_up, chirp_sin_down))

    trans_sw_cos = np.hstack((chirp_cos_up, chirp_cos_down))
    return (trans_sw_sin, trans_sw_cos, t)


#巴特沃斯带通
def butter_bandpass(lowcut, highcut, Fs, order=5):
    nyq = 0.5 * Fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, Fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, Fs, order)
    y = lfilter(b, a, data)
    return y

#求相关性
def correlation_lags(in1_len, in2_len, mode='full'):
    if mode == "full":
        lags = np.arange(-in2_len + 1, in1_len)
    elif mode == "same":
        lags = np.arange(-in2_len + 1, in1_len)
        mid = lags.size // 2
        lag_bound = in1_len // 2
        if in1_len % 2 == 0:
            lags = lags[(mid - lag_bound):(mid + lag_bound)]
        else:
            lags = lags[(mid - lag_bound):(mid + lag_bound) + 1]
    elif mode == "valid":
        lag_bound = in1_len - in2_len
        if lag_bound >= 0:
            lags = np.arange(lag_bound + 1)
        else:
            lags = np.arange(lag_bound, 1)
    return lags

#求接收信号和发射信号的延迟，以实现对齐
def delay_cal(data, ref_sig):
    correlation = signal.correlate(data, ref_sig, mode="full")
    lags = correlation_lags(data.size, ref_sig.size, mode="full")
    lag = lags[np.argmax(correlation)]
    # plt.figure()
    # plt.plot(lags, correlation)
    return lag

#发射信号和接收信号在对齐后进行相乘，得到混频信号
def mixed_sw(data, trans_sw_cos, trans_sw_sin, dist_min, dist_max, B, C, Tw):
    mix_sw_cos = data * trans_sw_cos
    mix_sw_sin = data * trans_sw_sin
    lowcut_deltaF = 2 * dist_min * B / C / Tw
    highcut_deltaF = 2 * dist_max * B / C / Tw

    mix_sw_cos_bpf = butter_bandpass_filter(mix_sw_cos, lowcut_deltaF, highcut_deltaF, Fs, order=3)
    mix_sw_sin_bpf = butter_bandpass_filter(mix_sw_sin, lowcut_deltaF, highcut_deltaF, Fs, order=3)

    array_mix_sw = mix_sw_cos_bpf + 1j * mix_sw_sin_bpf
    return mix_sw_cos_bpf, mix_sw_sin_bpf, array_mix_sw

#对混频信号作傅里叶变换可以得到各个bin的IQ值
def compute_IQ(mixed_chirp, dist_idx, Nfft):
    N = round(Nfft // 2)

    phasor = fftp.fft(mixed_chirp, Nfft)
    phasor = phasor[0: N]
    phasor = phasor[dist_idx]

    est_I_vec = np.real(phasor)
    est_Q_vec = np.imag(phasor)
    est_I_Q_vec = est_I_vec + 1j * est_Q_vec

    return est_I_vec, est_Q_vec, est_I_Q_vec

#移走异常点
def remove_outlier(data):
    for i in range(6, len(data) - 6):
        if abs(data[i] - data[i - 1]) > 2 * (max(data[i - 6:i]) - min(data[i - 6:i])) \
                and abs(data[i + 3] - data[i]) > 2 * (
                max(data[i + 3:i + 8]) - min(data[i + 3:i + 8])):
            data[i] = data[i - 1]
    return data


def select_bin(IQ):   #生成每个bin的相位和幅度信号
    I = np.real(IQ)
    Q = np.imag(IQ)
    IQ = []
    I_diff = []
    Q_diff = []
    for i in range(I.shape[0]):
        decompositionI = seasonal_decompose(I[i], freq=5, two_sided=False)
        trendI = decompositionI.trend
        decompositionQ = seasonal_decompose(Q[i], freq=5, two_sided=False)
        trendQ = decompositionQ.trend
        trendI = trendI[5:]
        trendI = gaussian_filter1d(I[i], sigma=3)
        trendQ = trendQ[5:]
        trendQ = gaussian_filter1d(Q[i], sigma=3)
        trendIQ = trendI + 1j*trendQ
        IQ.append(trendIQ)
        I_diff.append(np.diff(trendI))
        Q_diff.append(np.diff(trendQ))
    IQ = np.array(IQ)
    (num_of_bins, num_of_chirps) = IQ.shape
    I = np.real(IQ)
    Q = np.imag(IQ)
    am_map = np.abs(IQ)  # 幅度
    # plt.figure()
    # plt.pcolormesh(np.abs(am_map[:, :]), vmin=None, vmax=np.max(np.abs(am_map[:, :]))/1)
    # plt.colorbar()
    corr_scores = []   #表示前后两个chirp所有bin的幅度值的相关系数
    for i in range(1, num_of_chirps):
        cur = am_map[:, i]
        pre = am_map[:, i-1]
        corr_score = np.corrcoef(cur, pre)[0][1]
        corr_scores.append(corr_score)
    plt.figure()
    plt.plot(corr_scores)
    phase_map = np.angle(IQ)  # 相位
    phase_map = np.unwrap(phase_map, axis=1)
    am_map2 = []
    phase_map2 = []
    for bin_idx in range(0, num_of_bins):
        am_bin = am_map[bin_idx, :]
        phase_bin = phase_map[bin_idx, :]
        phase_bin = np.diff(phase_bin)
        am_bin = np.diff(am_bin)
        phase_bin = remove_outlier(phase_bin)
        # phase_bin = np.unwrap(phase_bin)
        am_bin = remove_outlier(am_bin)
        am_bin = am_bin
        phase_map2.append(phase_bin)
        am_map2.append(am_bin)
    phase_map2 = np.array(phase_map2)  #幅度和相位的差分
    am_map2 = np.array(am_map2)
    # plt.figure()
    # for i in range(10):
    #     plt.plot(phase_map2[i*4],'.-')
    return am_map2, phase_map2, np.array(I_diff), np.array(Q_diff)




def gen_IQ(original_data, return_data):
    if bi_flag == 0:   #根据单边还是双边信号选择发射信号
        trans_sw_sin, trans_sw_cos, t = generate_chirp(Fs, Tw, Fc, B)
    else:
        trans_sw_sin, trans_sw_cos, t = generate_chirp_bilateral(Fs, Tw, Fc, B)
    interested_signal = original_data
    if interested_signal.dtype == 'int16':
        interested_signal = interested_signal / 32768
    ref_data = trans_sw_cos   #发射信号
    lag = delay_cal(return_data[48000:48000+len_chirp * 3], ref_data) #求直达信号和发射信号的延迟
#过滤接收信号，只取FMCW带宽内的信号
    lowcut = Fc - 10
    highcut = Fc + B + 10
    interested_signal_filtered = butter_bandpass_filter(interested_signal, lowcut, highcut, Fs, order=5)
    ref_data_filtered = butter_bandpass_filter(ref_data, lowcut, highcut, Fs, order=5)
    lag2 = delay_cal(ref_data_filtered, ref_data)
    print(lag2, interested_signal_filtered.shape)
    freq_search = np.linspace(0, Fs // 2, Nfft // 2)   #选择混频信号的搜索频率
    if bi_flag == 0:
        dist_search = freq_search * C * Tw / (2 * B)
    else:
        dist_search = freq_search * C * Tw / (2 * B * 2)
    dist_idx = (dist_search >= dist_min) & (dist_search <= dist_max) #跟据距离的最大值最小值确定bin的区间范围

    lag = int((lag + lag2) % (Fs * Tw)) #确定延迟

    # 计算chirp数量
    sig_cycles = int((len(interested_signal_filtered) - (lag)) / len_chirp)
    print(sig_cycles)
    matrix_data = np.zeros((sig_cycles, len_cycle))
    for i in range(0, sig_cycles):  #取每个chirp的接收信号
        matrix_data[i] = interested_signal_filtered[ i* len_cycle + lag: i* len_cycle + lag + len_chirp]
    # 解调：与发射信号相乘
    mix_sw_cos = matrix_data * trans_sw_cos
    mix_sw_sin = matrix_data * trans_sw_sin

    lowcut_deltaF = 2 * dist_min * B / C / Tw   #根据最小最大距离确定混频信号转换成频域后的频率区间
    highcut_deltaF = 2 * dist_max * B / C / Tw
    mix_sw_cos_bpf = butter_bandpass_filter(mix_sw_cos, lowcut_deltaF, highcut_deltaF, Fs, order=3)
    mix_sw_sin_bpf = butter_bandpass_filter(mix_sw_sin, lowcut_deltaF, highcut_deltaF, Fs, order=3)
    array_mix_sw = mix_sw_cos_bpf + 1j * mix_sw_sin_bpf
    # 计算IQ
    N = round(Nfft // 2)
    phasor = fft(array_mix_sw[:, :], Nfft)
    phasor = phasor[:, 0: N]
    phasor = phasor[:, dist_idx]
    # shape = (bins, sig_cycles)
    est_I_Q_vec_sequence = phasor.T   #得到IQ信号
    print(est_I_Q_vec_sequence.shape)
    return est_I_Q_vec_sequence


def fmcw_pro(file_path, file, audio_type='wav'):
    original_data_mc = np.memmap(file_path, dtype=np.float32, mode='r')
    original_data_mc0 = original_data_mc[::2]
    original_data_mc1 = original_data_mc[1::2]
    original_data_mc = [original_data_mc0, original_data_mc1]
    original_data = np.array(original_data_mc)
    original_data_mc = np.transpose(original_data)
    return_data = original_data_mc[:, 0]
    arrayIQ = []
    am_maps = []
    phase_maps = []
    I_mics = []
    Q_mics = []
    for i in range(2):
        arrayIQ.append(gen_IQ(original_data_mc[int(24000*3):int(24000*-1), i], return_data))  # 生成2个麦克风的IQ信号
        am_map, phase_map, I_diff, Q_diff = select_bin(arrayIQ[-1])  # 选择第一个麦克风所有bin的幅度信号进行画图
        am_maps.append(am_map)
        phase_maps.append(phase_map)
        I_mics.append(I_diff)
        Q_mics.append(Q_diff)
    # am_map2, phase_map2, phase_map_undiff2 = select_bin(arrayIQ[1])  # 选择第2个麦克风所有bin的幅度信号进行画图
    # diff_phase = phase_map_undiff2-phase_map_undiff
    # diff_phase = np.unwrap(diff_phase, axis=1)
    # diff_phase = np.diff(diff_phase, axis=1)
    # plt.figure()
    # for i in range(10):
    #     plt.plot(diff_phase[i*16],'.-')
        tt = [i*Tw for i in range(am_map[:, :].shape[1])]
        if bi_flag == 1:
            bins = [i* C  / (2 * B * 2) for i in range(am_map[:, :].shape[0])]
        else:
            bins = [i * C / (2 * B) for i in range(am_map[:, :].shape[0])]
        plt.figure()
        plt.pcolormesh(tt, bins, np.abs(Q_diff[:, :]), vmin=None, vmax=np.max(np.abs(Q_diff[:, :]))/1)
        plt.colorbar()
        letter_size = 15
        font = {'weight': 'normal', 'size': letter_size}
        plt.xlabel('Time(s)', font)
        plt.ylabel('Distance(m)', font)
    # am_sum = np.sum(np.abs(am_map[:, :]), axis=0)
    # plt.figure()
    # plt.plot(am_sum)
    I_mics = np.array(I_mics)
    Q_mics = np.array(Q_mics)
    am_maps = np.array(am_maps)
    print(np.max(np.abs(am_maps[:, :])))
    phase_maps = np.array(phase_maps)
    amph_maps = np.vstack([am_maps, phase_maps])
    IQ = np.vstack([I_mics, Q_mics])
    # file_path_new = 'G:/experiments/2023/far_lip/farlip/pre/'
    # file_name_new = file_path_new + file[:-3] + 'npz'
    # np.savez_compressed(
    #     file_name_new,
    #     datapre= amph_maps)



if __name__ == '__main__':
    file_path = 'E:/zq_软件/llap/server/pcm/'
    file = '189.pcm'
    filename = file_path + file
    fmcw_pro(filename, file)
    plt.show()