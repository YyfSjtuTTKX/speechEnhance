import tensorflow as tf
import numpy as np
import time
import multi_ae_input_long
# import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.io as scio
import os
import csv
from tensorflow.python.keras.utils.vis_utils import plot_model
# import keras
from scipy.signal import butter, lfilter, find_peaks_cwt, stft, spectrogram, convolve2d

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#数据载入
my_data = multi_ae_input_long.Data_Control('../data_pro_wo_trend/',
                                      '../data_noise_stft/',
                                      '../data_stft/')

X_sense = my_data.trainsense
X_sense = X_sense.reshape(-1,my_data.trainsense.shape[1],my_data.trainsense.shape[2],6)
X_voice = my_data.traindata[:, :, :257]
Y = my_data.trainlabel[:, :, :257]
X_voice = X_voice.reshape(-1, X_voice.shape[1], X_voice.shape[2], 1)
Y = Y.reshape(-1, X_voice.shape[1], X_voice.shape[2], 1)


Keep_p = 0.6
batch_size = 32

Xtest_sense = my_data.testsense
Xtest_sense = Xtest_sense.reshape(-1, my_data.testsense.shape[1], my_data.testsense.shape[2], 6)
Xtest_voice = my_data.testdata[:, :, :257]
Ytest = my_data.testlabel[:, :, :257]
Xtest_voice = Xtest_voice.reshape(-1, Xtest_voice.shape[1], Xtest_voice.shape[2], 1)
Ytest = Ytest.reshape(-1, Xtest_voice.shape[1], Xtest_voice.shape[2], 1)

log_dir = "logs/new/train3"  # 指定TensorBoard日志的目录
summary_writer = tf.summary.create_file_writer(log_dir)

class Action_model(tf.keras.Model):     
    def __init__(self):
        super(Action_model, self).__init__()
        self.dw_conv1 = self.add_weight(
            name='dw_conv1',
            shape=(5, 5, 1, 1),
            initializer=tf.initializers.TruncatedNormal(mean=0.0, stddev=0.1),
            trainable=True
        )
        self.dw_conv2 = self.add_weight(
            name='dw_conv2',
            shape=(5, 5, 1, 1),
            initializer=tf.initializers.TruncatedNormal(mean=0.0, stddev=0.1),
            trainable=True
        )
        self.db_conv1 = tf.Variable(
            initial_value=tf.constant(0.1, shape=(1,)),  
            trainable=True
        )   
        self.db_conv2 = tf.Variable(
            initial_value=tf.constant(0.1, shape=(1,)), 
            trainable=True
        )   

        self.voiceConv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(9, 5), padding='same',  dilation_rate=(1, 1))
        self.voiceBn1 = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
        self.voiceConv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(9, 5), padding='same',  dilation_rate=(2, 2))
        self.voiceBn2 = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
        self.voiceConv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same',  dilation_rate=(2, 2))
        self.voiceBn3 = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
        self.voiceConv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same',  dilation_rate=(1, 1))
        self.voiceBn4 = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
        self.voiceConv5 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same',  dilation_rate=(2, 2))
        self.voiceBn5 = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
        self.voiceConv6 = tf.keras.layers.Conv2D(filters=4, kernel_size=(5, 5), padding='same',  dilation_rate=(1, 1))
        self.voiceBn6 = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)

        self.senseConv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(9, 5), padding='same',  dilation_rate=(1, 1))
        self.senseBn1 = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
        self.senseConv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='same',  dilation_rate=(2, 2))
        self.senseBn2 = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
        self.senseConv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same',  dilation_rate=(1, 1))
        self.senseBn3 = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
        self.senseConv4 = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same',  dilation_rate=(4, 1))
        self.senseBn4 = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
        self.senseConv5 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='same',  dilation_rate=(2, 1))
        self.senseBn5 = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)

        self.rConv1 = tf.keras.layers.Conv1D(filters=264, kernel_size=5, strides=1, padding='same', dilation_rate=1)
        self.rBn1 = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
        self.rConv2 = tf.keras.layers.Conv1D(filters=264, kernel_size=5, strides=1, padding='same', dilation_rate=21)
        self.rBn2 = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
        self.rConv3 = tf.keras.layers.Conv1D(filters=264, kernel_size=5, strides=1, padding='same', dilation_rate=1)
        self.rBn3 = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
        self.rConv4 = tf.keras.layers.Conv1D(filters=264, kernel_size=5, strides=1, padding='same', dilation_rate=2)
        self.rBn4 = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
        self.rConv5 = tf.keras.layers.Conv1D(filters=264, kernel_size=5, strides=1, padding='same', dilation_rate=1)
        self.rBn5 = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
        self.rConv6 = tf.keras.layers.Conv1D(filters=264, kernel_size=5, strides=1, padding='same', dilation_rate=2)
        self.rBn6 = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
        self.rConv7 = tf.keras.layers.Conv1D(filters=264, kernel_size=5, strides=1, padding='same')
        self.rBn7 = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
 
        self.endBn1 = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
        self.endBn2 = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
    def call(self,inputs_voice,inputs_sense,isTraining):
        # voice    
        voiceOutConv1 = self.voiceConv1(inputs_voice)
        voiceOutBn1 = self.voiceBn1(voiceOutConv1,isTraining)
        voiceRelu1 = tf.nn.relu(voiceOutBn1)
        voicePool1 = tf.nn.max_pool(voiceRelu1, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

        voiceOutConv2 = self.voiceConv2(voicePool1)
        voiceOutBn2 = self.voiceBn2(voiceOutConv2,isTraining)
        voiceRelu2 = tf.nn.relu(voiceOutBn2)
        voicePool2 = tf.nn.max_pool(voiceRelu2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

        voiceOutConv3 = self.voiceConv3(voicePool2)
        voiceOutBn3 = self.voiceBn3(voiceOutConv3,isTraining)
        voiceRelu3 = tf.nn.relu(voiceOutBn3)
        voicePool3 = tf.nn.max_pool(voiceRelu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        voiceOutConv4 = self.voiceConv4(voicePool3)
        voiceOutBn4 = self.voiceBn4(voiceOutConv4,isTraining)
        voiceRelu4 = tf.nn.relu(voiceOutBn4)
        voicePool4 = voiceRelu4 + voicePool3

        voiceOutConv5 = self.voiceConv5(voicePool4)
        voiceOutBn5 = self.voiceBn5(voiceOutConv5,isTraining)
        voiceRelu5 = tf.nn.relu(voiceOutBn5)
        voicePool5 = voiceRelu5 + voicePool4

        voiceOutConv6 = self.voiceConv6(voicePool5)
        voiceOutBn6 = self.voiceBn6(voiceOutConv6,isTraining)
        voiceRelu6 = tf.nn.relu(voiceOutBn6)
        voicePool6 = tf.reshape(voiceRelu6, [-1, voiceRelu6.shape[1], voiceRelu6.shape[2]*4, 1])
        #sense
        senseOutConv1 = self.senseConv1(inputs_sense)
        senseOutBn1 = self.senseBn1(senseOutConv1,isTraining)
        senseRelu1 = tf.nn.relu(senseOutBn1)
        sensePool1 = tf.nn.max_pool(senseRelu1, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

        senseOutConv2 = self.senseConv2(sensePool1)
        senseOutBn2 = self.senseBn2(senseOutConv2,isTraining)
        senseRelu2 = tf.nn.relu(senseOutBn2)
        sensePool2 = tf.nn.max_pool(senseRelu2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

        senseOutConv3 = self.senseConv3(sensePool2)
        senseOutBn3 = self.senseBn3(senseOutConv3,isTraining)
        senseRelu3 = tf.nn.relu(senseOutBn3)
        sensePool3 = tf.nn.max_pool(senseRelu3, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

        senseOutConv4 = self.senseConv4(sensePool3)
        senseOutBn4 = self.senseBn4(senseOutConv4,isTraining)
        senseRelu4 = tf.nn.relu(senseOutBn4)
        sensePool4 = senseRelu4 + sensePool3
    
        senseOutConv5 = self.senseConv5(sensePool4)
        senseOutBn5 = self.senseBn5(senseOutConv5,isTraining)
        senseRelu5 = tf.nn.relu(senseOutBn5)
        sensePool5 = tf.reshape(senseRelu5, [-1, senseRelu5.shape[1], senseRelu5.shape[2]*16, 1])

        h_sense = tf.image.resize(sensePool5, (voicePool6.shape[1], voicePool6.shape[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        h_voice = voicePool6
        h_conc = tf.concat([h_sense, h_voice], 2)
        ch_pool = h_conc
        ch_pool = tf.reshape(ch_pool, [-1, ch_pool.shape[1], ch_pool.shape[2]])
        nchannel = ch_pool.shape[2]
        #r
        rOutConv1 = self.rConv1(ch_pool)
        rOutBn1 = self.rBn1(rOutConv1,isTraining)
        rRelu1 = tf.nn.relu(rOutBn1)
        rPool1 = rRelu1 + rOutConv1
        
        rOutConv2 = self.rConv2(rPool1)
        rOutBn2 = self.rBn2(rOutConv2,isTraining)
        rRelu2 = tf.nn.relu(rOutBn2)
        rPool2 = rRelu2
        
        rOutConv3 = self.rConv3(rPool2)
        rOutBn3 = self.rBn3(rOutConv3,isTraining)
        rRelu3 = tf.nn.relu(rOutBn3)
        rPool3 = rRelu3 + rPool2

        rOutConv4 = self.rConv4(rPool3)
        rOutBn4 = self.rBn4(rOutConv4,isTraining)
        rRelu4 = tf.nn.relu(rOutBn4)
        rPool4 = rRelu4 + rPool3

        rOutConv5 = self.rConv5(rPool4)
        rOutBn5 = self.rBn5(rOutConv5,isTraining)
        rRelu5 = tf.nn.relu(rOutBn5)
        rPool5 = rRelu5 + rPool4

        rOutConv6 = self.rConv6(rPool5)
        rOutBn6 = self.rBn6(rOutConv6,isTraining)
        rRelu6 = tf.nn.relu(rOutBn6)
        rPool6 = rRelu6 + rPool5

        rOutConv7 = self.rConv7(rPool6)
        rOutBn7 = self.rBn7(rOutConv7,isTraining)
        rRelu7 = tf.nn.relu(rOutBn7)
        rPool7 = rRelu7

        rh_pool7 = tf.reshape(rPool7, [-1, rRelu7.shape[1], rRelu7.shape[2], 1])
        econv1_shape = inputs_voice.shape.as_list()
        econv1_shape[-1]=16
        dconv1 = tf.image.resize(rh_pool7, (econv1_shape[1], econv1_shape[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        dconv1 = tf.nn.conv2d(dconv1, self.dw_conv1, strides=[1, 1, 1, 1], padding='SAME')+self.db_conv1
        dBN_out1 = self.endBn1(dconv1,isTraining)
        dh_conv1 = tf.nn.relu(dBN_out1)
        dconv2 = tf.nn.conv2d(dh_conv1, self.dw_conv2, strides=[1, 1, 1, 1], padding='SAME')+self.db_conv2
        dBN_out2 = self.endBn2(dconv2,isTraining)
        dh_conv2 = tf.nn.sigmoid(dBN_out2)
        y_conv = dh_conv2
        return y_conv, h_sense

dcnn = Action_model()
# plot_model(dcnn, to_file='model.png', show_shapes=True, show_layer_names=True)

optimizer = tf.keras.optimizers.Adam()
tf.config.experimental_run_functions_eagerly(True)
# global_step = tf.Variable(0, trainable=False)
@tf.function  
def train_step(batch_x_sense, batch_x_voice, batch_y, isTraining):
    with tf.GradientTape() as tape:
        # 在 tf.GradientTape 中记录前向传播过程
        mask, h_sense = dcnn(batch_x_voice,batch_x_sense,isTraining)
        # <dtype: 'float32'> <dtype: 'float64'>
        # print(mask.dtype,batch_x_voice.dtype)
        batch_x_voice = tf.cast(batch_x_voice, tf.float32)
        logits = batch_x_voice*mask
        logits = tf.cast(logits, tf.float64)
        loss=tf.reduce_sum(tf.square(logits-batch_y))

    # global_step.assign_add(1)
    # 计算梯度
    gradients = tape.gradient(loss, dcnn.trainable_variables)
    # 使用优化器来更新模型权重
    optimizer.apply_gradients(zip(gradients, dcnn.trainable_variables))
    return loss,logits,batch_y

@tf.function  
def test_step(batch_x_sense, batch_x_voice, batch_y, isTraining):
    print("test",batch_x_voice.shape)
    mask, h_sense = dcnn(batch_x_voice,batch_x_sense,isTraining)
    batch_x_voice = tf.cast(batch_x_voice, tf.float32)
    logits = batch_x_voice*mask
    logits = tf.cast(logits, tf.float64)
    loss=tf.reduce_sum(tf.square(logits-batch_y))
    return loss,logits,mask

def generatebatch(X,step, batch_size):
    start = (step*batch_size)%len(X)
    if start + batch_size > len(X):
        start = ((step+1) * batch_size) % len(X)
    end = min(start + batch_size,len(X))
    return start, end # 生成每一个batch

starttime = time.time()
lasttime = time.time()
randomind = range(0, len(X_voice))
best_step = 0
best_loss = 1000
net_train=1

if net_train==1:
    tf.summary.trace_on(graph=True, profiler=True)
    for i in range(60000):
           start, end = generatebatch(randomind, i, batch_size)
           batch_x_voice = X_voice[start:end]
           batch_x_sense = X_sense[start:end]
           batch_y = Y[start:end]
           loss,_,_=train_step(batch_x_sense,batch_x_voice,batch_y,True)
           if i % 200 == 0:
               testloss,_,_ = test_step(Xtest_sense,Xtest_voice,Ytest,False)
               testloss = testloss*270/Ytest.shape[0]
               if testloss < best_loss:
                   best_loss = testloss
                   best_step = i
                #    dcnn.save('save/action_array/first.tf', save_format="tf")
                   dcnn.save_weights('save/first.tf', save_format="tf")
               nowtime = time.time()
               dur1 = nowtime-starttime
               dur2 = nowtime-lasttime
               lasttime = nowtime
               with summary_writer.as_default():
                    tf.summary.scalar('testloss', testloss, step=i) 
                    if i==200:
                        tf.summary.trace_export(name="modelTrace", step=i, profiler_outdir=log_dir) 
               print("step %d, %0fs, %0fs, loss %g, testloss %g" % (i, dur1, dur2, loss, testloss))
           if i % 200 == 0:
               randomind = list(range(X_voice.shape[0]))
               np.random.shuffle(randomind)
           if i % 10000 == 0 and i > 1:
               print("beststep %d, bestloss %g" % (best_step, best_loss))
               print()
    tf.summary.trace_off()
    
elif net_train==2:
    dcnn.load_weights("save/first.tf")
    testloss, signal_re, ma = test_step(Xtest_sense,Xtest_voice,Ytest)

    # ma = ma[0, :, :, 0]
    # ma = np.transpose(ma)
    SDRlist1 = []
    SDRlist2 = []
    SDRlist3 = []
    from pypesq import pesq
    from pystoi import stoi
    from scipy.io import wavfile


    def cal_pesq(st: np.ndarray, se: np.ndarray, fs=16000) -> float:
            score = pesq(ref=st, deg=se, fs=fs)
            return score


    def cal_stoi(st: np.ndarray, se: np.ndarray, fs=16000) -> float:
            score = stoi(st, se, fs, extended=False)
            return score

    def remove_dc(signal):
            """Normalized to zero mean"""
            mean = np.mean(signal)
            signal -= mean
            return signal


    def pow_np_norm(signal):
            """Compute 2 Norm"""
            return np.square(np.linalg.norm(signal, ord=2))


    def pow_norm(s1, s2):
            return np.sum(s1 * s2)


    def si_sdr(estimated, original):
            estimated = remove_dc(estimated)
            original = remove_dc(original)
            target = pow_norm(estimated, original) * original / pow_np_norm(original)
            noise = estimated - target
            return 10 * np.log10(pow_np_norm(target) / pow_np_norm(noise))

    pesqlist1 = []
    pesqlist2 = []
    pesqlist3 = []
    stoilist1 = []
    stoilist2 = []
    stoilist3 = []
    
    for sindex in range(Ytest.shape[0]):
        x_gt_am = my_data.testdata[sindex, :, :257]
        x_gt_ph = my_data.testdata[sindex, :, 257:514]
        x_gt = x_gt_am * np.cos(x_gt_ph) + 1j * x_gt_am * np.sin(x_gt_ph)
        fs = 16000
        t1_gt, xhat1_gt = signal.istft(np.transpose(x_gt), fs, window='hann', nperseg=fs * 0.032,
                                       noverlap=fs * 0.016,
                                       nfft=512)
        en_noise = np.sum(xhat1_gt ** 2)
        nameindex = sindex + 2000
        datapath1 = './data/se_generate/1/0-0-' + str(nameindex) + '.wav'
        wavfile.write(datapath1, fs, xhat1_gt.astype(np.float32))

        y_gt_am = Ytest[sindex, :, :257, 0]
        y_gt_ph = my_data.testlabel[sindex, :, 257:514]
        y_gt = y_gt_am * np.cos(y_gt_ph) + 1j * y_gt_am * np.sin(y_gt_ph)
        t2_gt, xhat2_gt = signal.istft(np.transpose(y_gt), fs, window='hann', nperseg=fs * 0.032,
                                       noverlap=fs * 0.016,
                                       nfft=512)
        en_clean = np.sum(xhat2_gt ** 2)
        datapath2 = './data/se_generate/2/0-0-' + str(nameindex) + '.wav'
        wavfile.write(datapath2, fs, xhat2_gt.astype(np.float32))

        s_re_am = signal_re[sindex, :, :, 0]
        s_re_am_numpy = s_re_am.numpy()
        s_re = s_re_am_numpy * np.cos(x_gt_ph) + 1j * s_re_am_numpy * np.sin(x_gt_ph)
        t3_gt, xhat3_gt = signal.istft(np.transpose(s_re), fs, window='hann', nperseg=fs * 0.032,
                                       noverlap=fs * 0.016,
                                       nfft=512)
        en_re = np.sum(xhat3_gt ** 2)
        datapath3 = './data/se_generate/3/0-0-' + str(nameindex) + '.wav'
        wavfile.write(datapath3, fs, xhat3_gt.astype(np.float32))

        s_re1 = s_re_am_numpy * np.cos(y_gt_ph) + 1j * s_re_am_numpy * np.sin(y_gt_ph)
        t4_gt, xhat4_gt = signal.istft(np.transpose(s_re1), fs, window='hann', nperseg=fs * 0.032,
                                       noverlap=fs * 0.016,
                                       nfft=512)
        datapath4 = './data/se_generate/4/0-0-' + str(nameindex) + '.wav'
        wavfile.write(datapath4, fs, xhat4_gt.astype(np.float32))

        snr1 = 10 * np.log10(np.abs(en_clean) / np.sum((xhat1_gt - xhat2_gt) ** 2))
        snr2 = 10 * np.log10(np.abs(en_clean) / np.sum((xhat3_gt - xhat2_gt) ** 2))
        snr3 = 10 * np.log10(np.abs(en_clean) / np.sum((xhat4_gt - xhat2_gt) ** 2))
        # snr1 = si_sdr(xhat1_gt, xhat2_gt)
        # snr2 = si_sdr(xhat3_gt, xhat2_gt)
        # snr3 = si_sdr(xhat4_gt, xhat2_gt)
        pesq1 = cal_pesq(xhat2_gt, xhat1_gt)
        pesq2 = cal_pesq(xhat2_gt, xhat3_gt)
        pesq3 = cal_pesq(xhat2_gt, xhat4_gt)
        # print(snr1, snr2, snr3)

        stoi1 = cal_stoi(xhat2_gt, xhat1_gt)
        stoi2 = cal_stoi(xhat2_gt, xhat3_gt)
        stoi3 = cal_stoi(xhat2_gt, xhat4_gt)

        SDRlist1.append(snr1)
        SDRlist2.append(snr2)
        SDRlist3.append(snr3)
        pesqlist1.append(pesq1)
        pesqlist2.append(pesq2)
        pesqlist3.append(pesq3)
        stoilist1.append(stoi1)
        stoilist2.append(stoi2)
        stoilist3.append(stoi3)
    with open(os.path.join('./data/', 'evaluation3.csv'), 'w') as f:
        w = csv.writer(f)
        for i in range(len(SDRlist1)):
            w.writerow([SDRlist1[i], SDRlist2[i], SDRlist3[i], pesqlist1[i], pesqlist2[i], pesqlist3[i], stoilist1[i], stoilist2[i], stoilist3[i]])
    SDRlist1 = np.array(SDRlist1)
    SDRlist1 = np.mean(SDRlist1)
    SDRlist2 = np.array(SDRlist2)
    SDRlist2 = np.mean(SDRlist2)
    SDRlist3 = np.array(SDRlist3)
    SDRlist3 = np.mean(SDRlist3)
    print(SDRlist1)
    print(SDRlist2)
    print(SDRlist3)
    print(np.mean(np.array(pesqlist1)))
    print(np.mean(np.array(pesqlist2)))
    print(np.mean(np.array(pesqlist3)))
    print(np.mean(np.array(stoilist1)))
    print(np.mean(np.array(stoilist2)))
    print(np.mean(np.array(stoilist3)))