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

log_dir = "logs/train4"  # 指定TensorBoard日志的目录
summary_writer = tf.summary.create_file_writer(log_dir)

def conv2d(x, W):
    return tf.nn.conv2d(input=x, filters=W, strides=[1, 1, 1, 1], padding='SAME')

# dcay改为momentum
# bn_layer = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
# def bn_layer_fun(inputs, scope=None):
#     return bn_layer(inputs)
def bn_layer_fun(inputs,scope,isTraining):
    bn_layer=tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
    return bn_layer(inputs,training=isTraining)

class voiceModulePooling(tf.keras.layers.Layer):
    def __init__(self, numFilters, kernelSize, dilationRate, paddingWord, poolSize,poolStrides,poolPadding, **kwargs):
        super(voiceModulePooling, self).__init__(**kwargs)
        self.poolSize = poolSize; self.poolStrides = poolStrides; self.poolPadding = poolPadding
        self.conv_layer = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=kernelSize, dilation_rate=dilationRate, padding=paddingWord)
        self.bn_layer = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
        self.relu_layer = tf.keras.layers.ReLU()

    def call(self, inputs,isTraining):
        voice_out = self.conv_layer(inputs)
        bn_out = self.bn_layer(voice_out,training=isTraining)
        rl_out = self.relu_layer(bn_out)
        pl_out = tf.nn.max_pool(rl_out, ksize=self.poolSize, strides=self.poolStrides, padding=self.poolPadding)
        return pl_out

class voiceModule(tf.keras.layers.Layer):
    def __init__(self, numFilters, kernelSize, dilationRate, paddingWord, **kwargs):
        super(voiceModule, self).__init__(**kwargs)
        self.conv_layer = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=kernelSize, dilation_rate=dilationRate, padding=paddingWord)
        self.bn_layer = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
        self.relu_layer = tf.keras.layers.ReLU()

    def call(self, inputs,isTraining):
        voice_out = self.conv_layer(inputs)
        bn_out = self.bn_layer(voice_out,training=isTraining)
        rl_out = self.relu_layer(bn_out)
        return rl_out
    
class senseModulePooling(tf.keras.layers.Layer):
    def __init__(self, numFilters, kernelSize, dilationRate, paddingWord, poolSize,poolStrides,poolPadding, **kwargs):
        super(senseModulePooling, self).__init__(**kwargs)
        self.poolSize = poolSize; self.poolStrides = poolStrides; self.poolPadding = poolPadding
        self.conv_layer = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=kernelSize, dilation_rate=dilationRate, padding=paddingWord)
        self.bn_layer = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
        self.relu_layer = tf.keras.layers.ReLU()

    def call(self, inputs, isTraining):
        sense_out = self.conv_layer(inputs)
        bn_out = self.bn_layer(sense_out,training=isTraining)
        rl_out = self.relu_layer(bn_out)
        pl_out = tf.nn.max_pool(rl_out, ksize=self.poolSize, strides=self.poolStrides, padding=self.poolPadding)
        return pl_out

class senseModule(tf.keras.layers.Layer):
    def __init__(self, numFilters, kernelSize, dilationRate, paddingWord, **kwargs):
        super(senseModule, self).__init__(**kwargs)
        self.conv_layer = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=kernelSize, dilation_rate=dilationRate, padding=paddingWord)
        self.bn_layer = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
        self.relu_layer = tf.keras.layers.ReLU()

    def call(self, inputs, isTraining):
        sense_out = self.conv_layer(inputs)
        bn_out = self.bn_layer(sense_out,training=isTraining)
        rl_out = self.relu_layer(bn_out)
        return rl_out

class rModel(tf.keras.layers.Layer):
    def __init__(self, numFilters, kernelSize, stridesNum, paddingWord, dilationRate=1, **kwargs):
        super(rModel, self).__init__(**kwargs)
        self.conv_layer = tf.keras.layers.Conv1D(filters=numFilters, kernel_size=kernelSize, strides=stridesNum,dilation_rate=dilationRate, padding=paddingWord)
        self.bn_layer = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True)
        self.relu_layer = tf.keras.layers.ReLU()

    def call(self, inputs,isTraining):
        r_out = self.conv_layer(inputs)
        bn_out = self.bn_layer(r_out,training=isTraining)
        rl_out = self.relu_layer(bn_out)
        return r_out,rl_out
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
        self.voice_1 = voiceModulePooling(16, (9, 5), (1, 1), 'same', [1, 1, 2, 1], [1, 1, 2, 1], 'SAME')
        self.voice_2 = voiceModulePooling(32, (9, 5), (2, 2), 'same', [1, 1, 2, 1], [1, 1, 2, 1], 'SAME')
        self.voice_3 = voiceModulePooling(64, (5, 5), (2, 2), 'same', [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        self.voice_4 = voiceModule(64, (5, 5), (1,1), 'same') 
        self.voice_5 = voiceModule(64, (5, 5), (2,2), 'same')   
        self.voice_6 = voiceModule(4, (5, 5), (1,1), 'same')
        self.sense_1 = senseModulePooling(16, (9, 5), (1, 1), 'same', [1, 1, 1, 1], [1, 1, 1, 1], 'SAME')          
        self.sense_2 = senseModulePooling(16, (5, 5), (2, 2), 'same', [1, 2, 1, 1], [1, 2, 1, 1], 'SAME')           
        self.sense_3 = senseModulePooling(32, (5, 5), (1, 1), 'same', [1, 1, 2, 1], [1, 1, 2, 1], 'SAME')           
        self.sense_4 = senseModule(32, (5, 5), (4, 1), 'same')
        self.sense_5 = senseModule(16, (5, 5), (2,1), 'same')   
        self.r_1 = rModel(264, 5, 1, 'same', 1)
        self.r_2 = rModel(264, 5, 1, 'same', 2)           
        self.r_3 = rModel(264, 5, 1, 'same', 1)        
        self.r_4 = rModel(264, 5, 1, 'same', 2)        
        self.r_5 = rModel(264, 5, 1, 'same', 1)        
        self.r_6 = rModel(264, 5, 1, 'same', 2)        
        self.r_7 = rModel(264, 5, 1, 'same')
 
    def call(self,inputs_voice,inputs_sense,isTraining):
        # voice    
        voice_1_output = self.voice_1(inputs_voice,isTraining)
        voice_2_output = self.voice_2(voice_1_output,isTraining)   
        voice_3_output = self.voice_3(voice_2_output,isTraining)         
        voice_4_output = self.voice_4(voice_3_output,isTraining)
        voice_4_output += voice_3_output      
        voice_5_output = self.voice_5(voice_4_output,isTraining)
        voice_5_output += voice_4_output       
        voice_6_output = self.voice_6(voice_5_output,isTraining)
        # print("voice_6_output",voice_6_output.shape)
        voice_6_re = tf.reshape(voice_6_output, [-1, voice_6_output.shape[1], voice_6_output.shape[2]*4, 1]) 

        sense_1_output = self.sense_1(inputs_sense,isTraining)       
        sense_2_output = self.sense_2(sense_1_output,isTraining)      
        sense_3_output = self.sense_3(sense_2_output,isTraining)     
        sense_4_output = self.sense_4(sense_3_output,isTraining)
        sense_4_output += sense_3_output         
        sense_5_output = self.sense_5(sense_4_output,isTraining)
        # print("sense_5_output",sense_5_output.shape)
        sense_5_re = tf.reshape(sense_5_output, [-1, sense_5_output.shape[1], sense_5_output.shape[2]*16, 1])
        # print("sense_5_re",sense_5_re.shape)
        # h_sense = tf.image.resize_nearest_neighbor(sense_5_re, (voice_6_re.shape[1], voice_6_re.shape[2]))
        h_sense = tf.image.resize(sense_5_re, (voice_6_re.shape[1], voice_6_re.shape[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # print("h_sense",h_sense.shape)
        h_voice = voice_6_re
        h_conc = tf.concat([h_sense, h_voice], 2)
        ch_pool = h_conc
        ch_pool = tf.reshape(ch_pool, [-1, ch_pool.shape[1], ch_pool.shape[2]])
        nchannel = ch_pool.shape[2]
        # print("nchannel",nchannel)
        # print("ch_pool",ch_pool.shape)
        r_layer_1,r_1_out = self.r_1(ch_pool,isTraining)
        r_pool_1 = r_layer_1 + r_1_out
        r_layer_2,r_2_out = self.r_2(r_pool_1,isTraining)
        r_pool_2 = r_2_out
        r_layer_3,r_3_out = self.r_3(r_pool_2,isTraining)
        r_pool_3 = r_3_out + r_pool_2
        r_layer_4,r_4_out = self.r_4(r_pool_3,isTraining)
        r_pool_4 = r_4_out + r_pool_3
        r_layer_5,r_5_out = self.r_5(r_pool_4,isTraining)
        r_pool_5 = r_5_out + r_pool_4
        r_layer_6,r_6_out = self.r_6(r_pool_5,isTraining)
        r_pool_6 = r_6_out + r_pool_5
        r_layer_7,r_7_out = self.r_7(r_pool_6,isTraining)
        r_pool_7 = r_7_out
        # print("r_pool_7",r_pool_7.shape)
        rh_pool7 = tf.reshape(r_pool_7, [-1, r_7_out.shape[1], r_7_out.shape[2], 1])
        # print("rh_pool7",rh_pool7.shape)
        econv1_shape = inputs_voice.shape.as_list()
        econv1_shape[-1]=16
        dconv1 = tf.image.resize(rh_pool7, (econv1_shape[1], econv1_shape[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # print("dconv1",dconv1.shape)
        dconv1 = conv2d(dconv1, self.dw_conv1) + self.db_conv1
        dBN_out1 = bn_layer_fun(dconv1, 'dBN1',isTraining)
        dh_conv1 = tf.nn.relu(dBN_out1)
        dconv2 = conv2d(dh_conv1, self.dw_conv2) + self.db_conv2
        dBN_out2 = bn_layer_fun(dconv2, 'dBN2',isTraining)
        dh_conv2 = tf.nn.sigmoid(dBN_out2)

        y_conv = dh_conv2
        return y_conv, h_sense

dcnn = Action_model()
# plot_model(dcnn, to_file='model.png', show_shapes=True, show_layer_names=True)

optimizer = tf.keras.optimizers.Adam()
tf.config.experimental_run_functions_eagerly(True)
global_step = tf.Variable(0, trainable=False)
@tf.function  
def train_step(batch_x_sense, batch_x_voice, batch_y, isTraining, global_step):
    with tf.GradientTape() as tape:
        # 在 tf.GradientTape 中记录前向传播过程
        mask, h_sense = dcnn(batch_x_voice,batch_x_sense,isTraining)
        # <dtype: 'float32'> <dtype: 'float64'>
        # print(mask.dtype,batch_x_voice.dtype)
        batch_x_voice = tf.cast(batch_x_voice, tf.float32)
        logits = batch_x_voice*mask
        logits = tf.cast(logits, tf.float64)
        loss=tf.reduce_sum(tf.square(logits-batch_y))

    global_step.assign_add(1)
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
           loss,_,_=train_step(batch_x_sense,batch_x_voice,batch_y,True,global_step)
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