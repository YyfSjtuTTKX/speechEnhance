import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import multi_ae_input_long
# import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.io as scio
import os
import csv
# import keras
from scipy.signal import butter, lfilter, find_peaks_cwt, stft, spectrogram, convolve2d
# my_data = multi_ae_input_long.Data_Control('./data/lipcontrol/cutdata17/5/sensing/',
#                                       './data/lipcontrol/cutdata17/5/noisy_stft/',
#                                       './data/lipcontrol/cutdata17/5/clean_stft/')

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


sess = tf.InteractiveSession() 
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

def bn_layer(inputs, phase_train, scope=None):
       return tf.cond(phase_train,
                      lambda: tf.contrib.layers .batch_norm(inputs, decay=0.9, is_training=True, scale=True,
            updates_collections=None, scope=scope),
                      lambda: tf.contrib.layers .batch_norm(inputs, decay=0.9,is_training=False, scale=True,
            updates_collections=None, scope=scope, reuse = True))


def dcnn(inputs_sense,inputs_voice, train, keep_prob, reuse=False, name='cnnBN'):
    with tf.variable_scope(name, reuse=reuse) as scope:
        econv1_voice = tf.layers.conv2d(inputs_voice, 16, (9, 5), dilation_rate=(1, 1), padding='same')
        BN_out1 = bn_layer(econv1_voice, train, scope='eBN1')
        eh_conv1_voice = tf.nn.relu(BN_out1)
        eh_pool1_voice = max_pool_2x2(eh_conv1_voice)

        econv2_voice = tf.layers.conv2d(eh_pool1_voice, 32, (9, 5), dilation_rate=(2, 2), padding='same')
        BN_out2 = bn_layer(econv2_voice,train, scope='eBN2')
        eh_conv2_voice = tf.nn.relu(BN_out2)
        eh_pool2_voice = tf.nn.max_pool(eh_conv2_voice, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

        econv3_voice = tf.layers.conv2d(eh_pool2_voice, 64, (5, 5), dilation_rate=(2,2), padding='same')
        BN_out3 = bn_layer(econv3_voice, train, scope='eBN3')
        eh_conv3_voice = tf.nn.relu(BN_out3)
        eh_pool3_voice = tf.nn.max_pool(eh_conv3_voice, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        econv4_voice = tf.layers.conv2d(eh_pool3_voice, 64, (5, 5), dilation_rate=(1,1), padding='same')
        BN_out4 = bn_layer(econv4_voice, train, scope='eBN4')
        eh_conv4_voice = tf.nn.relu(BN_out4)
        # eh_pool4_voice = tf.nn.max_pool(eh_conv4_voice, ksize=[1, 1, 1, 1], strides=[1, 2, 1, 1], padding='SAME')
        eh_pool4_voice = eh_conv4_voice + eh_pool3_voice

        econv5_voice = tf.layers.conv2d(eh_pool4_voice, 64, (5, 5), dilation_rate=(2, 2), padding='same')
        BN_out5 = bn_layer(econv5_voice, train, scope='eBN5')
        eh_conv5_voice = tf.nn.relu(BN_out5)
        eh_pool5_voice = eh_conv5_voice + eh_pool4_voice

        econv6_voice = tf.layers.conv2d(eh_pool5_voice, 4,  (5, 5), dilation_rate=(1, 1), padding='same')
        BN_out6 = bn_layer(econv6_voice, train, scope='eBN6')
        eh_conv6_voice = tf.nn.relu(BN_out6)
        print(eh_conv6_voice.shape)
        eh_pool6_voice = tf.reshape(eh_conv6_voice, [-1, eh_conv6_voice.shape[1], eh_conv6_voice.shape[2]*4, 1])


        conv1_sense = tf.layers.conv2d(inputs_sense, 16, (9, 5), dilation_rate=(1,1), padding='same')
        BN_out1_sense = bn_layer(conv1_sense, train, scope='BN1_sense')
        h_conv1_sense = tf.nn.relu(BN_out1_sense)
        h_pool1_sense = tf.nn.max_pool(h_conv1_sense, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')


        conv2_sense = tf.layers.conv2d(h_pool1_sense, 16, (5, 5), dilation_rate=(2, 2), padding='same')
        BN_out2_sense = bn_layer(conv2_sense, train, scope='BN2_sense')
        h_conv2_sense = tf.nn.relu(BN_out2_sense)
        h_pool2_sense = tf.nn.max_pool(h_conv2_sense, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')


        conv3_sense = tf.layers.conv2d(h_pool2_sense, 32, (5, 5), dilation_rate=(1, 1), padding='same')
        BN_out3_sense = bn_layer(conv3_sense, train, scope='BN3_sense')
        h_conv3_sense = tf.nn.relu(BN_out3_sense)
        h_pool3_sense = tf.nn.max_pool(h_conv3_sense, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')


        conv4_sense = tf.layers.conv2d(h_pool3_sense, 32, (5, 5), dilation_rate=(4, 1), padding='same')
        BN_out4_sense = bn_layer(conv4_sense, train, scope='BN4_sense')
        h_conv4_sense = tf.nn.relu(BN_out4_sense)
        h_pool4_sense = h_conv4_sense + h_pool3_sense

        conv5_sense = tf.layers.conv2d(h_pool4_sense, 16, (5, 5), dilation_rate=(2, 1), padding='same')
        BN_out5_sense = bn_layer(conv5_sense, train, scope='BN5_sense')
        h_conv5_sense = tf.nn.relu(BN_out5_sense)
        print(h_conv5_sense.shape)
        h_pool5_sense = tf.reshape(h_conv5_sense, [-1, h_conv5_sense.shape[1], h_conv5_sense.shape[2]*16, 1])
        print(h_pool5_sense.shape)

        # h_pool4_shape = eh_pool4_voice.get_shape().as_list()
        h_sense = tf.image.resize_nearest_neighbor(h_pool5_sense, (eh_pool6_voice.shape[1], eh_pool6_voice.shape[2]))
        print(h_sense.shape)
        h_voice = eh_pool6_voice

        h_conc = tf.concat([h_sense, h_voice], 2)
        # h_conc = h_voice



        ch_pool = h_conc

        ch_pool = tf.reshape(ch_pool, [-1, ch_pool.shape[1], ch_pool.shape[2]])
        nchannel = ch_pool.shape[2]
        print(ch_pool.shape)
        # ch_avg = tf.reduce_mean(ch_pool, 1)
        # nch1 = nchannel.value
        # nch2 = int(nchannel.value/2)
        #
        # W_fc1 = weight_variable([nch1, nch2])
        # b_fc1 = bias_variable([nch2])
        # z_embedding1 = tf.nn.relu(tf.matmul(ch_avg, W_fc1) + b_fc1)
        # W_fc2 = weight_variable([nch2, nch1])
        # b_fc2 = bias_variable([nch1])
        # z_embedding2 = tf.nn.relu(tf.matmul(z_embedding1, W_fc2) + b_fc2)
        # z_embedding2 = tf.nn.sigmoid(z_embedding2)
        # z_embedding2 = tf.reshape(z_embedding2, [-1, 1, nchannel.value])
        # ch_weight = ch_pool * z_embedding2
        ch_pool = ch_pool

        rconv1 = tf.layers.conv1d(ch_pool, nchannel.value, 5, strides=1, padding='same', dilation_rate=1)
        rBN_out1 = bn_layer(rconv1, train, scope='rBN1')
        rh_conv1 = tf.nn.relu(rBN_out1)
        rh_pool1 = rh_conv1 + rconv1

        rconv2 = tf.layers.conv1d(rh_pool1, nchannel.value, 5, strides=1, padding='same', dilation_rate=2)
        rBN_out2 = bn_layer(rconv2, train, scope='rBN2')
        rh_conv2 = tf.nn.relu(rBN_out2)
        rh_pool2 = rh_conv2

        rconv3 = tf.layers.conv1d(rh_pool2, nchannel.value, 5, strides=1, padding='same', dilation_rate=1)
        rBN_out3 = bn_layer(rconv3, train, scope='rBN3')
        rh_conv3 = tf.nn.relu(rBN_out3)
        rh_pool3 = rh_conv3 + rh_pool2

        rconv4 = tf.layers.conv1d(rh_pool3, nchannel.value, 5, strides=1, padding='same', dilation_rate=2)
        rBN_out4 = bn_layer(rconv4, train, scope='rBN4')
        rh_conv4 = tf.nn.relu(rBN_out4)
        rh_pool4 = rh_conv4 + rh_pool3

        rconv5 = tf.layers.conv1d(rh_pool4, nchannel.value, 5, strides=1, padding='same', dilation_rate=1)
        rBN_out5 = bn_layer(rconv5, train, scope='rBN5')
        rh_conv5 = tf.nn.relu(rBN_out5)
        rh_pool5 = rh_conv5 + rh_pool4

        rconv6 = tf.layers.conv1d(rh_pool5, nchannel.value, 5, strides=1, padding='same', dilation_rate=2)
        rBN_out6 = bn_layer(rconv6, train, scope='rBN6')
        rh_conv6 = tf.nn.relu(rBN_out6)
        rh_pool6 = rh_conv6 + rh_pool5

        rconv7 = tf.layers.conv1d(rh_pool6, nchannel.value, 5, strides=1, padding='same')
        rBN_out7 = bn_layer(rconv7, train, scope='rBN7')
        rh_conv7 = tf.nn.relu(rBN_out7)
        rh_pool7 = rh_conv7

        rh_pool7 = tf.reshape(rh_pool7, [-1, rh_conv7.shape[1], rh_conv7.shape[2], 1])
        print(rh_pool7.shape)
        dw_conv1 = weight_variable([5, 5, 1, 1])
        db_conv1 = bias_variable([1])
        econv1_shape = econv1_voice.get_shape().as_list()
        dconv1 = tf.image.resize_nearest_neighbor(rh_pool7, (econv1_shape[1], econv1_shape[2]))
        print(dconv1.shape)
        dconv1 = conv2d(dconv1, dw_conv1) + db_conv1
        dBN_out1 = bn_layer(dconv1, train, scope='dBN1')
        dh_conv1 = tf.nn.relu(dBN_out1)

        dw_conv2 = weight_variable([5, 5, 1, 1])
        db_conv2 = bias_variable([1])
        dconv2 = conv2d(dh_conv1, dw_conv2) + db_conv2
        dBN_out2 = bn_layer(dconv2, train, scope='dBN2')
        dh_conv2 = tf.nn.sigmoid(dBN_out2)

        y_conv = dh_conv2
        return y_conv, h_sense

xs_sense = tf.placeholder(tf.float32, [None, Xtest_sense.shape[1], Xtest_sense.shape[2], 6])
xs_voice = tf.placeholder(tf.float32, [None, Xtest_voice.shape[1], Xtest_voice.shape[2], 1])
ys = tf.placeholder(tf.float32, [None, Xtest_voice.shape[1], Xtest_voice.shape[2], 1])
# ys2 = tf.nn.sigmoid(ys)
# ys2 = ys2*2-1
keep_prob = tf.placeholder(tf.float32)
istraining = tf.placeholder(tf.bool)
mask, h_sense = dcnn(xs_sense, xs_voice, istraining, keep_prob, name='cnnBN')
logits = xs_voice*mask
loss_sum = tf.reduce_sum(tf.square(logits-ys))
cross_entropy = loss_sum
global_step = tf.Variable(0, trainable=False)
train_step = tf.train.AdamOptimizer().minimize(cross_entropy, global_step=global_step)

saver = tf.train.Saver()
tf.global_variables_initializer().run()
def generatebatch(X,step, batch_size):
    start = (step*batch_size)%len(X)
    if start + batch_size > len(X):
        start = ((step+1) * batch_size) % len(X)
    end = min(start + batch_size,len(X))
    return start, end

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    starttime = time.time()
    lasttime = time.time()
    randomind = range(0, len(X_voice))
    best_step = 0
    best_loss = 1000
    net_train = 1
    if net_train == 1:
        for i in range(60000):
           start, end = generatebatch(randomind, i, batch_size)
           batch_x_voice = X_voice[start:end]
           batch_x_sense = X_sense[start:end]
           batch_y = Y[start:end]
           train_step.run(feed_dict={xs_sense: batch_x_sense, xs_voice: batch_x_voice, ys: batch_y, keep_prob: Keep_p, istraining: True})
           if i % 200 == 0:
               loss, signal_re1,ysa1 = sess.run([cross_entropy, logits,ys],
                             feed_dict={xs_sense: batch_x_sense,
                                        xs_voice: batch_x_voice,
                                        ys: batch_y,
                                        keep_prob: Keep_p,
                                        istraining: True})
               testloss,signal_re, ysa = sess.run([cross_entropy, logits,ys],
                                                 feed_dict={xs_sense: Xtest_sense,
                                                            xs_voice: Xtest_voice,
                                                            ys: Ytest,
                                                            keep_prob: 1,
                                                            istraining:False})
               testloss = testloss*270/Ytest.shape[0]
               if testloss < best_loss:
                   best_loss = testloss
                   best_step = i
                #    model.save('./save3/savenet.h5')
               nowtime = time.time()
               dur1 = nowtime-starttime
               dur2 = nowtime-lasttime
               lasttime = nowtime
               print("step %d, %0fs, %0fs, loss %g, testloss %g" % (i, dur1, dur2, loss, testloss))
           if i % 200 == 0:
               randomind = list(range(X_voice.shape[0]))
               np.random.shuffle(randomind)
           if i % 10000 == 0 and i > 1:
               print("beststep %d, bestloss %g" % (best_step, best_loss))
               print()
    elif net_train == 0:
        saver.restore(sess, './save3/savenet.ckpt')
        # start, end = generatebatch(randomind, 32, batch_size)
        # batch_x_voice = X_voice[start:end]
        # batch_x_sense = X_sense[start:end]
        # batch_y = Y[start:end]
        # time1 = time.time()
        testloss, signal_re, ma = sess.run([cross_entropy, logits, mask],
                                       feed_dict={xs_sense: Xtest_sense,
                                                  xs_voice: Xtest_voice,
                                                  ys: Ytest,
                                                  keep_prob: 1,
                                                  istraining: False})
        # ma = ma[0, :, :, 0]
        # ma = np.transpose(ma)
        SDRlist1 = []
        SDRlist2 = []
        SDRlist3 = []
        from pypesq import pesq
        from pystoi import stoi


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
        from scipy.io import wavfile
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
            s_re = s_re_am * np.cos(x_gt_ph) + 1j * s_re_am * np.sin(x_gt_ph)
            t3_gt, xhat3_gt = signal.istft(np.transpose(s_re), fs, window='hann', nperseg=fs * 0.032,
                                           noverlap=fs * 0.016,
                                           nfft=512)
            en_re = np.sum(xhat3_gt ** 2)
            datapath3 = './data/se_generate/3/0-0-' + str(nameindex) + '.wav'
            wavfile.write(datapath3, fs, xhat3_gt.astype(np.float32))

            s_re_am = signal_re[sindex, :, :, 0]
            s_re1 = s_re_am * np.cos(y_gt_ph) + 1j * s_re_am * np.sin(y_gt_ph)
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
    else:
        saver.restore(sess, './save4/savenet.ckpt')
        # num = 480
        # randomind = range(0, num)
        # X_voice = Xtest_voice[:num]
        # X_sense = Xtest_sense[:num]
        # Y = Ytest[:num]
        # Xtest_voice = Xtest_voice[num:]
        # Xtest_sense = Xtest_sense[num:]
        # Ytest = Ytest[num:]
        # batch_size = 32
        for i in range(60000):
           start, end = generatebatch(randomind, i, batch_size)
           batch_x_voice = X_voice[start:end]
           batch_x_sense = X_sense[start:end]
           batch_y = Y[start:end]
           if i % 100 == 0:
               loss, signal_re1, ysa1 = sess.run([cross_entropy, logits, ys],
                             feed_dict={xs_sense: batch_x_sense,
                                        xs_voice: batch_x_voice,
                                        ys: batch_y,
                                        keep_prob: Keep_p,
                                        istraining: True})
               testloss,signal_re, ysa = sess.run([cross_entropy, logits, ys],
                                                 feed_dict={xs_sense: Xtest_sense,
                                                            xs_voice: Xtest_voice,
                                                            ys: Ytest,
                                                            keep_prob: 1,
                                                            istraining:False})
               testloss = testloss*270/Ytest.shape[0]
               if testloss < best_loss and i >= 600:
                   best_loss = testloss
                   best_step = i
                   save_path = saver.save(sess, './save5/savenet.ckpt')
               nowtime = time.time()
               dur1 = nowtime-starttime
               dur2 = nowtime-lasttime
               lasttime = nowtime
               print("step %d, %0fs, %0fs, loss %g, testloss %g" % (i, dur1, dur2, loss, testloss))
           if i % 10 == 0:
               randomind = list(range(X_voice.shape[0]))
               np.random.shuffle(randomind)
           if i % 10000 == 0 and i > 1:
               print("beststep %d, bestloss %g" % (best_step, best_loss))
               print()
           train_step.run(feed_dict={xs_sense: batch_x_sense, xs_voice: batch_x_voice, ys: batch_y, keep_prob: Keep_p,
                                     istraining: True})