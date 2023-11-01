import numpy as np
import time
import tensorflow as tf
from tensorflow.python.keras import Input
import cnn_lip_input2
from tensorflow.python.layers.core import Dense
import operator
from functools import reduce
import os
from tensorflow.python.keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sys
# from  tensorflow.keras.models import load_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

my_data = cnn_lip_input2.Data_Control(

    './data/sourceThird/NPZ/'
)
n_class = 20

X = my_data.traindata
print("X的尺度{}，{}，{}".format(X.shape[0],X.shape[1],X.shape[2]))
X = X.reshape(-1, my_data.traindata.shape[1], my_data.traindata.shape[2])
print("X的尺度{}，{}，{}".format(X.shape[0],X.shape[1],X.shape[2]))
Y = my_data.trainlabel

Keep_p = 0.6
batch_size = 32
train_writer = tf.summary.create_file_writer('logs/train/')
# 测试数据
Xtest = my_data.testdata

Xtest = Xtest.reshape(-1, my_data.testdata.shape[1], my_data.testdata.shape[2])
Ytest = my_data.testlabel
Ytestuser = my_data.testuser


# Create a callback 
tf_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs") 
 
# 训练
def trainFun():
    net_train = 1
    if net_train == 1:
        class Action_model(tf.keras.Model):
            def __init__(self, batch_sz, n_class):
                super(Action_model, self).__init__()
                self.batch_sz = batch_sz
                self.n_class = n_class
                self.Conv_1 = tf.keras.layers.Conv1D(filters=32, kernel_size=9, strides=1,
                                                     padding='same', trainable=True, use_bias=True,
                                                     bias_initializer=tf.keras.initializers.Constant(value=0.1))
                self.BN_1 = tf.keras.layers.BatchNormalization(trainable=True, scale=True)
                self.Relu_1 = tf.keras.layers.ReLU()
                self.Pool_1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=3, padding='same')

                self.Conv_2 = tf.keras.layers.Conv1D(filters=64, kernel_size=9, strides=1,
                                                     padding='same', trainable=True, use_bias=True,
                                                     bias_initializer=tf.keras.initializers.Constant(value=0.1))
                self.BN_2 = tf.keras.layers.BatchNormalization(trainable=True, scale=True)
                self.Relu_2 = tf.keras.layers.ReLU()
                self.Pool_2 = tf.keras.layers.MaxPool1D(pool_size=3, strides=3, padding='same')

                self.Conv_3 = tf.keras.layers.Conv1D(filters=128, kernel_size=9, strides=1,
                                                     padding='same', trainable=True, use_bias=True,
                                                     bias_initializer=tf.keras.initializers.Constant(value=0.1))
                self.BN_3 = tf.keras.layers.BatchNormalization(trainable=True, scale=True)
                self.Relu_3 = tf.keras.layers.ReLU()
                self.Pool_3 = tf.keras.layers.MaxPool1D(pool_size=3, strides=3, padding='same')

                self.Conv_4 = tf.keras.layers.Conv1D(filters=128, kernel_size=9, strides=1,
                                                     padding='same', trainable=True, use_bias=True,
                                                     bias_initializer=tf.keras.initializers.Constant(value=0.1))
                self.BN_4 = tf.keras.layers.BatchNormalization(trainable=True, scale=True)
                self.Relu_4 = tf.keras.layers.ReLU()
                self.Pool_4 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')

                self.Conv_5 = tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=1,
                                                     padding='same', trainable=True, use_bias=True,
                                                     bias_initializer=tf.keras.initializers.Constant(value=0.1))
                self.BN_5 = tf.keras.layers.BatchNormalization(trainable=True, scale=True)
                self.Relu_5 = tf.keras.layers.ReLU()
                self.Pool_5 = tf.keras.layers.MaxPool1D(pool_size=1, strides=1, padding='same')

                self.Conv_6 = tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=1,
                                                     padding='same', trainable=True, use_bias=True,
                                                     bias_initializer=tf.keras.initializers.Constant(value=0.1))
                self.BN_6 = tf.keras.layers.BatchNormalization(trainable=True, scale=True)
                self.Relu_6 = tf.keras.layers.ReLU()
                self.Pool_6 = tf.keras.layers.MaxPool1D(pool_size=1, strides=1, padding='same')

                self.Conv_7 = tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=1,
                                                     padding='same', trainable=True, use_bias=True,
                                                     bias_initializer=tf.keras.initializers.Constant(value=0.1))
                self.BN_7 = tf.keras.layers.BatchNormalization(trainable=True, scale=True)
                self.Relu_7 = tf.keras.layers.ReLU()
                self.Pool_7 = tf.keras.layers.MaxPool1D(pool_size=1, strides=1, padding='same')

                self.Conv_8 = tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=1,
                                                     padding='same', trainable=True, use_bias=True,
                                                     bias_initializer=tf.keras.initializers.Constant(value=0.1))
                self.BN_8 = tf.keras.layers.BatchNormalization(trainable=True, scale=True)
                self.Relu_8 = tf.keras.layers.ReLU()
                self.Pool_8 = tf.keras.layers.MaxPool1D(pool_size=1, strides=1, padding='same')

                self.Conv_9 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1,
                                                     padding='same', trainable=True, use_bias=True,
                                                     bias_initializer=tf.keras.initializers.Constant(value=0.1))
                self.BN_9 = tf.keras.layers.BatchNormalization(trainable=True, scale=True)
                self.Relu_9 = tf.keras.layers.ReLU()
                self.Pool_9 = tf.keras.layers.MaxPool1D(pool_size=1, strides=1, padding='same')

                self.FC1 = tf.keras.layers.Dense(128)
                self.Relu = tf.keras.layers.LeakyReLU()
                self.dp = tf.keras.layers.Dropout(0.4)
                self.FC2 = tf.keras.layers.Dense(self.n_class)

            def call(self, inputs, training=True):
                inputs = tf.reshape(inputs, [-1, my_data.traindata.shape[1], my_data.traindata.shape[2]])
                conv1 = self.Conv_1(inputs)
                BN_out1 = self.BN_1(conv1)
                h_conv1 = self.Relu_1(BN_out1)
                h_pool1 = self.Pool_1(h_conv1)

                conv2 = self.Conv_2(h_pool1)
                BN_out2 = self.BN_2(conv2)
                h_conv2 = self.Relu_2(BN_out2)
                h_pool2 = self.Pool_2(h_conv2)

                conv3 = self.Conv_3(h_pool2)
                BN_out3 = self.BN_3(conv3)
                h_conv3 = self.Relu_3(BN_out3)
                h_pool3 = self.Pool_3(h_conv3)

                conv4 = self.Conv_4(h_pool3)
                BN_out4 = self.BN_4(conv4)
                h_conv4 = self.Relu_4(BN_out4)
                h_pool4 = self.Pool_4(h_conv4)

                conv5 = self.Conv_5(h_pool4)
                BN_out5 = self.BN_5(conv5)
                h_conv5 = self.Relu_5(BN_out5)
                h_pool5 = self.Pool_5(h_conv5)

                conv6 = self.Conv_6(h_pool5)
                BN_out6 = self.BN_6(conv6)
                h_conv6 = self.Relu_6(BN_out6)
                h_pool6 = self.Pool_6(h_conv6)

                conv7 = self.Conv_7(h_pool6)
                BN_out7 = self.BN_7(conv7)
                h_conv7 = self.Relu_7(BN_out7)
                h_pool7 = self.Pool_7(h_conv7)

                conv8 = self.Conv_8(h_pool7)
                BN_out8 = self.BN_8(conv8)
                h_conv8 = self.Relu_8(BN_out8)
                h_pool8 = self.Pool_8(h_conv8)

                conv9 = self.Conv_9(h_pool8)
                BN_out9 = self.BN_9(conv9)
                h_conv9 = self.Relu_9(BN_out9)
                h_pool9 = self.Pool_9(h_conv9)

                h_conv = h_pool9
                h_conv_shape = h_conv.get_shape().as_list()
                h_flat = tf.reshape(h_conv, [-1, h_conv_shape[1] * h_conv_shape[2]])
                fc1 = self.dp(self.Relu(self.FC1(h_flat)))
                fc2 = self.FC2(fc1)

                return fc2

            # def initialize_hidden_state(self):
            #     return tf.zeros((self.batch_sz))


        BATCH_SIZE = 32
        action_model = Action_model(BATCH_SIZE, n_class)
        optimizer = tf.keras.optimizers.Adam()


        def loss_function(real, pred):
            cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
            loss = cross_entropy(y_true=real, y_pred=pred)
            return tf.reduce_mean(loss)


        def accuracy(real, pred):
            acc = tf.math.equal(pred, real)
            acc = tf.cast(acc, dtype=tf.float32)
            acc = tf.reduce_mean(acc)
            return acc


        def train_step(inp, targ):
            with tf.GradientTape() as tape:
                results = action_model(inp)
                real = targ
                loss = loss_function(real, results)
                prediction = tf.argmax(results, -1, output_type=tf.int32, name='prediction')
                acc = accuracy(real, prediction)
            variables = action_model.trainable_variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            return loss, acc


        def test_step(inp, targ):
            results = action_model(inp, training=False)
            real = targ
            loss = loss_function(real, results)
            prediction = tf.argmax(results, -1, output_type=tf.int32, name='prediction')
            acc = accuracy(real, prediction)
            return loss, acc, prediction


        def test_step2(inp):
            results = action_model(inp, training=False)
            prediction = tf.argmax(results, -1, output_type=tf.int32, name='prediction')
            return prediction


        def generatebatch(X, step, batch_size):
            start = (step * batch_size) % len(X)
            if start + batch_size > len(X):
                start = ((step + 1) * batch_size) % len(X)

            end = min(start + batch_size, len(X))
            batch_xs = X[start:end]
            return batch_xs  # 生成每一个batch


        step = 0
        best_step = 0
        best_acc = 0

        while step < 60000:
            batch_xs = generatebatch(X, step, BATCH_SIZE)
            batch_ys = generatebatch(Y, step, BATCH_SIZE)
            loss, acc = train_step(batch_xs, batch_ys)
            if step % 200 == 0:
                testloss, testacc, pred = test_step(Xtest, Ytest)
                pred = pred.numpy()
                if testacc >= best_acc and step >= 2000:
                    best_acc = testacc.numpy()
                    best_step = step
                    result = np.vstack((pred, Ytest))
                    result = np.transpose(result)
                    action_model.save('save/action_array/third.tf', save_format="tf")
                print("step %d, loss %g, acc %g, testloss %g, testacc %g" % (
                    step, loss, acc, testloss, testacc))
            with train_writer.as_default():
                tf.summary.scalar('loss', loss, step=step)
                tf.summary.scalar('accuracy', acc, step=step)
            if step % 200 == 0:
                print(best_acc, best_step)
            step += 1
    else:
        save_path = 'save/action_array/6.tf'
        action_model = tf.keras.models.load_model(save_path)
        results = action_model.predict(Xtest)  # 导入鼾声检测模型
        prediction = np.argmax(results, -1)  # 返回结果
        print(prediction)
        print(Ytest)
        result = np.vstack((pred, Ytest))
        result = np.transpose(result)
        for k in range(0, len(pred)):
            if Ytest[k] != pred[k]:
                print(Ytest[k], pred[k], my_data.test_index[k])
        print(pred)

trainFun()

save_path = 'save/action_array/third.tf'
action_model = tf.keras.models.load_model(save_path)
results = action_model.predict(Xtest)  # 导入鼾声检测模型
prediction = np.argmax(results, -1)  # 返回结果
print(prediction)
cnt=0
for i in range(Ytest.shape[0]):
    if(Ytest[i]==prediction[i]):
        cnt+=1
acc=cnt*1.0/Ytest.shape[0]
print(acc)

print(Ytest.shape)
rea_labels=[i for i in range(0,20)]
C=confusion_matrix(Ytest,prediction,labels=rea_labels)
plt.matshow(C,cmap=plt.cm.OrRd)
for i in range(len(C)):
    for j in range(len(C)):
        plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
plt.colorbar()
plt.title("混淆矩阵图(cnn)")
plt.ylabel('实际类别')
plt.xlabel('预测类别')
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
# plt.xticks(range(0,3), labels=['抬手','放手','其他']) # 将x轴或y轴坐标，刻度 替换为文字/字符
# plt.yticks(range(0,3), labels=['抬手','放手','其他'])
# plt.legend()
plt.show()

# Third:median_length:845,acc=78.5%