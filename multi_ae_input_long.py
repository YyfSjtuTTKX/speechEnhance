import numpy as np
import os
import re
import time


class Data_Control:
    def __init__(self, sensepath, filepathdata,filepathlabel):
        self.files_sense = os.listdir(filepathdata)
        self.files_voice = os.listdir(filepathdata)
        sensedata, senselen, senseflen, alldata, self.Xlen, alllabel, self.flen, self.allindex, self.alluser = self.loadfile(sensepath, filepathdata,filepathlabel)
        alldata, _, self.length = self.cnn_padding_audio(alldata, self.Xlen,self.flen)
        alllabel, _, _ = self.cnn_padding_audio(alllabel, self.Xlen, self.flen)
        sensedata, _, _ = self.cnn_padding_sense(sensedata, senselen, senseflen)
        trainindex, testindex = self.indexsplit(alldata.shape[0], False)
        print(len(testindex))
        # self.npfiles = np.array(self.files)
        self.trainsense = sensedata[trainindex]
        self.traindata = alldata[trainindex]
        self.trainlabel = alllabel[trainindex]
        self.trainuser = self.alluser[trainindex]
        self.testsense = sensedata[testindex]
        self.testdata = alldata[testindex]
        self.testlabel = alllabel[testindex]
        self.testuser = self.alluser[testindex]
        self.testind = self.allindex[testindex]
        self.batch_id = 0


    def indexsplit(self,indexlength,israndom):
        if israndom is True:
            randomind = list(range(indexlength))
            np.random.shuffle(randomind)
            trainindex = randomind[:int(len(randomind) * 0.8)]
            testindex = list(filter(lambda j: j not in trainindex, list(randomind)))
        else:
            trainindex = []
            testindex = []
            for i in range(indexlength):
                if self.allindex[i] >= 0:

                    if self.allindex[i] >= 160 and self.allindex[i] < 200 and self.allindex[i] % 1 == 0:
                        testindex.append(i)
                    elif self.alluser[i] != -1:
                        trainindex.append(i)
            print(len(trainindex))
            np.random.shuffle(trainindex)
            # np.random.shuffle(testindex)
        return trainindex, testindex

    def loadfile(self, sensepath, filepathdata, filepathlabel):
        sense_data = []
        sense_len = []
        raw_data = []
        raw_data_len = []
        raw_label = []
        raw_index = []
        raw_user = []
        starttime = time.time()
        lasttime = time.time()
        kk = 0
        for file_data in self.files_voice:
            pattern = re.compile(r'\d+')
            res = re.findall(pattern, file_data)
            if (len(res) == 2 and int(res[1]) <= 2):
                filename_data = filepathdata+file_data
                data = np.load(filename_data)
                sample = data['datapre']
                sample = sample[:, :257]+1j*sample[:, 257:514]
                sample = np.hstack([np.abs(sample),np.angle(sample)])
                # sample = np.abs(sample)
                sample = sample.astype(np.float32)
                featurelen = sample.shape[1]
                ind2 = int(res[1])
                file_label = 'datapre%d-%d.npz' % (int(res[0]), ind2)
                filename_label = filepathlabel + file_label
                data_label = np.load( filename_label)
                samplela = data_label['datapre']
                samplela = samplela[:, :257]+1j*samplela[:, 257:514]
                samplela = np.hstack([np.abs(samplela), np.angle(samplela)])
                # samplela = np.abs(samplela)
                samplela = samplela.astype(np.float32)


                file_sense = '%d_%d.npz' % (int(res[0]), ind2)
                filename_sense = sensepath+file_sense
                datasense = np.load(filename_sense)
                samplesense = datasense['datapre']
                samplesense = np.transpose(samplesense, [2, 1, 0])
                if np.max(np.abs(samplesense)) < 500:
                    samplesense = samplesense[:, 14:30, :6]
                    senseflen = samplesense.shape[1]


                    sense_data.append(samplesense)
                    sense_len.append(samplesense.shape[0])
                    raw_data.append(sample)
                    raw_data_len.append(sample.shape[0])

                    raw_label.append(samplela)
                    raw_index.append(int(res[0]))
                    raw_user.append(int(res[1]))
                kk = kk+1
                if kk % 1000 == 0:
                    nowtime = time.time()
                    print("%d, %0fs" % (kk, nowtime-starttime))
        raw_data = np.array(raw_data)
        raw_label = np.array(raw_label)
        raw_index = np.array(raw_index)
        raw_user = np.array(raw_user)
        return np.array(sense_data), np.array(sense_len), senseflen, raw_data, raw_data_len, raw_label,featurelen, raw_index, raw_user

    def cnn_padding_audio(self, data, slen, flen):
        raw_data = data
        lengths = slen
        median_length = int(np.percentile(lengths, 50))
        print(int(np.percentile(lengths, 5)))
        print(np.min(lengths))
        print(median_length)
        # median_length = 90
        num_samples = len(lengths)
        padding_data = np.zeros([num_samples, median_length, flen])
        for idx, seq in enumerate(raw_data):
            if len(seq) <= median_length:
                padding_data[idx, :len(seq), :] = seq
            else:
                padding_data[idx, :, :] = seq[:median_length]
        return padding_data, np.array(slen), median_length

    def cnn_padding_sense(self, data, slen, flen):
        raw_data = data
        lengths = slen
        median_length = int(np.percentile(lengths, 50))
        print(int(np.percentile(lengths, 5)))
        print(np.min(lengths))
        print(median_length)
        # median_length = 70
        num_samples = len(lengths)
        padding_data = np.zeros([num_samples, median_length, flen, 6])
        for idx, seq in enumerate(raw_data):
            if len(seq) <= median_length:
                padding_data[idx, :len(seq), :, :] = seq
            else:
                padding_data[idx, :, :, :] = seq[:median_length]
        return padding_data, np.array(slen), median_length
