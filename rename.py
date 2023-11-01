import os
import re
import sys
# userID = 0
# filepath = 'G:/experiments/2023/prp/zq/'
# files = os.listdir(filepath)
# for file in files:
#     pattern = re.compile(r'\d+')
#     res = re.findall(pattern, file)
#     if len(res) == 1 and file[-5] == 'l' and int(res[0]) >= 200:
#         filename = filepath + file
#         newname = str(int(res[0])) + "_" + str(userID)
#         os.rename(filename, "G:/experiments/2023/prp/se/data/" + newname + ".wav")


userID = 0
filepath = 'G:/experiments/2023/far_lip/farlip/fmcw15-4/fmcw15-4/'
files = os.listdir(filepath)
for file in files:
    pattern = re.compile(r'\d+')
    res = re.findall(pattern, file)
    if len(res) == 1  and int(res[0]) >= 0:
        filename = filepath + file
        newname = str(int(res[0])//10) + "_" + str(int(res[0])%10+40) + "_" + str(userID)
        os.rename(filename, "G:/experiments/2023/far_lip/farlip/rawdata/" + newname + ".pcm")