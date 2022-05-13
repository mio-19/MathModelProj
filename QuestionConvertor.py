
import time
import numpy as np
import torch
import torch.utils.data as data_utils
import scipy.signal as sig
import pandas as pd 

singlesize = 4
groupsize = 512
datasize = singlesize * groupsize

# https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-analysis-data
def read_gearbox_sensor(file, name):
    sheet = pd.read_excel(file, name)
    return sheet.iloc[:, 1:]

# https://stackoverflow.com/questions/48704526/split-pandas-dataframe-into-chunks-of-n
# check lost tail data
def chunkby(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq)-size, size))

# https://stackoverflow.com/questions/25440008/python-pandas-flatten-a-dataframe-to-a-list
def sensor_to_groups(input):
    return np.stack(c.to_numpy().flatten() for c in chunkby(input, groupsize))

def read_gearbox(file, name):
    return sensor_to_groups(read_gearbox_sensor(file, name))





def chunk1000to512(seq):
    return (seq[0:512],seq[488:1000])
def sensor2groups1000_512(input):
    return np.stack(c.to_numpy().flatten() for c in chunk1000to512(input))
def read_gearbox1000_512(file, name):
    return sensor2groups1000_512(read_gearbox_sensor(file, name))

# Loading the Data
xls2 = r'./2.xls'

# 1 to 12
ids = list(range(1, 13))

#sheets = list(read_gearbox(xls2, "test{}".format(i)) for i in ids)
sheets = list(read_gearbox1000_512(xls2, "test{}".format(i)) for i in ids)

for i in ids:
    data_file = open("test{}.txt".format(i), "w")
    print(sheets[i-1].shape)
    np.savetxt(data_file, sheets[i-1])
    data_file.close()

