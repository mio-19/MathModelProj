
import time
import numpy as np
import torch
import torch.utils.data as data_utils
import scipy.signal as sig
import pandas as pd 

singlesize = 4
groupsize = 512
datasize = singlesize * groupsize
skipsize = 128

# https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-analysis-data
def read_gearbox_sensor(file, name):
    sheet = pd.read_excel(file, name)
    return sheet.iloc[:, 1:]

# https://stackoverflow.com/questions/48704526/split-pandas-dataframe-into-chunks-of-n
def sliceby(seq, size, skip):
    return (seq[pos:pos + size] for pos in range(0, len(seq) - size, skip))

# https://stackoverflow.com/questions/25440008/python-pandas-flatten-a-dataframe-to-a-list
def sensor_to_groups(input):
    return np.stack(c.to_numpy().flatten() for c in sliceby(input, groupsize, skipsize))

def read_gearbox(file, name):
    return sensor_to_groups(read_gearbox_sensor(file, name))




def tag_output(input, tag):
    return np.full(input.shape[0], tag)

# Loading the Data
xls1 = r'./1.xls'
gearbox00 = read_gearbox(xls1,'gearbox00') 
print('Take a look at sample from gearbox00:') 
print(gearbox00) 
gearbox10 = read_gearbox(xls1,'gearbox10') 
gearbox20 = read_gearbox(xls1,'gearbox20') 
gearbox30 = read_gearbox(xls1,'gearbox30') 
gearbox40 = read_gearbox(xls1,'gearbox40') 
input_np = np.concatenate((gearbox00, gearbox10, gearbox20, gearbox30, gearbox40))
labels = (0, 1, 2, 3, 4)
output_np = np.concatenate((tag_output(gearbox00, 0),tag_output(gearbox10, 1),tag_output(gearbox20, 2),tag_output(gearbox30, 3),tag_output(gearbox40, 4)))

def write_np(file, data):
    f = open(file, "w")
    np.savetxt(file, data)
    f.close()

write_np("data.txt", input_np)
write_np("label.txt", output_np)


# For testing

write_np("gearbox00.txt", gearbox00)
write_np("gearbox10.txt", gearbox10)
write_np("gearbox20.txt", gearbox20)
write_np("gearbox30.txt", gearbox30)
write_np("gearbox40.txt", gearbox40)