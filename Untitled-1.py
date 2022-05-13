#!/usr/bin/python
# -*- coding:utf-8 -*-

# https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-analysis-data
import torch 
import pandas as pd 
import torch.nn as nn 
from torch.utils.data import random_split, DataLoader, TensorDataset 
import torch.nn.functional as F 
import numpy as np 
import torch.optim as optim 
from torch.optim import Adam 

import numpy

singlesize = 4
groupsize = 16
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
    return numpy.stack(c.to_numpy().flatten() for c in chunkby(input, groupsize))

def read_gearbox(file, name):
    return sensor_to_groups(read_gearbox_sensor(file, name))

def tag_output(input, tag):
    return numpy.full(tag, input.shape[0])

# Loading the Data
gearbox00 = read_gearbox(r'./1.xls','gearbox00') 
print('Take a look at sample from gearbox00:') 
print(gearbox00) 
gearbox10 = read_gearbox(r'./1.xls','gearbox10') 
gearbox20 = read_gearbox(r'./1.xls','gearbox20') 
gearbox30 = read_gearbox(r'./1.xls','gearbox30') 
gearbox40 = read_gearbox(r'./1.xls','gearbox40') 
input = numpy.concatenate((gearbox00, gearbox10, gearbox20, gearbox30, gearbox40))
output = numpy.concatenate((tag_output(gearbox00, 0),tag_output(gearbox10, 1),tag_output(gearbox20, 2),tag_output(gearbox30, 3),tag_output(gearbox40, 4)))




# ----------------------------inputsize >=28-------------------------------------------------------------------------
class WDCNN(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(WDCNN, self).__init__()


        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=64,stride=16,padding=24),  
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2,stride=2)
            )

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3,padding=1), 
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))  

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3,padding=1),  
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )  # 32, 12,12     (24-2) /2 +1

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3,padding=1),  
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )  # 32, 12,12     (24-2) /2 +1

        self.layer5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),  
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
            # nn.AdaptiveMaxPool1d(4)
        )  # 32, 12,12     (24-2) /2 +1

        self.fc=nn.Sequential(
            nn.Linear(192, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, out_channel)
        )


    def forward(self, x):
        # print(x.shape)
        x = self.layer1(x) #[16 64]
        # print(x.shape)
        x = self.layer2(x)  #[32 124]
        # print(x.shape)
        x = self.layer3(x)#[64 61]
        # print(x.shape)
        x = self.layer4(x)#[64 29]
        # print(x.shape)
        x = self.layer5(x)#[64 13]
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
