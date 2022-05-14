# -*- coding:utf-8 -*-

## initialize
from __future__ import print_function, division 
import torch.optim as optim
import torch.nn as nn
import data_loader
import iter_utils
import torch.utils.data as data_utils
from models import *
import numpy as np

for i in range(1, 13):
    # load data
    filename = r'test{}.txt'.format(i)

    data_arr_01 = data_loader.load_data(filename)
    # data_arr_03 = data_loader.load_data('data/pgb/SF03/vib_data_1.txt')
    # data_arr_01 = data_loader.resample_arr(data_arr_01, num=240) # add for Ince's model
    # data_arr_03 = data_loader.resample_arr(data_arr_03, num=240) # add for Ince's model
    # data_arr_01, _ = data_loader.fft_arr(data_arr_01) # add for fft wdcnn
    # data_arr_03, _ = data_loader.fft_arr(data_arr_03) # add for fft wdcnn
    # data_arr_01 = data_loader.stft_arr(data_arr_01) # add for stft-LeNet
    # data_arr_03 = data_loader.stft_arr(data_arr_03)

    ## make models
    model = wdcnn.Net(1, 5)

    # load model
    model.load_state_dict(torch.load('wdcnn.pth'))
    model.eval()

    inputs = torch.Tensor(data_arr_01)

    # from iter_utils.test
    
    with torch.no_grad():
        inputs = Variable(inputs)
        outputs = model(inputs)
        print(outputs.data)
        _, predicted = torch.max(outputs.data, 1)
        print(predicted)
        counts = np.bincount(predicted.numpy())
        predicted2 = np.argmax(counts)
        print(counts)
        print(filename)
        print(predicted2)
        f = open("{}.output.txt".format(filename), "w")
        # human readable
        np.savetxt(f, outputs.data, "%.4f")
        f.close()
