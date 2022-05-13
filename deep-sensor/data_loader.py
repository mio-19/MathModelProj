# -*- coding:utf-8 -*-
# python2

'''
functions:
- load data and label from txt files
- split train and test sets
- resample
- fast fourier transfrom
- short time fourier transform
'''

import time
import numpy as np
import torch
import torch.utils.data as data_utils
import scipy.signal as sig

def load_data(*data_file):
    '''
    data_file is a tuple with 1 or 2 elements;
    first is vibration matrix,
    second can be rotating speed matrix.

    input two files only for rsn,
    load_data twice if you have two vibration signal files.

    rsn is a preprocessing for signal, 
    it requires rotating speed matrix.

    see paper "Convolutional Neural Networks for Fault Diagnosis
    Using Rotating Speed Normalized Vibration".
    '''
    since = time.time()
    data_arr = np.loadtxt(data_file[0])
    if len(data_file)>1:
        print('rsn used (see the paper).')
        rpm_arr = np.loadtxt(data_file[1])
        mean_rpm = np.mean(rpm_arr)
        data_arr = np.power(mean_rpm,2)*data_arr / (rpm_arr*rpm_arr)
    time_elapsed = time.time() - since
    print('Data in file {} loaded in {:.0f}m {:.0f}s'.format(data_file, time_elapsed//60, time_elapsed%60))
    return np.expand_dims(data_arr, axis=1)

def load_label(label_file):
    '''
    load labels corrsponding to data
    '''
    return np.loadtxt(label_file, ndmin=1)

def split_set(data, label, p=0.8):
    '''
    split data and label array into train and test partitions
    '''
    n_total = np.shape(data)[0]
    n_train = int(n_total*p)
    n_test = n_total - n_train
    idx = np.random.permutation(n_total)
    train_mask = idx[:n_train]
    test_mask = idx[n_total-n_test:]
    trainset = arr_to_dataset(data[train_mask], label[train_mask])
    testset = arr_to_dataset(data[test_mask], label[test_mask])
    return trainset, testset

def arr_to_dataset(data_arr, label_vec):
    '''
    convert numpy array into tensor dataset
    dataset = (X,y)
    '''
    data_ten = torch.from_numpy(data_arr).float()
    label_ten = torch.from_numpy(label_vec).long()
    dataset = data_utils.TensorDataset(data_ten, label_ten)
    return dataset

def resample_arr(data_arr, num=2048, method='Fourier'):
    '''
    Resample input numpy array into legnthed at num
    methods option: 'Fourier' and 'Poly'
    '''
    if(method=='Fourier'):
        new_arr = sig.resample(data_arr, num, axis=2)
    elif(method=='Poly'):
        new_arr = sig.resample_poly(data_arr, num, data_arr.shape[2], axis=2) 
    return new_arr

def fft_arr(arr):
    '''
    Fourier transform for signals in a Numpy array
    '''
    (n, _, l) = arr.shape
    amp = np.zeros((n,1,l/2))
    ang = np.zeros((n,1,l/2))
    for idx in range(n):
        ft = np.fft.fft(arr[idx,:,:])[:,:l/2]
        amp[idx] = np.absolute(ft)
        ang[idx] = np.angle(ft)
    return amp, ang

def stft_arr(arr, output_size=(32,32)):
    '''
    Short Time Fourier Transform for signals in a Numpy array
    '''
    (n, _, l) = arr.shape
    spectrogram = np.zeros((n, 1, output_size[0], output_size[1]))
    for idx in range(n):
        f, t, S = sig.spectrogram(arr[idx,0,:], fs=10240, nperseg=output_size[0]*2, noverlap=0)
        spectrogram[idx, 0] = np.absolute(S[:(output_size[0]), :(output_size[1])])
    return spectrogram

'''
arr = np.random.rand(10, 1, 2048)
arr = resample_arr(arr, 240, method='Poly')
print(arr.shape)
'''

