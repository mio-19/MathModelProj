import time
import numpy as np
import torch
import torch.utils.data as data_utils
import scipy.signal as sig
import pandas as pd 

singlesize = 4
groupsize = 512
datasize = singlesize * groupsize
skipsize = 2

def read_gearbox_sensor(file, name):
    """
    Reads a gearbox sensor from an excel file.
    :param file: The excel file.
    :param name: The name of the sheet in the excel file.
    :return: The data as a pandas dataframe.
    """
    sheet = pd.read_excel(file, name)
    return sheet.iloc[:, 1:]

def sliceby(seq, size, skip):
    """
    Splits a sequence into chunks of size size, with a skip of skip.
    :param seq: The sequence to split.
    :param size: The size of the chunks.
    :param skip: The skip of the chunks.
    :return: A generator of chunks.
    """
    return (seq[pos:pos + size] for pos in range(0, len(seq) - size, skip))

def sensor_to_groups(input):
    """
    Converts a sensor to a list of groups.
    :param input: The sensor to convert.
    :return: A list of groups.
    """
    return np.stack(c.to_numpy().flatten() for c in sliceby(input, groupsize, skipsize))

def read_gearbox(file, name):
    """
    Reads a gearbox from an excel file.
    :param file: The excel file.
    :param name: The name of the sheet in the excel file.
    :return: The data as a list of groups.
    """
    return sensor_to_groups(read_gearbox_sensor(file, name))






# Loading the Data
xls2 = r'./2.xls'

# 1 to 12
ids = list(range(1, 13))

# Read the data from the excel files and convert it to a list of groups.
sheets = list(read_gearbox(xls2, "test{}".format(i)) for i in ids)

for i in ids:
    data_file = open("test{}.txt".format(i), "w")
    print(sheets[i-1].shape)
    np.savetxt(data_file, sheets[i-1])
    data_file.close()

