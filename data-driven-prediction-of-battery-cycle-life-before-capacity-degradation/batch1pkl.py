import h5py
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pickle

matFilename = './Data/2017-05-12_batchdata_updated_struct_errorcorrect.mat'
f = h5py.File(matFilename)

list(f.keys())

batch = f['batch']

print(list(batch.keys()))