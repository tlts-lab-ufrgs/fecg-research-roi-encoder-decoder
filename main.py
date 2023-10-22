#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fECG extraction with ML / DL

Created on Sunday Oct 22 08:30:00 2023

@author: juliacremus
"""
#%%
import mne
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from sklearn import preprocessing

import tensorflow as tf

from models.unet_model import unet

#%% Set model parameters

INPUT_SHAPE = [256, 262144, 4] # length, channels (stacked version), depth

#%% Data Import

PATH = '/home/julia/Documentos/ufrgs/Mestrado/fECG - research/scripts/fecg-denoising/data'

file_info = mne.io.read_raw_edf(f'{PATH}/r04.edf')
annotations = mne.read_annotations(f'{PATH}/r04.edf')

channel_names = file_info.ch_names

raw_data = file_info.get_data()

#%% Preprocess data 

data = np.zeros(shape=(256, 1024, 4))


for i in range(1,4):
    data[:, :, i-1] = np.split(raw_data[i][:262144], 256)[0]

abdominal_ecg = np.split(raw_data[0][:262144], 256)


#%% Unet 

unet_model = unet(INPUT_SHAPE, 1)

tf.keras.utils.plot_model(unet_model, show_shapes=True)
# %%

unet_model.fit(
    data, 
    abdominal_ecg, 
    batch_size=16, 
    epochs = 50
)