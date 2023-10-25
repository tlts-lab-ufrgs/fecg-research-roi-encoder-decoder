#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sand box 1D semantic segmantation

Created on Tue Oct 25 14:50:00 2023

@author: juliacremus
"""

#%%

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

#%% get data

PATH = '/home/julia/Documentos/ufrgs/Mestrado/fECG - research/scripts/fecg-denoising/data'

file_info = mne.io.read_raw_edf(f'{PATH}/r04.edf')
annotations = mne.read_annotations(f'{PATH}/r04.edf')

channel_names = file_info.ch_names

raw_data = file_info.get_data()

#%% constants

CHANNEL_NUMBER = 4

#%% Build Model 

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64, 1, activation='relu'), 
    # tf.keras.layers.Conv1D(32, 1, activation='relu'),
    # tf.keras.layers.Conv1D(16, 1, activation='relu'),
    # tf.keras.layers.Conv1D(8, 1, activation='relu'),
    # tf.keras.layers.Conv1D(4, 1, activation='relu'), 
    # tf.keras.layers.UpSampling1D(4), 
    # tf.keras.layers.UpSampling1D(4), 
    # tf.keras.layers.UpSampling1D(2)
])

#%% Run model

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


#%% Fit into data

history = model.fit(
    np.split(raw_data[1][:262144], 512),
    np.split(raw_data[0][:262144], 512), 
    epochs = 10
)

#%% SandBox

# sand_box_model = tf.keras.Sequential([
#     # Shape: (time, features) => (time*features)
#     tf.keras.layers.Conv1D(units=32, activation='relu'),
#     tf.keras.layers.Dense(units=32, activation='relu'),
#     tf.keras.layers.Dense(units=1),
#     # Add back the time dimension.
#     # Shape: (outputs) => (1, outputs)
#     tf.keras.layers.Reshape([1, -1]),
# ])


# def inverted_residual_block(x, expand=64, squeeze=16):
#   m = Layers.Conv1D(expand, (1,1), activation='relu')(x)
#   m = Layers.DepthwiseConv1D((3,3), activation='relu')(m)
#   m = Layers.Conv1D(squeeze, (1,1), activation='relu')(m)
#   return Layers.Add()([m, x])