#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoencoder - residual networks

base tutorial: https://medium.com/swlh/how-to-create-a-residual-network-in-tensorflow-and-keras-cd97f6c62557

Created on Sun Nov 26 18:29:15 2023

@author: julia
"""

import mne
import glob
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers as Layers

import matplotlib.pyplot as plt

#%% Parameters

DATA_PATH = "/home/julia/Documents/fECG_research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/"
EPOCHS = 500
FILENAMES = glob.glob(DATA_PATH + "*.edf")

LEN_DATA = 600
CHANNELS = 4

BATCH_SIZE = 32

QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = 100

data_store = np.empty(shape=(EPOCHS, LEN_DATA, CHANNELS))
fecg_store = np.empty(shape=(EPOCHS, LEN_DATA, 2))


#%% Resample data

for file in FILENAMES[0:2]:

    file_info = mne.io.read_raw_edf(file)
    filedata = file_info.get_data()

    annotations = mne.read_annotations(file)
    time_annotations = annotations.onset

    binary_mask = np.zeros(shape=file_info.times.shape)

    for step in time_annotations:

        center_index = np.where(file_info.times == step)[0][0]

        binary_mask[
            center_index - QRS_DURATION_STEP :
            center_index + QRS_DURATION_STEP 
        ] = 1

    chunked_data = np.array_split(filedata[1::] * 1e5, EPOCHS, axis = 1)
    chunked_data = [partial_data.transpose() for partial_data in chunked_data]
    
    chunked_fecg_data = np.array_split(np.array([filedata[0] * 1e5, binary_mask]), EPOCHS, axis = 1)
    chunked_fecg_data = [partial_data.transpose() for partial_data in chunked_fecg_data]    
    # chuncked_fecg_data = [partial_data.reshape(LEN_DATA, 1) for partial_data in chuncked_fecg_data]

    data_store = np.vstack((data_store, chunked_data))
    fecg_store = np.vstack((fecg_store, chunked_fecg_data))
    
#%% Model

# The inputs are 128-length vectors with 10 timesteps, and the
# batch size is 4.LEN_DATA
input_shape = (BATCH_SIZE, LEN_DATA, CHANNELS)
x = tf.keras.Input(batch_shape=input_shape)

# First block

y = Layers.Conv1D(64, 3, activation='relu',input_shape=input_shape[1:])(x)
y_block_1 = Layers.Conv1D(64, 3, activation='relu')(y)
y = Layers.MaxPooling1D(pool_size=2, padding='same')(y_block_1)

cropping_block_1 = Layers.Cropping1D(83)(y_block_1)  # it removes the initial and ends, 

print('block 1 encoder', y_block_1.shape)


# 2 Decoder

y = Layers.BatchNormalization(axis=2)(y)
y = Layers.Conv1D(128, 3, activation='relu')(y)
y_block_2 = Layers.Conv1D(128, 3, activation='relu')(y)
y = Layers.MaxPooling1D(pool_size=2, padding='same')(y_block_2)

cropping_block_2 = Layers.Cropping1D(37)(y_block_2)  # it removes the initial and ends, 


print('block 2 encoder', y_block_2.shape)


# 3 decoder

y = Layers.BatchNormalization(axis=2)(y)
y = Layers.Conv1D(256, 3, activation='relu')(y)
y_block_3 = Layers.Conv1D(256, 3, activation='relu')(y)
y = Layers.MaxPooling1D(pool_size=2, padding='same')(y_block_3)

print('block 3 encoder', y_block_3.shape)

cropping_block_3 = tf.keras.layers.Cropping1D(cropping=14)(y_block_3)

# 4 decoder
y = Layers.BatchNormalization(axis=2)(y)
y = Layers.Conv1D(512, 3, activation='relu')(y)
y_block_4 = Layers.Conv1D(512, 3, activation='relu')(y)
y = Layers.MaxPooling1D(pool_size=2, padding='same')(y_block_4)

print('block 4 encoder', y_block_4.shape)

cropping_block_4 = Layers.Cropping1D(3)(y_block_4)  # it removes the initial and ends, 

# 5 decoder
y = Layers.BatchNormalization(axis=2)(y)
y_block_5 = Layers.Conv1D(1024, 3, activation='relu')(y)
print('block 5 encoder', y_block_5.shape)

# encoder

# Bloco 1 encoder

print('---------------------')
print('first encoder block')

y = Layers.UpSampling1D(size=2)(y_block_5)
print('first upsampling', y.shape)
y = Layers.Conv1D(512, 3, activation='relu')(y)  # changed to conv 3x3 because of the output shape to do the cropping
# y = Layers.BatchNormalization(axis=2)(y)
print('first conv layer encoder', y.shape)
y = Layers.Concatenate(axis = 2)([cropping_block_4, y])
print('concat layers', y.shape)
y = Layers.Conv1D(512, 3, activation='relu')(y)
y_up_block_1 = Layers.Conv1D(512, 3, activation='relu')(y)

# Bloco 2 encoder 


print('---------------------')
print('second encoder block')


y = Layers.UpSampling1D(size=2)(y_up_block_1)
y = Layers.Conv1D(256, 2, activation='relu')(y)
print('first conv layer encoder', y.shape)
# y = Layers.BatchNormalization(axis=2)(y)
y = Layers.Concatenate(axis = 2)([cropping_block_3, y])
print('concat layers', y.shape)
y = Layers.Conv1D(256, 3, activation='relu')(y)
y_up_block_2 = Layers.Conv1D(256, 3, activation='relu')(y)

print('---------------------')

print('third encoder block')

y = Layers.UpSampling1D(size=2)(y_up_block_2)
y = Layers.Conv1D(128, 3, activation='relu')(y)
print('first conv layer encoder', y.shape)
# y = Layers.BatchNormalization(axis=2)(y)
y = Layers.Concatenate(axis = 2)([cropping_block_2, y])
print('concat layers', y.shape)
y = Layers.Conv1D(128, 3, activation='relu')(y)
y_up_block_3 = Layers.Conv1D(128, 3, activation='relu')(y)

print('---------------------')

print('fourth encoder block')

y = Layers.UpSampling1D(size=2)(y_up_block_3)
y = Layers.Conv1D(64, 3, activation='relu')(y)
print('first conv layer encoder', y.shape)
# y = Layers.BatchNormalization(axis=2)(y)
y = Layers.Concatenate(axis = 2)([cropping_block_1, y])
print('concat layers', y.shape)
y = Layers.Conv1D(64, 3, activation='relu')(y)
y_up_block_4 = Layers.Conv1D(64, 3, activation='relu')(y)

print('final block encoder', y_up_block_4.shape)

print('-----')
print('Transpose to get in the right length')

y = Layers.Conv1DTranspose(32, 21, activation='relu')(y_up_block_4)
y = Layers.Conv1DTranspose(16, 23, activation='relu')(y)
y = Layers.Conv1DTranspose(16, 25, activation='relu')(y)
y = Layers.Conv1DTranspose(16, 25, activation='relu')(y)
y = Layers.Conv1DTranspose(8, 27, activation='relu')(y)
y = Layers.Conv1DTranspose(4, 29, activation='relu')(y)
y_final_block = Layers.Conv1DTranspose(2, 31)(y)

print(y_final_block.shape)

#%% custom loss

# stack overflow: https://stackoverflow.com/questions/55445712/custom-loss-function-in-keras-based-on-the-input-data

# def mask_loss(input_layers):
    
#     def loss(y_true, y_pred):
                
#         return np.sqrt(np.sum(np.square(y_true - y_true[:, :, 1] * y_pred)) / y_pred.shape[1])
    
#     return loss


def mask_loss(y_true, y_pred):
    
    return np.sqrt(np.sum(np.square(y_true.numpy() - y_pred.numpy())) / y_pred.shape[1])
    

#%% model compile

model = tf.keras.Model(inputs=x, outputs=y_final_block)

#%% Training 

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss='mae', 
    metrics=['mean_squared_error']
    )

#%% Training with custom loss

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss=mask_loss, 
    metrics=['mean_squared_error']
    )

#%%

history = model.fit(data_store, fecg_store, 
          epochs=20, 
          batch_size=32,
          shuffle=True, 
    )