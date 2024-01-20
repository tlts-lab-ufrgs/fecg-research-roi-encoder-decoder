"""
Main python code for fECG extraction

juliacremus
Saturday 20 jan 2024
"""

#%% Imports

import mne
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as Layers

from models.linknet import linknet
from losses.mse_with_mask import mse_with_mask

#%% Parameters

# DATA_PATH = "/home/julia/Documents/fECG_research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/"

DATA_PATH = "/home/julia/Documentos/ufrgs/Mestrado/datasets - NI-fECG/abdominal-and-direct-fetal-ecg-database-1.0.0/"

EPOCHS = 500
FILENAMES = glob.glob(DATA_PATH + "*.edf")

LEN_DATA = 600
CHANNELS = 4

BATCH_SIZE = 32

QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = 100

data_store = np.empty(shape=(EPOCHS, LEN_DATA, CHANNELS))
fecg_store = np.empty(shape=(EPOCHS, LEN_DATA, 2))

#%% Data Loading 


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
    

#%% Data Preprocessing

# filter high and low frquency noise


#%% Model
    
input_shape = (BATCH_SIZE, LEN_DATA, CHANNELS)

model = linknet(input_shape, num_classes=1)


# %%

model.compile(optimizer='adam', loss=mse_with_mask, metrics=mse_with_mask)


#%%

history = model.fit(data_store, fecg_store, 
          epochs=20, 
          batch_size=32,
          shuffle=True, 
    )

# %%
