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

# data_store = np.empty(shape=[]) # shape=(EPOCHS, LEN_DATA, CHANNELS)
# fecg_store = np.empty(shape=[]) # shape=(EPOCHS, LEN_DATA, 2)
 
#%% Data Loading 


for file in FILENAMES[0:2]:

    file_info = mne.io.read_raw_edf(file)
    filedata = file_info.get_data()

    annotations = mne.read_annotations(file)
    time_annotations = annotations.onset


    # Generates Binary masks

    binary_mask = np.zeros(shape=file_info.times.shape)

    for step in time_annotations:

        center_index = np.where(file_info.times == step)[0][0]

        qrs_region = np.where(
            (file_info.times >= (step - QRS_DURATION_STEP)) &
            (file_info.times <= (step + QRS_DURATION_STEP))
        )[0]

        binary_mask[qrs_region] = 1


    for batch in range(0, np.shape(filedata)[1], LEN_DATA):


        chunked_data = filedata[1::, (batch): ((batch + LEN_DATA))].transpose() * 1e5
        
        chunked_fecg_real_data = filedata[0, (batch): (batch + LEN_DATA)] * 1e5
        chunked_fecg_binary_data = binary_mask[(batch): (batch + LEN_DATA)]

        chunked_fecg_data = np.array([
            chunked_fecg_real_data, 
            chunked_fecg_binary_data
        ]).transpose()


        if batch == 0:

            data_store = np.copy([chunked_data])
            fecg_store = np.copy([chunked_fecg_data])

        else:
            data_store = np.vstack((data_store, [chunked_data]))
            fecg_store = np.vstack((fecg_store, [chunked_fecg_data]))
    

#%% Data Preprocessing

# filter high and low frquency noise


#%% Model
    
input_shape = (BATCH_SIZE, LEN_DATA, CHANNELS)

model = linknet(input_shape, num_classes=1)


# %%

# model.compile(optimizer='adam', loss=mse_with_mask, metrics=mse_with_mask)

model.compile(optimizer='adam', loss=tf.keras.losses.MSE, metrics=['mean_squared_error'])




#%%

history = model.fit(data_store, fecg_store, 
          epochs=30, 
          batch_size=BATCH_SIZE,
          shuffle=True, 
    )

# %%

teste = model.predict(data_store, 32)
# %%
