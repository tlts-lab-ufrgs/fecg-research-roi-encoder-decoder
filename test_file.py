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
# import tensorflow.keras.layers as Layers, Input

from models.linknet import linknet
from models.ae_proposed_tests_file import proposed_ae

from losses.mse_with_mask import mse_with_mask
from utils.lr_scheduler import callback
from utils.training_patience import callback as patience_callback
from data_load import mask_as_input, signal_and_mask_as_output

from metrics.mse_mask import mse_mask
from metrics.mse_signal import mse_signal

#%% Parameters

DATA_PATH = "/home/julia/Documents/fECG_research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/"

# DATA_PATH = "/home/julia/Documentos/ufrgs/Mestrado/datasets - NI-fECG/abdominal-and-direct-fetal-ecg-database-1.0.0/"

EPOCHS = 512
FILENAMES = glob.glob(DATA_PATH + "*.edf")

LEN_DATA = 512
CHANNELS = 4

BATCH_SIZE = 4

DATA_BATCH = 4

QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = 50

INIT_LR = 0.0001 # 0.00005
 
#%% Data Loading 

# data_store, fecg_store = mask_as_input.load_data(
#     len_data = LEN_DATA, path = DATA_PATH, qrs_duration = QRS_DURATION
# )

to_be_read = glob.glob(DATA_PATH + "*.edf")[4:]

data_store, fecg_store = signal_and_mask_as_output.load_data(
    to_be_read, len_data = LEN_DATA, path = DATA_PATH, qrs_duration = QRS_DURATION, qrs_len = QRS_DURATION_STEP
)

# plt.plot(data_store[10])
#%%
plt.plot(fecg_store[2])
# plt.plot(data_store[-10])
plt.plot(data_store[2])

#% Data Preprocessing

# filter high and low frquency noise


#%% Model
    
input_shape = (DATA_BATCH, LEN_DATA, CHANNELS)

# model = linknet(input_shape, num_classes=1)

model = proposed_ae(input_shape, num_classes=1)

# def weighted_loss(y_true, y_pred, weights):
#       loss = tf.math.squared_difference(y_pred, y_true)
#       w_loss = tf.multiply(weights, loss)
#       return tf.reduce_mean(tf.reduce_sum(w_loss, axis=-1))
  
# targets = Input(BATCH_SIZE, LEN_DATA, 1)
# out = Input(BATCH_SIZE, LEN_DATA, 1)
# inp = Input(BATCH_SIZE, LEN_DATA, CHANNELS + 1)

# model.add_loss(weighted_loss(targets, out, inp))

model.compile(
    # optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR), 
    optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR), 
    loss=mse_with_mask, # 
    metrics=[
        tf.keras.metrics.RootMeanSquaredError(name='rmse'), 
        'mean_squared_error', 
        mse_signal, 
        mse_mask
    ]
    )

#%%

#%%

history = model.fit(data_store, fecg_store, 
          epochs=250, 
          batch_size=BATCH_SIZE,
          validation_split=0.25,
          shuffle=True, 
          callbacks=[
            callback,
            patience_callback('loss', 15)
        ],
    )

#%%

fig, ax = plt.subplots()

ax.plot(history.history['loss'], label='Training Loss')

ax.plot(history.history['val_loss'], label='Validation Loss')

# ax.plot(history.history['mean_squared_error'], label='MSE training')

ax.legend()

#%%

#%%

fig, ax = plt.subplots()

ax.plot(history.history['mse_mask'], label='MSE Mask')

ax.plot(history.history['mse_signal'], label='MSE Signal')

# ax.plot(history.history['mean_squared_error'], label='MSE training')

ax.legend()


#%% Parameters


def gaussian(x, mu, sig):
    
    signal = 1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    
    return signal / np.max(signal)


def load_data_to_predict(len_data = LEN_DATA, path = DATA_PATH, qrs_duration = QRS_DURATION):
    
    FILENAMES = glob.glob(path + "*.edf")


    for file in [FILENAMES[0]]:
        
        

        file_info = mne.io.read_raw_edf(file)
        filedata = file_info.get_data()
        
        annotations = mne.read_annotations(file)
        time_annotations = annotations.onset
        


        # Generates Binary masks

        binary_mask = np.zeros(shape=file_info.times.shape)

        for step in time_annotations:

            center_index = np.where(file_info.times == step)[0][0]

            qrs_region = np.where(
                (file_info.times >= (step - qrs_duration)) &
                (file_info.times <= (step + qrs_duration))
            )[0]
            

            binary_mask[qrs_region] = gaussian(qrs_region, center_index, QRS_DURATION_STEP / 2)


        for batch in range(0, 262144, len_data):


            chunked_data = filedata[1::, (batch): ((batch + len_data))].transpose()
            
            # chunked_data_with_noise = chunked_data + 0.01 * np.random.normal(0,1,len_data)    
                  
            chunked_fecg_real_data = filedata[0, (batch): (batch + len_data)]
            chunked_fecg_binary_data = binary_mask[(batch): (batch + len_data)]

            chunked_fecg_data = np.array([
                chunked_fecg_real_data, 
                chunked_fecg_binary_data
            ]).transpose()
            
            # Data Normalization
     

            chunked_data += np.abs(np.min(chunked_data)) # to zero things
            chunked_fecg_data += np.abs(np.min(chunked_fecg_data[:, 0])) # to zero things
            

            chunked_data *= (1 / np.abs(np.max(chunked_data)))
            chunked_fecg_data[:, 0] *= (1 / np.abs(np.max(chunked_fecg_data[:, 0])))
            
            

            if batch == 0:

                data_store = np.copy([chunked_data])
                fecg_store = np.copy([chunked_fecg_data])

            else:
                data_store = np.vstack((data_store, [chunked_data]))
                fecg_store = np.vstack((fecg_store, [chunked_fecg_data]))
    
    
    
    return data_store, fecg_store



data_store_predict, fecg_store_predict = load_data_to_predict(
    len_data = LEN_DATA, path = DATA_PATH, qrs_duration = QRS_DURATION
)
# %%
test = model.evaluate(data_store_predict, fecg_store_predict)
print(test)

#%%

predict_training = model.predict(data_store_predict)

# %%

index =  10

# from ecgdetectors import Detectors


# detector = Detectors(1000)

# r_peaks = detector.pan_tompkins_detector(predict[index, :, 0]),

# print(r_peaks)


fig, ax = plt.subplots()


# ax.plot(predict[1, :], color='orange')

# ax.plot(data_store[200], alpha = 0.5)
ax.plot(predict_training[index], label='predito')

ax.plot(fecg_store_predict[index], label='real')

# ax.vlines(ymin = 0, ymax = 1, x = r_peaks[0])

ax.legend()


# %%


predict = model.predict(data_store)
#%%
index =  500
# from ecgdetectors import Detectors


# detector = Detectors(1000)

# r_peaks = detector.pan_tompkins_detector(predict[index, :, 0]),

# print(r_peaks)


fig, ax = plt.subplots()


# ax.plot(predict[1, :], color='orange')

# ax.plot(data_store[200], alpha = 0.5)
ax.plot(predict[index], label='predito')

ax.plot(fecg_store[index], label='real')

# ax.vlines(ymin = 0, ymax = 1, x = r_peaks[0])

ax.legend()

#%%

# fig, ax = plt.subplots()


# # ax.plot(predict[1, :], color='orange')

# # ax.plot(data_store[200], alpha = 0.5)
# ax.plot(predict[index, :, 1], label='predito')

# ax.plot(fecg_store_predict[index, :, 1], label='real')

# # ax.vlines(ymin = 0, ymax = 1, x = r_peaks[0])

# ax.legend()


# %%

import pandas as pd

# results/4CH_MOD-LOSS_SCH-MOD-4_DATA-AUG-3_QRStime_0.1-LR_0.0001-W_MASK_0.3-W_SIG_0.1-LEFT_4/4CH_MOD-LOSS_SCH-MOD-4_DATA-AUG-3_QRStime_0.1-LR_0.0001-W_MASK_0.3-W_SIG_0.1-LEFT_4-training_history.csv

filename = '/home/julia/Documents/fECG_research/research_dev/autoencoder_with_mask/results/010324-4CH-VAL_LOSS-SCH_0.01-DROPOUT_RP10-LR_0.0001-W_MASK_0.3-W_SIG_0.1-LEFT_0/010324-4CH-VAL_LOSS-SCH_0.01-DROPOUT_RP10-LR_0.0001-W_MASK_0.3-W_SIG_0.1-LEFT_0-training_history.csv'

data = pd.read_csv(filename)
#%%

import matplotlib.pyplot as plt


# plt.plot(data['loss'], label='loss')
plt.plot(data['val_loss'], label='val loss')
# plt.plot(data['mse_signal'], label='mse siG')
# plt.plot(data['mse_mask'], label='mask')

plt.legend()

# %%


import numpy as np

def read_binary_file(ifname=None):
    if ifname is None:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        ifname = filedialog.askopenfilename(filetypes=[('Binary files', '*.bin')])

    fs = 0
    x = np.array([])

    try:
        with open(ifname, 'rb') as fileID:
            fs = np.frombuffer(fileID.read(8), dtype='float64', count=1, offset=0, )[0]
            rows = np.frombuffer(fileID.read(8), dtype='uint64', count=1, offset=0, )[0]
            cols = np.frombuffer(fileID.read(8), dtype='float64', count=1, offset=0, )[0]
            data = np.frombuffer(fileID.read(), dtype='float64', count=int(rows*cols), offset=0)
            fileID.close()

            if len(data) == int(rows * cols):
                x = data.reshape((int(rows), int(cols)), order='F')  # 'F' for column-wise order
                status = 0
            else:
                status = -1
    except IOError:
        status = -2

    return x, fs, status

#%%

filename = '/home/julia/Documents/fECG_research/datasets/ninfea-non-invasive-multimodal-foetal-ecg-doppler-dataset-for-antenatal-cardiology-research-1.0.0/bin_format_ecg_and_respiration/1.bin'

x, fs, status = read_binary_file(filename)
# %%

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from numba import cuda

from data_load.load_leave_one_out import data_loader

# Range in learning rate
UPPER_LIM_LR = 0.0001
LOWER_LIMIT_LR = 0.00098
LR_STEP = 0.00

# batch size
BATCH_SIZE=4

# files 
TOTAL_FILES = 5

RESULTS_PATH = "/home/julia/Documents/fECG_research/research_dev/autoencoder_with_mask/results/"
DATA_PATH =  "/home/julia/Documents/fECG_research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/"

CHANNELS = 4
LEN_BATCH = 512
QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = 50

MODEL_INPUT_SHAPE = (BATCH_SIZE, LEN_BATCH, CHANNELS)

training_data, testing_data = data_loader(
        DATA_PATH, 
        512, 
        QRS_DURATION, 
        QRS_DURATION_STEP,
        leave_for_testing=4,
        type_of_file='edf'
)


# %%

import mne
import glob
import numpy as np
import matplotlib.pyplot as plt


file = '/home/julia/Documents/fECG_research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/r07.edf'

file_info = mne.io.read_raw_edf(file)
raw_data = file_info.get_data()
annotations = mne.read_annotations(file)
time_annotations = annotations.onset

#%%

i = 200000
delta = 2048
lim_max = i + 2*delta

while i < lim_max - delta:
    
    fig, ax = plt.subplots(5, 1)
    
    ax[0].set_title(f'{i / 2048}')
    
    ax[0].plot(raw_data[1, i : (i+delta)], color='black')
    ax[1].plot(raw_data[2, i : (i+delta)], color='blue')
    ax[2].plot(raw_data[3, i : (i+delta)], color='green')
    ax[3].plot(raw_data[4, i : (i+delta)], color='orange')
    ax[4].plot(raw_data[0, i : (i+delta)], color='green')
    
    i += delta

