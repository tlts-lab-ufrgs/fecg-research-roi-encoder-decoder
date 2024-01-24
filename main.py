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
from losses.mse_with_mask import mse_with_mask
from utils.lr_scheduler import callback
from data_load import mask_as_input, signal_and_mask_as_output

#%% Parameters

DATA_PATH = "/home/julia/Documents/fECG_research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/"

# DATA_PATH = "/home/julia/Documentos/ufrgs/Mestrado/datasets - NI-fECG/abdominal-and-direct-fetal-ecg-database-1.0.0/"

EPOCHS = 500
FILENAMES = glob.glob(DATA_PATH + "*.edf")

LEN_DATA = 600
CHANNELS = 4

BATCH_SIZE = 32

QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = 100

INIT_LR = 0.01
 
#%% Data Loading 

# data_store, fecg_store = mask_as_input.load_data(
#     len_data = LEN_DATA, path = DATA_PATH, qrs_duration = QRS_DURATION
# )

data_store, fecg_store = signal_and_mask_as_output.load_data(
    len_data = LEN_DATA, path = DATA_PATH, qrs_duration = QRS_DURATION
)

#%% Data Preprocessing

# filter high and low frquency noise


#%% Model
    
input_shape = (BATCH_SIZE, LEN_DATA, CHANNELS)

model = linknet(input_shape, num_classes=1)

# def weighted_loss(y_true, y_pred, weights):
#       loss = tf.math.squared_difference(y_pred, y_true)
#       w_loss = tf.multiply(weights, loss)
#       return tf.reduce_mean(tf.reduce_sum(w_loss, axis=-1))
  
# targets = Input(BATCH_SIZE, LEN_DATA, 1)
# out = Input(BATCH_SIZE, LEN_DATA, 1)
# inp = Input(BATCH_SIZE, LEN_DATA, CHANNELS + 1)

# model.add_loss(weighted_loss(targets, out, inp))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR), 
    loss=tf.keras.losses.MSE, #mse_with_mask, # 
    metrics=['mean_squared_error']
    )


#%%

history = model.fit(data_store, fecg_store, 
          epochs=60, 
          batch_size=BATCH_SIZE,
          validation_split=0.2,
          shuffle=True, 
          callbacks=[callback],
    )

#%%

fig, ax = plt.subplots()

ax.plot(history.history['loss'], label='Training Loss')

ax.plot(history.history['val_loss'], label='Validation Loss')

ax.legend()

# %%



for file in FILENAMES[3:4]:

    file_info = mne.io.read_raw_edf(file)
    filedata = file_info.get_data()

    annotations = mne.read_annotations(file)
    time_annotations = annotations.onset


    # Generates Binary masks

    binary_mask = np.zeros(shape=file_info.times.shape)

    for step in time_annotations:

        center_index = np.where(file_info.times == step)[0][0]

        qrs_region = np.where(
            (file_info.times >= (step - QRS_DURATION)) &
            (file_info.times <= (step + QRS_DURATION))
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

            data_store_test = np.copy([chunked_data])
            fecg_store_test = np.copy([chunked_fecg_data])

        else:
            data_store_test = np.vstack((data_store_test, [chunked_data]))
            fecg_store_test = np.vstack((fecg_store_test, [chunked_fecg_data]))
    

#%%
teste = model.predict(data_store, 64)
# %%

fig, ax = plt.subplots()

ax.plot(fecg_store[6, :, 1])

ax1 = ax.twinx()

ax1.plot(teste[6, :, 1], color='orange')

# %%
