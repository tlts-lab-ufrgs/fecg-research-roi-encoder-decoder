import mne
import glob
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

#%% Read and re arange signal
"""
# 1) Read and re-arange signal

## Notes:
-------

We have 5 ECG samples, 
    this samples have 4 eletrodes with mECG signals 
    and 1 direct fECG signal


The ideia here is to concatenate this eletrodes in a array 
containing (5, TIME STEPS, 4 eletrodes)

After this separate the signal in x Epochs with TIME_STEPS / x steps each, so we have
(5 * x, TIME_STEPS / x, 4)


#%%

https://stackoverflow.com/questions/49290895/how-to-implement-a-1d-convolutional-auto-encoder-in-keras-for-vector-data
"""

DATA_PATH = "/home/julia/Documentos/ufrgs/Mestrado/datasets - NI-fECG/abdominal-and-direct-fetal-ecg-database-1.0.0/"
EPOCHS = 500
FILENAMES = glob.glob(DATA_PATH + "*.edf")

LEN_DATA = 600
CHANNELS = 4

QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = 100

data_store = np.empty(shape=(0, LEN_DATA, CHANNELS))
fecg_store = np.empty(shape=(0, LEN_DATA, 2))


#%%

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

    chuncked_data = np.array_split(filedata[1::] * 1e5, EPOCHS, axis = 1)
    chuncked_data = [partial_data.transpose() for partial_data in chuncked_data]
    
    chuncked_fecg_data = np.array_split(np.array([filedata[0] * 1e5, binary_mask]), EPOCHS, axis = 1)
    chuncked_fecg_data = [partial_data.transpose() for partial_data in chuncked_fecg_data]    
    # chuncked_fecg_data = [partial_data.reshape(LEN_DATA, 1) for partial_data in chuncked_fecg_data]

    data_store = np.vstack((data_store, chuncked_data))
    fecg_store = np.vstack((fecg_store, chuncked_fecg_data))
    
#%%

fig, ax = plt.subplots()

# ax.plot(data_store[5, , 1])

# %% Compose the model

# The inputs are 128-length vectors with 10 timesteps, and the
# batch size is 4.LEN_DATA
input_shape = (32, LEN_DATA, CHANNELS)


x = tf.keras.Input(batch_shape=input_shape)
y_ = tf.keras.layers.Conv1D(64, 3, activation='relu',input_shape=input_shape[1:])(x)
y = tf.keras.layers.Conv1D(64, 8, activation='relu')(y_)

max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=3,
   strides=1)(y)

y1_ = tf.keras.layers.Conv1D(32, 16, activation='relu')(max_pool_1d)
y1 = tf.keras.layers.Conv1D(32, 32, activation='relu')(y1_)

max_pool_1d_2 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(y1)

y2_ = tf.keras.layers.Conv1D(16, 64, activation='relu')(max_pool_1d_2)
y2 = tf.keras.layers.Conv1D(16, 64, activation='relu')(y2_)

max_pool_1d_3 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(y2)

y3_ = tf.keras.layers.Conv1D(8, 64, activation='relu')(max_pool_1d_3)
y3 = tf.keras.layers.Conv1D(8, 64, activation='relu')(y3_)

max_pool_1d_4 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(y3)

y4_ = tf.keras.layers.Conv1D(4, 128, activation='relu')(max_pool_1d_4)
y4 = tf.keras.layers.Conv1D(4, 128, activation='relu')(y4_)

max_pool_1d_5 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(y4)

x1 = tf.keras.layers.Conv1DTranspose(8, 256, activation='relu')(max_pool_1d_5)
x2 = tf.keras.layers.Conv1DTranspose(16, 128, activation='relu')(x1)
x3 = tf.keras.layers.Conv1DTranspose(16, 128, activation='relu')(x2)
x5 = tf.keras.layers.Conv1DTranspose(2, 55, activation='relu')(x3)
# x4 = tf.keras.layers.Conv1DTranspose(64, 3, activation='relu')(x3)

print(x5.shape)

#%%

layers = [
    tf.keras.layers.Conv1D(64, 3, activation='relu',input_shape=input_shape[1:]),
    tf.keras.layers.Conv1D(64, 6, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=3,
       strides=1, padding='valid'),
    tf.keras.layers.Conv1D(32, 12, activation='relu'),
    tf.keras.layers.Conv1D(32, 24, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='valid'),
    tf.keras.layers.Conv1D(16, 48, activation='relu'),
    tf.keras.layers.Conv1D(16, 48, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='valid'),
    tf.keras.layers.Conv1D(8, 48, activation='relu'),
    tf.keras.layers.Conv1D(8, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='valid'),
    tf.keras.layers.Conv1D(4, 3, activation='relu'),
    tf.keras.layers.Conv1D(4, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='valid'),
    tf.keras.layers.Conv1DTranspose(8, 2, activation='relu'),
    tf.keras.layers.Conv1DTranspose(16, 4, activation='relu'),
    tf.keras.layers.Conv1DTranspose(32, 26, activation='relu'),
    tf.keras.layers.Conv1DTranspose(1, 2, activation='relu')
    ]

#%%


# model = tf.keras.Sequential(layers)

model = tf.keras.Model(inputs=x, outputs=x5)



#%% Training 

model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.01)


, loss='mse', metrics=['mean_squared_error'])

#%%

history = model.fit(data_store, fecg_store, 
          epochs=20, 
          batch_size=32,
          shuffle=True, 
    )

#%%

test_data_store = np.empty(shape=(0, LEN_DATA, CHANNELS))
test_fecg_store = np.empty(shape=(0, LEN_DATA, 2))

# for file in FILENAMES[0:2]:

file_info = mne.io.read_raw_edf(FILENAMES[-1])
filedata = file_info.get_data()

annotations = mne.read_annotations(FILENAMES[-1])
time_annotations = annotations.onset

binary_mask = np.zeros(shape=file_info.times.shape)

for step in time_annotations:

    center_index = np.where(file_info.times == step)[0][0]

    binary_mask[
        center_index - QRS_DURATION_STEP :
        center_index + QRS_DURATION_STEP 
    ] = 1

chuncked_data = np.array_split(filedata[1::], EPOCHS, axis = 1)
chuncked_data = [partial_data.transpose() for partial_data in chuncked_data]

chuncked_fecg_data = np.array_split(np.array([filedata[0], binary_mask]), EPOCHS, axis = 1)
chuncked_fecg_data = [partial_data.transpose() for partial_data in chuncked_fecg_data]    
# chuncked_fecg_data = [partial_data.reshape(LEN_DATA, 1) for partial_data in chuncked_fecg_data]

test_data_store = np.array(chuncked_data)
test_fecg_store = np.array(chuncked_fecg_data)

# data_store = filedata[1::]
# test_fecg_data = np.array([filedata[0], binary_mask])
#%%

result = model.predict(test_data_store)

#%%

fig, ax = plt.subplots()

# ax.plot(result[:, :, 1])

ax1 = ax.twinx()

ax1.plot(test_fecg_store[:, :, 1])
