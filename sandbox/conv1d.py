import mne
import glob
import numpy as np
import tensorflow as tf

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
"""

DATA_PATH = "/home/julia/Documentos/ufrgs/Mestrado/datasets - NI-fECG/abdominal-and-direct-fetal-ecg-database-1.0.0/"
EPOCHS = 500
FILENAMES = glob.glob(DATA_PATH + "*.edf")

LEN_DATA = 600
CHANNELS = 4

data_store = np.empty(shape=(EPOCHS, LEN_DATA, CHANNELS))
fecg_store = np.empty(shape=(EPOCHS, LEN_DATA, 1))

for file in FILENAMES:

    file_info = mne.io.read_raw_edf(file)
    filedata = file_info.get_data()

    chuncked_data = np.array_split(filedata[1::], EPOCHS, axis = 1)
    chuncked_data = [partial_data.transpose() for partial_data in chuncked_data]
    
    chuncked_fecg_data = np.array_split(filedata[0], EPOCHS)
    chuncked_fecg_data = [partial_data.reshape(LEN_DATA, 1) for partial_data in chuncked_fecg_data]

    data_store = np.vstack((data_store, chuncked_data))
    fecg_store = np.vstack((fecg_store, chuncked_fecg_data))

# %% Compose the model

# The inputs are 128-length vectors with 10 timesteps, and the
# batch size is 4.LEN_DATA
input_shape = (32, LEN_DATA, CHANNELS)
x = tf.random.normal(input_shape)
y_ = tf.keras.layers.Conv1D(64, 3, activation='relu',input_shape=input_shape[1:])(x)
y = tf.keras.layers.Conv1D(64, 3, activation='relu')(y_)

max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=3,
   strides=1, padding='valid')(y)

y1_ = tf.keras.layers.Conv1D(32, 3, activation='relu')(max_pool_1d)
y1 = tf.keras.layers.Conv1D(32, 3, activation='relu')(y1_)

max_pool_1d_2 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='valid')(y1)

y2_ = tf.keras.layers.Conv1D(16, 3, activation='relu')(max_pool_1d_2)
y2 = tf.keras.layers.Conv1D(16, 3, activation='relu')(y2_)

max_pool_1d_3 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='valid')(y2)

y3_ = tf.keras.layers.Conv1D(8, 3, activation='relu')(max_pool_1d_3)
y3 = tf.keras.layers.Conv1D(8, 3, activation='relu')(y3_)

max_pool_1d_4 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='valid')(y3)

y4_ = tf.keras.layers.Conv1D(4, 3, activation='relu')(max_pool_1d_4)
y4 = tf.keras.layers.Conv1D(4, 3, activation='relu')(y4_)

max_pool_1d_5 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='valid')(y4)

x1 = tf.keras.layers.Conv1DTranspose(8, 2, activation='relu')(max_pool_1d_5)
x2 = tf.keras.layers.Conv1DTranspose(16, 4, activation='relu')(x1)
x3 = tf.keras.layers.Conv1DTranspose(32, 26, activation='relu')(x2)
x5 = tf.keras.layers.Conv1DTranspose(1, 2, activation='relu')(x3)
# x4 = tf.keras.layers.Conv1DTranspose(64, 3, activation='relu')(x3)

print(max_pool_1d_5.shape)
print(x5.shape)

#%%

layers = [
    tf.keras.layers.Conv1D(64, 3, activation='relu',input_shape=input_shape[1:]),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=3,
       strides=1, padding='valid'),
    tf.keras.layers.Conv1D(32, 3, activation='relu'),
    tf.keras.layers.Conv1D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='valid'),
    tf.keras.layers.Conv1D(16, 3, activation='relu'),
    tf.keras.layers.Conv1D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='valid'),
    tf.keras.layers.Conv1D(8, 3, activation='relu'),
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


model = tf.keras.Sequential(layers)


#%% Training 

model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

#%%

history = model.fit(data_store, fecg_store, 
          epochs=20, 
          batch_size=32,
          shuffle=True, 
    )

