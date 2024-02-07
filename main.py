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

#%% Parameters

DATA_PATH = "/home/julia/Documents/fECG_research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/"

# DATA_PATH = "/home/julia/Documentos/ufrgs/Mestrado/datasets - NI-fECG/abdominal-and-direct-fetal-ecg-database-1.0.0/"

EPOCHS = 512
FILENAMES = glob.glob(DATA_PATH + "*.edf")

LEN_DATA = 512
CHANNELS = 4

BATCH_SIZE = 4

DATA_BATCH = 4

QRS_DURATION = 0.2  # seconds, max
QRS_DURATION_STEP = 100

INIT_LR = 0.0001
 
#%% Data Loading 

# data_store, fecg_store = mask_as_input.load_data(
#     len_data = LEN_DATA, path = DATA_PATH, qrs_duration = QRS_DURATION
# )

data_store, fecg_store = signal_and_mask_as_output.load_data(
    len_data = LEN_DATA, path = DATA_PATH, qrs_duration = QRS_DURATION
)

#%%

# plt.plot(data_store[10])

plt.plot(fecg_store[10])

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
    optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR), 
    loss=mse_with_mask, # 
    metrics=[
        tf.keras.metrics.RootMeanSquaredError(name='rmse'), 
        'mean_squared_error'
    ]
    )



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

# test = model.evaluate(data_store, fecg_store[4])
# print(test)



#%% Parameters


def gaussian(x, mu, sig):
    
    signal = 1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    
    return signal / np.max(signal)


def load_data_to_predict(len_data = LEN_DATA, path = DATA_PATH, qrs_duration = QRS_DURATION):
    
    FILENAMES = glob.glob(path + "*.edf")


    for file in FILENAMES[3:4]:
        
        

        file_info = mne.io.read_raw_edf(file)
        filedata = file_info.get_data()
        
        # filedata *= 1E5
        
        filedata += np.abs(np.min(filedata)) # to zero things
        
        max_absolute_value = np.max(np.abs(filedata))
        
        
        
        filedata *= (1 / max_absolute_value)

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


#%%

predict = model.predict(data_store)

# %%

index = 12 


from ecgdetectors import Detectors


detector = Detectors(1000)

r_peaks = detector.pan_tompkins_detector(predict[index, :, 0]),

print(r_peaks)


fig, ax = plt.subplots()


# ax.plot(predict[1, :], color='orange')

# ax.plot(data_store[200], alpha = 0.5)
ax.plot(predict[index, :, 0], label='predito')

ax.plot(fecg_store[index], label='real')

ax.vlines(ymin = 0, ymax = 1, x = r_peaks[0])

ax.legend()

#%%
    
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


order = 10
fs = 1000.0       # sample rate, Hz
cutoff = 300  # desired cutoff frequency of the filter, Hz

index = 16 

# Filter the data, and plot both the original and filtered signals.
y = butter_lowpass_filter(predict[index, :, 0], cutoff, fs, order)



from ecgdetectors import Detectors


detector = Detectors(1000)

r_peaks = detector.pan_tompkins_detector(y),

print(r_peaks)


fig, ax = plt.subplots()


# ax.plot(predict[1, :], color='orange')

# ax.plot(data_store[200], alpha = 0.5)
ax.plot(y, label='predito')

ax.plot(fecg_store[index], label='real')

ax.vlines(ymin = 0, ymax = 1, x = r_peaks[0])

ax.legend()
# %%
