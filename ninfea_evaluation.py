"""
Evluate FECG Syn dataset
"""

#%%

import glob
import wfdb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from models.ae_proposed import Metric, Loss
from utils.gaussian_function import gaussian
from data_load.load_leave_one_out import data_resizer
from utils.lr_scheduler import callback as lr_scheduler

from scipy.signal import resample
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
from scipy.io import loadmat

#%%

model = tf.keras.models.load_model(
    '/home/julia/Documents/fECG_research/research_dev/autoencoder_with_mask/final_model_3ch/', 
    custom_objects = {
        'mse_mask': Metric.mse_mask,
        'mse_signal': Metric.mse_signal, 
        'loss': Loss.loss, 
        'lr': lr_scheduler
    }
)


#%%

QRS_LEN = 100 # +- 125 pontos fazem o esmo que no abcd, fs = 250
LEN_DATA = 1024

DATA_PATH = '/home/julia/Documents/fECG_research/datasets/ninfea-non-invasive-multimodal-foetal-ecg-doppler-dataset-for-antenatal-cardiology-research-1.0.0/'

#%%
doppler_signals = loadmat(f'{DATA_PATH}/pwd_signals/{4}envelopes')

#%% load data

high_risk = [
    6,
    7,
    8,
    12,
    18,
    19,
    24,
    27,
    34,
    37,
    38,
    47,
    48,
    49,
    50,
    53,
    54,
    58
]

# signal, label = wfdb.rdsamp(
#     f'{PATH}/wfdb_format_ecg_and_respiration/23')
INIT_SUB = 60
END_SUB = 60

for i in range(INIT_SUB, END_SUB + 1, 1):  # subjects
    
    
    mecg_file = f'{DATA_PATH}/wfdb_format_ecg_and_respiration/{i}'
    
    mecg_signal, labels_mecg = wfdb.rdsamp(mecg_file)
    


    filtered_channels = mecg_signal[:, [7, 14, 11]]
        
    size_array = int(labels_mecg['sig_len'])
    
    
    resampled_signal = np.zeros(shape=(int(size_array / 2) - 100, 3))
    
    for j in range(3):
        
        resampled_signal[:, j] = resample(filtered_channels[:, j], int(size_array / 2))[100:]
    
    
    transformer = FastICA()
    X_transformed = transformer.fit_transform(resampled_signal)
    

        # mask = np.zeros(shape=(size_array))

        # for step in time_annotations.sample:

        #     # center_index = np.where(time_annotations.sample == step)[0][0]

        #     qrs_region = np.where(
        #         (np.arange(0, size_array, 1) >= (step - 50)) &
        #         (np.arange(0, size_array, 1) <= (step + 50))
        #     )[0]
        
            
        #     mask[qrs_region] = gaussian(qrs_region, step, 12)
            
            
            
            

        # Resize data to be in the desire batch size
        
        # if training:
            # UPPER_LIMIT = int(np.power(2, np.round(np.log2(2 * file_info.times.shape[0]), 0)))
        # else: 
    UPPER_LIMIT = int(size_array/2) - 100- LEN_DATA
    
    batch = 0
    
    while batch < UPPER_LIMIT:        

        chunked_data = np.copy(resampled_signal[(batch): ((batch + LEN_DATA)), :])
            
            # chunked_fecg_real_data = filtered_channels_fecg[(batch): (batch + LEN_DATA)]
            # chunked_fecg_binary_data = mask[(batch): (batch + LEN_DATA)]

            
            # Data Normalization

        min_data = np.min(chunked_data)
        chunked_data -= min_data # to zero things
            # chunked_fecg_real_data += np.abs(np.min(chunked_fecg_real_data)) # to zero things
            
        max_data = np.max(chunked_data)

        chunked_data *= (1 / max_data)
            # chunked_fecg_real_data *= (1 / np.abs(np.max(chunked_fecg_real_data)))
            
            
            # chunked_fecg_data = np.array([
            #     chunked_fecg_real_data, 
            #     chunked_fecg_binary_data
            # ]).transpose()

        if i == INIT_SUB and batch == 0:

            aECG_store = np.copy([chunked_data])
                # fECG_store = np.copy([chunked_fecg_data])

        else:
            aECG_store = np.vstack((aECG_store, [chunked_data]))
                # fECG_store = np.vstack((fECG_store, [chunked_fecg_data]))
    
        batch += LEN_DATA

#%%%
prediction = model.predict(aECG_store)

#%%

index = 1
plt.plot(aECG_store[index, :])

# plt.plot(prediction[index])

# plt.plot(fECG_store[2])

#%%
# if syn_case == 5:
#     fecg_file2 = f'{DATA_PATH}/snr{snr}dB/sub{i}_snr{snr}dB_l{j}_c{syn_case}_fecg2'

# # combine this two files
# mecg_signal, mecg_labels = wfdb.rdsamp(mecg_file)

# # get only the right channels




# mecg_signal_training = data_resizer(
#     []
# )
# # %%

# import os
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from keras import backend as K
# from numba import cuda

# from data_load.load_leave_one_out import data_loader
# from models.ae_proposed import ProposedAE

# #%% constants

# # Range in learning rate
# UPPER_LIM_LR = 0.00001
# LOWER_LIMIT_LR = 0.00098
# LR_STEP = 0.00

# # batch size
# BATCH_SIZE=4

# # files 
# TOTAL_FILES = 5

# RESULTS_PATH = "/home/julia/Documents/fECG_research/research_dev/autoencoder_with_mask/results/"
# DATA_PATH =  "/home/julia/Documents/fECG_research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/"

# CHANNELS = 4
# LEN_BATCH = 512

# MODEL_INPUT_SHAPE = (BATCH_SIZE, LEN_BATCH, CHANNELS)


# W_MASK = 0.2
# W_SIGNAL = 0.1
# W_COMBINED = 1 - W_MASK - W_SIGNAL
# #%%

# model = ProposedAE(
#     MODEL_INPUT_SHAPE, 
#     BATCH_SIZE, 
#     UPPER_LIM_LR, 
#     W_MASK, 
#     W_SIGNAL, 
#     W_COMBINED, 
#     training_data=aECG_store, 
#     ground_truth=fECG_store,
#     testing_data=None, 
#     ground_truth_testing=None, 
# )

# history, _, _ = model.fit_and_evaluate()
# %%
