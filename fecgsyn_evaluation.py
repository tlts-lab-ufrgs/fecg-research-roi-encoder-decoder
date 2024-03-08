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

#%%

model = tf.keras.models.load_model(
    '/home/julia/Documents/fECG_research/research_dev/autoencoder_with_mask/final_model_3ch_512/', 
    custom_objects = {
        'mse_mask': Metric.mse_mask,
        'mse_signal': Metric.mse_signal, 
        'loss': Loss.loss, 
        'lr': lr_scheduler
    }
)


#%%

QRS_LEN = 50 # +- 125 pontos fazem o esmo que no abcd, fs = 250
LEN_DATA = 512

DATA_PATH = '/home/julia/Documents/fECG_research/datasets/our_fecgsyn_db'

# %% load data

syn_case = 3
snr = '00'

for i in range(1, 2, 1):  # subjects
    
    i = f'0{i}' if i < 10 else 10 
    
    for j in range(1,6):  #repetitions
    
    
        mecg_file = f'{DATA_PATH}/{i}_snr{snr}dB_l{j}_c{syn_case}_mecg'
        fecg_file = f'{DATA_PATH}/{i}_snr{snr}dB_l{j}_c{syn_case}_fecg1'
        noise_file = f'{DATA_PATH}/{i}_snr{snr}dB_l{j}_c{syn_case}_noise2'


        mecg_signal, labels_mecg = wfdb.rdsamp(mecg_file)
        fecg_signal, labels_fecg = wfdb.rdsamp(fecg_file)
        noise_signal, labels_noise = wfdb.rdsamp(noise_file)
        
        # mecg_signal += noise_signal
        
        filtered_channels = mecg_signal[:, [3,9,13]]
        filtered_channels_fecg = fecg_signal[:, 3]

        # time_annotations = wfdb.rdann(
        #     fecg_file,
        #     extension='qrs'
        # )
        
        # Generates masks
        
        size_array = int(labels_mecg['sig_len'])
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
        UPPER_LIMIT = int(np.power(2, np.round(np.log2(float(labels_mecg['sig_len'])), 0)))
        
        for batch in range(0, UPPER_LIMIT, LEN_DATA):


            chunked_data = filtered_channels[(batch): ((batch + LEN_DATA)), :]
            
            chunked_fecg_real_data = filtered_channels_fecg[(batch): (batch + LEN_DATA)]
            # chunked_fecg_binary_data = mask[(batch): (batch + LEN_DATA)]

            
            # Data Normalization

            chunked_data += np.abs(np.min(chunked_data)) # to zero things
            chunked_fecg_real_data += np.abs(np.min(chunked_fecg_real_data)) # to zero things
            

            chunked_data *= (1 / np.abs(np.max(chunked_data)))
            chunked_fecg_real_data *= (1 / np.abs(np.max(chunked_fecg_real_data)))
            
            
            chunked_fecg_data = np.array([
                chunked_fecg_real_data, 
                # chunked_fecg_binary_data
            ]).transpose()

            if i == '01' and j == 1 and batch == 0:

                aECG_store = np.copy([chunked_data])
                fECG_store = np.copy([chunked_fecg_data])

            else:
                aECG_store = np.vstack((aECG_store, [chunked_data]))
                fECG_store = np.vstack((fECG_store, [chunked_fecg_data]))
    
#%%

predict = model.predict(aECG_store)
#%%
index= 10

plt.plot(fECG_store[index])
plt.plot(predict[index])
# plt.plot(aECG_store[index])

# plt.plot(prediction[9])

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
