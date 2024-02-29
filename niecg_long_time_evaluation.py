"""
Evluate FECG Syn dataset
"""

#%%

import glob
import wfdb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data_load.load_leave_one_out import data_loader
from models.ae_proposed import Metric, Loss
from utils.gaussian_function import gaussian
from data_load.load_leave_one_out import data_resizer
from utils.lr_scheduler import callback as lr_scheduler

#%%

model = tf.keras.models.load_model(
    '/home/julia/Documents/fECG_research/research_dev/autoencoder_with_mask/final_model/', 
    custom_objects = {
        'mse_mask': Metric.mse_mask,
        'mse_signal': Metric.mse_signal, 
        'loss': Loss.loss, 
        'lr': lr_scheduler
    }
)


#%%

QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = 50
LEN_BATCH = 512

DATA_PATH = '/home/julia/Documents/fECG_research/datasets/non-invasive-fetal-ecg-database-1.0.0/'

#%%
training_data, testing_data = data_loader(
                    DATA_PATH, 
                    LEN_BATCH, 
                    QRS_DURATION, 
                    QRS_DURATION_STEP,
                    whole_dataset_training=True,
                    leave_for_testing=0,
                    type_of_file='edf'
            )
    
#%%

prediction = model.predict(training_data[0])

#%%

for i in range(1, 100):
    
    fig, ax = plt.subplots()

    ax.plot(prediction[i])

    ax.plot(training_data[1][i])

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
