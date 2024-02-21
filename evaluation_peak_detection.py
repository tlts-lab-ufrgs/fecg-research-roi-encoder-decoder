"""
Evaluate results from loop in hyperparameters

18 februaty 2024
"""


#%% import 

import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import norm
import matplotlib.pyplot as plt

from data_load.load_leave_one_out import data_loader

    
#%% constants 


RESULTS_PATH = "/home/julia/Documents/fECG_research/research_dev/autoencoder_with_mask/results/"
DATA_PATH =  "/home/julia/Documents/fECG_research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/"

CHANNELS = 4
LEN_BATCH = 512
QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = 100

TEST_FILE = 4

#%%

def mse_function(y_true, y_pred):
    
    mse_value = np.mean(
        np.power((y_true - y_pred), 2)
    )
    
    return mse_value

#%% data load

_, testing_data = data_loader(
                DATA_PATH, 
                LEN_BATCH, 
                QRS_DURATION, 
                QRS_DURATION_STEP,
                leave_for_testing=TEST_FILE,
                type_of_file='edf'
            )

fecg_testing_data = testing_data[1]
fecg_roi = fecg_testing_data[:, :, 0] * fecg_testing_data[:, :, 1]
#%% concat results of the same dir

results_dir = glob.glob(RESULTS_PATH + 'LR*')
results_rows = []

for i in [
    [0.0, 0.7], [0.2, 0.2], [0.1, 0.1, 0.3, 0.4]
]:
    
    dir = f'LR_0.001-W_MASK_{i[0]}-W_SIG_{i[1]}-LEFT_4'
    
    w_mask = i[0]
    w_signal = i[1]
    
    this_row = []

    result_files = glob.glob(dir + '/' + '*prediction_*')
    
    mse_signal, mse_mask, mse_combined = 0, 0, 0
    
    for file in result_files:
        
        prediction_index = int(file.split('-prediction_')[1].split('-')[0].replace('.csv', ''))
        
        prediction_data = pd.read_csv(file, names=['signal', 'mask'])
        prediction_data['combined'] = prediction_data['signal'] * prediction_data['mask']
        
        mean, std = norm.fit(prediction_data['mask'])
        
        this_row.append(mean)

    
    results_rows.append(this_row)

#%% form data frame

metrics_dataframe = pd.DataFrame(
    np.array(results_rows), columns=['w_mask', 'w_signal', 'mse_signal', 'mse_mask', 'mse_combined']
)

#%%

metrics_dataframe.sort_values(by = 'mse_mask', inplace=True)

#%%

import seaborn as sns

sns.heatmap(metrics_dataframe[['w_mask', 'w_signal']])
# %%


