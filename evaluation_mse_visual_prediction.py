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
import matplotlib.pyplot as plt

from data_load.load_leave_one_out import data_loader
    
#%% constants 


RESULTS_PATH = "/home/julia/Documents/fECG_research/research_dev/autoencoder_with_mask/results/"
DATA_PATH =  "/home/julia/Documents/fECG_research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/"

CHANNELS = 4
LEN_BATCH = 512
QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = 50

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

results_dir = glob.glob(RESULTS_PATH + 'RP*')
results_rows = []

for i in results_dir:
    
    w_mask = float(i.split('-W_MASK_')[1].split('-')[0])
    w_signal = float(i.split('-W_SIG_')[1].split('-')[0])
    
    this_row = [w_mask, w_signal]

    result_files = glob.glob(i + '/' + '*prediction_*')
    
    mse_signal, mse_mask, mse_combined = 0, 0, 0
    
    for file in result_files:
        
        prediction_index = int(file.split('-prediction_')[1].split('-')[0].replace('.csv', ''))
        
        prediction_data = pd.read_csv(file, names=['signal', 'mask'])
        prediction_data['combined'] = prediction_data['signal'] * prediction_data['mask']
        
        mse_signal_partial = mse_function(fecg_testing_data[prediction_index, :, 0], prediction_data['signal'])
        mse_mask_partial = mse_function(fecg_testing_data[prediction_index, :, 1], prediction_data['mask'])
        mse_combined_partial = mse_function(fecg_roi[prediction_index], prediction_data['combined'])
           
        mse_signal += mse_signal_partial
        mse_mask += mse_mask_partial
        mse_combined += mse_combined_partial
        
        if prediction_index in [89]:
            
            fig, ax = plt.subplots()
            
            ax.set_title(f'W mask {w_mask}, W signal {w_signal} - {i.split("-")[-1]}')
            
            ax.plot(fecg_testing_data[prediction_index], label='fECG')
            ax.plot(prediction_data['signal'], label='Model Signal')
            ax.plot(prediction_data['mask'], label='Model Mask')
            
            # ax.plot(fecg_roi[prediction_index], label='fECG')
            # ax.plot(prediction_data['combined'], label='Model Signal')
            
            ax.legend()
       
    this_row.append(mse_signal / len(result_files))
    this_row.append(mse_mask / len(result_files))
    this_row.append(mse_combined / len(result_files))
    
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


