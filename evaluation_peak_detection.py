"""
Evaluate results - QRS detection 

18 februaty 2024
"""

"""
- loop in  all files
    - fit a norm distribution in the segments
    - get the mean of this 
    - add into a list
    - append segments in a full list
    
- pass into pan and tom
- evaluate how different from ground truth


"""


# #%% import 

import os
import mne
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from ecgdetectors import Detectors
from scipy.signal import find_peaks

from data_load.load_leave_one_out import data_loader

#%% definition for fitting

# Define the function for a Gaussian distribution
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)

# Define the function for the sum of two Gaussians
def double_gaussian(x, amplitude1, mean1, stddev1, amplitude2, mean2, stddev2):
    return (
        gaussian(x, amplitude1, mean1, stddev1) +
        gaussian(x, amplitude2, mean2, stddev2)
    )
    
def single_gaussian(x, amplitude1, mean1, stddev1):
    return (
        gaussian(x, amplitude1, mean1, stddev1)
    )
        
    
def mse_function(y_true, y_pred):
    
    mse_value = np.mean(
        np.power((y_true - y_pred), 2)
    )
    
    return mse_value
    
#%% constants 


RESULTS_PATH = "/home/julia/Documents/fECG_research/research_dev/autoencoder_with_mask/results/"
DATA_PATH =  "/home/julia/Documents/fECG_research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/"

CHANNELS = 4
LEN_BATCH = 512
QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = 100

TEST_FILE = 4

#%% data load

#%% data load

annotations_data = {}

filenames = glob.glob(DATA_PATH + '/*.edf')

for file in glob.glob(DATA_PATH + '/*.edf'):

    annotations = mne.read_annotations(file)
    
    annotations_data[f'{filenames.index(file)}'] = annotations.onset

#%% concat results of the same dir

results_dir = glob.glob(RESULTS_PATH + 'LR*')

result_qrs = {}

detectors = Detectors(1000) # fs = frequencia de sampling

this_weights_results = {}

for i in [
    [0.1, 0.7000000000000001], 
    [0.1, 0.6000000000000001]
]:
    
    for j in range(5):
    
        dir = f'QRStime_0.1-LR_0.0001-W_MASK_{i[0]}-W_SIG_{i[1]}-LEFT_{j}'
        
        w_mask = i[0]
        w_signal = i[1]
        
        qrs_detection = []
        pan_tom_qrs_detection = []
        pan_tom_qrs_detection_signal = []

        result_files = glob.glob(RESULTS_PATH + '/' +  dir + '/' + '*prediction_*')
        
        mse_signal, mse_mask, mse_combined = 0, 0, 0
        
       
        for file in result_files:
            
            prediction_index = int(file.split('-prediction_')[1].split('-')[0].replace('.csv', ''))
            
            prediction_data = pd.read_csv(file, names=['signal', 'mask'])
            prediction_data['combined'] = prediction_data['signal'] * prediction_data['mask']
            
            # mean, std = norm.fit(prediction_data['mask'])
            
            # Fit the double Gaussian to the data
            
            peaks_proposed = find_peaks(prediction_data['mask'].values, height=0.8, distance=50)
            
            for p in peaks_proposed[0]:
                qrs_detection.append(int(p + prediction_index * 512))
        
                
            r_peaks_combined = detectors.pan_tompkins_detector(prediction_data['combined'].values)
            r_peaks_signal = detectors.pan_tompkins_detector(prediction_data['signal'].values)

            for r in r_peaks_combined:
                pan_tom_qrs_detection.append(r  + prediction_index * 512)
                
            for r in r_peaks_signal:
                pan_tom_qrs_detection_signal.append(r  + prediction_index * 512)
        
        this_weights_results[f'{dir}-proposed'] = qrs_detection
        this_weights_results[f'{dir}-pan-combined'] = pan_tom_qrs_detection
        this_weights_results[f'{dir}-pan-signal'] = pan_tom_qrs_detection_signal

#%% form data frame

# peaks_qrs = [round(qrs_detection[i], 0) for i in qrs_detection]
# peaks_qrs_pan = [round(pan_tom_qrs_detection[i], 0) for i in pan_tom_qrs_detection]

#%%

file_info = mne.io.read_raw_edf(filenames[-1])
filedata = file_info.get_data()

peaks_qrs = [filedata[0][int(i)] for i in qrs_detection]
peaks_qrs_pan = [filedata[0][int(i)] for i in pan_tom_qrs_detection]



#%%

# plt.plot(np.sort(np.array(peaks_qrs)))

# plt.plot(np.sort(np.array(pan_tom_qrs_detection)))


plt.plot(filedata[0], alpha=0.5)
plt.scatter(qrs_detection, np.zeros(shape=(np.shape(qrs_detection))), marker='x')
plt.scatter(pan_tom_qrs_detection, np.zeros(shape=(np.shape(pan_tom_qrs_detection))), marker='.')


plt.xlim(10000, 20000)
annotations.onset
# plt.plot(pan_tom_qrs_detection)

# metrics_dataframe = pd.DataFrame(
#     np.array(results_rows), columns=['w_mask', 'w_signal', 'mse_signal', 'mse_mask', 'mse_combined']
# )

# #%%

# metrics_dataframe.sort_values(by = 'mse_mask', inplace=True)

# #%%

# import seaborn as sns

# sns.heatmap(metrics_dataframe[['w_mask', 'w_signal']])
# # %%



# %%
