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


#%% import 

import os
import mne
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from ecgdetectors import panPeakDetect, Detectors
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
QRS_DURATION_STEP = 50

TEST_FILE = 4

#%% data load

annotations_data = {}
fecg_real_data = {}

filenames = glob.glob(DATA_PATH + '/*.edf')

for file in glob.glob(DATA_PATH + '/*.edf'):

    annotations = mne.read_annotations(file)
    
    file_info = mne.io.read_raw_edf(file)
    filedata = file_info.get_data()
    
    fecg_real_data[f'{filenames.index(file)}'] = filedata[0]
    annotations_data[f'{filenames.index(file)}'] = annotations.onset
  
#%%

for file in filenames:
    
    file_info = mne.io.read_raw_edf(file)
    filedata = file_info.get_data()

    fig, ax = plt.subplots(5, 1)
    
    ax[0].plot(filedata[0], label='fECG')
    ax[1].plot(filedata[1], label='aECG 1')
    ax[2].plot(filedata[2], label='aECG 2')
    ax[3].plot(filedata[3], label='aECG 3')
    ax[4].plot(filedata[4], label='aECG 4')
    
    # fig, ax = plt.subplots()
    
    # ax.plot(filedata)
    
#%%

testing_data = {}

for i in range(5):
    
    _, this_testing_data = data_loader(
                DATA_PATH, 
                LEN_BATCH, 
                QRS_DURATION, 
                QRS_DURATION_STEP,
                leave_for_testing=i,
                type_of_file='edf'
            )

    fecg_testing_data = this_testing_data[1]
    fecg_roi = fecg_testing_data[:, :, 0] * fecg_testing_data[:, :, 1]
    
    
    testing_data[i] = {
        'signal': fecg_testing_data,
        'roi_signal': fecg_roi
    }

#%% concat results of the same dir

results_dir = glob.glob(RESULTS_PATH + 'QRStime_0.1-LR_0.0001*')

result_qrs = {}

detectors = Detectors(1000) # fs = frequencia de sampling


this_weights_results = {}

for i in [
    # [0.3, 0.30000000000000004], 
    [0.2, 0.1]
    # [0.1, 0.6000000000000001]
]:
    
    for j in range(5):
    
        dir = f'QRStime_0.1-LR_0.0001-W_MASK_{i[0]}-W_SIG_{i[1]}-LEFT_{j}'
        
        w_mask = i[0]
        w_signal = i[1]
        
        qrs_detection = []
        pan_tom_qrs_detection = []
        pan_tom_qrs_detection_signal = []

        result_files = glob.glob(RESULTS_PATH + '/' +  dir + '/' + '*prediction_*')
        
        for file in result_files:
            
            prediction_index = int(file.split('-prediction_')[1].split('-')[0].replace('.csv', ''))
            
            prediction_data = pd.read_csv(file, names=['signal', 'mask'])
            prediction_data['binary_mask'] = prediction_data['mask'].where(prediction_data['mask'] == 0, 1)
            
            prediction_data['combined'] = prediction_data['signal'] * prediction_data['mask']
            prediction_data['combined-binary'] = prediction_data['signal'] * prediction_data['binary_mask']
            # mean, std = norm.fit(prediction_data['mask'])
            
            # Fit the double Gaussian to the data
            
            peaks_proposed = find_peaks(prediction_data['mask'].values, height=0.7, distance=50)
                       
                      
            # if len(peaks_proposed[0]) == 0:
            #     print(prediction_index)
                
            #     fig, ax = plt.subplots()
                
            #     ax.set_title(f'{prediction_index} - file {j}')
            #     ax.plot(prediction_data['mask'])
            #     ax.plot(testing_data[j]['signal'][prediction_index], label='fECG')
            
            if len(peaks_proposed[0]) == 0:
                print(prediction_index)
            
            for p in peaks_proposed[0]:
                qrs_detection.append(int(p + prediction_index * 512))
                
            this_real = fecg_real_data[f'{j}'][prediction_index * 512 : prediction_index * 512 + 512]
            
            prediction_data['combined-not-norm'] = np.max(this_real + np.abs(np.min(this_real))) * (prediction_data['combined-binary']) - np.abs(np.min(this_real))
            prediction_data['signal-not-norm'] = np.max(this_real + np.abs(np.min(this_real))) * (prediction_data['signal']) - np.abs(np.min(this_real))
            
                
            r_peaks_combined = panPeakDetect(prediction_data['combined-not-norm'].values, 200)
            r_peaks_signal = panPeakDetect(prediction_data['signal-not-norm'].values, 200)

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

file_info = mne.io.read_raw_edf(filenames[1])
filedata = file_info.get_data()

plt.plot(filedata[0], alpha=0.5)
plt.scatter(qrs_detection, np.zeros(shape=(np.shape(qrs_detection))), marker='x', color='black')
plt.scatter(pan_tom_qrs_detection, np.zeros(shape=(np.shape(pan_tom_qrs_detection))), marker='.', color='red')
plt.scatter(annotations.onset * 1000, 5e-4 + np.zeros(shape=np.shape(annotations.onset)))

plt.vlines(70, ymin=-5e-4, ymax=5e-4)
plt.vlines(110, ymin=-5e-4, ymax=5e-4)

plt.xlim(10000, 20000)

#%%

print('id\tf1_pt\tacc_pt')

for i in range(5):
    
    r_peaks = detectors.pan_tompkins_detector(fecg_real_data[f'{i}'])
    
    true_positive = 0
    false_positive = 0
    false_negative = 0
    total_peaks = 0
    
    true_positive_pt = 0
    false_positive_pt = 0
    false_negative_pt = 0

    limit = 262144

    for peak in annotations_data[f'{i}'] * 1000:
        
        if peak <= limit:
            
            total_peaks += 1
        
        
            lower_limit = peak - 30
            upper_limit = peak + 30
            
            
            peak_found_pt = np.where(
                (np.array(r_peaks >= lower_limit)) & 
                (np.array(r_peaks <= upper_limit))
            )
            
                
            if len(peak_found_pt[0]) > 0:
            
                true_positive_pt += 1
            else:
                false_negative_pt += 1
        


    f1_pt = true_positive_pt / (
        true_positive_pt + 0.5 * (false_positive_pt + false_negative_pt)
    )
    
    
    acc_pt = true_positive_pt / (
        (total_peaks)
    )
    
    print(
        '\t'.join(
            [
                f'{i}', 
                f'{f1_pt}', 
                f'{acc_pt}'
            ]
        )
    )

#%%

print('id\tf1\tf1_pt\tacc\tacc_pt')

for j in range(5):
    
    dir = f'QRStime_0.1-LR_0.0001-W_MASK_{i[0]}-W_SIG_{i[1]}-LEFT_{j}'
    
    true_positive = 0
    false_positive = 0
    false_negative = 0
    total_peaks = 0
    
    true_positive_pt = 0
    false_positive_pt = 0
    false_negative_pt = 0

    limit = 262144

    for peak in annotations_data[f'{j}'] * 1000:
        
        if peak <= limit:
            
            total_peaks += 1
        
        
            lower_limit = peak - 30
            upper_limit = peak + 30
            
            peak_found = np.where(
                (np.array(this_weights_results[f'{dir}-proposed']) >= lower_limit) & 
                (np.array(this_weights_results[f'{dir}-proposed']) <= upper_limit)
            )
            
            peak_found_pt = np.where(
                (np.array(this_weights_results[f'{dir}-pan-signal']) >= lower_limit) & 
                (np.array(this_weights_results[f'{dir}-pan-signal']) <= upper_limit)
            )
            
            if len(peak_found[0]) > 0:
            
                true_positive += 1
            else:
                false_negative += 1
                
            if len(peak_found_pt[0]) > 0:
            
                true_positive_pt += 1
            else:
                false_negative_pt += 1
        


    f1 = true_positive / (
        true_positive + 0.5 * (false_positive + false_negative)
    )
    
    f1_pt = true_positive_pt / (
        true_positive_pt + 0.5 * (false_positive_pt + false_negative_pt)
    )
    
    acc = true_positive / (
        (total_peaks)
    )
    
    acc_pt = true_positive_pt / (
        (total_peaks)
    )
    
    print(
        '\t'.join(
            [
                f'{j}', 
                f'{f1}', 
                f'{f1_pt}', 
                f'{acc}', 
                f'{acc_pt}'
            ]
        )
    )

#%%

for i in range(5):
    
    fig, ax = plt.subplots()
    
    ax.plot(fecg_real_data[f'{i}'])
    
    mean = np.mean(fecg_real_data[f'{i}'])
    std = np.std(fecg_real_data[f'{i}'])
    
    ax.hlines(y=mean, xmin = 0,xmax = 3e5, color='black')
    ax.hlines(y=mean + std, xmin = 0,xmax = 3e5, color='black')
    ax.hlines(y=mean - std, xmin = 0,xmax = 3e5, color='black')

#%%
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
