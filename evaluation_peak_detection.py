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
from scipy.stats import shapiro

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

CHANNELS = 3
LEN_BATCH = 512
QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = 50

TEST_FILE = 4

#%% data load

annotations_data = {}
fecg_real_data = {}
testing_data = {}

filenames = glob.glob(DATA_PATH + '/*.edf')

for file in glob.glob(DATA_PATH + '/*.edf'):

    annotations = mne.read_annotations(file)
    
    file_info = mne.io.read_raw_edf(file)
    filedata = file_info.get_data()
    
    fecg_real_data[f'{filenames.index(file)}'] = filedata[0]
    annotations_data[f'{filenames.index(file)}'] = annotations.onset

    
    _, this_testing_data = data_loader(
                DATA_PATH, 
                LEN_BATCH, 
                QRS_DURATION, 
                QRS_DURATION_STEP,
                leave_for_testing=filenames.index(file),
                type_of_file='edf'
            )

    fecg_testing_data = this_testing_data[1]
    fecg_roi = fecg_testing_data[:, :, 0] * fecg_testing_data[:, :, 1]
    
    
    testing_data[filenames.index(file)] = {
        'signal': fecg_testing_data,
        'roi_signal': fecg_roi
    }
  
#%%

# for file in filenames:
    
#     file_info = mne.io.read_raw_edf(file)
#     filedata = file_info.get_data()

#     fig, ax = plt.subplots(5, 1)
    
#     ax[0].plot(filedata[0], label='fECG')
#     ax[1].plot(filedata[1], label='aECG 1')
#     ax[2].plot(filedata[2], label='aECG 2')
#     ax[3].plot(filedata[3], label='aECG 3')
#     ax[4].plot(filedata[4], label='aECG 4')
    
#     # fig, ax = plt.subplots()
    
#     # ax.plot(filedata)

#%% concat results of the same dir

this_files = '010324-3CH-VAL_LOSS-MOD_DA6-LR_0.0001'

results_dir = glob.glob(RESULTS_PATH + this_files + '*')

result_qrs = {}

detectors = Detectors(1000) # fs = frequencia de sampling

to_remove = [
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    56,
    57,
    58,
    59,
    60,
    177,
    179,
    180,
    181,
    182,183,
    184,
    185,
    186,
    187,
    188,
    189,
    190,
    197,
    198,
    199,
    200,
    201,
    202,
    203,
    204,
    205,
    206,
    207,
    311,
    312,
    313,
    314,
    315,
    316,
    317,
    318,
    319,
    320,
    335, 
    336, 
    338,
    339,
    340,
    341,
    343,
    365,
    371,
    393,
    394,
    395,
    396,
    397,
    398,
    399,
    400,
    401,
    402,
    403,
    404,
    405,
    406,
    407,
    408,
    409,
    410,
    411,
    412,
]

this_weights_results = {}

for w in [
    # [0.3, 0.3], 
    [0.3, 0.1]
    # [0.1, 0.6000000000000001]
]:

    for j in range(5):

        print(j)        
        dir = f'{this_files}-W_MASK_{w[0]}-W_SIG_{w[1]}-LEFT_{j}'
        
        w_mask = w[0]
        w_signal = w[1]
        
        qrs_detection = []
        pan_tom_qrs_detection = []
        pan_tom_qrs_detection_signal = []

        result_files = glob.glob(RESULTS_PATH + '/' +  dir + '/' + '*prediction_*')
        
        for file in result_files:
            
            prediction_index = int(file.split('-prediction_')[1].split('-')[0].replace('.csv', ''))
        
            
            if j == 4 and prediction_index in to_remove:
                continue
            
            prediction_data = pd.read_csv(file, names=['signal', 'mask'])
            prediction_data['binary_mask'] = prediction_data['mask'].where(prediction_data['mask'] == 0, 1)
            
            prediction_data['combined'] = prediction_data['signal'] * prediction_data['mask']
            prediction_data['combined-binary'] = prediction_data['signal'] * prediction_data['binary_mask']
            # mean, std = norm.fit(prediction_data['mask'])
            
            # Fit the double Gaussian to the data
            
            peaks_proposed = find_peaks(prediction_data['mask'].values, height=0.7, distance=300)
              
           
            for p in peaks_proposed[0]:
            
                lower_limit_mask = 0 if p - 50 < 0 else p - 50
                upper_limit_mask = LEN_BATCH if p + 50 > LEN_BATCH else p + 50

                roi_predicted = prediction_data['mask'][lower_limit_mask : upper_limit_mask]
                
                if roi_predicted.diff().max() < 0.5:
                # stat, p_value = shapiro(roi_predicted)

                # print(f'Statistics={stat:.3f}, p-value={p_value:.3f}')

                # # Interpret the p-value
                # alpha = 0.05
                # if p_value > alpha:
                
                    qrs_detection.append(int(p + prediction_index * LEN_BATCH))
                # else:
                #     fig, ax = plt.subplots()
                #     ax.set_title(f'{j} - {p} - {prediction_index}')
                #     ax.plot(roi_predicted)
                #     ax.plot(roi_predicted.diff())
                #     ax.plot(roi_predicted.diff().diff())
                
            this_real = fecg_real_data[f'{j}'][prediction_index * LEN_BATCH : prediction_index * LEN_BATCH + LEN_BATCH]
            
            # prediction_data['combined-not-norm'] = np.max(this_real + np.abs(np.min(this_real))) * (prediction_data['combined-binary']) - np.abs(np.min(this_real))
            # prediction_data['signal-not-norm'] = np.max(this_real + np.abs(np.min(this_real))) * (prediction_data['signal']) - np.abs(np.min(this_real))
            
                
            r_peaks_combined = panPeakDetect(prediction_data['combined-binary'].values, 200)
            r_peaks_signal = panPeakDetect(prediction_data['signal'].values, 200)
            
            for r in r_peaks_combined:
                pan_tom_qrs_detection.append(r  + prediction_index * LEN_BATCH)
                
            for r in r_peaks_signal:
                pan_tom_qrs_detection_signal.append(r  + prediction_index * LEN_BATCH)
        
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
# plt.scatter(pan_tom_qrs_detection, np.zeros(shape=(np.shape(pan_tom_qrs_detection))), marker='.', color='red')
plt.scatter(annotations.onset * 1000, 5e-4 + np.zeros(shape=np.shape(annotations.onset)))

plt.vlines(70, ymin=-5e-4, ymax=5e-4)
plt.vlines(110, ymin=-5e-4, ymax=5e-4)

plt.xlim(0, 1000)

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

f1_store = []
recall_store = []
precision_store = []

print('id\tf1\tf1_pt\trecall\trecall_pt\ts\ts_pt\acc')

for j in [0, 1, 2, 3, 4]:
    
    dir = f'{this_files}-W_MASK_{w[0]}-W_SIG_{w[1]}-LEFT_{j}'
    
    true_positive = 0
    false_positive = 0
    false_negative = 0
    total_peaks = 0
    true_positive_peaks = []
    true_positive_peaks_pt = []
    
    true_positive_pt = 0
    false_positive_pt = 0
    false_negative_pt = 0

    limit = 298976

    for peak in annotations_data[f'{j}'] * 1000:
        
        if j == 4 and int(np.floor(peak / LEN_BATCH)) in to_remove:
            continue
        
        if peak <= limit:
            
            total_peaks += 1
        
        
            lower_limit = peak - 30
            upper_limit = peak + 30
            
            peak_found = np.where(
                (np.array(this_weights_results[f'{dir}-proposed']) >= lower_limit) & 
                (np.array(this_weights_results[f'{dir}-proposed']) <= upper_limit)
            )
            
            peak_found_pt = np.where(
                (np.array(this_weights_results[f'{dir}-pan-combined']) >= lower_limit) & 
                (np.array(this_weights_results[f'{dir}-pan-combined']) <= upper_limit)
            )
            
            if len(peak_found[0]) > 0:
                true_positive_peaks.append(peak_found[0][0])
                true_positive += 1
            else:
                false_negative += 1
                
            if len(peak_found_pt[0]) > 0:
                true_positive_peaks_pt.append(peak_found_pt[0][0])
                true_positive_pt += 1
            else:
                false_negative_pt += 1
    
    already_mentioned_peaks = []  
    
    for peak in this_weights_results[f'{dir}-proposed']:
        
        already_counted = np.where(
            (np.array(already_mentioned_peaks) >= peak - 50) & 
            (np.array(already_mentioned_peaks) <= peak + 50)
        )
        
        # this case is specially for roi at the end or beggining of an file
        already_counted_as_true = np.where(
                (np.array(true_positive_peaks) >= lower_limit) & 
                (np.array(true_positive_peaks) <= upper_limit)
            )
        
        if len(already_counted[0]) == 0 and len(already_counted_as_true[0]) == 0:
            lower_limit = peak - 30
            upper_limit = peak + 30
                
            peak_found = np.where(
                (np.array(annotations_data[f'{j}'] * 1000) >= lower_limit) & 
                (np.array(annotations_data[f'{j}'] * 1000) <= upper_limit)
            )
            
            if len(peak_found[0]) == 0:
                
                false_positive += 1
            
        already_mentioned_peaks.append(peak)
    # print(total_peaks)
    
    already_counted = []
    
    for peak in this_weights_results[f'{dir}-pan-combined']:
        
        already_counted = np.where(
            (np.array(already_mentioned_peaks) >= peak - 50) & 
            (np.array(already_mentioned_peaks) <= peak + 50)
        )
        
        # this case is specially for roi at the end or beggining of an file
        already_counted_as_true = np.where(
                (np.array(true_positive_peaks_pt) >= lower_limit) & 
                (np.array(true_positive_peaks_pt) <= upper_limit)
            )
        
        if len(already_counted[0]) == 0 and len(already_counted_as_true[0]) == 0:
            lower_limit = peak - 30
            upper_limit = peak + 30
                
            peak_found = np.where(
                (np.array(annotations_data[f'{j}'] * 1000) >= lower_limit) & 
                (np.array(annotations_data[f'{j}'] * 1000) <= upper_limit)
            )
            
            if len(peak_found[0]) == 0:
                
                false_positive_pt += 1
            
        already_mentioned_peaks.append(peak)
    # print(total_peaks)

    f1 = true_positive / (
        true_positive + 0.5 * (false_positive + false_negative)
    )
    
    recall = true_positive / (
        true_positive + (false_negative)
    )
    
    s = true_positive / (
        true_positive + (false_positive)
    )
    
    f1_pt = true_positive_pt / (
        true_positive_pt + 0.5 * (false_positive_pt + false_negative_pt)
    )
    
    recall_pt = true_positive_pt / (
        true_positive_pt + (false_negative_pt)
    )
    
    s_pt = true_positive_pt / (
     true_positive_pt + (false_positive_pt)
    )
    
    acc = true_positive / (
        (total_peaks + false_positive)
    )
    
    acc_pt = true_positive_pt / (
        (total_peaks + false_positive)
    )
    
    f1_store.append(f1)
    recall_store.append(recall)
    precision_store.append(s)
    
    print(
        '\t'.join(
            [
                f'{j}', 
                f'{f1}', 
                f'{f1_pt}', 
                f'{recall}', 
                f'{recall_pt}', 
                f'{s}', 
                f'{s_pt}', 
                f'{acc}'
            ]
        )
    )
# %%

import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.norm.ppf((1 + confidence) / 2.)
    return m, se, m-h, m+h


#%%

print(mean_confidence_interval(f1_store))
print(mean_confidence_interval(recall_store))
print(mean_confidence_interval(precision_store))

# %%
