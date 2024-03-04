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
import matplotlib.pyplot as plt
from ecgdetectors import panPeakDetect, Detectors
from scipy.signal import find_peaks

from data_load.load_leave_one_out import data_loader
from utils.mean_confidence_interval import mean_confidence_interval

#%% definition for fitting
       
    
def mse_function(y_true, y_pred):
    
    mse_value = np.mean(
        np.power((y_true - y_pred), 2)
    )
    
    return mse_value

#%% constants 

FILES_TO_CALCULATE = '040324-RESAMPLED_SIGNAL_250-LR_0.0001'

# [w_mask, w_signal]
WEIGHTS_TO_EVAL = [
    [0.3, 0.1]
] 

SAMPLING_FREQ = 250

CHANNELS = 3
RESAMPLING_FREQUENCY_RATIO = int(1000 / SAMPLING_FREQ)

RESULTS_PATH = "/home/julia/Documents/fECG_research/research_dev/autoencoder_with_mask/results/"
DATA_PATH =  "/home/julia/Documents/fECG_research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/"

NUMBER_OF_FILES = 5

# All constants are defined based on a 1000Hz fs
LEN_BATCH = int(512 / RESAMPLING_FREQUENCY_RATIO)
LIMT_GAUS = int(50 / RESAMPLING_FREQUENCY_RATIO)
QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = int(50 / RESAMPLING_FREQUENCY_RATIO)
MIN_QRS_DISTANCE = int(300 / RESAMPLING_FREQUENCY_RATIO) # fs = 1000Hz
MASK_MIN_HEIGHT = 0.7

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
                type_of_file='edf', 
                resample_fs=RESAMPLING_FREQUENCY_RATIO
            )

    fecg_testing_data = this_testing_data[1]
    fecg_roi = fecg_testing_data[:, :, 0] * fecg_testing_data[:, :, 1]
    
    testing_data[filenames.index(file)] = {
        'signal': fecg_testing_data,
        'roi_signal': fecg_roi
    }
  

#%% concat results of the same dir

results_dir = glob.glob(RESULTS_PATH + FILES_TO_CALCULATE + '*')

result_qrs = {}

detectors = Detectors(SAMPLING_FREQ) # fs = frequencia de sampling

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
    182,
    183,
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

for w in WEIGHTS_TO_EVAL:

    for j in range(NUMBER_OF_FILES):

        print(j)        
        dir = f'{FILES_TO_CALCULATE}-W_MASK_{w[0]}-W_SIG_{w[1]}-LEFT_{j}'
        
        w_mask = w[0]
        w_signal = w[1]
        
        qrs_detection = []
        pan_tom_qrs_detection = []
        pan_tom_qrs_detection_signal = []

        result_files = glob.glob(RESULTS_PATH + '/' +  dir + '/' + '*prediction_*')
        
        for file in result_files:
            
            prediction_index = int(file.split('-prediction_')[1].split('-')[0].replace('.csv', ''))
        
            
            if j == 4 and int(prediction_index / RESAMPLING_FREQUENCY_RATIO) in to_remove:
                continue
            
            prediction_data = pd.read_csv(file, names=['signal', 'mask'])
            prediction_data['binary_mask'] = prediction_data['mask'].where(prediction_data['mask'] == 0, 1)
            
            prediction_data['combined'] = prediction_data['signal'] * prediction_data['mask']
            prediction_data['combined-binary'] = prediction_data['signal'] * prediction_data['binary_mask']
            # mean, std = norm.fit(prediction_data['mask'])
            
            # Fit the double Gaussian to the data
            
            peaks_proposed = find_peaks(
                prediction_data['mask'].values, 
                height=MASK_MIN_HEIGHT, 
                distance=MIN_QRS_DISTANCE
            )
              
           
            for p in peaks_proposed[0]:
            
                lower_limit_mask = 0 if p - QRS_DURATION_STEP < 0 else p - QRS_DURATION_STEP
                upper_limit_mask = LEN_BATCH if p + QRS_DURATION_STEP > LEN_BATCH else p + QRS_DURATION_STEP

                roi_predicted = prediction_data['mask'][lower_limit_mask : upper_limit_mask]
                
                if roi_predicted.diff().max() < 0.5:
                    qrs_detection.append(int(p + prediction_index * LEN_BATCH))

                
            this_real = fecg_real_data[f'{j}'][prediction_index * LEN_BATCH : prediction_index * LEN_BATCH + LEN_BATCH]

            r_peaks_combined = panPeakDetect(prediction_data['combined-binary'].values, int(SAMPLING_FREQ / 5))
            r_peaks_signal = panPeakDetect(prediction_data['signal'].values, int(SAMPLING_FREQ / 5))
            
            for r in r_peaks_combined:
                pan_tom_qrs_detection.append(r  + prediction_index * LEN_BATCH)
                
            for r in r_peaks_signal:
                pan_tom_qrs_detection_signal.append(r  + prediction_index * LEN_BATCH)
        
        this_weights_results[f'{dir}-proposed'] = qrs_detection
        this_weights_results[f'{dir}-pan-combined'] = pan_tom_qrs_detection
        this_weights_results[f'{dir}-pan-signal'] = pan_tom_qrs_detection_signal


#%% calculate metrics

f1_store = []
recall_store = []
precision_store = []

print('file_id\tf1\tf1_pt\trecall\trecall_pt\tprecision\tprecision_pt\acc')

for j in range(NUMBER_OF_FILES):
    
    dir = f'{FILES_TO_CALCULATE}-W_MASK_{w[0]}-W_SIG_{w[1]}-LEFT_{j}'
    
    true_positive = 0
    false_positive = 0
    false_negative = 0
    total_peaks = 0
    true_positive_peaks = []
    true_positive_peaks_pt = []
    
    true_positive_pt = 0
    false_positive_pt = 0
    false_negative_pt = 0

    limit = 29696# 149760

    for peak in annotations_data[f'{j}'] * SAMPLING_FREQ:
        
        if j == 4 and int(np.floor(peak / LEN_BATCH) * RESAMPLING_FREQUENCY_RATIO) in to_remove:
            continue
        
        if peak <= limit:
            
            total_peaks += 1
        
        
            lower_limit = peak - LIMT_GAUS
            upper_limit = peak + LIMT_GAUS
            
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
            (np.array(already_mentioned_peaks) >= peak - LIMT_GAUS) & 
            (np.array(already_mentioned_peaks) <= peak + LIMT_GAUS)
        )
        
        # this case is specially for roi at the end or beggining of an file
        already_counted_as_true = np.where(
                (np.array(true_positive_peaks) >= lower_limit) & 
                (np.array(true_positive_peaks) <= upper_limit)
            )
        
        if len(already_counted[0]) == 0 and len(already_counted_as_true[0]) == 0:
            lower_limit = peak - LIMT_GAUS
            upper_limit = peak + LIMT_GAUS
                
            peak_found = np.where(
                (np.array(annotations_data[f'{j}'] * SAMPLING_FREQ) >= lower_limit) & 
                (np.array(annotations_data[f'{j}'] * SAMPLING_FREQ) <= upper_limit)
            )
            
            if len(peak_found[0]) == 0:
                
                false_positive += 1
            
        already_mentioned_peaks.append(peak)
    # print(total_peaks)
    
    already_counted = []
    
    for peak in this_weights_results[f'{dir}-pan-combined']:
        
        already_counted = np.where(
            (np.array(already_mentioned_peaks) >= peak - LIMT_GAUS) & 
            (np.array(already_mentioned_peaks) <= peak + LIMT_GAUS)
        )
        
        # this case is specially for roi at the end or beggining of an file
        already_counted_as_true = np.where(
                (np.array(true_positive_peaks_pt) >= lower_limit) & 
                (np.array(true_positive_peaks_pt) <= upper_limit)
            )
        
        if len(already_counted[0]) == 0 and len(already_counted_as_true[0]) == 0:
            lower_limit = peak - LIMT_GAUS
            upper_limit = peak + LIMT_GAUS
                
            peak_found = np.where(
                (np.array(annotations_data[f'{j}'] * SAMPLING_FREQ) >= lower_limit) & 
                (np.array(annotations_data[f'{j}'] * SAMPLING_FREQ) <= upper_limit)
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
    
    precision = true_positive / (
        true_positive + (false_positive)
    )
    
    f1_pt = true_positive_pt / (
        true_positive_pt + 0.5 * (false_positive_pt + false_negative_pt)
    )
    
    recall_pt = true_positive_pt / (
        true_positive_pt + (false_negative_pt)
    )
    
    precision_pt = true_positive_pt / (
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
    precision_store.append(precision)
    
    print(
        '\t'.join(
            [
                f'{j}', 
                f'{f1}', 
                f'{f1_pt}', 
                f'{recall}', 
                f'{recall_pt}', 
                f'{precision}', 
                f'{precision_pt}', 
                f'{acc}'
            ]
        )
    )


#%%

print(mean_confidence_interval(f1_store, name='f1-score'))
print(mean_confidence_interval(recall_store))
print(mean_confidence_interval(precision_store))

# %%
