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

from data_load.data_loader import DataLoader
# from utils.mean_confidence_interval import mean_confidence_interval

#%% constants 

FILES_TO_CALCULATE = '2024-08-15-MASK_gaussian-DECODER_BY_convtranspose-ED_rev0-B2-500hz-LR_0.0001'

# [w_mask, w_signal]
WEIGHTS_TO_EVAL = [
    [0.3, 0.1]
] 

SAMPLING_FREQ = 500

CHANNELS = 3
RESAMPLING_FREQUENCY_RATIO = int(1000 / SAMPLING_FREQ)

RESULTS_PATH = "/home/julia/Documents/research/sprint_1/results/ablation_extended/"
DATA_PATH = "/home/julia/Documents/research/datasets/b2-records/B2_Labour_dataset/"

NUMBER_OF_FILES = 12

# All constants are defined based on a 1000Hz fs
LEN_BATCH = int(512 / RESAMPLING_FREQUENCY_RATIO)
LIMT_GAUS = int(50 / RESAMPLING_FREQUENCY_RATIO)
QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = int(50 / RESAMPLING_FREQUENCY_RATIO)
MIN_QRS_DISTANCE = int(200 / RESAMPLING_FREQUENCY_RATIO) # fs = 1000Hz
MASK_MIN_HEIGHT = 0.7

LIMIT = int(300000 / RESAMPLING_FREQUENCY_RATIO) - LEN_BATCH

type_of_mask = 'gaussian'

#%% 

data_loader = DataLoader(
    DATA_PATH, 
    LEN_BATCH, 
    RESAMPLING_FREQUENCY_RATIO, 
    'txt', 
    QRS_DURATION_STEP, 
    QRS_DURATION, 
    load_training_set=False,
    type_of_mask=type_of_mask, 
)

#%% concat results of the same dir

results_dir = glob.glob(RESULTS_PATH + FILES_TO_CALCULATE + '*')

result_qrs = {}

detectors = Detectors(SAMPLING_FREQ) # fs = frequencia de sampling

this_weights_results = {}

for w in WEIGHTS_TO_EVAL:

    for j in range(NUMBER_OF_FILES):

        concat = np.empty(shape=0)
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
            
            prediction_data = pd.read_csv(file, names=['signal', 'mask'])
            prediction_data['binary_mask'] = prediction_data['mask'].where(prediction_data['mask'] == 0, 1)
            prediction_data['combined'] = prediction_data['signal'] * prediction_data['mask']
            prediction_data['combined-binary'] = prediction_data['signal'] * prediction_data['binary_mask']
            # mean, std = norm.fit(prediction_data['mask'])
            
            concat = np.concatenate([concat, prediction_data['signal']])
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
                
                #if roi_predicted.diff().max() < 0.5:
                qrs_detection.append(int(p + prediction_index * LEN_BATCH))

                
            r_peaks_combined = detectors.pan_tompkins_detector(prediction_data['mask'].values)
            r_peaks_signal = detectors.pan_tompkins_detector(prediction_data['signal'].values)

            # r_peaks_combined = panPeakDetect(prediction_data['combined-binary'].values, SAML)
            # r_peaks_signal = panPeakDetect(prediction_data['signal'].values, SAML)
            
            for r in r_peaks_combined:
                pan_tom_qrs_detection.append(r  + prediction_index * LEN_BATCH)
                
            for r in r_peaks_signal:
                pan_tom_qrs_detection_signal.append(r  + prediction_index * LEN_BATCH)
        

        # test passing all the signal to the pan and tompikins algo
        pt_detection = detectors.pan_tompkins_detector(concat)


        this_weights_results[f'{dir}-proposed'] = qrs_detection
        this_weights_results[f'{dir}-pan-combined'] = pan_tom_qrs_detection
        this_weights_results[f'{dir}-pan-signal'] = pt_detection

#%% calculate metrics

f1_store = []
recall_store = []
precision_store = []

f1_store_pt = []
recall_store_pt = []
precision_store_pt = []

print('file_id\tf1\tf1_pt\trecall\trecall_pt\tprecision\tprecision_pt\acc')

for j in range(NUMBER_OF_FILES):
    
    dir = f'{FILES_TO_CALCULATE}-W_MASK_{w[0]}-W_SIG_{w[1]}-LEFT_{j}'

    _, testing_data = data_loader.data_load(j)
    fr_ann = data_loader.load_rpeak_annotations(j)

    # print(dir)
    
    true_positive = 0
    false_positive = 0
    false_negative = 0
    total_peaks = 0
    true_positive_peaks = []
    true_positive_peaks_pt = []
    
    true_positive_pt = 0
    false_positive_pt = 0
    false_negative_pt = 0

    for peak in fr_ann:
        
        if peak <= LIMIT:
            
            total_peaks += 1
        
            lower_limit = peak - LIMT_GAUS
            upper_limit = peak + LIMT_GAUS
            
            peak_found = np.where(
                (np.array(this_weights_results[f'{dir}-proposed']) >= lower_limit) & 
                (np.array(this_weights_results[f'{dir}-proposed']) <= upper_limit)
            )
            
            peak_found_pt = np.where(
                (np.array(this_weights_results[f'{dir}-pan-signal']) >= lower_limit) & 
                (np.array(this_weights_results[f'{dir}-pan-signal']) <= upper_limit)
            )
            
            if len(peak_found[0]) > 0:
                for k in peak_found[0]:
                    true_positive_peaks.append(k)
                    
                true_positive += 1
            else:
                # print('False negative', peak, peak / LEN_BATCH)
                false_negative += 1
                
            if len(peak_found_pt[0]) > 0:
                for k in peak_found_pt[0]:
                    true_positive_peaks_pt.append(k)
                true_positive_pt += 1
            else:
                # print('False negative', j, peak, peak / LEN_BATCH)
                false_negative_pt += 1
    
    for peak_predicted in this_weights_results[f'{dir}-proposed']:
        if peak_predicted not in true_positive_peaks and peak_predicted <= LIMIT:
            
            possible_ann = np.where(
                (fr_ann >= peak_predicted - LIMT_GAUS) &
                (fr_ann <= peak_predicted + LIMT_GAUS)
            )[0]

            if len(possible_ann) == 0:
                # print('False positive', peak_predicted, peak_predicted / LEN_BATCH)
                false_positive += 1
                
    for peak_predicted in this_weights_results[f'{dir}-pan-signal']:
        if peak_predicted not in true_positive_peaks_pt and peak_predicted <= LIMIT:
            
            possible_ann = np.where(
                (fr_ann >= peak_predicted - LIMT_GAUS) &
                (fr_ann <= peak_predicted + LIMT_GAUS)
            )[0]

            if len(possible_ann) == 0:

                false_positive_pt += 1



    f1 = true_positive / (
        true_positive + 0.5 * (false_positive + false_negative)
    )
    
    recall = true_positive / (
        true_positive + (false_negative)
    )
    
    try:
        precision = true_positive / (
             true_positive + (false_positive)
        )
    except:
        precision = 0

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

    f1_store_pt.append(f1_pt)
    recall_store_pt.append(recall_pt)
    precision_store_pt.append(precision_pt)
    
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

from utils.stats_fn import mean_confidence_interval

print(mean_confidence_interval(f1_store, name='f1-score'))
print(mean_confidence_interval(recall_store, name='recall'))
print(mean_confidence_interval(precision_store, name='precision'))

# %%

print('Without the problematic')

f1_store.pop(5)
recall_store.pop(5)
precision_store.pop(5)

print(mean_confidence_interval(f1_store, name='f1-score'))
print(mean_confidence_interval(recall_store, name='recall'))
print(mean_confidence_interval(precision_store, name='precision'))

# %%


print(mean_confidence_interval(f1_store_pt, name='f1-score'))
print(mean_confidence_interval(recall_store_pt, name='recall'))
print(mean_confidence_interval(precision_store_pt, name='precision'))
#%%

print('Without the problematic')

f1_store_pt.pop(5)
recall_store_pt.pop(5)
precision_store_pt.pop(5)


print(mean_confidence_interval(f1_store_pt, name='f1-score'))
print(mean_confidence_interval(recall_store_pt, name='recall'))
print(mean_confidence_interval(precision_store_pt, name='precision'))
# %%
