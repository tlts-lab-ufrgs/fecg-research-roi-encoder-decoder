"""
Run regression and classification evaluations

"""
#%% Packages 

import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.signal import find_peaks

from data_load.data_loader import DataLoader
from utils.stats_fn import mean_confidence_interval, mae_function
from utils.segments_to_remove_adfecg import to_remove

#%% Constants 

# Testing results

RESULTS_PATH = "/home/julia/Documents/research/sprint_1/results/ablation_extended/"
ABLATION_TEST = '2024-08-13-MASK_gaussian-DECODER_BY_convtransp-ED_rev0-500hz-testelr-LR_0.0001-W_MASK_0.3-W_SIG_0.1'
SAMPLING_FREQ = 500
TYPE_OF_MASK = 'gaussian'


# Data 

DATA_PATH =  "/home/julia/Documents/research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/"
# DATA_PATH = "/home/julia/Documents/research/datasets/b2-records/B2_Labour_dataset/"
TYPE_OF_FILE = 'edf'
NUMBER_OF_FILES = 1



# Constants based on sampling frequency

RESAMPLING_FREQUENCY_RATIO = int(1000 / SAMPLING_FREQ)
LEN_BATCH = int(512 / RESAMPLING_FREQUENCY_RATIO)
LIMT_GAUS = int(30 / RESAMPLING_FREQUENCY_RATIO)
QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = int(50 / RESAMPLING_FREQUENCY_RATIO)
MIN_QRS_DISTANCE = int(200 / RESAMPLING_FREQUENCY_RATIO) # fs = 1000Hz
LIMIT = int(300000 / RESAMPLING_FREQUENCY_RATIO)# - LEN_BATCH



# Evaluation limits 

MASK_MIN_HEIGHT = 0.7


#%% Initialize variables

data_loader = DataLoader(
    DATA_PATH, 
    LEN_BATCH, 
    RESAMPLING_FREQUENCY_RATIO, 
    TYPE_OF_FILE, 
    QRS_DURATION_STEP, 
    QRS_DURATION, 
    load_training_set=False, 
    type_of_mask=TYPE_OF_MASK
)

results_per_file = []

#%%

results_dir = glob.glob(RESULTS_PATH + ABLATION_TEST + '*')

#%% Loop in files to evaluate
 
for file_number in range(NUMBER_OF_FILES):

    prediction_folder =  ABLATION_TEST + f"-LEFT_{file_number}"

    _, testing_data = data_loader.data_load(file_number)
    fr_ann = data_loader.load_rpeak_annotations(file_number)

    # Binary true mask 
    binary_true_mask = np.where(
                testing_data[1][:, :, 1] != 0, 
                1, 
                0
            )
    
    # Add RoI signals into testing_data:
    qrs_signals = np.multiply(
            testing_data[1][:, :, 0], 
            binary_true_mask
    )

    # qrs_signals = np.multiply(
            # testing_data[1][:, :, 0], 
            # testing_data[1][:, :, 1]
    # )



    prediction_files = glob.glob(RESULTS_PATH + prediction_folder + '/' + '*prediction_*')


    # Initialize classification values
    true_positives, false_positives, false_negatives = 0, 0, 0
    mae_signal, mae_roi, mae_qrs = 0,0,0

    random_index_for_plot = [0,5,10,20,30,40,45,55,50,150,180,256]

    # Calculate per segment
    for prediction in prediction_files:
        
        index = int(prediction.split('-prediction_')[1].split('-')[0].replace('.csv', ''))

        if file_number == 4 and int(index * RESAMPLING_FREQUENCY_RATIO) in to_remove:
            continue
 
        # Read prediction:
        prediction_data = pd.read_csv(prediction, names=['signal', 'mask'])
        prediction_data['binary_mask'] = prediction_data['mask'].where(prediction_data['mask'] == 0, 1)
        prediction_data['combined'] = prediction_data['signal'] * binary_true_mask[index, :]


        # # Plot to visualize:

        # if prediction_files.index(prediction) in random_index_for_plot:

            # fig, ax = plt.subplots()

            # ax.set_title(f'File number: {file_number}, segment index: {index}')

            # ax.plot(
            #     testing_data[1][index, :, 0], 
            #     label='Ground truth signal', 
            #     )

            # # ax.plot(
            # #     qrs_signals[index, :], 
            # #     label='Ground truth signal', 
            # #     )
            
            
            # ax.plot(prediction_data['signal'], label='Predicted Signal')
            
            # ax1 = ax.twinx()
            
            # ax1.plot(
            #     testing_data[1][index, :, 1], 
            #     label='Ground truth RoI', 
            #     color='green'
            #     )
            # ax1.plot(prediction_data['mask'], label='Predicted RoI', color='purple')
        
            
            # ax.set_xlabel('Time steps')
            # ax.set_ylabel('fECG normalized')
            # ax1.set_ylabel('RoI signal')
            
            # #Shrink current axis's height by 10% on the bottom
            # box = ax.get_position()
            # ax.set_position([box.x0, box.y0 + box.height * 0.1,
            #                 box.width, box.height * 0.9])

            # #Put a legend below current axis
            # ax.legend(loc='upper center', bbox_to_anchor=(0.1, -0.15),
            #         fancybox=True, shadow=True, ncol=2)
            
            # ax1.legend(loc='upper center', bbox_to_anchor=(0.9, -0.15),
            #         fancybox=True, shadow=True, ncol=2)
            
        
            # ax.grid()


        # Calc MAE values:
        mae_signal_pt = mae_function(testing_data[1][index, :, 0], prediction_data['signal'])
        mae_roi_pt    = mae_function(testing_data[1][index, :, 1], prediction_data['mask'])
        mae_qrs_pt    = mae_function(qrs_signals[index, :], prediction_data['combined'])
        # if len(qrs_signals[index, np.where(qrs_signals[index, :] != 0)[0]]) == 0:
        #     mae_qrs_pt = mae_signal_pt
        # else:    
        #     mae_qrs_pt    = (
        #         np.sum(np.abs((qrs_signals[index, :] - prediction_data['combined']))) / len(qrs_signals[index, np.where(qrs_signals[index, :] != 0)[0]])
        #     )
        
        
        # mae_function(qrs_signals[index, np.where(qrs_signals[index, :] != 0)[0]], prediction_data[prediction_data['combined'] != 0]['combined'])

        mae_signal += mae_signal_pt
        mae_roi += mae_roi_pt
        mae_qrs += mae_qrs_pt

        # Check fQRS annotations:      
        segment_interval = (index * LEN_BATCH, (index + 1) * LEN_BATCH)

        fR_index_on_segment = np.where(
            (fr_ann >= segment_interval[0]) & (fr_ann <= segment_interval[1])
        )[0]

        fR_peaks_pred = find_peaks(
                prediction_data['signal'].values, 
                height=MASK_MIN_HEIGHT, 
                distance=MIN_QRS_DISTANCE
            )[0]
        
        # False negatives
        for j in fR_index_on_segment:

            this_peak = fr_ann[j]

            tolerance_interval = (
                this_peak - int(QRS_DURATION_STEP / 2), 
                this_peak + int(QRS_DURATION_STEP / 2)
            )

            candidates = np.where(
                ((fR_peaks_pred + segment_interval[0]) >= tolerance_interval[0])
                &
                ((fR_peaks_pred + segment_interval[0]) <= tolerance_interval[1])
            )[0] 

            if len(candidates) == 0:
                false_negatives += 1
                print(fR_peaks_pred)
                fig, ax = plt.subplots()

                ax.set_title(f'File number: {file_number}, segment index: {index}, this peak {this_peak}')

                ax.plot(
                    testing_data[1][index, :, 0], 
                    label='Ground truth signal', 
                    )

                # ax.plot(
                #     qrs_signals[index, :], 
                #     label='Ground truth signal', 
                #     )
                
                
                ax.plot(prediction_data['signal'], label='Predicted Signal')
                
                ax1 = ax.twinx()
                
                ax1.plot(
                    testing_data[1][index, :, 1], 
                    label='Ground truth RoI', 
                    color='green'
                    )
                ax1.plot(prediction_data['mask'], label='Predicted RoI', color='purple')
            
                
                ax.set_xlabel('Time steps')
                ax.set_ylabel('fECG normalized')
                ax1.set_ylabel('RoI signal')
                
                #Shrink current axis's height by 10% on the bottom
                box = ax.get_position()
                ax.set_position([box.x0, box.y0 + box.height * 0.1,
                                box.width, box.height * 0.9])

                #Put a legend below current axis
                ax.legend(loc='upper center', bbox_to_anchor=(0.1, -0.15),
                        fancybox=True, shadow=True, ncol=2)
                
                ax1.legend(loc='upper center', bbox_to_anchor=(0.9, -0.15),
                        fancybox=True, shadow=True, ncol=2)
                
            
                ax.grid()
            else:
                true_positives += 1

        # False positives
        false_positives += abs(len(fR_peaks_pred) - len(fR_index_on_segment))

    # Calculate file_number metrics
    f1 = true_positives / (
        true_positives + 0.5 * (false_positives + false_negatives)
    )
    
    recall = true_positives / (
        true_positives + (false_negatives)
    )
    
    precision = true_positives / (
         true_positives + (false_positives)
    )

    # mean value of mae in the subject

    LEN_SEGMENTS = len(prediction_files)

    mae_signal *= 1 / LEN_SEGMENTS
    mae_roi    *= 1 / LEN_SEGMENTS
    mae_qrs    *= 1 / LEN_SEGMENTS


    # results_per_file.append([glob.glob(DATA_PATH + 'B*')[file_number].replace(DATA_PATH, ''), f1, recall, precision, mae_signal, mae_roi, mae_qrs])

    results_per_file.append([file_number, f1, recall, precision, mae_signal, mae_roi, mae_qrs])

#%% Calculate per file and dataset     


metrics_dataframe = pd.DataFrame(
    np.array(results_per_file), 
    columns=[
        'test_file', 
        'f1', 
        'recall', 
        'precision',
        'mae_signal', 
        'mae_mask', 
        'mae_combined', 
        ], 
)

#%%

metrics_dataframe

#%%

print(mean_confidence_interval(metrics_dataframe['mae_signal'].astype(float), 'MAE signal'))
print(mean_confidence_interval(metrics_dataframe['mae_combined'].astype(float), 'MAE qrs'))
print(mean_confidence_interval(metrics_dataframe['mae_mask'].astype(float), 'MAE roi'))
print(mean_confidence_interval(metrics_dataframe['f1'].astype(float), 'f1-score'))
print(mean_confidence_interval(metrics_dataframe['recall'].astype(float), 'recall'))
print(mean_confidence_interval(metrics_dataframe['precision'].astype(float), 'precision'))

#%%

# # remove file number 5

# metrics_dataframe = metrics_dataframe[metrics_dataframe.index != 5]
# # %%

# print('After removing labour 03')


# print(mean_confidence_interval(metrics_dataframe['mae_signal'].astype(float), 'MAE signal'))
# print(mean_confidence_interval(metrics_dataframe['mae_combined'].astype(float), 'MAE qrs'))
# print(mean_confidence_interval(metrics_dataframe['mae_mask'].astype(float), 'MAE roi'))
# print(mean_confidence_interval(metrics_dataframe['f1'].astype(float), 'f1-score'))
# print(mean_confidence_interval(metrics_dataframe['recall'].astype(float), 'recall'))
# print(mean_confidence_interval(metrics_dataframe['precision'].astype(float), 'precision'))


# %%
