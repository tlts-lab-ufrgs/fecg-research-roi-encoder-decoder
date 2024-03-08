
#%%
import glob
import wfdb
import mne
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data_load.load_leave_one_out import data_loader

from models.ae_proposed import Metric, Loss
from utils.gaussian_function import gaussian
from data_load.load_leave_one_out import data_resizer
from utils.lr_scheduler import callback as lr_scheduler
from scipy.signal import find_peaks

from scipy.signal import resample
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
from scipy.io import loadmat

from utils.mean_confidence_interval import mean_confidence_interval

#%%

model = tf.keras.models.load_model(
    '/home/julia/Documents/fECG_research/research_dev/autoencoder_with_mask/final_model_3ch/', 
    custom_objects = {
        'mse_mask': Metric.mse_mask,
        'mse_signal': Metric.mse_signal, 
        'loss': Loss.loss, 
        'lr': lr_scheduler
    }
)

#%% constants

# Range in learning rate
UPPER_LIM_LR = 0.0001

SAMPLING_FREQ = 1000

RESAMPLE_FREQ_RATIO = 1

# batch size
BATCH_SIZE=4

# files 
FILES_TO_READ = [154, 192, 244, 274, 290, 323, 368, 444, 597, 733, 746, 811, 826, 906,]

RESULTS_PATH = "/home/julia/Documents/fECG_research/research_dev/autoencoder_with_mask/results/"
DATA_PATH =  "/home/julia/Documents/fECG_research/datasets/non-invasive-fetal-ecg-database-1.0.0/"

CHANNELS = 3
LEN_BATCH = 512
QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = 50

RESAMPLING_FREQUENCY_RATIO = 1


LEN_BATCH = int(512 / RESAMPLING_FREQUENCY_RATIO)
LIMT_GAUS = int(50 / RESAMPLING_FREQUENCY_RATIO)
QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = int(50 / RESAMPLING_FREQUENCY_RATIO)
MIN_QRS_DISTANCE = int(300 / RESAMPLING_FREQUENCY_RATIO) # fs = 1000Hz
MASK_MIN_HEIGHT = 0.8

LIMIT = int(300000 / RESAMPLING_FREQUENCY_RATIO)# - LEN_BATCH

#%%

def mae_function(y_true, y_pred):
    
    mse_value = np.mean(
        np.abs((y_true - y_pred)) # , 2)
    )
    
    return mse_value

#%%

filenames = glob.glob(DATA_PATH + '*.edf')
filenames = [i for i in filenames if int(i.split('ecgca')[-1].replace('.edf', '')) in FILES_TO_READ ]


global_f1_score = []
global_recall = []
global_precision = []
global_mae_signal = []
global_mae_mask = []
global_mae_roi = []

for file in filenames:
    
    # Get data
    
    aECG, fECG =  data_resizer(    
        [file],
        LEN_BATCH, 
        QRS_DURATION, 
        QRS_DURATION_STEP,
        type_of_file='edf', 
        resample_fs=1, 
        channels=CHANNELS, 
        fecg_on_gt=False
    )
    
    annotations =  mne.read_annotations(file)
    time_annotations = annotations.onset

    # Model prediction

    predict = model.predict(aECG)
    
    # ----------------- fECG extraction assessment
    
    mse_signal_partial = 0
    mse_mask_partial = 0
    mse_combined_partial = 0
    
    for i in range(np.shape(predict)[0]):
        binary_mask = np.where(
            fECG[i, :, 1] != 0, 
            1, 
            0
        )
        roi_true_signal = fECG[i, :, 0] * binary_mask
        
        binary_predicted_mask = np.where(
            predict[i, :, 1] != 0, 
            1, 
            0
        )
        roi_predicted_signal = predict[i, :, 0] * binary_predicted_mask
        
        mse_signal_partial += mae_function(fECG[i, :, 0], predict[i, :, 0])
        mse_mask_partial += mae_function(fECG[i, :, 1], predict[i, :, 1])
        mse_combined_partial += mae_function(
            roi_true_signal, 
            roi_predicted_signal)
            
    global_mae_signal.append(mse_signal_partial / np.shape(predict)[0])
    global_mae_mask.append(mse_mask_partial / np.shape(predict)[0])
    global_mae_roi.append(mse_combined_partial / np.shape(predict)[0])
        
    
    
    # ----------------- fQSR detection metric assessment
    
    qrs_detection = []
    
    true_positive = 0
    false_positive = 0
    false_negative = 0
    total_peaks = 0
    true_positive_peaks = []
    
    for i in range(np.shape(predict)[0]):
    
        peaks_proposed = find_peaks(
            predict[i, :, 1], 
            height=MASK_MIN_HEIGHT, 
            distance=MIN_QRS_DISTANCE
        )[0]
        
        for p in peaks_proposed:
            
            lower_limit_mask = 0 if p - QRS_DURATION_STEP < 0 else p - QRS_DURATION_STEP
            upper_limit_mask = LEN_BATCH if p + QRS_DURATION_STEP > LEN_BATCH else p + QRS_DURATION_STEP

            roi_predicted = predict[i, lower_limit_mask : upper_limit_mask, 1]

            if np.max(np.diff(roi_predicted)) < 0.5:
                qrs_detection.append(int(p + i * LEN_BATCH))

                
            
    for peak in time_annotations * SAMPLING_FREQ:
    
        if peak <= np.shape(aECG)[0] * np.shape(aECG)[1]:
            
            total_peaks += 1
        
            peak_found = np.where(
                (np.array(qrs_detection) >= peak - LIMT_GAUS) & 
                (np.array(qrs_detection) <= peak + LIMT_GAUS)
            )
            
            
            if len(peak_found[0]) > 0:
                for k in peak_found[0]:
                    true_positive_peaks.append(k)
                    
                true_positive += 1
            else:
                false_negative += 1
                
    
    for peak_predicted in qrs_detection:
        if peak_predicted not in true_positive_peaks and peak_predicted <= LIMIT:
            
            possible_ann = np.where(
                (time_annotations * SAMPLING_FREQ >= peak_predicted - LIMT_GAUS) &
                (time_annotations * SAMPLING_FREQ <= peak_predicted + LIMT_GAUS)
            )[0]

            if len(possible_ann) == 0:

                false_positive += 1
   
    f1 = true_positive / (
        true_positive + 0.5 * (false_positive + false_negative)
    )

    recall = true_positive / (
        true_positive + (false_negative)
    )

    precision = true_positive / (
        true_positive + (false_positive)
    )


    global_f1_score.append(f1)
    global_recall.append(recall)
    global_precision.append(precision)

#%%

print(mean_confidence_interval(global_f1_score, name='f1-score'))
print(mean_confidence_interval(global_recall))
print(mean_confidence_interval(global_precision))
# %%

print(mean_confidence_interval(global_mae_signal))
print(mean_confidence_interval(global_mae_mask))
print(mean_confidence_interval(global_mae_roi))
# %%
