
#%%
import glob
import wfdb
import mne
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data_load.load_leave_one_out import data_loader

from models.ae_proposed import Metric, Loss
from utils.masks_function import gaussian
from data_load.load_leave_one_out import data_resizer
from utils.lr_scheduler import callback as lr_scheduler
from scipy.signal import find_peaks

from scipy.signal import resample
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
from scipy.io import loadmat
from ecgdetectors import panPeakDetect, Detectors

from utils.stats_fn import mean_confidence_interval

from models.ae_proposed import ProposedAE

from data_load.data_loader import DataLoader

#%% constants

# Range in learning rate
UPPER_LIM_LR = 0.0001

SAMPLING_FREQ = 500

# batch size
BATCH_SIZE=4

# files 
FILES_TO_READ = [154, 192, 244, 274, 290, 323, 368, 444, 597, 733, 746, 811, 826, 906,]

RESULTS_PATH = "/home/julia/Documents/fECG_research/research_dev/autoencoder_with_mask/results/"
DATA_PATH =  "/home/julia/Documents/research/datasets/non-invasive-fetal-ecg-database-1.0.0/"

CHANNELS = 3

RESAMPLING_FREQUENCY_RATIO = int(1000 / SAMPLING_FREQ)

detectors = Detectors(SAMPLING_FREQ) # fs = frequencia de sampling


LEN_BATCH = int(512 / RESAMPLING_FREQUENCY_RATIO)
LIMT_GAUS = int(30 / RESAMPLING_FREQUENCY_RATIO)
QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = int(50 / RESAMPLING_FREQUENCY_RATIO)
MIN_QRS_DISTANCE = int(200 / RESAMPLING_FREQUENCY_RATIO) # fs = 1000Hz
MASK_MIN_HEIGHT = 0.7

MODEL_INPUT_SHAPE = (BATCH_SIZE, LEN_BATCH, CHANNELS)
LIMIT = int(300000 / RESAMPLING_FREQUENCY_RATIO)# - LEN_BATCH

#%%

# data_loader = DataLoader(
#     DATA_PATH, 
#     LEN_BATCH, 
#     RESAMPLE_FREQ_RATIO, 
#     'edf', 
#     QRS_DURATION_STEP, 
#     QRS_DURATION, 
#     load_training_set=True,
#     load_testing_set=False,
#     type_of_mask='gaussian', 
#     # filters=True
# )

#%%
# training_data, _ = data_loader.data_load(0)

model = ProposedAE(
    MODEL_INPUT_SHAPE, 
    BATCH_SIZE, 
    UPPER_LIM_LR, 
    0.3, 
    0.1, 
    0.6, 
    training_data=[], 
    ground_truth=[],
    testing_data=None, 
    ground_truth_testing=None, 
    epochs=100
)

model.linknet()

#%%
model.model.load_weights('/home/julia/Documents/research/sprint_1/rev1_weights/weights.h5')
#%%

def mae_function(y_true, y_pred):
    
    mse_value = np.mean(
        np.abs((y_true - y_pred)) # , 2)
    )
    
    return mse_value

#%%

filenames = glob.glob(DATA_PATH + '*.edf')
filenames = [i for i in filenames if int(i.split('ecgca')[-1].replace('.edf', '')) in FILES_TO_READ ]


false_positive_peaks_per_file = []

global_f1_score = []
global_recall = []
global_precision = []
global_f1_score_pt = []
global_recall_pt = []
global_precision_pt = []
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
        resample_fs=RESAMPLING_FREQUENCY_RATIO, 
        channels=CHANNELS, 
        fecg_on_gt=False
    )
    
    if len(aECG) == 0:
        continue
    
    annotations =  mne.read_annotations(file)
    time_annotations = annotations.onset

    # Model prediction

    predict = model.model.predict(aECG)
    
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
        

        if i in [10,20,30,40]:
            fig, ax = plt.subplots()
            
            ax.set_title(file)

            ax.plot(
                fECG[i, :, 0], 
                label='Ground truth signal', 
                )
            
            
            ax.plot( predict[i, :, 0], label='Predicted Signal')
            
            ax1 = ax.twinx()
            
            ax1.plot(
                fECG[i, :, 1], 
                label='Ground truth RoI', 
                color='green'
                )
            ax1.plot(predict[i, :, 1], label='Predicted RoI', color='purple')
        
            
            ax.set_xlabel('Time steps')
            ax.set_ylabel('fECG normalized')
            ax1.set_ylabel('RoI signal')
            
            # Shrink current axis's height by 10% on the bottom
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                            box.width, box.height * 0.9])

            # Put a legend below current axis
            ax.legend(loc='upper center', bbox_to_anchor=(0.1, -0.15),
                    fancybox=True, shadow=True, ncol=2)
            
            ax1.legend(loc='upper center', bbox_to_anchor=(0.9, -0.15),
                    fancybox=True, shadow=True, ncol=2)
            
            
            
            ax.grid()
            
    global_mae_signal.append(mse_signal_partial / np.shape(predict)[0])
    global_mae_mask.append(mse_mask_partial / np.shape(predict)[0])
    global_mae_roi.append(mse_combined_partial / np.shape(predict)[0])
        
    
    
    # ----------------- fQSR detection metric assessment
    
    
    qrs_detection = []
    
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_positive_pt = 0
    false_positive_pt = 0
    false_negative_pt = 0
    total_peaks = 0
    true_positive_peaks = []
    false_positive_peaks = []
    true_positive_peaks_pt = []
    false_positive_peaks_pt = []
    
    concat = np.empty(shape=0)

    for i in range(np.shape(predict)[0]):
        
        concat = np.concatenate([concat, predict[i, :, 0]])

    r_peaks_signal = detectors.pan_tompkins_detector(concat)


    
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
                
            peak_found_pt = np.where(
                (np.array(r_peaks_signal) >= peak - LIMT_GAUS) & 
                (np.array(r_peaks_signal) <= peak + LIMT_GAUS)
            )
            
            
            if len(peak_found_pt[0]) > 0:
                for k in peak_found_pt[0]:
                    true_positive_peaks_pt.append(k)
                    
                true_positive_pt += 1
            else:
                false_negative_pt += 1
                
    
    for peak_predicted in qrs_detection:
        if peak_predicted not in true_positive_peaks and peak_predicted <= LIMIT:
            
            possible_ann = np.where(
                (time_annotations * SAMPLING_FREQ >= peak_predicted - LIMT_GAUS) &
                (time_annotations * SAMPLING_FREQ <= peak_predicted + LIMT_GAUS)
            )[0]

            if len(possible_ann) == 0:
                false_positive_peaks.append(peak_predicted)
                false_positive += 1
   
    for peak_predicted in r_peaks_signal:
        if peak_predicted not in true_positive_peaks and peak_predicted <= LIMIT:
            
            possible_ann = np.where(
                (time_annotations * SAMPLING_FREQ >= peak_predicted - LIMT_GAUS) &
                (time_annotations * SAMPLING_FREQ <= peak_predicted + LIMT_GAUS)
            )[0]

            if len(possible_ann) == 0:
                false_positive_peaks_pt.append(peak_predicted)
                false_positive_pt += 1
   
   
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


    global_f1_score.append(f1)
    global_recall.append(recall)
    global_precision.append(precision)
    
    global_f1_score_pt.append(f1_pt)
    global_recall_pt.append(recall_pt)
    global_precision_pt.append(precision_pt)
    
    false_positive_peaks_per_file.append(false_positive_peaks)

#%%

print(mean_confidence_interval(global_f1_score, name='f1-score'))
print(mean_confidence_interval(global_recall))
print(mean_confidence_interval(global_precision))

#%%

print(mean_confidence_interval(global_f1_score_pt, name='f1-score'))
print(mean_confidence_interval(global_recall_pt))
print(mean_confidence_interval(global_precision_pt))
# %%

print(mean_confidence_interval(global_mae_signal))
print(mean_confidence_interval(global_mae_mask))
print(mean_confidence_interval(global_mae_roi))
# %%

# for index in range(0, 150):
    
#     fig, ax = plt.subplots()
    
#     ax.set_title(index)
    
#     ax.plot(aECG[index], label='aecg')
#     ax.plot(predict[index, :, 1], label='predict')
#     ax.plot(fECG[index, :, 1], label='fecg')
    
#     ax.legend()
# %%

# for file in filenames:

# aECG, fECG =  data_resizer(    
#     [file],
#     256, 
#     QRS_DURATION, 
#     50,
#     type_of_file='edf', 
#     resample_fs=2, 
#     channels=CHANNELS, 
#     fecg_on_gt=False
# )

# for i in range(np.shape(fECG)[0]):
    
#     concat = np.concatenate([concat, fECG[i, :, 0]])

# r_peaks_signal = detectors.pan_tompkins_detector(concat)


# %%

for file in filenames:
    name_f = file.replace('/home/julia/Documents/research/datasets/non-invasive-fetal-ecg-database-1.0.0/ecgca', '')
    number = name_f.replace('.edf', '')

    index_number = filenames.index(file)

    precision_mod = round(global_precision[index_number] * 100, 2)
    recall_mod    = round(global_recall[index_number] * 100, 2)
    f1_mod        = round(global_f1_score[index_number] * 100, 2)

    print(f'{number} & {precision_mod} & {recall_mod} & {f1_mod} \\\\')
# %%
