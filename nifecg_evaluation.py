
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

#%%

model = tf.keras.models.load_model(
    '/home/julia/Documents/fECG_research/research_dev/autoencoder_with_mask/final_model_3ch_512/', 
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
LOWER_LIMIT_LR = 0.00098
LR_STEP = 0.00

# batch size
BATCH_SIZE=4

# files 
TOTAL_FILES = 5

RESULTS_PATH = "/home/julia/Documents/fECG_research/research_dev/autoencoder_with_mask/results/"
DATA_PATH =  "/home/julia/Documents/fECG_research/datasets/non-invasive-fetal-ecg-database-1.0.0/"

CHANNELS = 3
LEN_BATCH = 512
QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = 50

MODEL_INPUT_SHAPE = (BATCH_SIZE, LEN_BATCH, CHANNELS)


#%% loop in variables
filenames = glob.glob(DATA_PATH + "*.edf")
    
to_consider = [154, 192, 244, 274, 290, 323, 368, 444, 597, 733, 746, 811, 826, 906,]

for i in range(0, len(filenames)):

    if int(filenames[i].split('ecgca')[-1].replace('.edf', '')) in to_consider:

        aECG, fECG = data_resizer(
            [filenames[i]],
            LEN_BATCH, 
            QRS_DURATION, 
            QRS_DURATION_STEP,
            type_of_file='edf', 
            training=False
        )
        
        annotations = mne.read_annotations(filenames[i])
        time_annotations = annotations.onset

        #%%

        predict = model.predict(aECG)
    

        qrs_detection = []
        annotations = []

        for prediction_index in range(0, np.shape(predict[:, :, 1])[0]):

            peaks_proposed = find_peaks(predict[prediction_index, :, 1], height=0.7, distance=200)
                        
                    
            for p in peaks_proposed[0]:
                        
                lower_limit_mask = 0 if p - 50 < 0 else p - 50
                upper_limit_mask = LEN_BATCH if p + 50 > LEN_BATCH else p + 50

                roi_predicted = predict[:, lower_limit_mask : upper_limit_mask]
                            
                if np.max(np.diff(roi_predicted)) < 0.5:

                            
                    qrs_detection.append(int(p + prediction_index * LEN_BATCH))

                    
        # r_peaks_combined = panPeakDetect(prediction_data['combined-binary'].values, 200)
        # r_peaks_signal = panPeakDetect(prediction_data['signal'].values, 200)
                
        # for r in r_peaks_combined:
        #     pan_tom_qrs_detection.append(r  + prediction_index * LEN_BATCH)
                    
        # for r in r_peaks_signal:
        #     pan_tom_qrs_detection_signal.append(r  + prediction_index * LEN_BATCH)
            
# %%

# true_positive = 0
#     false_positive = 0
#     false_negative = 0
#     total_peaks = 0
#     true_positive_peaks = []
#     true_positive_peaks_pt = []
    
#     true_positive_pt = 0
#     false_positive_pt = 0
#     false_negative_pt = 0

#     limit = 298976

#     for peak in annotations_data[f'{j}'] * 1000:
        
#         if peak <= limit:
            
#             total_peaks += 1
        
        
#             lower_limit = peak - 30
#             upper_limit = peak + 30
            
#             peak_found = np.where(
#                 (np.array(this_weights_results[f'{dir}-proposed']) >= lower_limit) & 
#                 (np.array(this_weights_results[f'{dir}-proposed']) <= upper_limit)
#             )
            
#             peak_found_pt = np.where(
#                 (np.array(this_weights_results[f'{dir}-pan-combined']) >= lower_limit) & 
#                 (np.array(this_weights_results[f'{dir}-pan-combined']) <= upper_limit)
#             )
            
#             if len(peak_found[0]) > 0:
#                 true_positive_peaks.append(peak_found[0][0])
#                 true_positive += 1
#             else:
#                 false_negative += 1
                
#             if len(peak_found_pt[0]) > 0:
#                 true_positive_peaks_pt.append(peak_found_pt[0][0])
#                 true_positive_pt += 1
#             else:
#                 false_negative_pt += 1
    
#     already_mentioned_peaks = []  
    
#     for peak in this_weights_results[f'{dir}-proposed']:
        
#         already_counted = np.where(
#             (np.array(already_mentioned_peaks) >= peak - 50) & 
#             (np.array(already_mentioned_peaks) <= peak + 50)
#         )
        
#         # this case is specially for roi at the end or beggining of an file
#         already_counted_as_true = np.where(
#                 (np.array(true_positive_peaks) >= lower_limit) & 
#                 (np.array(true_positive_peaks) <= upper_limit)
#             )
        
#         if len(already_counted[0]) == 0 and len(already_counted_as_true[0]) == 0:
#             lower_limit = peak - 30
#             upper_limit = peak + 30
                
#             peak_found = np.where(
#                 (np.array(annotations_data[f'{j}'] * 1000) >= lower_limit) & 
#                 (np.array(annotations_data[f'{j}'] * 1000) <= upper_limit)
#             )
            
#             if len(peak_found[0]) == 0:
                
#                 false_positive += 1
            
#         already_mentioned_peaks.append(peak)
#     # print(total_peaks)
    
#     already_counted = []
    
#     for peak in :
        
#         already_counted = np.where(
#             (np.array(already_mentioned_peaks) >= peak - 50) & 
#             (np.array(already_mentioned_peaks) <= peak + 50)
#         )
        
#         # this case is specially for roi at the end or beggining of an file
#         already_counted_as_true = np.where(
#                 (np.array(true_positive_peaks_pt) >= lower_limit) & 
#                 (np.array(true_positive_peaks_pt) <= upper_limit)
#             )
        
#         if len(already_counted[0]) == 0 and len(already_counted_as_true[0]) == 0:
#             lower_limit = peak - 30
#             upper_limit = peak + 30
                
#             peak_found = np.where(
#                 (np.array(annotations_data[f'{j}'] * 1000) >= lower_limit) & 
#                 (np.array(annotations_data[f'{j}'] * 1000) <= upper_limit)
#             )
            
#             if len(peak_found[0]) == 0:
                
#                 false_positive_pt += 1
            
#         already_mentioned_peaks.append(peak)
#     # print(total_peaks)

#     f1 = true_positive / (
#         true_positive + 0.5 * (false_positive + false_negative)
#     )
    
#     recall = true_positive / (
#         true_positive + (false_negative)
#     )
    
#     s = true_positive / (
#         true_positive + (false_positive)
#     )
    
#     f1_pt = true_positive_pt / (
#         true_positive_pt + 0.5 * (false_positive_pt + false_negative_pt)
#     )
    
#     recall_pt = true_positive_pt / (
#         true_positive_pt + (false_negative_pt)
#     )
    
#     s_pt = true_positive_pt / (
#      true_positive_pt + (false_positive_pt)
#     )
    
#     acc = true_positive / (
#         (total_peaks + false_positive)
#     )
    
#     acc_pt = true_positive_pt / (
#         (total_peaks + false_positive)
#     )
    
#     f1_store.append(f1)
#     recall_store.append(recall)
#     precision_store.append(s)
    
#     print(
#         '\t'.join(
#             [
#                 f'{j}', 
#                 f'{f1}', 
#                 f'{f1_pt}', 
#                 f'{recall}', 
#                 f'{recall_pt}', 
#                 f'{s}', 
#                 f'{s_pt}', 
#                 f'{acc}'
#             ]
#         )
#     )
