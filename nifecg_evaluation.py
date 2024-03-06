
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
MASK_MIN_HEIGHT = 0.7

LIMIT = int(300000 / RESAMPLING_FREQUENCY_RATIO)# - LEN_BATCH

#%%

filenames = glob.glob(DATA_PATH + '*.edf')
filenames = [i for i in filenames if int(i.split('ecgca')[-1].replace('.edf', '')) in FILES_TO_READ ]


global_f1_score = []
global_recall = []
global_precision = []

for file in filenames:
    
    # Get data
    
    aECG, fECG =  data_resizer(    
        [file],
        LEN_BATCH, 
        QRS_DURATION, 
        QRS_DURATION_STEP,
        type_of_file='edf', 
        training=False, 
        resample_fs=1
    )
    
    annotations =  mne.read_annotations(file)
    time_annotations = annotations.onset

    # Model prediction

    predict = model.predict(aECG)
    
    # Metric assessment
    qrs_detection = []
    
    true_positive = 0
    false_positive = 0
    false_negative = 0
    total_peaks = 0
    true_positive_peaks = []
    
    for i in range(np.shape(predict)[0]):
    
        peaks_proposed = find_peaks(
            predict[i, :, 0], 
            height=MASK_MIN_HEIGHT, 
            distance=MIN_QRS_DISTANCE
        )
                
            
        for p in time_annotations * SAMPLING_FREQ:
                
            lower_limit_mask = 0 if p - QRS_DURATION_STEP < 0 else p - QRS_DURATION_STEP
            upper_limit_mask = LEN_BATCH if p + QRS_DURATION_STEP > LEN_BATCH else p + QRS_DURATION_STEP

            roi_predicted = predict[i, int(lower_limit_mask) : int(upper_limit_mask), 0]
                  
            if len(roi_predicted) == 0:
                continue
                    
            if np.max(np.diff(roi_predicted)) < 0.5:
                qrs_detection.append(int(p + i * LEN_BATCH))
                

            already_mentioned_peaks = []  
    
        for peak in qrs_detection:
            
            lower_limit = peak - LIMT_GAUS
            upper_limit = peak + LIMT_GAUS
            
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
                    (np.array(time_annotations * SAMPLING_FREQ) >= lower_limit) & 
                    (np.array(time_annotations * SAMPLING_FREQ) <= upper_limit)
                )
                
                if len(peak_found[0]) == 0:
                    
                    false_positive += 1
                
            already_mentioned_peaks.append(peak)
        
        already_counted = []
        

        f1 = true_positive / (
            true_positive + 0.5 * (false_positive + false_negative)
        )
        
        recall = true_positive / (
            true_positive + (false_negative)
        )
        
        precision = true_positive / (
            true_positive + (false_positive)
        )
        
       
        acc = true_positive / (
            (total_peaks + false_positive)
        )
        
        global_f1_score.append(f1)
        global_recall.append(recall)
        global_precision.append(precision)

#%%