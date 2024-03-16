"""
Evluate FECG Syn dataset
"""

#%%

import glob
import wfdb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ecgdetectors import panPeakDetect, Detectors

from scipy.io import loadmat
from scipy.signal import resample
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from models.ae_proposed import Metric, Loss

from utils.gaussian_function import gaussian
from utils.lr_scheduler import callback as lr_scheduler
from utils.mean_confidence_interval import mean_confidence_interval

#%% Load model

model = tf.keras.models.load_model(
    '/home/julia/Documents/fECG_research/research_dev/autoencoder_with_mask/final_model_3ch/', 
    custom_objects = {
        'mse_mask': Metric.mse_mask,
        'mse_signal': Metric.mse_signal, 
        'loss': Loss.loss, 
        'lr': lr_scheduler
    }
)


#%%

QRS_LEN = 50 # +- 125 pontos fazem o esmo que no abcd, fs = 250
LEN_BATCH = 512
QRS_DURATION = 0.1

DATA_PATH = '/home/julia/Documents/fECG_research/datasets/ninfea-non-invasive-multimodal-foetal-ecg-doppler-dataset-for-antenatal-cardiology-research-1.0.0/'

to_not_use = [
    9,12,18,22,27,38,42,49,59,60,
    1,2,4,5,6,17,25,34,37
]  # based on https://www.frontiersin.org/articles/10.3389/fbioe.2023.1059119/full
# if the file has less than 3 electrods channels identified as informative, it was not evaluated


INIT_SUB = 1
END_SUB = 60

RESAMPLE_FS = 2
CHANNELS = 3


DOPPLER_SAMPLING_FREQ = 284
ECG_SAMPLING_FREQ = 2048
SAMPLING_FREQ = 1024

MASK_MIN_HEIGHT = 0.8
LIMT_GAUS = 50
MIN_QRS_DISTANCE = 300

detectors = Detectors(SAMPLING_FREQ) # fs = frequencia de sampling


#%%

annotations = loadmat('/home/julia/Documents/fECG_research/datasets/ninfea-non-invasive-multimodal-foetal-ecg-doppler-dataset-for-antenatal-cardiology-research-1.0.0/annotated_V_dataset.mat')


#%% data load

counter = 0

for i in range(INIT_SUB, END_SUB + 1, 1):  # subjects
    
    if i in to_not_use:
        continue
    
    mecg_file = f'{DATA_PATH}/wfdb_format_ecg_and_respiration/{i}'
    mecg_signal, labels_mecg = wfdb.rdsamp(mecg_file)
    
    raw_data = mecg_signal[:, [21, 8,15,11]] # the 1 electrode will not be used, it will be replaced by extracted fecg
    time_steps = np.linspace(0, np.shape(raw_data)[0] * (1 / ECG_SAMPLING_FREQ), np.shape(raw_data)[0])
    
    if RESAMPLE_FS != 1:
        filedata = np.zeros(shape=(int(np.shape(raw_data)[0] / RESAMPLE_FS), 4))
        for j in range(4):
            filedata[:, j] = resample(raw_data[:, j], 
                                      int(np.shape(raw_data)[0] / RESAMPLE_FS))
    else:
        filedata = np.copy(raw_data)

    
    # If fecg dont exist in dataset, extract it from BSS ICA method
    
    tmpdata = filedata[:, 1]  # randomly choose this channel to retrieve fecg
    # calculate the number of components using eigenvalues
    pca = PCA()
    pca.fit(tmpdata.reshape(-1, 1))
    perc = np.cumsum(pca.explained_variance_ratio_)
    number_components = np.argmax(perc >= 0.999) + 1            
    transformer = FastICA(number_components)
    fecg_retrieved = transformer.fit_transform(tmpdata.reshape(-1, 1)) 
    filedata[:, 0] = fecg_retrieved[:, 0]

    
    time_annotations =   annotations['V'][i-1][1][0] * (
        ECG_SAMPLING_FREQ / DOPPLER_SAMPLING_FREQ
    ) * (1 / RESAMPLE_FS)
    
    # detectors.pan_tompkins_detector(filedata[:, 0])
    
    # Generates masks
    mask = np.zeros(shape=(int(np.shape(filedata)[0])))

    for step in time_annotations:

        center_index = int(step) #%%np.mean(np.where((time_steps < step + 0.001) & (time_steps > step - 0.001)))

        qrs_region = np.arange(
            int(step-QRS_LEN*2) if int(step-QRS_LEN*2) > 0 else 0, 
            int(step+QRS_LEN*2) if int(step+QRS_LEN*2) < int(np.shape(filedata)[0]) else int(np.shape(filedata)[0]) - 1, 
            dtype=np.int64
        )
        
        mask[qrs_region] = gaussian(qrs_region, center_index , QRS_LEN / 2)
        
   
    # Loop in data
    
    batch = 0
    index = 0              
            
    while batch <= np.shape(filedata)[0] - LEN_BATCH:
        

        chunked_data = filedata[(batch): ((batch + LEN_BATCH)), 1::]
        
        chunked_fecg_real_data = filedata[(batch): (batch + LEN_BATCH), 0]
        chunked_fecg_binary_data = mask[(batch): (batch + LEN_BATCH)]

        
        # Data Normalization

        chunked_data -= np.min(chunked_data) # to zero things
        chunked_fecg_real_data -= np.min(chunked_fecg_real_data) # to zero things
        
        max_abdominal = np.abs(np.max(chunked_data)) if np.abs(np.max(chunked_data)) != 0 else 1e-7
        max_fecg = np.abs(np.max(chunked_fecg_real_data)) if np.abs(np.max(chunked_fecg_real_data)) != 0 else 1e-7
        

        chunked_data *= (1 / max_abdominal) 
        chunked_fecg_real_data *= (1 / max_fecg)
        
        chunked_fecg_data = np.array([
            chunked_fecg_real_data, 
            chunked_fecg_binary_data
        ]).transpose()
        
        

        if counter == 0 and batch == 0:

            aECG_store = np.copy([chunked_data])
            fECG_store = np.copy([chunked_fecg_data])

        else:
            aECG_store = np.vstack((aECG_store, [chunked_data]))
            fECG_store = np.vstack((fECG_store, [chunked_fecg_data]))

        batch += LEN_BATCH
        index += 1
    
    counter += 1



#%%


concat = np.empty(shape=0)

for i in range(np.shape(aECG_store)[0]):
    
    concat = np.concatenate([concat, fECG_store[i, :, 0]])

r_peaks_signal = detectors.pan_tompkins_detector(concat)



#%%

# Model prediction

predict = model.predict(aECG_store)
# %%

index = 20

plt.plot(aECG_store[index])
plt.plot(fECG_store[index], label='fecg')
plt.plot(predict[index], label='predict')

plt.legend()

#%%

def mae_function(y_true, y_pred):
    
    mse_value = np.mean(
        np.abs((y_true - y_pred)) # , 2)
    )
    
    return mse_value
# %%

global_f1_score = []
global_recall = []
global_precision = []
global_mae_signal = []
global_mae_mask = []
global_mae_roi = []

 # ----------------- fECG extraction assessment
    
mse_signal_partial = 0
mse_mask_partial = 0
mse_combined_partial = 0

for i in range(np.shape(predict)[0]):
    binary_mask = np.where(
            fECG_store[i, :, 1] != 0, 
            1, 
            0
        )
    roi_true_signal = fECG_store[i, :, 0] * binary_mask
        
    binary_predicted_mask = np.where(
            predict[i, :, 1] != 0, 
            1, 
            0
        )
    roi_predicted_signal = predict[i, :, 0] * binary_predicted_mask

    mse_signal_partial += mae_function(fECG_store[i, :, 0], predict[i, :, 0])
    mse_mask_partial += mae_function(fECG_store[i, :, 1], predict[i, :, 1])
    mse_combined_partial += mae_function(
            roi_true_signal, 
            roi_predicted_signal)
            
    global_mae_signal.append(mse_signal_partial / np.shape(predict)[0])
    global_mae_mask.append(mse_mask_partial / np.shape(predict)[0])
    global_mae_roi.append(mse_combined_partial / np.shape(predict)[0])
        
#%%  
    
# ----------------- fQSR detection metric assessment
    
qrs_detection = []

true_positive = 0
false_positive = 0
false_negative = 0
total_peaks = 0
true_positive_peaks = []
false_positive_peaks = []

for i in range(np.shape(predict)[0]):
    
    peaks_proposed = find_peaks(
        predict[i, :, 1], 
        height=MASK_MIN_HEIGHT, 
        distance=MIN_QRS_DISTANCE
    )[0]

    for p in peaks_proposed:

        lower_limit_mask = 0 if p - QRS_LEN < 0 else p - QRS_LEN
        upper_limit_mask = LEN_BATCH if p + QRS_LEN > LEN_BATCH else p + QRS_LEN

        roi_predicted = predict[i, lower_limit_mask : upper_limit_mask, 1]

        if np.max(np.diff(roi_predicted)) < 0.5:
            qrs_detection.append(int(p + i * LEN_BATCH))

                
            
for peak in r_peaks_signal:
    
    if peak <= np.shape(aECG_store)[0] * np.shape(aECG_store)[1]:
            
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
    if peak_predicted not in true_positive_peaks:
            
        possible_ann = np.where(
            (np.array(r_peaks_signal) >= peak_predicted - LIMT_GAUS) &
            (np.array(r_peaks_signal) <= peak_predicted + LIMT_GAUS)
        )[0]

        if len(possible_ann) == 0:
            false_positive_peaks.append(peak_predicted)
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
    
print(mean_confidence_interval(global_f1_score, name='f1-score'))
print(mean_confidence_interval(global_recall))
print(mean_confidence_interval(global_precision))
# %%

print(mean_confidence_interval(global_mae_signal))
print(mean_confidence_interval(global_mae_mask))
print(mean_confidence_interval(global_mae_roi))

#%%