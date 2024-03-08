"""
Evluate FECG Syn dataset
"""

#%%

import glob
import wfdb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import resample
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from models.ae_proposed import Metric, Loss

from utils.gaussian_function import gaussian
from utils.lr_scheduler import callback as lr_scheduler

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

high_risk = [
    6,
    7,
    8,
    12,
    18,
    19,
    24,
    27,
    34,
    37,
    38,
    47,
    48,
    49,
    50,
    53,
    54,
    58
]

INIT_SUB = 4
END_SUB = 5

RESAMPLE_FS = 2
CHANNELS = 3


DOPPLER_SAMPLING_FREQ = 284
ECG_SAMPLING_FREQ = 2048


#%% data load

counter = 0

for i in range(INIT_SUB, END_SUB + 1, 1):  # subjects
    
    if i in high_risk:
        continue
    
    mecg_file = f'{DATA_PATH}/wfdb_format_ecg_and_respiration/{i}'
    mecg_signal, labels_mecg = wfdb.rdsamp(mecg_file)
    
    raw_data = mecg_signal[:, [1, 7, 14, 11]] # the 1 electrode will not be used, it will be replaced by extracted fecg
    time_steps = np.linspace(0, np.shape(raw_data)[0] * (1 / ECG_SAMPLING_FREQ), np.shape(raw_data)[0])
    
    if RESAMPLE_FS != 1:
        filedata = np.zeros(shape=(int(np.shape(raw_data)[0] / RESAMPLE_FS), 4))
        for j in range(4):
            filedata[:, j] = resample(raw_data[:, j], 
                                      int(np.shape(raw_data)[0] / RESAMPLE_FS))
    else:
        filedata = np.copy(raw_data)
    
            
    # If fecg dont exist in dataset, extract it from BSS ICA method
    
    tmpdata = filedata[:, 2]  # randomly choose this channel to retrieve fecg
    # calculate the number of components using eigenvalues
    pca = PCA()
    pca.fit(tmpdata.reshape(-1, 1))
    perc = np.cumsum(pca.explained_variance_ratio_)
    number_components = np.argmax(perc >= 0.999) + 1            
    transformer = FastICA(number_components)
    fecg_retrieved = transformer.fit_transform(tmpdata.reshape(-1, 1)) 
    filedata[:, 0] = fecg_retrieved[:, 0]


    # Get annotations:
    
    doppler_signals = loadmat(f'{DATA_PATH}/pwd_signals/{i}envelopes')
    peaks, _ = find_peaks(
        np.abs(doppler_signals['x_down'][0]), 
        distance=30,
        width=15
    )
    
    time_annotations = peaks * (1 / DOPPLER_SAMPLING_FREQ)
    
    # Generates masks
    mask = np.zeros(shape=(int(np.shape(filedata)[0])))



    for step in time_annotations:

        center_index = np.mean(np.where((time_steps < step + 0.001) & (time_steps > step - 0.001)))

        qrs_region = np.where(
            (time_steps[::RESAMPLE_FS] > (step - QRS_DURATION)) &
            (time_steps[::RESAMPLE_FS] < (step + QRS_DURATION))
        )[0]
        
        mask[qrs_region] = gaussian(qrs_region, center_index / RESAMPLE_FS, QRS_LEN / 2)
        
   
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

# Model prediction

predict = model.predict(aECG_store)
# %%
