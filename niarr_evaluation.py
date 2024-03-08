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

QRS_LEN = 40 # +- 125 pontos fazem o esmo que no abcd, fs = 250
LEN_BATCH = 512
QRS_DURATION = 0.1

DATA_PATH = '/home/julia/Documents/fECG_research/datasets/non-invasive-fetal-ecg-arrhythmia-database-1.0.0/'

INIT_SUB = 1
END_SUB = 5

RESAMPLE_FS = 1
CHANNELS = 3


ECG_SAMPLING_FREQ = 1000


#%% data load

subjects = [10]

counter = 0

for i in subjects:  # subjects
    
    mecg_file = f'{DATA_PATH}/ARR_0{i}' if i < 10 else f'{DATA_PATH}/ARR_{i}'
    mecg_signal, labels_mecg = wfdb.rdsamp(mecg_file)

    filedata = mecg_signal[:, [4,1,2,3]] # the 1 electrode will not be used, it will be replaced by extracted fecg
            
    # If fecg dont exist in dataset, extract it from BSS ICA method
    
    tmpdata = np.copy(filedata[:, 1])  # randomly choose this channel to retrieve fecg
    # calculate the number of components using eigenvalues
    pca = PCA()
    pca.fit(tmpdata.reshape(-1, 1))
    perc = np.cumsum(pca.explained_variance_ratio_)
    number_components = np.argmax(perc >= 0.999) + 1            
    transformer = FastICA(number_components)
    fecg_retrieved = transformer.fit_transform(tmpdata.reshape(-1, 1)) 
    filedata[:, 0] = np.copy(fecg_retrieved[:, 0])


    # # Get annotations:
    try:
        time_annotations = wfdb.rdann(
            mecg_file, 
            extension='hea'
        ).sample
    except:
        print(i)
        continue

    
    # Generates masks
    mask = np.zeros(shape=(int(np.shape(filedata)[0])))

    # for step in time_annotations:

    #     qrs_region = np.arange(step - 75, step + 75 + 1, 1)
    #     mask[qrs_region] = gaussian(qrs_region, step, QRS_LEN / 2)
        

    # Loop in data
    
    batch = 0
    index = 0              
            
    while batch <= np.shape(filedata)[0] - LEN_BATCH:
        

        chunked_data = np.copy(filedata[(batch): ((batch + LEN_BATCH)), 1::])
        
        chunked_fecg_real_data = np.copy(filedata[(batch): (batch + LEN_BATCH), 0])
        chunked_fecg_binary_data = np.copy(mask[(batch): (batch + LEN_BATCH)])

        
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

index = 4

plt.plot(aECG_store[index])
plt.plot(fECG_store[index])

#%%

# Model prediction

predict = model.predict(aECG_store)
# %%

index = 9

plt.plot(aECG_store[index, :, 0])
plt.plot(fECG_store[index, :, 0])
plt.plot(predict[index])

# %%
