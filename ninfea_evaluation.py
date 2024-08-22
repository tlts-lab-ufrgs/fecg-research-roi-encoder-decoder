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

from scipy.signal import resample
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA

from utils.stats_fn import mean_confidence_interval

from models.ae_proposed import ProposedAE

#%% Load model

# Range in learning rate
UPPER_LIM_LR = 0.0001

SAMPLING_FREQ = 512

RESAMPLE_FREQ_RATIO = 4

# batch size
BATCH_SIZE=4

# files 
FILES_TO_READ = [154, 192, 244, 274, 290, 323, 368, 444, 597, 733, 746, 811, 826, 906,]

RESULTS_PATH = "/home/julia/Documents/fECG_research/research_dev/autoencoder_with_mask/results/"
DATA_PATH =  "/home/julia/Documents/research/datasets/ninfea-dataset/"

CHANNELS = 3

detectors = Detectors(SAMPLING_FREQ) # fs = frequencia de sampling


LEN_BATCH = int(2 * 512 / RESAMPLE_FREQ_RATIO)
LIMT_GAUS = int(2 * 50 / RESAMPLE_FREQ_RATIO)
QRS_DURATION = 0.1  # seconds, max
QRS_LEN = int(2 * 50 / RESAMPLE_FREQ_RATIO)
MIN_QRS_DISTANCE = int(2 * 200 / RESAMPLE_FREQ_RATIO) # fs = 1000Hz
MASK_MIN_HEIGHT = 0.7

MODEL_INPUT_SHAPE = (BATCH_SIZE, LEN_BATCH, CHANNELS)
LIMIT = int(2 * 300000 / RESAMPLE_FREQ_RATIO)# - LEN_BATCH

INIT_SUB = 1
END_SUB = 60

ECG_SAMPLING_FREQ = 2048

detectors = Detectors(SAMPLING_FREQ) # fs = frequencia de sampling

#%%


model = ProposedAE(
    MODEL_INPUT_SHAPE, 
    BATCH_SIZE, 
    UPPER_LIM_LR, 
    w_mask=0.3, 
    w_signal=0.1, 
    w_combined=0.6, 
    training_data=[], 
    ground_truth=[],
    testing_data=None, 
    ground_truth_testing=None, 
    epochs=100
)

model.linknet()

model.model.load_weights('/home/julia/Documents/research/sprint_1/rev1_weights/weights.h5')


to_not_use = [
    9,12,18,22,27,38,42,49,59,60,
    1,2,4,5,6,17,25,34,37
]  # based on https://www.frontiersin.org/articles/10.3389/fbioe.2023.1059119/full
# if the file has less than 3 electrods channels identified as informative, it was not evaluated


#%% data load

counter = 0

for i in range(INIT_SUB, END_SUB + 1, 1):  # subjects
    
    if i in to_not_use:
        continue
    
    mecg_file = f'{DATA_PATH}/wfdb_format_ecg_and_respiration/{i}'
    mecg_signal, labels_mecg = wfdb.rdsamp(mecg_file)
    
    raw_data = mecg_signal[:, [21, 8,15,11]] # the 1 electrode will not be used, it will be replaced by extracted fecg
    time_steps = np.linspace(0, np.shape(raw_data)[0] * (1 / ECG_SAMPLING_FREQ), np.shape(raw_data)[0])
    
    if RESAMPLE_FREQ_RATIO != 1:
        filedata = np.zeros(shape=(int(np.shape(raw_data)[0] / RESAMPLE_FREQ_RATIO), 4))
        for j in range(4):
            filedata[:, j] = resample(raw_data[:, j], 
                                      int(np.shape(raw_data)[0] / RESAMPLE_FREQ_RATIO))
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

   
    # Loop in data
    
    batch = 0
    index = 0              
            
    while batch <= np.shape(filedata)[0] - LEN_BATCH:
        

        chunked_data = filedata[(batch): ((batch + LEN_BATCH)), 1::]
        
        chunked_fecg_real_data = filedata[(batch): (batch + LEN_BATCH), 0]
        
        # Data Normalization

        chunked_data -= np.min(chunked_data) # to zero things
        chunked_fecg_real_data -= np.min(chunked_fecg_real_data) # to zero things
        
        max_abdominal = np.abs(np.max(chunked_data)) if np.abs(np.max(chunked_data)) != 0 else 1e-7
        max_fecg = np.abs(np.max(chunked_fecg_real_data)) if np.abs(np.max(chunked_fecg_real_data)) != 0 else 1e-7
        

        chunked_data *= (1 / max_abdominal) 
        chunked_fecg_real_data *= (1 / max_fecg)
        
        chunked_fecg_data = np.array([chunked_fecg_real_data]).transpose()
                

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

predict = model.model.predict(aECG_store)
# %%

index = 95

plt.plot(aECG_store[index], label='aecg')
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
    # binary_mask = np.where(
    #         fECG_store[i, :, 1] != 0, 
    #         1, 
    #         0
    #     )
    # roi_true_signal = fECG_store[i, :, 0] * binary_mask

    binary_mask = np.where(
            predict[i, :, 1] != 0, 
            1, 
            0
        )
        
    mse_signal_partial += mae_function(fECG_store[i, :, 0], predict[i, :, 0])
    mse_mask_partial += 0 # mae_function(fECG_store[i, :, 1], predict[i, :, 1])
    mse_combined_partial += mae_function(
       fECG_store[i, :, 0] * binary_mask, 
       predict[i, :, 0] * binary_mask
    )
            
    global_mae_signal.append(mse_signal_partial / np.shape(predict)[0])
    global_mae_mask.append(mse_mask_partial / np.shape(predict)[0])
    global_mae_roi.append(mse_combined_partial / np.shape(predict)[0])
        
# %%

print(mean_confidence_interval(global_mae_signal))
print(mean_confidence_interval(global_mae_mask))
print(mean_confidence_interval(global_mae_roi))

#%%