"""
Signal and mask as output

juliacremus
Wednesday 24 jan 09h18
"""

#%% Imports

import mne
import glob
import numpy as np


#%% Parameters

DATA_PATH = "/home/julia/Documents/fECG_research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/"

LEN_DATA = 600

QRS_DURATION = 0.05  # seconds, max

QRS_DURATION_STEP = 26

def gaussian(x, mu, sig):
    
    signal = 1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    
    return signal / np.max(signal)


#%% Data Loading 

def load_data(len_data = LEN_DATA, path = DATA_PATH, qrs_duration = QRS_DURATION):
    
    FILENAMES = glob.glob(path + "*.edf")


    for file in FILENAMES[0:3]:

        file_info = mne.io.read_raw_edf(file)
        filedata = file_info.get_data()
        
        # filedata *= 1E5
        
        filedata += np.abs(np.min(filedata)) # to zero things
        
        max_absolute_value = np.max(np.abs(filedata))
        
        
        
        filedata *= (1 / max_absolute_value)

        annotations = mne.read_annotations(file)
        time_annotations = annotations.onset
        


        # Generates Binary masks

        binary_mask = np.zeros(shape=file_info.times.shape)

        for step in time_annotations:

            center_index = np.where(file_info.times == step)[0][0]

            qrs_region = np.where(
                (file_info.times >= (step - qrs_duration)) &
                (file_info.times <= (step + qrs_duration))
            )[0]
            

            binary_mask[qrs_region] = gaussian(qrs_region, center_index, QRS_DURATION_STEP / 2)


        for batch in range(0, 262144, len_data):


            chunked_data = filedata[1::, (batch): ((batch + len_data))].transpose()
            
            # chunked_data_with_noise = chunked_data + 0.01 * np.random.normal(0,1,len_data)    
                  
            chunked_fecg_real_data = filedata[0, (batch): (batch + len_data)]
            chunked_fecg_binary_data = binary_mask[(batch): (batch + len_data)]

            chunked_fecg_data = np.array([
                chunked_fecg_real_data, 
                chunked_fecg_binary_data
            ]).transpose()
            

            if batch == 0:

                data_store = np.copy([chunked_data])
                fecg_store = np.copy([chunked_fecg_data])

            else:
                data_store = np.vstack((data_store, [chunked_data]))
                fecg_store = np.vstack((fecg_store, [chunked_fecg_data]))
    
    
    return data_store, fecg_store

# %%
