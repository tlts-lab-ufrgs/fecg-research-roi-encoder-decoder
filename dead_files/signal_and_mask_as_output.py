"""
Signal and mask as output

juliacremus
Wednesday 24 jan 09h18
"""

#%% Imports

import mne
import glob
import numpy as np

from signal_preprocessing.threshold import filtering_coef
from signal_preprocessing.haar_tranform import HaarTransform


#%% Parameters

DATA_PATH = "/home/julia/Documents/fECG_research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/"
LEN_DATA = 512
QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = 50

LEVEL = 1
FS = 1 / 2  # just a factor

def add_baseline_wandering(x, num_components=10, amplitude=1e-6, fs=1000):
    t = np.arange(len(x)) / fs
    baseline_wandering = np.zeros_like(x)

    for _ in range(num_components):
        frequency = np.random.uniform(low=0.01, high=0.1)  # Random low frequency
        phase = np.random.uniform(0, 2 * np.pi)  # Random phase
        component = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        baseline_wandering += component

    x_with_baseline = x + baseline_wandering
    return x_with_baseline

#%% Gaussian function

def gaussian(x, mu, sig):
    
    signal = 1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    
    return signal / np.max(signal)

def remove_tendency(original_data, len_data):

    filtered_data = np.copy(original_data)

    # Remove tendency

    for i in range(512):
        
        for j in range(4):
         
            haar = HaarTransform(original_data[i, :, j], LEVEL, FS)
            details = haar.run_cascade_multiresolution_transform()

                
            # Filter coefs
            # 'minimax', 'alice', 'han', 'universal'
            data_filtered = filtering_coef(
                len_data, 
                details, 
                LEVEL, 
                'han', 
                'hard'
                )

            # Recosntruct signal filtered
            reconstruct_data = HaarTransform(data_filtered, LEVEL, FS)
            tendency_signal = reconstruct_data.run_cascade_multiresolution_inv_transform()

            filtered_data[i, :, j] -= tendency_signal
        
    return filtered_data


from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff_frequency, sampling_frequency, order=4):
    nyquist = 0.5 * sampling_frequency
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Example usage:
# Assuming you have a signal 'signal_with_baseline'
original_sampling_frequency = 1000  # Replace this with your actual sampling frequency
cutoff_frequency = 70  # Cutoff frequency for the low-pass filter

#%% Data Loading 

def load_data(
    filenames, 
    type_of_file='edf', 
    len_data = LEN_DATA, 
    path = DATA_PATH, 
    qrs_duration = QRS_DURATION, 
    qrs_len = QRS_DURATION_STEP
):

    for file in filenames:
        
        # Read data and annotations
        
        if type_of_file == 'edf':
            file_info = mne.io.read_raw_edf(file)
            raw_data = file_info.get_data()
            annotations = mne.read_annotations(file)
            time_annotations = annotations.onset
        
        
        filedata = np.empty(shape=(5, int(2 * np.shape(raw_data)[1])))
                # Add data augmentation
        if True:
            
            for i in range(1, 5):
                
                augmented_data = np.copy(raw_data[i])
                
                # # add noise to it
                # mu = 0
                # sigma = 0.1    
                # noise = 5e-6 * np.random.normal(mu, sigma, size=np.shape(augmented_data)) 
                # augmented_data += noise
                
                # add baseline wandering
                # Apply the low-pass filter
                augmented_data = butter_lowpass_filter(augmented_data, cutoff_frequency, original_sampling_frequency)

                augmented_data = add_baseline_wandering(augmented_data)
            
                filedata[i] = np.append(raw_data[i], augmented_data)
            
            # add ground truth as well
            
            duplicate_fecg = np.copy(raw_data[0])
            filedata[0] = np.append(raw_data[0], duplicate_fecg)

            print(np.shape(filedata))


        # Generates masks
        mask = np.zeros(shape=file_info.times.shape)

        for step in time_annotations:

            center_index = np.where(file_info.times == step)[0][0]

            qrs_region = np.where(
                (file_info.times >= (step - qrs_duration)) &
                (file_info.times <= (step + qrs_duration))
            )[0]
            
            mask[qrs_region] = gaussian(qrs_region, center_index, qrs_len / 2)

        # Resize data to be in the desire batch size
        mask = np.append(mask, mask)
        
        UPPER_LIMIT = int(np.power(2, np.round(np.log2(2 * file_info.times.shape[0]), 0)))
    
        print(UPPER_LIMIT)
        
        batch = 0
        
        while batch < 300000 - len_data:
        
        # for batch in range(0, UPPER_LIMIT, len_data):


            chunked_data = filedata[1::, (batch): ((batch + len_data))].transpose()
            
            chunked_fecg_real_data = filedata[0, (batch): (batch + len_data)]
            chunked_fecg_binary_data = mask[(batch): (batch + len_data)]
            
            
        
            # Data Normalization
     

            chunked_data += np.abs(np.min(chunked_data)) # to zero things
            chunked_fecg_real_data += np.abs(np.min(chunked_fecg_real_data)) # to zero things
            

            chunked_data *= (1 / np.abs(np.max(chunked_data)))
            chunked_fecg_real_data *= (1 / np.abs(np.max(chunked_fecg_real_data)))
        
            
            chunked_fecg_data = np.array([
                chunked_fecg_real_data, 
                chunked_fecg_binary_data
            ]).transpose()

            if filenames.index(file) == 0 and batch == 0:

                data_store = np.copy([chunked_data])
                fecg_store = np.copy([chunked_fecg_data])

            else:
                data_store = np.vstack((data_store, [chunked_data]))
                fecg_store = np.vstack((fecg_store, [chunked_fecg_data]))
    
    
            batch += len_data
    # remove tendencies
    
    # data_store = remove_tendency(data_store, len_data)

    
    return data_store, fecg_store
