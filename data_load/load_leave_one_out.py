import mne
import glob
import numpy as np


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
cutoff_frequency = 100  # Cutoff frequency for the low-pass filter


from utils.gaussian_function import gaussian

def data_resizer(    
    filenames,
    len_data, 
    qrs_duration, 
    qrs_len,
    type_of_file='edf', 
    training=False
):
    
        # training file
    for file in filenames:
        
        # Read data and annotations
        
        if type_of_file == 'edf':
            file_info = mne.io.read_raw_edf(file)
            raw_data = file_info.get_data()
            annotations = mne.read_annotations(file)
            time_annotations = annotations.onset
        
        print(np.shape(raw_data))
        
        # if 'r10' in file:
            
        #     filedata = raw_data[[0,1,2,4]]
            
        # elif 'r07' in file:
            
        #     filedata = raw_data[[0,2,3,4]]
        
        # elif 'r04' in file:
            
        #     filedata = raw_data[[0,2,3,4]]
            
        # else:
        #     filedata = raw_data[[0,4,2,3]]
    
            
        # Generates masks
        mask = np.zeros(shape=file_info.times.shape)

        for step in time_annotations:

            center_index = np.where(file_info.times == step)[0][0]

            qrs_region = np.where(
                (file_info.times >= (step - qrs_duration)) &
                (file_info.times <= (step + qrs_duration))
            )[0]
            
            mask[qrs_region] = gaussian(qrs_region, center_index, qrs_len / 2)
            
            
        # Add data augmentation
        # if training:
            
        #     filedata = np.empty(shape=(5, int(2 * np.shape(raw_data)[1])))
            
        #     for i in range(1, 5):
                
        #         augmented_data = np.copy(raw_data[i])
                
        #         # # add noise to it
        #         # mu = 0
        #         # sigma = 0.1    
        #         # noise = 5e-6 * np.random.normal(mu, sigma, size=np.shape(augmented_data)) 
        #         # augmented_data += noise
                
        #         # add baseline wandering
        #         augmented_data = add_baseline_wandering(augmented_data)
            
        #         filedata[i] = np.append(raw_data[i], augmented_data)
            
        #     # add ground truth as well
            
        #     duplicate_fecg = np.copy(raw_data[0])
        #     filedata[0] = np.append(raw_data[0], duplicate_fecg)

        #     print(np.shape(filedata))
            
        #     mask = np.append(mask, mask)

        # else: 
        filedata = np.copy(raw_data)

        # Resize data to be in the desire batch size
        
        # if training:
            # UPPER_LIMIT = int(np.power(2, np.round(np.log2(2 * file_info.times.shape[0]), 0)))
        # else: 
        UPPER_LIMIT = int(np.power(2, np.round(np.log2(file_info.times.shape[0]), 0)))
        
        for batch in range(0, UPPER_LIMIT, len_data):


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

                aECG_store = np.copy([chunked_data])
                fECG_store = np.copy([chunked_fecg_data])

            else:
                aECG_store = np.vstack((aECG_store, [chunked_data]))
                fECG_store = np.vstack((fECG_store, [chunked_fecg_data]))
    
    
    
    return aECG_store, fECG_store

def data_loader(
    path, 
    len_data, 
    qrs_duration, 
    qrs_len,
    leave_for_testing,
    whole_dataset_training=False,
    type_of_file='edf'
):
    
    # get the filenames and filter the left out
    
    filenames = glob.glob(path + "*." + type_of_file)
    
    if not whole_dataset_training:
    
        test_file = filenames.pop(leave_for_testing)
    
    training_data = data_resizer(
        filenames, 
        len_data, 
        qrs_duration, 
        qrs_len, 
        type_of_file, 
    )
    
    if whole_dataset_training:
        testing_data = None
    
    else:
        testing_data = data_resizer(
            [test_file], 
            len_data, 
            qrs_duration, 
            qrs_len, 
            type_of_file
        )
    
    
    
    return training_data, testing_data