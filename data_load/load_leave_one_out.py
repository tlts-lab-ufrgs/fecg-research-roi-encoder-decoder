import mne
import glob
import numpy as np

from utils.gaussian_function import gaussian

def data_resizer(    
    filenames,
    len_data, 
    qrs_duration, 
    qrs_len,
    type_of_file='edf'
):
    
        # training file
    for file in filenames:
        
        # Read data and annotations
        
        if type_of_file == 'edf':
            file_info = mne.io.read_raw_edf(file)
            filedata = file_info.get_data()
            annotations = mne.read_annotations(file)
            time_annotations = annotations.onset
        
        
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
    type_of_file='edf'
):
    
    # get the filenames and filter the left out
    
    filenames = glob.glob(path + "*." + type_of_file)
    
    test_file = filenames.pop(leave_for_testing)

    training_data = data_resizer(
        filenames, 
        len_data, 
        qrs_duration, 
        qrs_len, 
        type_of_file
    )
    
    testing_data = data_resizer(
        [test_file], 
        len_data, 
        qrs_duration, 
        qrs_len, 
        type_of_file
    )
    
    
    
    return training_data, testing_data