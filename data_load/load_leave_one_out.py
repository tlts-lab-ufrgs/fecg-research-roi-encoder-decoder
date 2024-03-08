import mne
import glob
import numpy as np
from scipy.signal import resample
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

from utils.gaussian_function import gaussian

to_remove = [
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    56,
    57,
    58,
    59,
    60,
    177,
    179,
    180,
    181,
    182,183,
    184,
    185,
    186,
    187,
    188,
    189,
    190,
    197,
    198,
    199,
    200,
    201,
    202,
    203,
    204,
    205,
    206,
    207,
    311,
    312,
    313,
    314,
    315,
    316,
    317,
    318,
    319,
    320,
    335, 
    336, 
    338,
    339,
    340,
    341,
    343,
    365,
    371,
    393,
    394,
    395,
    396,
    397,
    398,
    399,
    400,
    401,
    402,
    403,
    404,
    405,
    406,
    407,
    408,
    409,
    410,
    411,
    412,
]


def data_resizer(    
    filenames,
    len_data, 
    qrs_duration, 
    qrs_len,
    type_of_file='edf', 
    resample_fs=1, 
    channels = 3, 
    fecg_on_gt = True
):
    
        # training file
    for file in filenames:
        
        # Read data and annotations
        
        if type_of_file == 'edf':
            
            try:
                file_info = mne.io.read_raw_edf(file)
                raw_data = file_info.get_data()
                annotations = mne.read_annotations(file)
                time_annotations = annotations.onset
            except:
                continue
    
        
        if resample_fs != 1:
            resampled_signal = np.zeros(shape=(5, int(np.shape(raw_data)[-1] / resample_fs)))
            for j in range(5):
                resampled_signal[j, :] = resample(raw_data[j, :], int(np.shape(raw_data)[-1] / resample_fs))
        else:
            resampled_signal = np.copy(raw_data)
    
            
        # Generates masks
        mask = np.zeros(shape=int(np.shape(raw_data)[-1] / resample_fs))

        for step in time_annotations:

            center_index = np.where(file_info.times == step)[0][0]

            qrs_region = np.where(
                (file_info.times[::resample_fs] > (step - qrs_duration)) &
                (file_info.times[::resample_fs] < (step + qrs_duration))
            )[0]
            
            mask[qrs_region] = gaussian(qrs_region, center_index / resample_fs, qrs_len / 2)
            
        # Number of channels:
        
        if channels == 3:    
            if 'r10' in file: # abcd  file with wrong channel 
                filedata = resampled_signal[[0, 1, 2, 4]]
            else:
                filedata = resampled_signal[[0, 2, 3, 4]]
        if channels == 4:
            filedata = np.copy(raw_data)
        
        # If fecg dont exist in dataset, extract it from BSS ICA method
        
        if not fecg_on_gt:
            
            tmpdata = filedata[2]  # randomly choose this channel to retrieve fecg
            
            # calculate the number of components using eigenvalues
            pca = PCA()
            pca.fit(tmpdata.reshape(-1, 1))
            perc = np.cumsum(pca.explained_variance_ratio_)
            number_components = np.argmax(perc >= 0.999) + 1            

                        
            transformer = FastICA(number_components)
            fecg_retrieved = transformer.fit_transform(tmpdata.reshape(-1, 1)) 
        
            print('fecg retrieved shappe', np.shape(fecg_retrieved))
        
            filedata[0] = fecg_retrieved[:, 0]
        
        # Loop in data
        
        batch = 0
        index = 0              
               
        while batch <= np.shape(filedata)[-1] - len_data:
            
            if 'r10' in file and index in to_remove:
                batch += len_data
                index += 1
                continue
            

            chunked_data = filedata[1::, (batch): ((batch + len_data))].transpose()
            
            chunked_fecg_real_data = filedata[0, (batch): (batch + len_data)]
            chunked_fecg_binary_data = mask[(batch): (batch + len_data)]

            
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
            
            

            if filenames.index(file) == 0 and batch == 0:

                aECG_store = np.copy([chunked_data])
                fECG_store = np.copy([chunked_fecg_data])

            else:
                aECG_store = np.vstack((aECG_store, [chunked_data]))
                fECG_store = np.vstack((fECG_store, [chunked_fecg_data]))

            batch += len_data
            index += 1
    
    
    return aECG_store, fECG_store

def data_loader(
    path, 
    len_data, 
    qrs_duration, 
    qrs_len,
    leave_for_testing,
    whole_dataset_training=False,
    type_of_file='edf', 
    resample_fs=1,
    dataset = '', 
    channels = 3, 
    fecg_on_gt = True
):
    
    # get the filenames and filter the left out
    
    filenames = glob.glob(path + "*." + type_of_file)
    
    if dataset == 'nifecg':
        
        to_consider = [154, 192, 244, 274, 290, 323, 368, 444, 597, 733, 746, 811, 826, 906,]

        filenames = [i for i in filenames if int(i.split('ecgca')[-1].replace('.edf', '')) in to_consider ]
    
    if not whole_dataset_training:
    
        test_file = filenames.pop(leave_for_testing)
    
    training_data = data_resizer(
        filenames, 
        len_data, 
        qrs_duration, 
        qrs_len, 
        type_of_file, 
        resample_fs=resample_fs, 
        channels = channels,  
        fecg_on_gt = fecg_on_gt
    )
    
    if whole_dataset_training:
        testing_data = None
    
    else:
        testing_data = data_resizer(
            [test_file], 
            len_data, 
            qrs_duration, 
            qrs_len, 
            type_of_file, 
            resample_fs=resample_fs,
            channels = channels, 
            fecg_on_gt = fecg_on_gt
        )
    
    
    
    return training_data, testing_data