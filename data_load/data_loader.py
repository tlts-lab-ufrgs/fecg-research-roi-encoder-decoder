
import os
import mne
import glob
import wfdb
import pandas as pd
import numpy as np
import scipy.io
from scipy.signal import butter, lfilter, freqz, filtfilt, resample

from sklearn.preprocessing import scale
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

from utils.segments_to_remove_adfecg import to_remove
from utils.masks_function import gaussian, triangle

class DataLoader: 

    def __init__(
            self, 
            data_path, 
            interval_size, 
            resample_fs, 
            file_extension, 
            qrs_interval = 50, 
            std_gaussian = 0.1, 
            load_training_set=True, 
            load_testing_set=True,
            channels = 3,
            fecg_on_gt = True,
            overlap_data = False, 
            type_of_mask = 'gaussian', 
            filters=False
        ):

     
        self.INTERVAL_SIZE = interval_size
        self.RESAMPLE_FS_RATIO = resample_fs

        self.DATA_PATH = data_path
        self.FILE_EXTENSION = file_extension
        self.QRS_INTERVAL = qrs_interval
        self.STD_GAUSSIAN = std_gaussian

        self.CHANNELS = channels
        self.CALC_FECG_FROM_BSS = fecg_on_gt

        self.LOAD_TRAINING_SET = load_training_set
        self.LOAD_TESTING_SET = load_testing_set
        self.OVERLAP_DATA = overlap_data 

        self.TYPE_OF_MASK = type_of_mask

        self.FILTERS_ON = filters

        pass


    def __butter_bandpass(self, lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')

        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=3, axis=-1):
        b, a = self.__butter_bandpass(lowcut, highcut, fs, order=order)
        y = scale(filtfilt(b, a, data))
        return y

    def data_load(self, leave_for_testing):
        
        training_data = None
        testing_data = None

        if self.LOAD_TESTING_SET:

            match self.FILE_EXTENSION:

                case 'txt':
                    filenames = glob.glob(self.DATA_PATH + 'B*')

                    test_file = filenames.pop(leave_for_testing)

                    if self.LOAD_TRAINING_SET:

                        training_data = self.load_b2_dataset(
                            filenames,
                        )

                    testing_data = self.load_b2_dataset(
                        [test_file],
                    )

                case 'edf':
                    filenames = glob.glob(self.DATA_PATH + "*." + self.FILE_EXTENSION) 

                    test_file = filenames.pop(leave_for_testing)

                    if self.LOAD_TRAINING_SET:

                        training_data = self.load_adfecg_dataset(
                            filenames, 
                        )

                    testing_data = self.load_adfecg_dataset(
                        [test_file]
                    )


        return training_data, testing_data
    
    def load_adfecg_dataset(self, filenames):

        """
        Load ADFECG files (edf)
        
        """

        # training file
        for file in filenames:
            
            # Read data and annotations
            
            
            try:
                file_info = mne.io.read_raw_edf(file)
                raw_data = file_info.get_data()
                annotations = mne.read_annotations(file)
                time_annotations = annotations.onset
            except:
                continue
                           
            if self.RESAMPLE_FS_RATIO != 1:
                resampled_signal = np.zeros(shape=(5, int(np.shape(raw_data)[-1] / self.RESAMPLE_FS_RATIO)))
                for j in range(5):  # because there are 5 channels in the data
                    resampled_signal[j, :] = resample(raw_data[j, :], int(np.shape(raw_data)[-1] / self.RESAMPLE_FS_RATIO))
            else:
                resampled_signal = np.copy(raw_data)
    
                
            # Generates masks


            mask = np.zeros(shape=int(np.shape(raw_data)[-1] / self.RESAMPLE_FS_RATIO))

            for step in time_annotations:

                center_index = np.where(file_info.times == step)[0][0]

                qrs_region = np.where(
                    (file_info.times[::self.RESAMPLE_FS_RATIO] > (step - self.STD_GAUSSIAN)) &
                    (file_info.times[::self.RESAMPLE_FS_RATIO] < (step + self.STD_GAUSSIAN))
                )[0]

                if self.TYPE_OF_MASK == 'none':
                    mask = np.ones(shape=int(np.shape(raw_data)[-1] / self.RESAMPLE_FS_RATIO))

                if self.TYPE_OF_MASK == 'gaussian':
                    mask[qrs_region] = gaussian(qrs_region, center_index / self.RESAMPLE_FS_RATIO, self.QRS_INTERVAL / 2)
                
                if self.TYPE_OF_MASK == 'triangle':
                    mask[qrs_region] = triangle(qrs_region, center_index / self.RESAMPLE_FS_RATIO, self.QRS_INTERVAL / 2)
                
            # Number of channels:
            if self.CHANNELS == 3:    
                if 'r10' in file: # abcd  file with wrong channel 
                    filedata = resampled_signal[[0, 1, 2, 4]]
                else:
                    filedata = resampled_signal[[0, 2, 3, 4]]
            
            if self.CHANNELS == 4:
                filedata = np.copy(raw_data)
            
            # If fecg dont exist in dataset, extract it from BSS ICA method
            if not self.CALC_FECG_FROM_BSS:
                
                tmpdata = filedata[2]  # randomly choose this channel to retrieve fecg
                
                # calculate the number of components using eigenvalues
                pca = PCA()
                pca.fit(tmpdata.reshape(-1, 1))
                perc = np.cumsum(pca.explained_variance_ratio_)
                number_components = np.argmax(perc >= 0.999) + 1            

                            
                transformer = FastICA(number_components)
                fecg_retrieved = transformer.fit_transform(tmpdata.reshape(-1, 1)) 
            
                # print('fecg retrieved shappe', np.shape(fecg_retrieved))
            
                filedata[0] = fecg_retrieved[:, 0]
            
            
            
            # Loop in data
            
            batch = 0
            index = 0              
                
            while batch <= np.shape(filedata)[-1] - self.INTERVAL_SIZE:
                
                if 'r10' in file and index in to_remove:
                    batch += self.INTERVAL_SIZE
                    index += 1
                    continue
                

                chunked_data = np.copy(filedata[1::, (batch): ((batch + self.INTERVAL_SIZE))].transpose())
                
                if np.shape(chunked_data.transpose())[1] != self.INTERVAL_SIZE:
                    continue
                
                chunked_fecg_real_data = np.copy(filedata[0, (batch): (batch + self.INTERVAL_SIZE)])
                chunked_fecg_binary_data = np.copy(mask[(batch): (batch + self.INTERVAL_SIZE)])
               
                # Data Normalization

                chunked_data -= np.min(chunked_data) # to zero things
                chunked_fecg_real_data -= np.min(chunked_fecg_real_data) # to zero things
                
                max_abdominal = np.abs(np.max(chunked_data)) if np.abs(np.max(chunked_data)) != 0 else 1e-7
                max_fecg = np.abs(np.max(chunked_fecg_real_data)) if np.abs(np.max(chunked_fecg_real_data)) != 0 else 1e-7
                

                chunked_data *= (1 / max_abdominal) 
                chunked_fecg_real_data *= (1 / max_fecg)


                if self.FILTERS_ON:
                    for i in range(3):
                        # print(np.shape(chunked_data[:, i]))
                        chunked_data[:, i] = self.butter_bandpass_filter(np.copy(chunked_data[:, i]), 1, 100, 1000 / self.RESAMPLE_FS_RATIO)
                    chunked_fecg_real_data = self.butter_bandpass_filter(np.copy(chunked_fecg_real_data), 1, 100, 1000 / self.RESAMPLE_FS_RATIO)

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

                    if self.OVERLAP_DATA and ((batch + self.INTERVAL_SIZE + int(self.INTERVAL_SIZE / 2))) <= np.shape(filedata)[-1]:

                        init_interval = batch + int(self.INTERVAL_SIZE / 2)
                        end_interval  = init_interval + self.INTERVAL_SIZE

                        overlapped_segment_aceg = np.copy(filedata[1::, init_interval: end_interval].transpose())

                        overlapped_segment_dfceg = np.copy(filedata[0, init_interval: end_interval])                       
                        overlapped_segment_mask = np.copy(mask[init_interval: end_interval])

                        # Normalization
                        overlapped_segment_aceg -= np.min(overlapped_segment_aceg)
                        overlapped_segment_dfceg -= np.min(overlapped_segment_dfceg)

                        max_abdominal = np.abs(np.max(overlapped_segment_aceg)) if np.abs(np.max(overlapped_segment_aceg)) != 0 else 1e-7
                        max_fecg = np.abs(np.max(overlapped_segment_dfceg)) if np.abs(np.max(overlapped_segment_dfceg)) != 0 else 1e-7
                

                        overlapped_segment_aceg *= (1 / max_abdominal) 
                        overlapped_segment_dfceg *= (1 / max_fecg)


                        overlapped_ground_truth = np.array([
                            overlapped_segment_dfceg, 
                            overlapped_segment_mask
                        ]).transpose()



                        aECG_store = np.vstack((aECG_store, [overlapped_segment_aceg]))
                        fECG_store = np.vstack((fECG_store, [overlapped_ground_truth]))
                

                batch += self.INTERVAL_SIZE
                index += 1


        try:
            return aECG_store, fECG_store
        except:
            return np.empty(shape=(0)), np.empty(shape=(0))

    def load_b2_dataset(self, dirs):

        """
        The b2 records are subdivided in folders for each pregancy, so you have to enter 
        the folder to get the information. 

        authors provide four electrode channels of aECG and four others with the 
        supression of mECG, that in our case where used as ground truth
        """

        aECG_data = []
        fECG_data = [] 

        for dir in dirs:
            
            print('Reading data from:', dir.replace(self.DATA_PATH, ''))    

            # Read the data

            # os.path.

            abdominal_file = glob.glob(dir + '/*abSignals*.txt')[0]
            dfECG_file = glob.glob(dir + '/*dFECG*.txt')[0]
            fetal_R_file   = glob.glob(dir + '/*Fetal_R*.txt')[0]

            aecg_signals = pd.read_csv(abdominal_file, delimiter='\t', header=None)
            dfecg_signals = pd.read_csv(dfECG_file, delimiter='\t', header=None)
            fR_annotation = pd.read_csv(fetal_R_file, delimiter='\t', header=None)

            DATA_LEN = aecg_signals[0].size

            # generate the chuncked arrays

            # the signal is read as a string, so u have to replace the commas before
            for i in range(8): # 8 columns
                aecg_signals[i] = aecg_signals[i].str.replace(',', '.').astype(np.float64)

            for i in range(2): # 2 columns
                dfecg_signals[i] = dfecg_signals[i].str.replace(',', '.').astype(np.float64)        

            aecg_signals = aecg_signals.to_numpy()
            dfecg_signals = dfecg_signals.to_numpy()
           
            # Direct fECG from B2 pregancy dataset has fs = 1kHz and the abdominal part 500Hz, so we downsample
            # the fECG to 500Hz

            resampled_signal = np.zeros(shape=(int(DATA_LEN), 2))
            
            for j in range(2):
                resampled_signal[:, j] = resample(dfecg_signals[:, j], int(DATA_LEN))

            dfecg_signals = np.copy(resampled_signal)
            del resampled_signal

            # Generate the mask

            mask = np.zeros(shape = (DATA_LEN))

            # the labour fetal R dont have the column 0 or 1 as the pregnancy one

            for r_peak in fR_annotation.to_numpy():
                r_peak = int(r_peak / 2)

                begin_of_interval = 0 if r_peak - 2 * self.QRS_INTERVAL < 0 else r_peak - 2 * self.QRS_INTERVAL
                end_of_interval = DATA_LEN if r_peak + 2 * self.QRS_INTERVAL > DATA_LEN else r_peak + 2 * self.QRS_INTERVAL

                if end_of_interval > begin_of_interval:
                    qrs_region = np.arange(begin_of_interval, end_of_interval, step=1)
                    mask[qrs_region] = gaussian(qrs_region, r_peak, self.QRS_INTERVAL/2)


            

            # reshape

            batch = 0

            while batch <= DATA_LEN - self.INTERVAL_SIZE:

                # print('linha inicial', np.shape(aECG_data), np.shape(fECG_data))
                
                aecg_seg = np.copy(aecg_signals[batch : batch + self.INTERVAL_SIZE, 0:3])
                fecg_seg = np.copy(dfecg_signals[batch : batch + self.INTERVAL_SIZE, 0])          
            
                # Normalize
                aecg_seg -= np.min(aecg_seg)
                max_aecg = np.abs(np.max(aecg_seg)) if np.abs(np.max(aecg_seg)) != 0 else 1e-7 
                aecg_seg *= 1 / max_aecg

                fecg_seg -= np.min(fecg_seg)
                max_fecg = np.abs(np.max(fecg_seg)) if np.abs(np.max(fecg_seg)) != 0 else 1e-7 
                fecg_seg *= 1 / max_fecg

                if self.FILTERS_ON:
                    for i in range(3):
                        aecg_seg[:, i]  = self.butter_bandpass_filter(np.copy(aecg_seg[:, i]), 1, 100, 1000 / self.RESAMPLE_FS_RATIO)
                    fecg_seg = self.butter_bandpass_filter(np.copy(fecg_seg), 1, 100, 1000 / self.RESAMPLE_FS_RATIO)
                
                    # Normalize
                    aecg_seg -= np.min(aecg_seg)
                    max_aecg = np.abs(np.max(aecg_seg)) if np.abs(np.max(aecg_seg)) != 0 else 1e-7 
                    aecg_seg *= 1 / max_aecg
    
                    fecg_seg -= np.min(fecg_seg)
                    max_fecg = np.abs(np.max(fecg_seg)) if np.abs(np.max(fecg_seg)) != 0 else 1e-7 
                    fecg_seg *= 1 / max_fecg

                # print('normalized', np.shape(aecg_seg), np.shape(fecg_seg))

                # print('mask', np.shape(mask[batch : batch + self.INTERVAL_SIZE]))

                mask_and_signal = np.array([
                    fecg_seg,
                    mask[batch : batch + self.INTERVAL_SIZE], 
                ]).transpose()

                # print('mask and signal', np.shape(mask_and_signal))


                # Append to array
                if len(aECG_data) == 0:

                    aECG_data = np.copy([aecg_seg])
                    fECG_data = np.array([mask_and_signal])

                else:

                    aECG_data = np.vstack((aECG_data, [aecg_seg]))
                    fECG_data = np.vstack((fECG_data, [mask_and_signal]))

                # print('linha final', np.shape(aECG_data), np.shape(fECG_data))


                batch += self.INTERVAL_SIZE


        return aECG_data, fECG_data

    def load_rpeak_annotations(self, read_file):

        fR_annotations = None

        match self.FILE_EXTENSION:

            case 'txt':

                dirnames = glob.glob(self.DATA_PATH + 'B*')
                dir = dirnames.pop(read_file)

                fetal_R_file   = glob.glob(dir + '/*Fetal_R*.txt')[0]
                fR_annotations = pd.read_csv(fetal_R_file, delimiter='\t', header=None)
                fR_annotations = fR_annotations.to_numpy().transpose()[0]

                fR_annotations = np.multiply(fR_annotations, 1 / self.RESAMPLE_FS_RATIO).astype(np.int32)
            
            case 'edf':

                filenames = glob.glob(self.DATA_PATH + "*." + self.FILE_EXTENSION) 
                test_file = filenames.pop(read_file)

                annotations = mne.read_annotations(test_file)
                fR_annotations = np.copy(annotations.onset)

                # the annnotations are in seconds, so I multiply by the sapling frequency
                fR_annotations = np.multiply(fR_annotations, 1000 / self.RESAMPLE_FS_RATIO).astype(np.int32)


        return fR_annotations
