
#%% Imports
import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from numba import cuda
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from data_load.load_leave_one_out import data_loader, data_resizer
from models.ae_proposed import ProposedAE

#%% To run other experiments please change this below

# CHANGEBLE VARIABLES ---------------------------------------------------------------------------------------------- 
TOTAL_FILES = 55
CHANNELS = 3
RESAMPLE_FREQ_RATIO = 1
HAVE_DIRECT_FECG = False

RESULTS_PATH = "/home/julia/Documents/fECG_research/research_dev/autoencoder_with_mask/results/"
DATA_PATH =  "/home/julia/Documents/fECG_research/datasets/non-invasive-fetal-ecg-database-1.0.0/"
# -------------------------------------------------------------------------------------------------------------------

#%% Model constants

BATCH_SIZE=4
UPPER_LIM_LR = 0.0001
LEN_BATCH = int(512 / RESAMPLE_FREQ_RATIO)
QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = int(50 / RESAMPLE_FREQ_RATIO)

MODEL_INPUT_SHAPE = (BATCH_SIZE, LEN_BATCH, CHANNELS)

w_mask = 0.3
w_signal = 0.1
w_combined = 1 - w_mask - w_signal

#%% Loop in files to run cross-validation 

for i in range(0, 1, 1):
    
    prefix_id = f'080324-NIFECG-3CH-1000-LR_{UPPER_LIM_LR}-W_MASK_{w_mask}-W_SIG_{w_signal}-LEFT_{i}'
    
    print(prefix_id)
    
    # try:
        # training_data, testing_data = data_loader(
        #         DATA_PATH, 
        #         LEN_BATCH, 
        #         QRS_DURATION, 
        #         QRS_DURATION_STEP,
        #         leave_for_testing=i,
        #         type_of_file='edf', 
        #         resample_fs=RESAMPLE_FREQ_RATIO, 
        #         whole_dataset_training=True,
        #         channels=CHANNELS, 
        #         fecg_on_gt=HAVE_DIRECT_FECG
        # )
    filenames = glob.glob(DATA_PATH + "*.edf")
    # test_file = filenames[1]
    
    
    all_data = data_resizer(
        filenames, 
        LEN_BATCH, 
        QRS_DURATION, 
        QRS_DURATION_STEP, 
        'edf', 
        resample_fs=RESAMPLE_FREQ_RATIO, 
        channels = CHANNELS,  
        fecg_on_gt = HAVE_DIRECT_FECG
    )
    
    training_data, testing_data = train_test_split(all_data, test_size=0.2)
    
        # testing_data = data_resizer(
        #     [test_file], 
        #     LEN_BATCH, 
        #     QRS_DURATION, 
        #     QRS_DURATION_STEP, 
        #     'edf', 
        #     resample_fs=RESAMPLE_FREQ_RATIO,
        #     channels = CHANNELS, 
        #     fecg_on_gt = HAVE_DIRECT_FECG
        # )

    
    model = ProposedAE(
        MODEL_INPUT_SHAPE, 
        BATCH_SIZE, 
        UPPER_LIM_LR, 
        w_mask, 
        w_signal, 
        w_combined, 
        training_data=training_data[0], 
        ground_truth=training_data[1],
        testing_data=testing_data[0], 
        ground_truth_testing=testing_data[1], 
        epochs=100
    )

    history, testing_metrics, predict = model.fit_and_evaluate()


    print(prefix_id)
    print(testing_metrics)


    # save things

    this_dir = os.path.join(RESULTS_PATH, prefix_id)


    os.mkdir(this_dir)

    pd.DataFrame.from_dict(history.history).to_csv(this_dir + '/' + prefix_id + '-training_history.csv')

    index = 0
    for batch_pred in predict:
        np.savetxt(this_dir + '/' + prefix_id + f'-prediction_{index}.csv', batch_pred, delimiter=',')
        index += 1

    del training_data
    del testing_data
    del model
    del history
    del predict  


# %%
