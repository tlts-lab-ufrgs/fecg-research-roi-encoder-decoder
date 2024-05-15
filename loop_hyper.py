
#%% Imports
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from numba import cuda
import matplotlib.pyplot as plt

from data_load.load_leave_one_out import data_loader
from models.ae_proposed import ProposedAE

#%% To run other experiments please change this below

# CHANGEBLE VARIABLES ---------------------------------------------------------------------------------------------- 
TOTAL_FILES = 5
CHANNELS = 3
RESAMPLE_FREQ_RATIO = 1
HAVE_DIRECT_FECG = True

RESULTS_PATH = "/home/julia/Documents/research/sprint_1/results/ablation_extended/"
DATA_PATH =  "/home/julia/Documents/research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/"
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

#%% If you want to loop in weight parameters


# for w_mask in np.arange(0.8, 1.1, 0.1):
        
#     w_signal_upper_bound = 1 - w_mask
    
#     w_mask=0.5

#     for w_signal in np.arange(0.5, w_signal_upper_bound + 0.1, 0.1):

# # for i in [
# #     [0, 0.7], [0.2, 0.2], [0.1, 0.1], [0.3, 0.4]
# # ]:
    
#     # w_mask = i[0]
#     # w_signal = i[1]

#%% Loop in files to run cross-validation 

for i in range(0, TOTAL_FILES, 1):
    
    prefix_id = f'140524-sin_act-upsampling-LR_{UPPER_LIM_LR}-W_MASK_{w_mask}-W_SIG_{w_signal}-LEFT_{i}'
    
    print(prefix_id)
    
    training_data, testing_data = data_loader(
            DATA_PATH, 
            LEN_BATCH, 
            QRS_DURATION, 
            QRS_DURATION_STEP,
            leave_for_testing=i,
            type_of_file='edf', 
            resample_fs=RESAMPLE_FREQ_RATIO, 
            channels=CHANNELS, 
            fecg_on_gt=HAVE_DIRECT_FECG
    )
    
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
