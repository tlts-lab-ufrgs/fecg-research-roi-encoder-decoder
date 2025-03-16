
#%% Imports
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from numba import cuda
import matplotlib.pyplot as plt
from datetime import datetime

from data_load.data_loader import DataLoader
from models.ae_proposed_3blocks import ProposedAE

#%% To run other experiments please change this below

# CHANGEBLE VARIABLES ---------------------------------------------------------------------------------------------- 
TOTAL_FILES = 12
CHANNELS = 3
RESAMPLE_FREQ_RATIO = 2
HAVE_DIRECT_FECG = True

RESULTS_PATH = "/home/julia/Documents/research/sprint_1/results/ablation_extended/"
DATA_PATH = "/home/julia/Documents/research/datasets/b2-records/B2_Labour_dataset/"
# DATA_PATH =  "/home/julia/Documents/research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/"
# -------------------------------------------------------------------------------------------------------------------

#%% Model constants

BATCH_SIZE=4
UPPER_LIM_LR = 0.0001
LEN_BATCH = int(512 / RESAMPLE_FREQ_RATIO)
QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = int(50 / RESAMPLE_FREQ_RATIO)

MODEL_INPUT_SHAPE = (BATCH_SIZE, LEN_BATCH, CHANNELS)

TYPE_OF_FILE = 'txt'

w_mask = 0.3
w_signal = 0.1
w_combined = 1 - w_mask - w_signal

type_of_mask = 'gaussian'
decoder_type = 'transpose'

NUMBER_OF_EPOCHS = 100


today = datetime.today().strftime('%Y-%m-%d')

#%% If you want to loop in weight parameters

data_loader = DataLoader(
    DATA_PATH, 
    LEN_BATCH, 
    RESAMPLE_FREQ_RATIO, 
    TYPE_OF_FILE, 
    QRS_DURATION_STEP, 
    QRS_DURATION, 
    load_training_set=True,
    type_of_mask=type_of_mask, 
    # filters=True
)

#%% Loop in files to run cross-validation 

for i in  range(0, TOTAL_FILES):
    
    prefix_id = f'{today}-3B-4B-MASK_{type_of_mask}-DECODER_BY_{decoder_type}-LR_{UPPER_LIM_LR}-W_MASK_{w_mask}-W_SIG_{w_signal}-LEFT_{i}'
    
    print(prefix_id)

        # save things

    this_dir = os.path.join(RESULTS_PATH, prefix_id)
    os.mkdir(this_dir)

    # os.mkdir(this_dir + '/model')
    # os.mkdir(this_dir + '/qz_model')

    training_data, testing_data = data_loader.data_load(i)
       
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
        epochs=NUMBER_OF_EPOCHS, 
        # saved_model_dir=this_dir
    )

    history, testing_metrics, predict = model.fit_and_evaluate()


    # print(prefix_id)
    # print(testing_metrics)


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

#%%
