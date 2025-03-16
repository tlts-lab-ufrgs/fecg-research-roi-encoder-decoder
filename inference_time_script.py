#%% Imports
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from numba import cuda
import matplotlib.pyplot as plt
from datetime import datetime

from data_load.data_loader import DataLoader
from models.ae_proposed_upsampling import ProposedAE

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


#%%
model = ProposedAE(
    MODEL_INPUT_SHAPE, 
    BATCH_SIZE, 
    UPPER_LIM_LR, 
    w_mask, 
    w_signal, 
    w_combined, 
    training_data=[], 
    ground_truth=[],
    testing_data=None, 
    ground_truth_testing=None, 
    epochs=100
)

model.linknet()

#%%
model.model.load_weights('/home/julia/Documents/research/sprint_1/4blocks_interpolation/weights.h5')

#%%

data_loader = DataLoader(
    DATA_PATH, 
    LEN_BATCH, 
    RESAMPLE_FREQ_RATIO, 
    TYPE_OF_FILE, 
    QRS_DURATION_STEP, 
    QRS_DURATION, 
    load_training_set=True,
    load_testing_set=False,
    type_of_mask='gaussian', 
    # filters=True
)

#%%
training_data, _ = data_loader.data_load(0)

#%%

from time import time

start_time = time()

model.model.predict(training_data[0])

end_time = time()

len_file = 5 # minutes

total_time_per_min_ecg = (end_time - start_time) / len_file / TOTAL_FILES

print(f'Time per subject per minute {total_time_per_min_ecg}')

#%%

model.model.summary()