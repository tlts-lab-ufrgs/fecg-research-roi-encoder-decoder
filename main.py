
#%%
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from numba import cuda

from data_load.data_loader import DataLoader
from models.ae_proposed_3blocks_upsampling import ProposedAE

#%% constants

# sa,mpling frequency
RESAMPLE_FREQ_RATIO = 2

# Range in learning rate
UPPER_LIM_LR = 0.0001

# batch size
BATCH_SIZE=4

# files 
TOTAL_FILES = 12

RESULTS_PATH = "/home/julia/Documents/research/sprint_1/results/"
#DATA_PATH =  "/home/julia/Documents/research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/"
DATA_PATH = "/home/julia/Documents/research/datasets/b2-records/B2_Labour_dataset/"
SAVE_MODEL_PATH = "/home/julia/Documents/research/sprint_1/3blocks_interpolation"
TYPE_OF_FILE='txt'

CHANNELS = 3
LEN_BATCH = int(1024 / RESAMPLE_FREQ_RATIO)
QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = int(50 / RESAMPLE_FREQ_RATIO)

MODEL_INPUT_SHAPE = (BATCH_SIZE, LEN_BATCH, CHANNELS)


W_MASK = 0.3
W_SIGNAL = 0.1
W_COMBINED = 1 - W_MASK - W_SIGNAL
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

model = ProposedAE(
    MODEL_INPUT_SHAPE, 
    BATCH_SIZE, 
    UPPER_LIM_LR, 
    W_MASK, 
    W_SIGNAL, 
    W_COMBINED, 
    training_data=training_data[0], 
    ground_truth=training_data[1],
    testing_data=None, 
    ground_truth_testing=None, 
    epochs=100
)

#%%

history, _, _ = model.fit_and_evaluate()

#%%
#model.model.export(SAVE_MODEL_PATH)


model.model.save(f'{SAVE_MODEL_PATH}/weights.h5')

# %%

# from models.ae_proposed import Metric, Loss
# from utils.masks_function import gaussian
# from data_load.load_leave_one_out import data_resizer
# from utils.lr_scheduler import callback as lr_scheduler

# %%

# model = tf.keras.models.load_model(
    # '/home/julia/Documents/research/sprint_1/model_rev0', 
    # custom_objects = {
        # 'mse_mask': Metric.mse_mask,
        # 'mse_signal': Metric.mse_signal, 
        # 'loss': Loss.loss, 
        # 'lr': lr_scheduler
    # }
# )

# model = tf.keras.layers.TFSMLayer(
    # '/home/julia/Documents/research/sprint_1/model_rev0', 
    # call_endpoint='serving_default'
    # # custom_objects = {
    # #     'mse_mask': Metric.mse_mask,
    # #     'mse_signal': Metric.mse_signal, 
    # #     'loss': Loss.loss, 
    # #     'lr': lr_scheduler
    # # }
# )

# %%
# predict = model.predict(training_data[0])
# # %%
# import matplotlib.pyplot as plt

# for index in range(2540, 2550):
    
    # fig, ax = plt.subplots()
    # # index = 2140
    # ax.plot(predict[index])

    # ax.plot(training_data[1][index])# %%

# %%
