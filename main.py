
#%%
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from numba import cuda

from data_load.load_leave_one_out import data_loader
from models.ae_proposed_rev0 import ProposedAE

#%% constants

# Range in learning rate
UPPER_LIM_LR = 0.001

# batch size
BATCH_SIZE=4

# files 
TOTAL_FILES = 5

RESULTS_PATH = "/home/julia/Documents/research/sprint_1/results/"
DATA_PATH =  "/home/julia/Documents/research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/"
SAVE_MODEL_PATH = "/home/julia/Documents/research/sprint_1/model_rev2/"

CHANNELS = 3
LEN_BATCH = 512
QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = 50

MODEL_INPUT_SHAPE = (BATCH_SIZE, LEN_BATCH, CHANNELS)


W_MASK = 0.3
W_SIGNAL = 0.1
W_COMBINED = 1 - W_MASK - W_SIGNAL
#%%

training_data, testing_data = data_loader(
                    DATA_PATH, 
                    LEN_BATCH, 
                    QRS_DURATION, 
                    QRS_DURATION_STEP,
                    whole_dataset_training=True,
                    leave_for_testing=0,
                    type_of_file='edf'
            )

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
    epochs=150
)

#%%

history, _, _ = model.fit_and_evaluate()

#%%
model.save(SAVE_MODEL_PATH)

#%%

from models.ae_proposed import Metric, Loss
from utils.masks_function import gaussian
from data_load.load_leave_one_out import data_resizer
from utils.lr_scheduler import callback as lr_scheduler

#%%

model = tf.keras.models.load_model(
    '/home/julia/Documents/research/sprint_1/model_rev0', 
    custom_objects = {
        'mse_mask': Metric.mse_mask,
        'mse_signal': Metric.mse_signal, 
        'loss': Loss.loss, 
        'lr': lr_scheduler
    }
)

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
predict = model.predict(training_data[0])
# # %%
# import matplotlib.pyplot as plt

# for index in range(2540, 2550):
    
#     fig, ax = plt.subplots()
#     # index = 2140
#     ax.plot(predict[index])

#     ax.plot(training_data[1][index])# %%

# %%
