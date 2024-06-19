#%% import 

import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
# from data_load.load_leave_one_out import data_loader
# from utils.mean_confidence_interval import mean_confidence_interval
    
#%% constants 


RESULTS_PATH = "/home/julia/Documents/research/sprint_1/results/ablation_extended/"
DATA_PATH =  "/home/julia/Documents/research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/"

#%%

TEST_TO_READ = ''
NUMBER_OF_FILES = 2

#%% 

#%%

prefix = '/home/julia/Documents/research/sprint_1/results/ablation_extended/160524-sin_act-wt_droout-upsampling-LR_0.0001-W_MASK_0.3-W_SIG_0.1-LEFT_4/160524-sin_act-wt_droout-upsampling-LR_0.0001-W_MASK_0.3-W_SIG_0.1-LEFT_4-training_history.csv'

fig, ax = plt.subplots()

for i in range(NUMBER_OF_FILES):
    ax.plot()


ax.set_xlabel('Epochs')
ax.set_ylabel('Loss ')

#%%

data = pd.read_csv(
    '/home/julia/Documents/research/sprint_1/results/ablation_extended/160524-sin_act-wt_droout-upsampling-LR_0.0001-W_MASK_0.3-W_SIG_0.1-LEFT_4/160524-sin_act-wt_droout-upsampling-LR_0.0001-W_MASK_0.3-W_SIG_0.1-LEFT_4-training_history.csv'
)
# %%

fig, ax = plt.subplots()

ax.plot(data['loss'], label='Loss')

ax.plot(data['mse_signal'], label='MSE Signal')

ax.plot(data['mse_mask'], label='MSE Mask')

ax.legend()
# %%
