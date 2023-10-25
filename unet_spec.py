#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fECG extraction with ML / DL

Created on Sunday Oct 22 08:30:00 2023

@author: juliacremus
"""
#%%
import mne
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from sklearn import preprocessing

import tensorflow as tf

from models.unet_model import unet

#%% Set model parameters

INPUT_SHAPE = [512, 512, 4] # length, channels (stacked version), depth
OUTPUT_CHANNELS = 512

#%% Data Import

PATH = '/home/julia/Documentos/ufrgs/Mestrado/fECG - research/scripts/fecg-denoising/data'

file_info = mne.io.read_raw_edf(f'{PATH}/r04.edf')
annotations = mne.read_annotations(f'{PATH}/r04.edf')

channel_names = file_info.ch_names

raw_data = file_info.get_data()