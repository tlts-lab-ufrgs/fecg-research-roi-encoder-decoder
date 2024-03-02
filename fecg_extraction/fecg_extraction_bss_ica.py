"""
Open mECG files and extract using blind source ICA
"""

#%% Bibs 
import glob
import wfdb
import pprint  # doctest: +SKIP
import numpy as np
import pandas as pd
from oct2py import Oct2Py
import matplotlib.pyplot as plt

#%% Constants 

DATA_PATH = '/home/julia/Documents/fECG_research/datasets/ninfea-non-invasive-multimodal-foetal-ecg-doppler-dataset-for-antenatal-cardiology-research-1.0.0/wfdb_format_ecg_and_respiration'
RESULT_PATH = '/home/julia/Documents/fECG_research/datasets/ninfea-non-invasive-multimodal-foetal-ecg-doppler-dataset-for-antenatal-cardiology-research-1.0.0/csv_fecg_extracted'
SCRIPT_PATH = '/home/julia/fecgsyn/fecgsyn-master/subfunctions/extraction-methods/'
SCRIPT_NAME = 'FECGSYN_bss_extraction.m'

#%% Setup octave

oc = Oct2Py()
# to add a folder use:
oc.addpath(SCRIPT_PATH)  # doctest: +SKIP
# to add folder with all subfolder in it use:
# octave.addpath(octave.genpath("/path/to/directory"))  # doctest: +SKIP
# to run the .m file :

#%% Loop in files

files = glob.glob(f'{DATA_PATH}/*.hea')[0:1]


for file in files:
    signal, labels = wfdb.rdsamp(f'{DATA_PATH}/1') 

    result = oc.feval(
        SCRIPT_NAME, 
        signal, 
        'ICA', 
        2048, 
        512, 
        0,
        0
    )  # doctest: +SKIP
    # use nout='max_nout' to automatically choose max possible nout
    # octave.addpath("./example")  # doctest: +SKIP
    # out, oclass = octave.roundtrip(signal, nout='max_nout')  # doctest: +SKIP
    # pprint.pprint([out, oclass, out.dtype])  # doctest: +SKIP

#%% SandBox

# from oct2py import Oct2Py
# oc = Oct2Py()


script = "function y = myScript(x)\n" \
         "    y = x-5" \
         "end"

with open("myScript.m","w+") as f:
    f.write(script)

oc.myScript(7)