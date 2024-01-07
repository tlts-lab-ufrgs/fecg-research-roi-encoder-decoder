
#%%
# 
import wfdb

record = wfdb.rdrecord('/home/julia/Documents/fECG_research/datasets/ninfea-non-invasive-multimodal-foetal-ecg-doppler-dataset-for-antenatal-cardiology-research-1.0.0/wfdb_format_ecg_and_respiration/3') 
# wfdb.plot_wfdb(record=record, title='Record a103l from PhysioNet Challenge 2015') 

#%%

# sig_name > signal name, all columns 
# p_signal > measured signal

record.sig_name
record.p_signal

#%%

import matplotlib.pyplot as plt


fig, ax = plt.subplots()

ax.plot(record.p_signal[:, record.sig_name.index("matrsp")])

ax1 = ax.twinx()

ax1.plot(record.p_signal[:, record.sig_name.index("bi_tho3")], color='orange')

plt.show()
# %%
