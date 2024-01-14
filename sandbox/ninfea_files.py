
#%%
# 
import numpy as np
import wfdb
from scipy.io import loadmat
import matplotlib.pyplot as plt

#%%

FILENUMBER = '3'

PATH = '/home/julia/Documents/fECG_research/datasets/ninfea-non-invasive-multimodal-foetal-ecg-doppler-dataset-for-antenatal-cardiology-research-1.0.0/'

#%%
record = wfdb.rdrecord(f'{PATH}wfdb_format_ecg_and_respiration/{FILENUMBER}') 
# wfdb.plot_wfdb(record=record, title='Record a103l from PhysioNet Challenge 2015') 


#%%

doppler_signals = loadmat(f'{PATH}pwd_signals/{FILENUMBER}envelopes.mat')

#%%

# sig_name > signal name, all columns 
# p_signal > measured signal

record.sig_name
record.p_signal

#%%


fig, ax = plt.subplots(figsize=(16,9))

ax.set_title('Thoraxic ECG per time step')


ax.plot(
    np.arange(0, 500, 1), 
    record.p_signal[:, record.sig_name.index("bi_tho1")][:500], 
    label='Channel Thorax 1'
)

ax.plot(
    np.arange(0, 500, 1), 
    record.p_signal[:, record.sig_name.index("bi_tho2")][:500], 
    label='Channel Thorax 2'
)

ax.plot(
    np.arange(0, 500, 1), 
    record.p_signal[:, record.sig_name.index("bi_tho3")][:500], 
    label='Channel Thorax 3'
)

ax.set_ylabel('Thoraxic ECG')
ax.set_xlabel('Timesteps')

# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
          fancybox=True, shadow=True, ncol=5)

ax.grid()

plt.show()


# %%


ig, ax = plt.subplots(figsize=(16,9))

ax.set_title('Abdominal ECG per time step - File 3')

for i in range(1,24,1):

    ax.plot(
        np.arange(0, 500, 1), 
        record.p_signal[:, record.sig_name.index(f"uni_abd{i}")][:500], 
        label=f'Channel Abdominal {i}'
    )


ax.set_ylabel('Abdominal ECG')
ax.set_xlabel('Timesteps')

# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
          fancybox=True, shadow=True, ncol=5)

ax.grid()

plt.show()

# %%
