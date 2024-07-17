#%%

import pandas as pd
import matplotlib.pyplot as plt



RESULTS_PATH = "/home/julia/Documents/research/sprint_1/results/ablation_extended/"

ABLATION_TEST = '2024-07-10-MASK_gaussian-DECODER_BY_convtransp-two_partial_branches-LR_0.0001-W_MASK_0.3-W_SIG_0.1-LEFT_4'

data = pd.read_csv(f'{RESULTS_PATH}{ABLATION_TEST}/{ABLATION_TEST}-training_history.csv')

plt.plot(data['loss'], label='Loss')
# plt.plot(data['mse_signal'], label='MSE Signal')
plt.plot(data['mse_mask'], label='MSE Mask')

# plt.vlines(x=10, ymin=0, ymax=0.1)

plt.legend()
plt.grid()
# plt.plot(data['val_loss'])
# %%
