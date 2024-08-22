#%%

import pandas as pd
import matplotlib.pyplot as plt

RESULTS_PATH = "/home/julia/Documents/research/sprint_1/results/ablation_extended/"
#%%

ABLATION_TEST = '2024-08-14-MASK_gaussian-DECODER_BY_upsampling-ED_rev0-ABCD-500hz-LR_0.0001-W_MASK_0.3-W_SIG_0.1-LEFT_1'

data = pd.read_csv(f'{RESULTS_PATH}{ABLATION_TEST}/{ABLATION_TEST}-training_history.csv')

plt.plot(data['loss'], label='Loss')
plt.plot(data['mse_signal'], label='MSE Signal')
plt.plot(data['mse_mask'], label='MSE Mask')

# plt.vlines(x=10, ymin=0.0, ymax=0.04)

# plt.ylim(0.0, 0.01)

plt.legend()
plt.grid()
# plt.plot(data['val_loss'])

# %% em loop


ABLATION_TEST = '2024-08-12-MASK_gaussian-DECODER_BY_convtransp-ED_rev0-500hz-ABCD-3CH-DA_with_noise-LR_0.0001-W_MASK_0.3-W_SIG_0.1-LEFT_'

for i in range(0, 5):

    data = pd.read_csv(f'{RESULTS_PATH}{ABLATION_TEST}{i}/{ABLATION_TEST}{i}-training_history.csv')

    #plt.plot(data['loss'], label=f'{i} - Loss')
    plt.plot(data['mse_signal'], label=f'{i} - MSE Signal')
    plt.plot(data['mse_mask'], label=f'{i} - MSE Mask')

# plt.vlines(x=10, ymin=0.0, ymax=0.04)

# plt.ylim(0.0, 0.01)

plt.legend()
plt.grid()
# plt.plot(data['val_loss'])



# %%
