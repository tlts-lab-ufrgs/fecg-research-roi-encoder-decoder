import numpy as np


#%% Gaussian function

def gaussian(x, mu, sig):
    
    signal = 1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    
    return signal / np.max(signal)

def triangle(x, mu, sig):

    a_coef_increasing_fc = 1 / (mu - x[0])
    b_coef_increasing_fc = - a_coef_increasing_fc * x[0]

    a_coef_decreasing_fc = 1 / ( mu - x[-1] )
    b_coef_decreasing_fc = - a_coef_decreasing_fc * x[-1]

    increasing_line = a_coef_increasing_fc * x[:int(len(x) / 2)] + b_coef_increasing_fc
    decreasing_line = a_coef_decreasing_fc * x[int(len(x) / 2)::] + b_coef_decreasing_fc

    mask = np.append(increasing_line, decreasing_line)

    return mask / np.max(mask)

# %%
