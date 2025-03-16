#%%
import numpy as np
import matplotlib.pyplot as plt

subjects = np.arange(1,13)

metrics_model1_pre = [
    99.38, 95.28, 74.68, 96.62, 99.69, 99.85, 97.65, 99.22, 98.20, 97.80, 98.31, 97.70, 
]

metrics_model1_re = [
    99.22, 98.58, 66.11, 97.77, 99.09, 99.71, 98.73, 98.91, 97.03, 99.36, 99.38, 97.26, 
]

metrics_model2_pre = [
    99.53, 94.40, 77.40, 96.0, 99.54, 99.41, 97.44, 98.76, 96.10, 99.51, 98.17, 97.49, 
]

metrics_model2_re = [
    99.22,  98.28,  58.54,  95.29,  98.94,  99.12,  96.35,  99.38,  95.39,  97.28,  99.53,  94.66,  
]

WF_dB = [6.7, 5.0, -2.7, 2.3, 3.3, 3.7, 0.0, 7.0, 3.2, 7.1, 6.6, 3.8]

loss_percent = [1.7, 5.8, 27.5, 2.5, 1.7, 0.0, 0.0, 0.0, 0.8, 0.0, 3.3, 0.0]


fig, ax = plt.subplots()

difference = np.array(metrics_model1_pre) - np.array(metrics_model2_pre)

# ax.bar(subjects, difference)
ax.plot(subjects, metrics_model1_re)
ax.plot(subjects, metrics_model2_re)

# ax.set_ylim(75,100)

ax1 = ax.twinx()

ax1.plot(subjects, loss_percent, color='black')

# %%
