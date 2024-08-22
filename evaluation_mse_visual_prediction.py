"""
Evaluate results from loop in hyperparameters

18 februaty 2024
"""


#%% import 

import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from data_load.load_leave_one_out import data_loader
from utils.stats_fn import mean_confidence_interval

from data_load.data_loader import DataLoader
    
#%% constants 


RESULTS_PATH = "/home/julia/Documents/research/sprint_1/results/ablation_extended/"
DATA_PATH =  "/home/julia/Documents/research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/"
#DATA_PATH = "/home/julia/Documents/research/datasets/b2-records/B2_Labour_dataset/"

ABLATION_TEST = '2024-08-15-MASK_gaussian-DECODER_BY_convtranspose-ED_rev0-B2-500hz-LR_0.0001'


SAMPLING_FREQ = 500

CHANNELS = 3
RESAMPLING_FREQUENCY_RATIO = int(1000 / SAMPLING_FREQ)

# All constants are defined based on a 1000Hz fs
LEN_BATCH = int(512 / RESAMPLING_FREQUENCY_RATIO)
LIMT_GAUS = int(30 / RESAMPLING_FREQUENCY_RATIO)
QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = int(50 / RESAMPLING_FREQUENCY_RATIO)
MIN_QRS_DISTANCE = int(300 / RESAMPLING_FREQUENCY_RATIO) # fs = 1000Hz
MASK_MIN_HEIGHT = 0.7

LIMIT = int(300000 / RESAMPLING_FREQUENCY_RATIO)# - LEN_BATCH

TEST_FILE = 0

TYPE_OF_FILE = 'edf'

type_of_mask = 'gaussian'

NUMBER_OF_FILES = 5

plt.rcParams["font.family"] = "Times New Roman"

#%%

def mse_function(y_true, y_pred):
    
    mse_value = np.mean(
        np.abs((y_true - y_pred)) # , 2)
    )
    
    return mse_value


data_loader = DataLoader(
    DATA_PATH, 
    LEN_BATCH, 
    RESAMPLING_FREQUENCY_RATIO, 
    TYPE_OF_FILE, 
    QRS_DURATION_STEP, 
    QRS_DURATION, 
    load_training_set=False, 
)

#%% data load

testing_data = {}

for i in range(NUMBER_OF_FILES):
    
    _, this_testing_data = data_loader.data_load(i)

    fecg_testing_data = this_testing_data[1] # the first is the aECG, the second is the ground truth signal
    #fecg_roi = fecg_testing_data[:, :, 0] * fecg_testing_data[:, :, 1]
    fecg_roi = fecg_testing_data[:, :, 0] * np.where(
        fecg_testing_data[:, :, 1] == 0, 
        0, 
        1
    )
    
    
    testing_data[i] = {
        'signal': fecg_testing_data,
        'roi_signal': fecg_roi, 
        'binary_true_mask': np.where(
            fecg_testing_data[:, :, 1] == 0, 
            0, 
            1
        )
    }


#%% concat results of the same dir
results_dir = glob.glob(RESULTS_PATH + ABLATION_TEST + '*')
results_rows = []

for i in results_dir:
    
    w_mask = float(i.split('-W_MASK_')[1].split('-')[0])
    w_signal = float(i.split('-W_SIG_')[1].split('-')[0])

    if i.split("-")[-1] == 'model':
        continue 

    test_file = int(i.split("-")[-1].replace('LEFT_', ''))
    
    this_row = [test_file, w_mask, w_signal]

    result_files = glob.glob(i + '/' + '*prediction_*')
    
    mse_signal, mse_mask, mse_combined = 0, 0, 0
    r_squared_signal, r_squared_mask, r_squared_combined = 0, 0, 0
    
    for file in result_files:
        
        prediction_index = int(file.split('-prediction_')[1].split('-')[0].replace('.csv', ''))
               
        prediction_data = pd.read_csv(file, names=['signal', 'mask'])
        prediction_data['binary_mask'] = prediction_data['mask'].where(prediction_data['mask'] == 0, 1)
        
        
        prediction_data['combined'] = prediction_data['signal'] * prediction_data['binary_mask']
        mse_mask_partial = mse_function(testing_data[test_file]['signal'][prediction_index, :, 1], prediction_data['mask'])
        prediction_data['roi_signal'] = prediction_data['signal'] * testing_data[test_file]['binary_true_mask'][prediction_index] #prediction_data['mask'] # testing_data[test_file]['binary_true_mask'][prediction_index] # 
        
        mse_signal_partial = mse_function(testing_data[test_file]['signal'][prediction_index, :, 0], prediction_data['signal'])
        mse_mask_partial = mse_function(testing_data[test_file]['signal'][prediction_index, :, 1], prediction_data['mask'])
        mse_combined_partial = mse_function(
            testing_data[test_file]['roi_signal'][prediction_index], 
            prediction_data['roi_signal'])
           
        mse_signal += mse_signal_partial
        mse_mask += mse_mask_partial
        mse_combined += mse_combined_partial
        
    

        r_squared_signal_partial = (r2_score(testing_data[test_file]['signal'][prediction_index, :, 0], prediction_data['signal']))
        r_squared_mask_partial = r2_score(testing_data[test_file]['signal'][prediction_index, :, 1], prediction_data['mask'])
        r_squared_combined_partial = r2_score(
            testing_data[test_file]['roi_signal'][prediction_index], 
            prediction_data['roi_signal'])
           
        r_squared_signal += (r_squared_signal_partial)
        r_squared_mask += (r_squared_mask_partial)
        r_squared_combined += (r_squared_combined_partial)
        
        # false_positive = [
        # ]
        
        # if prediction_index in [int(i / 512) for i in false_positive] and test_file == 0:
           
        if prediction_index in [
            0,11,20,30,40,42,50,60,70,80,90,100,150,250,300 
        ]: 
            fig, ax = plt.subplots()
            
            # ax.set_title('')
            ax.set_title(f'W mask {w_mask}, W signal {w_signal} - {prediction_index} - {test_file}')
            
 
            ax.plot(
                testing_data[test_file]['signal'][prediction_index, :, 0], 
                label='Ground truth signal', 
                )
            
            
            ax.plot(prediction_data['signal'], label='Predicted Signal')
            
            ax1 = ax.twinx()
            
            ax1.plot(
                testing_data[test_file]['signal'][prediction_index, :, 1], 
                label='Ground truth RoI', 
                color='green'
                )
            ax1.plot(prediction_data['mask'], label='Predicted RoI', color='purple')
        
            
            ax.set_xlabel('Time steps')
            ax.set_ylabel('fECG normalized')
            ax1.set_ylabel('RoI signal')
            
            # Shrink current axis's height by 10% on the bottom
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                            box.width, box.height * 0.9])

            # Put a legend below current axis
            ax.legend(loc='upper center', bbox_to_anchor=(0.1, -0.15),
                    fancybox=True, shadow=True, ncol=2)
            
            ax1.legend(loc='upper center', bbox_to_anchor=(0.9, -0.15),
                    fancybox=True, shadow=True, ncol=2)
            
            
            
            ax.grid()
            
            # ax.plot(testing_data[test_file]['roi_signal'][prediction_index], label='fECG')
            # ax.plot(prediction_data['combined'], label='Model Signal')
            
            # ax.legend()
       
    this_row.append(mse_signal / len(result_files))
    this_row.append(mse_mask / len(result_files))
    this_row.append(mse_combined / len(result_files))
    this_row.append(r_squared_signal / len(result_files))
    this_row.append(r_squared_mask / len(result_files))
    this_row.append(r_squared_combined / len(result_files))
    
    results_rows.append(this_row)

#%% form data frame

metrics_dataframe = pd.DataFrame(
    np.array(results_rows), 
    columns=[
        'test_file', 
        'w_mask', 
        'w_signal', 
        'mse_signal', 
        'mse_mask', 
        'mse_combined', 
        'r2_signal', 
        'r2_mask', 
        'r2_combined']
)

#%%

metrics_dataframe.sort_values(by = ['mse_mask'], inplace=True)

#%%

a = metrics_dataframe.groupby(['w_mask', 'w_signal']).mean()

#%%

print(mean_confidence_interval(
    metrics_dataframe['mse_signal'].values, 'MAE Signal'
))

print(mean_confidence_interval(
    metrics_dataframe['mse_mask'].values, 'MAE Mask'
))

print(mean_confidence_interval(
    metrics_dataframe['mse_combined'].values, 'MAE RoI'
))

print(mean_confidence_interval(
    metrics_dataframe['r2_signal'].values, 'R2 Signal'
))

print(mean_confidence_interval(
    metrics_dataframe['r2_mask'].values, 'R2 Mask'
))

print(mean_confidence_interval(
    metrics_dataframe['r2_combined'].values, 'R2 RoI'
))
# %% plot single results

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif"
# })

# test_subject = 5

# EXP_TO_PLT = f"2024-08-15-MASK_gaussian-DECODER_BY_convtranspose-ED_rev0-B2-500hz-LR_0.0001-W_MASK_0.3-W_SIG_0.1-LEFT_{test_subject}"

# index = 80

# file_to_plot = RESULTS_PATH + EXP_TO_PLT + "/" + EXP_TO_PLT + f"-prediction_{index}.csv"

# prediction_data = pd.read_csv(file_to_plot, names=['signal', 'mask'])
# prediction_data['binary_mask'] = prediction_data['mask'].where(prediction_data['mask'] == 0, 1)


# prediction_data['combined'] = prediction_data['signal'] * prediction_data['binary_mask']
# prediction_data['roi_signal'] = prediction_data['signal'] * prediction_data['mask'] #testing_data[test_file]['binary_true_mask'][prediction_index]


# fig, ax = plt.subplots(figsize=(8,3))
            
# #ax.set_title('')
# #ax.set_title(f'W mask {w_mask}, W signal {w_signal} - {prediction_index} - {test_file}')


# ax.plot(
#     testing_data[test_subject]['signal'][index, :, 0], 
#     label='$\mathbf{s}$', 
#     )


# ax.plot(prediction_data['signal'], label='$\mathbf{\\bar{s}}$')

# ax1 = ax.twinx()

# ax1.plot(
#     testing_data[test_subject]['signal'][index, :, 1], 
#     label='$\mathbf{m}$', 
#     color='green'
#     )
# ax1.plot(prediction_data['mask'], label='$\mathbf{\\bar{m}}$', color='purple')


# ax.set_xlabel('Time steps', fontsize='large')
# ax.set_ylabel('fECG normalized',  fontsize='large')
# ax1.set_ylabel('RoI signal', fontsize='large')

# #Shrink current axis's height by 10% on the bottom
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                 box.width, box.height * 0.9])

# #Put a legend below current axis
# ax.legend(loc='upper center', bbox_to_anchor=(0.3, -0.20),
#         fancybox=True, shadow=True, ncol=2)

# ax1.legend(loc='upper center', bbox_to_anchor=(0.6, -0.20),
#         fancybox=True, shadow=True, ncol=2)



# ax.grid()


# fig.savefig('r03-prediction_index-80-b2_dataset-model_rev1.pdf', bbox_inches='tight')
# %% compare the mse results for different roi masks

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

test_subject = 4


EXP_TO_PLT_WITH_MASK = f"2024-08-13-MASK_gaussian-DECODER_BY_convtransp-ED_rev0-500hz-testelr-LR_0.0001-W_MASK_0.3-W_SIG_0.1-LEFT_{test_subject}"

EXP_TO_PLT_WT_MASK = f"2024-08-20-MASK_none-DECODER_BY_contranspose-ED_rev0-ABCD-wt_RoI-500hz-LR_0.0001-W_MASK_0.0-W_SIG_1-LEFT_{test_subject}"


index = 50

file_to_plot_with_mask = RESULTS_PATH + EXP_TO_PLT_WITH_MASK + "/" + EXP_TO_PLT_WITH_MASK + f"-prediction_{index}.csv"
file_to_plot_wt_mask = RESULTS_PATH + EXP_TO_PLT_WT_MASK + "/" + EXP_TO_PLT_WT_MASK + f"-prediction_{index}.csv"


prediction_data = pd.read_csv(file_to_plot_with_mask, names=['signal', 'mask'])
prediction_data['binary_mask'] = prediction_data['mask'].where(prediction_data['mask'] == 0, 1)


prediction_data['combined'] = prediction_data['signal'] * prediction_data['binary_mask']
prediction_data['roi_signal'] = prediction_data['signal'] * prediction_data['mask'] #testing_data[test_file]['binary_true_mask'][prediction_index]


prediction_data_wt_mask = pd.read_csv(file_to_plot_wt_mask, names=['signal', 'mask'])
prediction_data_wt_mask['binary_mask'] = prediction_data_wt_mask['mask'].where(prediction_data_wt_mask['mask'] == 0, 1)


prediction_data_wt_mask['combined'] = prediction_data_wt_mask['signal'] * prediction_data_wt_mask['binary_mask']
prediction_data_wt_mask['roi_signal'] = prediction_data_wt_mask['signal'] * prediction_data_wt_mask['mask'] #testing_data[test_file]['binary_true_mask'][prediction_index]






fig, ax = plt.subplots(figsize=(8,3))
            

ax.plot(
    testing_data[test_subject]['signal'][index, :, 0], 
    label='$\mathbf{s}$', 
    )


ax.plot(prediction_data['signal'], label='$\mathbf{\\bar{s}_{RoI}}$')

ax.plot(prediction_data_wt_mask['signal'], label='$\mathbf{\\bar{s}_{WTRoI}}$')


ax.set_xlabel('Time steps', fontsize='large')
ax.set_ylabel('fECG normalized', fontsize='large')

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                box.width, box.height * 0.9])

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20),
        fancybox=True, shadow=True, ncol=3)




ax.grid()

fig.savefig('r10-prediction_index-50-abcd_dataset-model_rev0.pdf', bbox_inches='tight')
# %%
