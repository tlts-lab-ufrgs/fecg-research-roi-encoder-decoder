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
from utils.mean_confidence_interval import mean_confidence_interval
    
#%% constants 


RESULTS_PATH = "/home/julia/Documents/research/sprint_1/results/ablation_extended/"
DATA_PATH =  "/home/julia/Documents/research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/"

ABLATION_TEST = '2024-06-19-MASK_triangle-DECODER_BY_upsampling'

CHANNELS = 3
LEN_BATCH = 512
QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = 50

TEST_FILE = 0

#%%

def mse_function(y_true, y_pred):
    
    mse_value = np.mean(
        np.abs((y_true - y_pred)) # , 2)
    )
    
    return mse_value

#%% data load

testing_data = {}

for i in range(5):
    
    _, this_testing_data = data_loader(
                DATA_PATH, 
                LEN_BATCH, 
                QRS_DURATION, 
                QRS_DURATION_STEP,
                leave_for_testing=i,
                type_of_file='edf', 
                type_of_mask="triangle"
            )

    fecg_testing_data = this_testing_data[1]
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
        prediction_data['roi_signal'] = prediction_data['signal'] * testing_data[test_file]['binary_true_mask'][prediction_index]
        
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
            1,2,3,4,5,6,7,8,9,10,11,12,13
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
# %%


# #%%

# model = '060324-3CH-MOD2-LR_0.0001-W_MASK_0.3-W_SIG_0.1-LEFT_'

# plot1 = pd.read_csv(f'{RESULTS_PATH}{model}0/{model}0-prediction_1.csv', names=['signal', 'mask'])
# plot2 = pd.read_csv(f'{RESULTS_PATH}{model}1/{model}1-prediction_5.csv', names=['signal', 'mask'])
# plot3 = pd.read_csv(f'{RESULTS_PATH}{model}2/{model}2-prediction_70.csv', names=['signal', 'mask'])


# fig, ax = plt.subplots(3, 1, sharex=True)

# # High SNR
# ax[0].plot(
#     testing_data[1]['signal'][5, :, 0], 
#     label='Ground truth signal', 
#     )

# ax[0].plot(plot2['signal'], label='Predicted Signal')

# ax1 = ax[0].twinx()

# ax1.plot(
#     testing_data[1]['signal'][5, :, 1], 
#     label='Ground truth RoI', 
#     color='green'
#     )
# ax1.plot(plot2['mask'], label='Predicted RoI', color='purple')

# # Noisy enviroment
# ax[1].plot(
#     testing_data[0]['signal'][1, :, 0], 
#     label='Ground truth signal', 
#     )

# ax[1].plot(plot1['signal'], label='Predicted Signal')

# ax1 = ax[1].twinx()

# ax1.plot(
#     testing_data[0]['signal'][1, :, 1], 
#     label='Ground truth RoI', 
#     color='green'
#     )
# ax1.plot(plot1['mask'], label='Predicted RoI', color='purple')

# # Two masks enviroment
# ax[2].plot(
#     testing_data[2]['signal'][70, :, 0], 
#     label='Ground truth signal', 
#     )

# ax[2].plot(plot3['signal'], label='Predicted Signal')

# ax1 = ax[2].twinx()

# ax1.plot(
#     testing_data[2]['signal'][70, :, 1], 
#     label='Ground truth RoI', 
#     color='green'
#     )
# ax1.plot(plot1['mask'], label='Predicted RoI', color='purple')



# # ax.set_xticklabels(labels,rotation=0, fontsize=fs)

# ax[2].set_xlabel('Time steps')

# fig.text(0.04, 0.5, 'fECG normalized', va='center', rotation='vertical')
# # ax[0].set_ylabel('fECG normalized')
# # ax1[0].set_ylabel('RoI signal')



# # Shrink current axis's height by 10% on the bottom
# box = ax[2].get_position()
# ax[2].set_position([box.x0, box.y0 + box.height * 0.1,
#                 box.width, box.height * 0.9])

# # Put a legend below current axis
# ax[2].legend(loc='upper center', bbox_to_anchor=(0.1, -0.15),
#         fancybox=True, shadow=True, ncol=2)

# ax1.legend(loc='upper center', bbox_to_anchor=(0.9, -0.15),
#         fancybox=True, shadow=True, ncol=2)



# ax[0].grid()
# ax[1].grid()
# ax[2].grid()

# #%%

# import pandas 
# import matplotlib.pyplot as plt

# data = pd.read_csv('/home/julia/Documents/fECG_research/research_dev/autoencoder_with_mask/results/060324-3CH-MOD2-LR_0.0001-W_MASK_0.3-W_SIG_0.1-LEFT_2/060324-3CH-MOD2-LR_0.0001-W_MASK_0.3-W_SIG_0.1-LEFT_2-training_history.csv')

# plt.plot(data['loss'])
# plt.plot(data['val_loss'])
# %%
