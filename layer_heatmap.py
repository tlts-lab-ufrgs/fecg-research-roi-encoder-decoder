import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras

# Display
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
from numba import cuda
from datetime import datetime

from data_load.data_loader import DataLoader
from models.ae_proposed import ProposedAE, Metric, Loss

from utils.lr_scheduler import callback as lr_scheduler

#%% To run other experiments please change this below

# CHANGEBLE VARIABLES ---------------------------------------------------------------------------------------------- 
TOTAL_FILES = 12
CHANNELS = 3
RESAMPLE_FREQ_RATIO = 2
HAVE_DIRECT_FECG = True

RESULTS_PATH = "/home/julia/Documents/research/sprint_1/results/ablation_extended/"
DATA_PATH = "/home/julia/Documents/research/datasets/b2-records/B2_Labour_dataset/"
# DATA_PATH =  "/home/julia/Documents/research/datasets/abdominal-and-direct-fetal-ecg-database-1.0.0/"
# -------------------------------------------------------------------------------------------------------------------

#%% Model constants

BATCH_SIZE=4
UPPER_LIM_LR = 0.0001
LEN_BATCH = int(512 / RESAMPLE_FREQ_RATIO)
QRS_DURATION = 0.1  # seconds, max
QRS_DURATION_STEP = int(50 / RESAMPLE_FREQ_RATIO)

MODEL_INPUT_SHAPE = (BATCH_SIZE, LEN_BATCH, CHANNELS)

TYPE_OF_FILE = 'txt'

w_mask = 0.3
w_signal = 0.1
w_combined = 1 - w_mask - w_signal

type_of_mask = 'gaussian'
decoder_type = 'convtransp'

NUMBER_OF_EPOCHS = 100
#%%
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
    
        print(np.shape(last_conv_layer_output), np.shape(preds))

    #     if pred_index is None:
    #         pred_index = tf.argmax(preds[0])
    #     class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(preds, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4):

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))




#%%
# 
model = ProposedAE(
    input_shape=MODEL_INPUT_SHAPE, 
    batch_size=BATCH_SIZE, 
    init_lr=UPPER_LIM_LR,
    w_mask=0.3, 
    w_signal=0.1, 
    w_combined=0.6, 
    training_data=None, 
    ground_truth=None, 
    testing_data=None, 
    ground_truth_testing=None
)

model.linknet()

model.model.load_weights('/home/julia/Documents/research/sprint_1/rev1_weights/weights.h5')

#%% If you want to loop in weight parameters

data_loader = DataLoader(
    DATA_PATH, 
    LEN_BATCH, 
    RESAMPLE_FREQ_RATIO, 
    TYPE_OF_FILE, 
    QRS_DURATION_STEP, 
    QRS_DURATION, 
    load_training_set=True,
    type_of_mask=type_of_mask, 
    # filters=True
)


training_data, testing_data = data_loader.data_load(0)
#%%
# change this name to show weights from layers

for layer in model.model.layers:

    last_conv_layer_name = 'add_7'

    if not 'act' in last_conv_layer_name:
        continue

# Generate class activation heatmap
# heatmap = make_gradcam_heatmap(testing_data[0][0:1, :, :], model.model, last_conv_layer_name)


    grad_model = keras.models.Model(
        model.model.inputs, [model.model.get_layer(last_conv_layer_name).output, model.model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(testing_data[0][4:5, :, :])

    ##%%

    concatenate = np.empty(shape=(256,))

    for i in range(np.shape(last_conv_layer_output)[2]):

        # Interpolação
        yinterp = np.interp(
            x  = np.linspace(0, 256, 256), 
            xp = np.linspace(0, 256, np.shape(last_conv_layer_output)[1]), 
            fp = last_conv_layer_output[0, :, i]
        )

        x = np.copy(np.where(
            ((yinterp < -1e6) | (yinterp > 1e6)), 
            np.nan, 
            yinterp
        ))

            # Adicionar ao 'concatenate' se for válido
        concatenate = np.vstack((concatenate, x))
    ##%%

    print(np.shape(concatenate))

    fig, axs = plt.subplot_mosaic(
        [['aecg'], ['fecg'], ['imshow']],
        constrained_layout=True,
        gridspec_kw={'width_ratios':[1],'height_ratios':[1,1,1]}, 
        figsize=(6,3)
    )

    fig.suptitle(last_conv_layer_name)

    axs['fecg'].sharex(axs['imshow'])
    axs['aecg'].sharex(axs['fecg'])


    axs['aecg'].plot(testing_data[0][4, :, :])

    axs['fecg'].plot(preds[0, :, :], color='black')

    im = axs['imshow'].imshow(concatenate, cmap='PRGn', aspect=0.01)

    # fig.colorbar(im, ax=axs['imshow'])

    axs['aecg'].grid()
    axs['fecg'].grid()

    plt.show()

    del x



#%%

second_conv_layer = 'add_1'
last_conv_layer_name = 'add_5'

index = 5 # 105,256, 23

grad_model = keras.models.Model(
    model.model.inputs, [model.model.get_layer(last_conv_layer_name).output, model.model.output]
)

grad_model_second = keras.models.Model(
    model.model.inputs, [model.model.get_layer(second_conv_layer).output, model.model.output]
)

# Then, we compute the gradient of the top predicted class for our input image
# with respect to the activations of the last conv layer
# with tf.GradientTape() as tape:
last_conv_layer_output, preds = grad_model(testing_data[0][index:index+1, :, :])

second_conv_layer_output, preds = grad_model_second(testing_data[0][index:index+1, :, :])

##%%

concatenate = np.empty(shape=(256,))

for i in range(np.shape(last_conv_layer_output)[2]):

    # Interpolação
    yinterp = np.interp(
        x  = np.linspace(0, 256, 256), 
        xp = np.linspace(0, 256, np.shape(last_conv_layer_output)[1]), 
        fp = last_conv_layer_output[0, :, i]
    )

    x = np.copy(np.where(
        ((yinterp < -1e6) | (yinterp > 1e6)), 
        np.nan, 
        yinterp
    ))

    # Adicionar ao 'concatenate' se for válido
    concatenate = np.vstack((concatenate, x))

concatenate *= 1 / np.max(concatenate)


concatenate_second = np.empty(shape=(256,))

for i in range(np.shape(second_conv_layer_output)[2]):

    # Interpolação
    yinterp = np.interp(
        x  = np.linspace(0, 256, 256), 
        xp = np.linspace(0, 256, np.shape(second_conv_layer_output)[1]), 
        fp = second_conv_layer_output[0, :, i]
    )

    x = np.copy(np.where(
        ((yinterp < -1e6) | (yinterp > 1e6)), 
        np.nan, 
        yinterp
    ))

    # Adicionar ao 'concatenate_second' se for válido
    concatenate_second = np.vstack((concatenate_second, x))

concatenate_second *= 1 / np.max(concatenate_second)
##%%

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})


fig, axs = plt.subplot_mosaic(
    [['aecg'], ['fecg'], ['init_layer'], ['last_layer']],
    constrained_layout=True,
    gridspec_kw={'width_ratios':[1],'height_ratios':[1,1,1,1]}, 
    figsize=(6,3)
)

# fig.suptitle(last_conv_layer_name)

axs['init_layer'].sharex(axs['last_layer'])
axs['fecg'].sharex(axs['init_layer'])
axs['aecg'].sharex(axs['fecg'])


axs['aecg'].plot(testing_data[0][index, :, :])

axs['fecg'].plot(preds[0, :, 0], color='C7')
axs['fecg'].plot(preds[0, :, 1], color='C6')

im = axs['init_layer'].imshow(concatenate_second, cmap='PRGn', aspect=0.3)

axs['last_layer'].imshow(concatenate, cmap='PRGn', aspect=0.08)

# axs['aecg'].set_xticks([])
# axs['fecg'].set_xticks([])
axs['init_layer'].set_xticks([])

axs['aecg'].set_yticks([])
axs['fecg'].set_yticks([])
axs['init_layer'].set_yticks([])
axs['last_layer'].set_yticks([])

axs['aecg'].set_ylabel('aECG')
axs['fecg'].set_ylabel('$\mathbf{\\bar{s}}, \mathbf{\\bar{m}}$')
axs['init_layer'].set_ylabel('1st EB')
axs['last_layer'].set_ylabel('3th EB')

axs['aecg'].grid(True)
axs['fecg'].grid(True)

plt.show()

#%%

fig.savefig('heatmap-r01-prediction_index-5-b2_dataset-model_rev1.pdf', bbox_inches='tight')

#%%
# if pred_index is None:
#         pred_index = tf.argmax(preds[0])
#     class_channel = preds[:, pred_index]

# # This is the gradient of the output neuron (top predicted or chosen)
# # with regard to the output feature map of the last conv layer
# grads = tape.gradient(preds, last_conv_layer_output)

# # This is a vector where each entry is the mean intensity of the gradient
# # over a specific feature map channel
# pooled_grads = tf.reduce_max(grads, axis=(0,1,2))

# # We multiply each channel in the feature map array
# # by "how important this channel is" with regard to the top predicted class
# # then sum all the channels to obtain the heatmap class activation
# # last_conv_layer_output = last_conv_layer_output[0]
# # heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
# heatmap = last_conv_layer_output[0, :, :] * pooled_grads #tf.squeeze(heatmap)

# # For visualization purpose, we will also normalize the heatmap between 0 & 1
# heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

# # # %%
# # save_and_display_gradcam(testing_data[0][0:1, :, :], heatmap)

# # Rescale heatmap to a range 0-255
# heatmap = np.uint8(255 * heatmap)

# # Use jet colormap to colorize heatmap
# jet = mpl.colormaps["jet"]

# # Use RGB values of the colormap
# jet_colors = jet(np.arange(256))[:, :3]
# jet_heatmap = jet_colors[heatmap]

# # Create an image with RGB colorized heatmap
# jet_heatmap = keras.utils.array_to_img(jet_heatmap)

# ##%%
# jet_heatmap = jet_heatmap.resize((256, 3))
# jet_heatmap = keras.utils.img_to_array(jet_heatmap)

# # Superimpose the heatmap on original image
# superimposed_img = jet_heatmap * 0.4 + testing_data[0][0:1, :, :]
# superimposed_img = keras.utils.array_to_img(superimposed_img)


# # #%%

# fig, (ax,ax2,ax3) = plt.subplots(nrows=3, sharex=True)

# ax.set_title(last_conv_layer_name)

# ax.imshow(
#     superimposed_img, 
#     cmap="plasma", 
#     aspect="auto", 
#     # extent=extent
# )

# ax2.plot(testing_data[0][0, :, :])

# ax3.plot(testing_data[1][0, :, :])

# plt.show()


# %%
