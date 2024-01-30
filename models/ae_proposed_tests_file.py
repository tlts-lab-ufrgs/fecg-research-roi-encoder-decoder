#%%

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, 
    Conv1D, 
    BatchNormalization, 
    Activation, 
    MaxPooling1D, 
    Add, 
    Conv1DTranspose, 
    UpSampling1D, 
    Reshape, 
    Dropout, 
    Dense
)

def downsampling(inputs, num_filters, stride):
    
    x = Conv1D(num_filters, kernel_size=1, strides=stride, padding='same')(inputs)
    x = BatchNormalization()(x)
    
    return x

def conv_block(inputs, num_filters, kernel_size=3, stride=1, padding='same'):
    x = Conv1D(num_filters, kernel_size, strides=stride, padding=padding)(inputs)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def linknet_block(inputs, num_filters):
    x1 = conv_block(inputs, num_filters, stride=2)
    x1 = conv_block(x1, num_filters)
 
    reshaped_inputs = downsampling(inputs, num_filters, stride=2)
    
    print('Reshaped inputs', np.shape(reshaped_inputs))
    
    x = Add()([x1, reshaped_inputs])
    
    x2 = conv_block(x, num_filters)
    x2 = conv_block(x2, num_filters)
    
    x = Add()([x1, x2])
    return x

def decoder_block(inputs, skip_connection, filters_num, kernel_size=3, stride=2, output_padding=0):

    print('Decoder input', np.shape(inputs))
    
    x = conv_block(inputs, num_filters=filters_num, kernel_size=2, padding='valid')
    # ((timesteps - 1) * strides + kernel_size - 2 * padding + output_padding)
    x = Conv1DTranspose(
        filters_num, 
        kernel_size=kernel_size, 
        activation="relu", 
        strides=stride, 
        output_padding=output_padding
    )(x)
    x = conv_block(x, num_filters=filters_num, kernel_size=1, padding='valid')
    
    x = Add()([x, skip_connection])
    
    print('x signals', np.shape(x))

    return x


def mask_decoder_block(x, encoder_block1, encoder_block2, encoder_block3, encoder_block4):
    
    print(np.shape(x))
    print(np.shape(encoder_block3))
    
    x = conv_block(x, num_filters=256, kernel_size=2, padding='valid')
    # ((timesteps - 1) * strides + kernel_size - 2 * padding + output_padding)
    x = Conv1DTranspose(
        256, 
        kernel_size=4, 
        activation="relu", 
        strides=2,
        output_padding=0
    )(x)
    x = conv_block(x, num_filters=256, kernel_size=1, padding='valid')
    
    
        # # Decoder
    # decoder = decoder_block(x, encoder_block4, 256, kernel_size=4)
    decoder = decoder_block(x, encoder_block3, 256, kernel_size=4)
    decoder = decoder_block(decoder, encoder_block2, 128, kernel_size=4)
    decoder = decoder_block(decoder, encoder_block1, 64, kernel_size=4)

    # Last upsampling
    x = UpSampling1D(2)(decoder)
    x = conv_block(x, num_filters=1, kernel_size=1, stride=1)
                
    decode_mask = Activation('softmax')(x)
    
    # decode_mask = Dense(units = 2, activation='softmax')(x)
        
    return decode_mask

def signal_decoder_block(x, encoder_block1, encoder_block2, encoder_block3, encoder_block4):
    
    x = conv_block(x, num_filters=256, kernel_size=2, padding='valid')
    # ((timesteps - 1) * strides + kernel_size - 2 * padding + output_padding)
    x = Conv1DTranspose(
        256, 
        kernel_size=4, 
        activation="relu", 
        strides=2,
        output_padding=0
    )(x)
    x = conv_block(x, num_filters=256, kernel_size=1, padding='valid')
    
            # # Decoder
    # decoder = decoder_block(x, encoder_block4, 256, kernel_size=4)
    decoder = decoder_block(x, encoder_block3, 256, kernel_size=4)
    decoder = decoder_block(decoder, encoder_block2, 128, kernel_size=4)
    decoder = decoder_block(decoder, encoder_block1, 64, kernel_size=4)

    # Last upsampling
    x = UpSampling1D(2)(decoder)
    x = conv_block(x, num_filters=1, kernel_size=1, stride=1)
                
    decode_signal = Activation('relu')(x)
    
    # decode_signal = Dense(units = 2, activation='softmax')(x)
        
    return decode_signal


def proposed_ae(input_shape=(256, 1), num_classes=21):  # Adjust input_shape and num_classes as needed
    inputs = Input(batch_shape=input_shape)

    # Encoder
    encoder_block1 = linknet_block(inputs, num_filters=64)
    print('Encoder Block 1', np.shape(encoder_block1))

    encoder_block2 = linknet_block(encoder_block1, num_filters=128)
    print('Encoder Block 2', np.shape(encoder_block2))

    encoder_block3 = linknet_block(encoder_block2, num_filters=256)
    print('Encoder Block 3', np.shape(encoder_block3))

    encoder_block4 = linknet_block(encoder_block3, num_filters=512)
    print('Encoder Block 4', np.shape(encoder_block4))

    bottleneck = linknet_block(encoder_block4, num_filters=1024)
    print('Bottle neck', np.shape(bottleneck))

   
    mask_decoded = mask_decoder_block(bottleneck[:, :, 0:512], encoder_block1, encoder_block2, encoder_block3, encoder_block4)
    signal_decoded = signal_decoder_block(bottleneck[:, :, 512:1024], encoder_block1, encoder_block2, encoder_block3, encoder_block4)

    # Output
    outputs = tf.concat([signal_decoded, mask_decoded], 2)
    

    print('Output form', np.shape(outputs))

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='linknet')
    return model

# # Create LinkNet model for 1D signals
# model = linknet()
# model.summary()

#%%

# linknet((32, 600, 4), 2)
# %%
