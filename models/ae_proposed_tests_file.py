#%%

import numpy as np
import tensorflow as tf

from tensorflow.keras.activations import (
    softplus
)

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

def downsampling(inputs, num_filters, stride, remove_normalization = False):
    
    x = Conv1D(num_filters, kernel_size=1, strides=stride, padding='same')(inputs)
    
    x = BatchNormalization()(x)
    
    return x

def conv_block(inputs, num_filters, kernel_size=3, stride=1, padding='same', activation='relu'):
    x = Conv1D(num_filters, kernel_size, strides=stride, padding=padding)(inputs)
    x = Activation('relu')(x)
    
    return x

def linknet_block(inputs, num_filters, activation='relu'):
    x1 = conv_block(inputs, num_filters, stride=2, activation=activation)
    x1 = conv_block(x1, num_filters, activation=activation)
 
    reshaped_inputs = downsampling(inputs, num_filters, stride=2)
    
    print('Reshaped inputs', np.shape(reshaped_inputs))
    
    x = Add()([x1, reshaped_inputs])
    
    x2 = conv_block(x, num_filters, activation=activation)
    x2 = conv_block(x2, num_filters, activation=activation)
    
    x = Add()([x1, x2])
    
    return x

def decoder_block(inputs, skip_connection, filters_num, kernel_size=3, stride=2, output_padding=0, activation='relu'):

    print('Decoder input', np.shape(inputs))
    
    x = conv_block(inputs, num_filters=filters_num, kernel_size=2, padding='valid', activation=activation)
    x = Conv1DTranspose(
        filters_num, 
        kernel_size=kernel_size, 
        activation=activation, 
        strides=stride, 
        output_padding=output_padding
    )(x)
    x = conv_block(x, num_filters=filters_num, kernel_size=1, padding='valid', activation = activation) 
    
    x = Add()([x, skip_connection])
    
    print('x signals', np.shape(x))

    return x


def mask_decoder_block(x, encoder_block1, encoder_block2, encoder_block3, encoder_block4):
    
    # decoder = decoder_block(x, encoder_block4[:, :, 256:512], 256, kernel_size=4)
    # decoder = decoder_block(decoder, encoder_block3[:, :, 128:256], 128, kernel_size=4)
    # decoder = decoder_block(decoder, encoder_block2[:, :, 64:128], 64, kernel_size=4)
    # decoder = decoder_block(decoder, encoder_block1[:, :, 32:64], 32, kernel_size=4)

    x = conv_block(x, num_filters=512, kernel_size=2, padding='valid', activation='relu')
    # ((timesteps - 1) * strides + kernel_size - 2 * padding + output_padding)
    x = Conv1DTranspose(
        512, 
        kernel_size=4, 
        activation="relu", 
        strides=2,
        output_padding=0
    )(x)
    x = conv_block(x, num_filters=512, kernel_size=1, padding='valid', activation='relu')
    
    decoder = decoder_block(x, encoder_block3, 256, kernel_size=4, activation='relu')
    decoder = decoder_block(decoder, encoder_block2, 128, kernel_size=4, activation='relu')
    decoder = decoder_block(decoder, encoder_block1, 64, kernel_size=4, activation='relu')
    
    

    # Last upsampling
    x = conv_block(decoder, num_filters=512, kernel_size=2, padding='valid', activation='relu')
    x = Conv1DTranspose(
        512, 
        kernel_size=4, 
        activation="relu", 
        strides=2,
        output_padding=0
    )(x)
    x = conv_block(x, num_filters=16, kernel_size=1, padding='valid', activation='relu')
    
    
    x = conv_block(x, num_filters=1, kernel_size=1, stride=1)
                
    decode_mask = Activation('relu')(x)
        
    return decode_mask

def signal_decoder_block(x, encoder_block1, encoder_block2, encoder_block3, encoder_block4):
    
    
    
    # x = conv_block(x, num_filters=256, kernel_size=2, padding='valid', activation='relu')
    # # ((timesteps - 1) * strides + kernel_size - 2 * padding + output_padding)
    # x = Conv1DTranspose(
    #     256, 
    #     kernel_size=4, 
    #     activation="relu", 
    #     strides=2,
    #     output_padding=0
    # )(x)
    # x = conv_block(x, num_filters=512, kernel_size=1, padding='valid', activation='relu')
    
    # decoder = decoder_block(x, encoder_block3, 256, kernel_size=4, activation='relu')
    # decoder = decoder_block(decoder, encoder_block2, 128, kernel_size=4, activation='relu')
    # decoder = decoder_block(decoder, encoder_block1, 64, kernel_size=4, activation='relu')
    
    decoder = decoder_block(x, encoder_block4[:, :, 0:256], 256, kernel_size=4)
    decoder = decoder_block(decoder, encoder_block3[:, :, 0:128], 128, kernel_size=4)
    decoder = decoder_block(decoder, encoder_block2[:, :, 0:64], 64, kernel_size=4)
    decoder = decoder_block(decoder, encoder_block1[:, :, 0:32], 32, kernel_size=4)

   
    x = conv_block(decoder, num_filters=512, kernel_size=2, padding='valid', activation='relu')
    
    x = Conv1DTranspose(
        512, 
        kernel_size=4, 
        activation="relu", 
        strides=2,
        output_padding=0
    )(x)
    x = conv_block(x, num_filters=64, kernel_size=1, padding='valid', activation='relu')
    
    
    x = conv_block(x, num_filters=1, kernel_size=1, stride=1)
                
    decode_signal = Activation('relu')(x)
        
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

   
    mask_decoded = mask_decoder_block(bottleneck[:, :, 512:1024], encoder_block1, encoder_block2, encoder_block3, encoder_block4)
    signal_decoded = signal_decoder_block(bottleneck[:, :, 0:512], encoder_block1, encoder_block2, encoder_block3, encoder_block4)

    # Output
    outputs = tf.concat([signal_decoded, mask_decoded], 2)
    

    print('Output form', np.shape(outputs))

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='linknet')
    return model

