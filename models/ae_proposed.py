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
    Reshape
)

class ProposedAE:
    
    def __init__(self):
        pass
    

    def downsampling(inputs, num_filters, stride):
        
        x = Conv1D(num_filters, kernel_size=1, strides=stride, padding='same')(inputs)
        x = BatchNormalization()(x)
        
        return x

    def conv_block(inputs, num_filters, kernel_size=3, stride=1, padding='same'):
        x = Conv1D(num_filters, kernel_size, strides=stride, padding=padding)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def encoder_block():
        
        return

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


    def mask_decoder_block(x):
        
        
        
        decode_mask = Activation('softmax')(x)
        
        return decode_mask

    def signal_decoder_block(x):

        
        decode_signal = Activation('relu')(x)
        
        return decode_signal

    def linknet(self, input_shape=(256, 1), num_classes=21):  # Adjust input_shape and num_classes as needed
        inputs = Input(batch_shape=input_shape)

        # Encoder
        encoder_block1 = self.linknet_block(inputs, num_filters=64)
        print('Encoder Block 1', np.shape(encoder_block1))

        encoder_block2 = self.linknet_block(encoder_block1, num_filters=128)
        print('Encoder Block 2', np.shape(encoder_block2))

        encoder_block3 = self.linknet_block(encoder_block2, num_filters=256)
        print('Encoder Block 3', np.shape(encoder_block3))

        encoder_block4 = self.linknet_block(encoder_block3, num_filters=512)
        print('Encoder Block 4', np.shape(encoder_block4))

        bottleneck = self.linknet_block(encoder_block4, num_filters=1024)
        print('Bottle neck', np.shape(bottleneck))

        # # Decoder
        
        outputs = 1


        print('Output form', np.shape(outputs))

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='linknet')
        return model

