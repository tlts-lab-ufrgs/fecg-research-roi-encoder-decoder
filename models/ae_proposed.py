#%%

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, Conv1DTranspose, Concatenate

def conv_block(inputs, num_filters, kernel_size=3, stride=1):
    x = Conv1D(num_filters, kernel_size, strides=stride, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def linknet_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    x = conv_block(x, num_filters)
    return x

def decoder_block(inputs, skip_connection, filters_num):

    print('Decoder input', np.shape(inputs))

    mul_number = np.shape(skip_connection)[1] - np.shape(inputs)[1] + 1

    x = Conv1DTranspose(filters_num, mul_number, activation="relu")(inputs)

    print('x signals', np.shape(x))

    x = Concatenate()([x, skip_connection])
    x = conv_block(x, num_filters=filters_num)  # Adjust the number of filters based on your needs
    return x


def linknet(input_shape=(256, 1), num_classes=21):  # Adjust input_shape and num_classes as needed
    inputs = Input(batch_shape=input_shape)

    # Encoder
    encoder_block1 = linknet_block(inputs, num_filters=64)
    encoder_pool1 = MaxPooling1D(pool_size=2)(encoder_block1)

    print('Encoder Block 1', np.shape(encoder_block1))

    encoder_block2 = linknet_block(encoder_pool1, num_filters=128)
    encoder_pool2 = MaxPooling1D(pool_size=2)(encoder_block2)

    print('Encoder Block 2', np.shape(encoder_block2))

    encoder_block3 = linknet_block(encoder_pool2, num_filters=256)
    encoder_pool3 = MaxPooling1D(pool_size=2)(encoder_block3)

    print('Encoder Block 3', np.shape(encoder_block3))

    encoder_block4 = linknet_block(encoder_pool3, num_filters=512)
    encoder_pool4 = MaxPooling1D(pool_size=2)(encoder_block4)

    print('Encoder Block 4', np.shape(encoder_block4))

    # Bottleneck
    bottleneck = linknet_block(encoder_pool4, num_filters=1024)

    print('Bottle neck', np.shape(bottleneck))

    # # Decoder
    decoder = decoder_block(bottleneck, encoder_block4, 512)
    decoder = decoder_block(decoder, encoder_block3, 256)
    decoder = decoder_block(decoder, encoder_block2, 128)
    decoder = decoder_block(decoder, encoder_block1, 64)

    # Final classification layer
    # x= bottleneck
    x = conv_block(decoder, num_filters=1, kernel_size=1, stride=1)

    # Output
    outputs = Activation('linear')(x)

    print('Output form', np.shape(outputs))
    print(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='linknet')
    return model

# # Create LinkNet model for 1D signals
# model = linknet()
# model.summary()

#%%

# linknet((32, 600, 4), 2)
# %%
