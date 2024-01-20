import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, UpSampling1D, Concatenate

def conv_block(inputs, num_filters, kernel_size=3, stride=1):
    x = Conv1D(num_filters, kernel_size, strides=stride, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def linknet_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    x = conv_block(x, num_filters)
    return x

def decoder_block(inputs, skip_connection):
    x = UpSampling1D(size=2)(inputs)
    x = Concatenate()([x, skip_connection])
    x = conv_block(x, num_filters=256)  # Adjust the number of filters based on your needs
    return x

def linknet(input_shape=(256, 1), num_classes=21):  # Adjust input_shape and num_classes as needed
    inputs = Input(shape=input_shape)

    # Encoder
    encoder_block1 = linknet_block(inputs, num_filters=64)
    encoder_pool1 = MaxPooling1D(pool_size=2)(encoder_block1)

    encoder_block2 = linknet_block(encoder_pool1, num_filters=128)
    encoder_pool2 = MaxPooling1D(pool_size=2)(encoder_block2)

    encoder_block3 = linknet_block(encoder_pool2, num_filters=256)
    encoder_pool3 = MaxPooling1D(pool_size=2)(encoder_block3)

    encoder_block4 = linknet_block(encoder_pool3, num_filters=512)
    encoder_pool4 = MaxPooling1D(pool_size=2)(encoder_block4)

    # Bottleneck
    bottleneck = linknet_block(encoder_pool4, num_filters=1024)

    # Decoder
    decoder = decoder_block(bottleneck, encoder_block4)
    decoder = decoder_block(decoder, encoder_block3)
    decoder = decoder_block(decoder, encoder_block2)
    decoder = decoder_block(decoder, encoder_block1)

    # Final classification layer
    x = conv_block(decoder, num_filters=num_classes, kernel_size=1, stride=1)

    # Output
    outputs = Activation('softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='linknet')
    return model

# Create LinkNet model for 1D signals
model = linknet()
model.summary()
