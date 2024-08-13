#%%

import numpy as np
import tensorflow as tf
import keras

from keras.layers import (
    Input, 
    Conv1D, 
    BatchNormalization, 
    Activation, 
    MaxPooling1D, 
    Add, 
    Conv1DTranspose, 
    UpSampling1D, 
    Reshape,
    Dropout
)

from utils.lr_scheduler import callback as lr_scheduler
from custom_data_aug import CustomDataAugmentation

class Metric: 
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def mse_mask(y_true, y_pred):

        error_mask = y_true[:, :, 1] - y_pred[:, :, 1]
        
        loss_mask = tf.reduce_mean(tf.math.square(error_mask))
        
        return loss_mask
    
    def mse_signal(y_true, y_pred):

        error = y_true[:, :, 0] - y_pred[:, :, 0]
        
        loss = tf.reduce_mean(tf.math.square(error))
        
        return loss
    
    def mse_combined(): 
        
        return
    
class Loss:
    
    def __init__(self, w_mask, w_signal, w_combined):
        
        self.w_signal = w_signal
        self.w_mask = w_mask
        self.w_combined = w_combined
        
        pass
    
    def loss(self, y_true, y_pred):
        
        y_true_mod = tf.multiply(y_true[:, :, 0], y_true[:, :, 1])
        
        y2_pred_combined = tf.multiply(y_pred[:, :, 0], y_true[:, :, 1])
        y1_pred_combined = tf.multiply(y_true[:, :, 0], y_pred[:, :, 1])
        
        y_pred_combined = y1_pred_combined + y2_pred_combined

        
        loss_combined = (
            tf.keras.losses.logcosh(y_true_mod, y2_pred_combined) + 
            tf.keras.losses.logcosh(y_true_mod, y1_pred_combined)
        )
        
        loss_signal = tf.keras.losses.logcosh(y_true[:, :, 0], y_pred[:, :, 0]) 
        loss_mask_mse = tf.keras.losses.logcosh(y_true[:, :, 1], y_pred[:, :, 1]) 
        
        loss = self.w_mask * loss_mask_mse + self.w_combined * loss_combined + self.w_signal * loss_signal

        return loss
               
class ProposedAE:
    
    def __init__(self, 
        input_shape, 
        batch_size, 
        init_lr, 
        w_mask, 
        w_signal, 
        w_combined, 
        training_data, 
        ground_truth, 
        testing_data, 
        ground_truth_testing, 
        epochs = 250, 
        epochs_in_patience = 15):
        
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.init_lr = init_lr
        self.total_epochs = epochs
        self.epochs_in_patience = epochs_in_patience
        
        self.w_signal = w_signal
        self.w_mask = w_mask
        self.w_combined = w_combined
        
        self.training_data = training_data
        self.ground_truth = ground_truth
        self.testing_data = testing_data
        self.ground_truth_testing = ground_truth_testing
        
        pass
    
    @staticmethod
    def downsampling(inputs, num_filters, stride):
        
        x = Conv1D(num_filters, kernel_size=1, strides=stride, padding='same')(inputs)
        x = BatchNormalization()(x)
        
        return x

    @staticmethod
    def conv_block(inputs, num_filters, kernel_size=3, stride=1, padding='same', activation='relu'):
        x = Conv1D(num_filters, kernel_size, strides=stride, padding=padding)(inputs)
        x = Activation(activation)(x)
        return x

    def encoder_block(self, inputs, num_filters):
        
        x1 = self.conv_block(inputs, num_filters, stride=2)
        x1 = self.conv_block(x1, num_filters)
    
        reshaped_inputs = self.downsampling(inputs, num_filters, stride=2)
        
        print('Reshaped inputs', np.shape(reshaped_inputs))
        
        x = Add()([x1, reshaped_inputs])
        
        x2 = self.conv_block(x, num_filters)
        x2 = self.conv_block(x2, num_filters)
        
        x = Add()([x1, x2])
        return x
        
    def decoder_block(self, inputs, skip_connection, filters_num, kernel_size=3, stride=2, output_padding=1, activation="relu"):

        print('Decoder input', np.shape(inputs))
        
        x = self.conv_block(inputs, num_filters=filters_num, kernel_size=2, padding='valid')
        x = Conv1DTranspose(
            filters_num, 
            kernel_size=kernel_size, 
            activation=activation, 
            strides=stride, 
            #output_padding=output_padding
        )(x)
        x = self.conv_block(x, num_filters=filters_num, kernel_size=1, padding='valid')
        
        x = Add()([x, skip_connection])
        
        print('x signals', np.shape(x))

        return x


    def mask_decoder_block(self, x, encoder_block1, encoder_block2, encoder_block3, encoder_block4):
        
        conv1 = self.conv_block(x, num_filters=512, kernel_size=2, padding='valid', activation='relu')
        deconv1 = Conv1DTranspose(
            512, 
            kernel_size=4, 
            activation="relu", 
            strides=2,
            #output_padding=1
        )(conv1)
        conv2 = self.conv_block(deconv1, num_filters=512, kernel_size=1, padding='valid', activation='relu')
        
        decoder_2 = self.decoder_block(conv2, encoder_block3, 256, kernel_size=4, activation='relu')
        decoder_3 = self.decoder_block(decoder_2, encoder_block2, 128, kernel_size=4, activation='relu')
        decoder_4 = self.decoder_block(decoder_3, encoder_block1, 64, kernel_size=4, activation='relu')
        

        # Last upsampling
        conv5 = self.conv_block(decoder_4, num_filters=512, kernel_size=2, padding='valid', activation='relu')
        deconv5 = Conv1DTranspose(
            512, 
            kernel_size=4, 
            activation="relu", 
            strides=2,
            #output_padding=1
        )(conv5)
        deconv_drop = Dropout(0.2)(deconv5)
        conv6 = self.conv_block(deconv_drop, num_filters=16, kernel_size=1, padding='valid', activation='relu')
        
        print(np.shape(conv6))
        
        mask = self.conv_block(conv6, num_filters=1, kernel_size=1, stride=1)
                    
        decode_mask = Activation('relu')(mask)
            
        return decode_mask

    def signal_decoder_block(self, x, encoder_block1, encoder_block2, encoder_block3, encoder_block4):

        decoder_1 = self.decoder_block(x, encoder_block4[:, :, 0:256], 256, kernel_size=4)
        decoder_2 = self.decoder_block(decoder_1, encoder_block3[:, :, 0:128], 128, kernel_size=4)
        decoder_3 = self.decoder_block(decoder_2, encoder_block2[:, :, 0:64], 64, kernel_size=4)
        decoder_4 = self.decoder_block(decoder_3, encoder_block1[:, :, 0:32], 32, kernel_size=4)

        conv_5 = self.conv_block(decoder_4, num_filters=512, kernel_size=2, padding='valid', activation='relu')
        
        decoder_5 = Conv1DTranspose(
            512, 
            kernel_size=4, 
            activation="relu", 
            strides=2,
            #output_padding=0
        )(conv_5)
        conv_6 = self.conv_block(decoder_5, num_filters=64, kernel_size=1, padding='valid', activation='relu')
        
        print(np.shape(conv_6))
        
        signal = self.conv_block(conv_6, num_filters=1, kernel_size=1, stride=1)
                    
        decode_signal = Activation('relu')(signal)
            
        return decode_signal

    def linknet(self): 
        inputs = Input(batch_shape=self.input_shape)

        aug_inputs = CustomDataAugmentation(
            num_components=15, 
            amplitude=0.2, 
            fs=500, 
        )(inputs)

        print('Input shape', np.shape(aug_inputs))
        
        #drop_inputs = Dropout(0.2)(aug_inputs)
        # Encoder
        encoder_block1 = self.encoder_block(aug_inputs, num_filters=64)
        print('Encoder Block 1', np.shape(encoder_block1))
        encoder_block1_drop = Dropout(0.2)(encoder_block1)

        encoder_block2 = self.encoder_block(encoder_block1_drop, num_filters=128)
        print('Encoder Block 2', np.shape(encoder_block2))
        encoder_block2_drop = Dropout(0.2)(encoder_block2)

        encoder_block3 = self.encoder_block(encoder_block2_drop, num_filters=256)
        print('Encoder Block 3', np.shape(encoder_block3))
        encoder_block3_drop = Dropout(0.2)(encoder_block3)

        encoder_block4 = self.encoder_block(encoder_block3_drop, num_filters=512)
        print('Encoder Block 4', np.shape(encoder_block4))
        encoder_block4_drop = Dropout(0.2)(encoder_block4)

        bottleneck = self.encoder_block(encoder_block4_drop, num_filters=1024)
        print('Bottle neck', np.shape(bottleneck))
        bottleneck_drop = Dropout(0.2)(bottleneck)

    
        mask_decoded = self.mask_decoder_block(bottleneck_drop[:, :, 256:512], encoder_block1, encoder_block2, encoder_block3, encoder_block4)
        signal_decoded = self.signal_decoder_block(bottleneck_drop[:, :, 0:256], encoder_block1, encoder_block2, encoder_block3, encoder_block4)

        # Output

        outputs = keras.ops.concatenate([signal_decoded, mask_decoded], 2)
        

        print('Output form', np.shape(outputs))

        self.model = keras.Model(inputs=inputs, outputs=outputs, name='linknet')
        
        
        return

    def fit_and_evaluate(self):
        
        self.linknet()

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.init_lr), 
            loss=Loss(self.w_signal, self.w_signal, self.w_combined).loss, # 
            metrics=[
                Metric.mse_signal, 
                Metric.mse_mask
            ]
        )

        history = self.model.fit(
                self.training_data, 
                self.ground_truth, 
                epochs=self.total_epochs, 
                batch_size=self.batch_size,
                shuffle=True, 
                callbacks=[
                    lr_scheduler,
                ],
            )
        
        if self.testing_data is None:
            return history, [], []
        else:
            test = self.model.evaluate(self.testing_data, self.ground_truth_testing)
            
            prediction = self.model.predict(self.testing_data)
            
            return history, test, prediction
    
    def save(self, path_dir):
        
        self.model.save(path_dir)
        
