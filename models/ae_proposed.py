#%%

import numpy as np
import tensorflow as tf
from utils.training_patience import callback as patience_callback
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

class Metric: 
    def __init__(self) -> None:
        pass
    
    def mse_mask():
        
        return
    
    def mse_signal():
        
        return 
    
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
        y_pred_mod = tf.multiply(y_pred[:, :, 0], y_pred[:, :, 1])

        
        loss_combined =  tf.keras.losses.logcosh(y_true_mod, y_pred_mod) 
        loss_signal = tf.keras.losses.logcosh(y_true[:, :, 0], y_pred[:, :, 0]) 
        loss_mask_mse = tf.keras.losses.logcosh(y_true[:, :, 1], y_pred[:, :, 1]) 
        
        loss = self.w_mask * loss_mask_mse + self.w_combined * loss_combined + self.w_signal * loss_signal

        return loss
        
class LearningRate:
    
    def __init__(self) -> None:
        self.WAIT_UNTIL = 10
        pass
    
    def lr_decay(self, epoch, lr):
        
        if epoch < self.WAIT_UNTIL:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
        
        return
class ProposedAE:
    
    def __init__(self, input_shape, batch_size, init_lr, w_mask, w_signal, w_combined, training_data, ground_truth, epochs = 250, epochs_in_patience = 15):
        
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
        
        pass
    
    @staticmethod
    def downsampling(self, inputs, num_filters, stride):
        
        x = Conv1D(num_filters, kernel_size=1, strides=stride, padding='same')(inputs)
        x = BatchNormalization()(x)
        
        return x

    @staticmethod
    def conv_block(inputs, num_filters, kernel_size=3, stride=1, padding='same'):
        x = Conv1D(num_filters, kernel_size, strides=stride, padding=padding)(inputs)
        x = Activation('relu')(x)
        return x

    def encoder_block(self, inputs, num_filters):
        
        x1 = self.conv_block(inputs, num_filters, stride=2)
        x1 = self.conv_block(x1, num_filters)
    
        reshaped_inputs = self.downsampling(inputs, num_filters, stride=2)
        
        print('Reshaped inputs', np.shape(reshaped_inputs))
        
        x = Add()([x1, reshaped_inputs])
        
        x2 = self.onv_block(x, num_filters)
        x2 = self.conv_block(x2, num_filters)
        
        x = Add()([x1, x2])
        return x
        
    def decoder_block(self, inputs, skip_connection, filters_num, kernel_size=3, stride=2, output_padding=0):

        print('Decoder input', np.shape(inputs))
        
        x = self.conv_block(inputs, num_filters=filters_num, kernel_size=2, padding='valid')
        x = Conv1DTranspose(
            filters_num, 
            kernel_size=kernel_size, 
            activation="relu", 
            strides=stride, 
            output_padding=output_padding
        )(x)
        x = self.conv_block(x, num_filters=filters_num, kernel_size=1, padding='valid')
        
        x = Add()([x, skip_connection])
        
        print('x signals', np.shape(x))

        return x


    def mask_decoder_block(self, x, encoder_block3, encoder_block2, encoder_block1):
        
        x = self.conv_block(x, num_filters=512, kernel_size=2, padding='valid', activation='relu')
        x = Conv1DTranspose(
            512, 
            kernel_size=4, 
            activation="relu", 
            strides=2,
            output_padding=0
        )(x)
        x = self.conv_block(x, num_filters=512, kernel_size=1, padding='valid', activation='relu')
        
        decoder = self.decoder_block(x, encoder_block3, 256, kernel_size=4, activation='relu')
        decoder = self.decoder_block(decoder, encoder_block2, 128, kernel_size=4, activation='relu')
        decoder = self.decoder_block(decoder, encoder_block1, 64, kernel_size=4, activation='relu')
        
        

        # Last upsampling
        x = self.conv_block(decoder, num_filters=512, kernel_size=2, padding='valid', activation='relu')
        x = Conv1DTranspose(
            512, 
            kernel_size=4, 
            activation="relu", 
            strides=2,
            output_padding=0
        )(x)
        x = self.conv_block(x, num_filters=16, kernel_size=1, padding='valid', activation='relu')
        
        
        x = self.conv_block(x, num_filters=1, kernel_size=1, stride=1)
                    
        decode_mask = Activation('relu')(x)
            
        return decode_mask

    def signal_decoder_block(self, x, encoder_block4, encoder_block3, encoder_block2, encoder_block1):

        decoder = self.decoder_block(x, encoder_block4[:, :, 0:256], 256, kernel_size=4)
        decoder = self.decoder_block(decoder, encoder_block3[:, :, 0:128], 128, kernel_size=4)
        decoder = self.decoder_block(decoder, encoder_block2[:, :, 0:64], 64, kernel_size=4)
        decoder = self.decoder_block(decoder, encoder_block1[:, :, 0:32], 32, kernel_size=4)

    
        x = self.conv_block(decoder, num_filters=512, kernel_size=2, padding='valid', activation='relu')
        
        x = Conv1DTranspose(
            512, 
            kernel_size=4, 
            activation="relu", 
            strides=2,
            output_padding=0
        )(x)
        x = self.conv_block(x, num_filters=64, kernel_size=1, padding='valid', activation='relu')
        
        
        x = self.conv_block(x, num_filters=1, kernel_size=1, stride=1)
                    
        decode_signal = Activation('relu')(x)
            
        return decode_signal

    def linknet(self): 
        inputs = Input(batch_shape=self.input_shape)

        # Encoder
        encoder_block1 = self.encoder_block(inputs, num_filters=64)
        print('Encoder Block 1', np.shape(encoder_block1))

        encoder_block2 = self.encoder_block(encoder_block1, num_filters=128)
        print('Encoder Block 2', np.shape(encoder_block2))

        encoder_block3 = self.encoder_block(encoder_block2, num_filters=256)
        print('Encoder Block 3', np.shape(encoder_block3))

        encoder_block4 = self.encoder_block(encoder_block3, num_filters=512)
        print('Encoder Block 4', np.shape(encoder_block4))

        bottleneck = self.encoder_block(encoder_block4, num_filters=1024)
        print('Bottle neck', np.shape(bottleneck))

    
        mask_decoded = self.mask_decoder_block(bottleneck[:, :, 512:1024], encoder_block1, encoder_block2, encoder_block3, encoder_block4)
        signal_decoded = self.signal_decoder_block(bottleneck[:, :, 0:512], encoder_block1, encoder_block2, encoder_block3, encoder_block4)

        # Output
        outputs = tf.concat([signal_decoded, mask_decoded], 2)
        

        print('Output form', np.shape(outputs))

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='linknet')
        return model

    def compile(self):
        
        model = self.linknet()
        
        model.compile(
            # optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR), 
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.init_lr), 
            loss=Loss(self.w_signal, self.w_signal, self.w_combined), # 
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(name='rmse'), 
                'mean_squared_error', 
                Metric.mse_signal, 
                Metric.mse_mask
            ]
            )

        history = model.fit(
                self.training_data, 
                self.ground_truth, 
                epochs=self.total_epochs, 
                batch_size=self.batch_size,
                validation_split=0.25,
                shuffle=True, 
                callbacks=[
                    LearningRate.lr_decay(),
                    patience_callback('loss', self.epochs_in_patience)
                ],
            )
        
        return 
