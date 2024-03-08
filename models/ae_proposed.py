#%%

import numpy as np
import tensorflow as tf

from utils.training_patience import callback as patience_callback
from utils.lr_scheduler import callback as lr_scheduler

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
    Dropout
)

from utils.gaussian_function import gaussian


def add_baseline_wandering(x, num_components=5, amplitude=1, fs=1000):
    t = np.arange(len(x)) / fs
    baseline_wandering = np.zeros_like(x)

    for _ in range(int(np.random.uniform(low=0, high=num_components))):
        frequency = np.random.uniform(low=0.1, high=1)  # Random low frequency
        phase = np.random.uniform(0, 2 * np.pi)  # Random phase
        component = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        baseline_wandering += component

    x_with_baseline = x + baseline_wandering
    
    max_baseline = np.max(x_with_baseline) if np.max(x_with_baseline) != 0 else 1e-7
    
    # normalization
    x_with_baseline = x_with_baseline / max_baseline
    
    return x_with_baseline

class CustomDataAugmentation(tf.keras.layers.Layer):
    def __init__(self, num_components=15, amplitude=1, fs=1000, **kwargs):
        super(CustomDataAugmentation, self).__init__(**kwargs)
        self.num_components = num_components
        self.amplitude = amplitude
        self.fs = fs

    def call(self, inputs, training=None):
        if training:
            # baseline input
            add_baseline = np.random.randint(0, 2)
            if add_baseline != 0:
                augmented_inputs = tf.numpy_function(add_baseline_wandering, [inputs, self.num_components, self.amplitude, self.fs], tf.float32)   
            else:
                augmented_inputs = np.copy(inputs)            
            
            len_batch = np.shape(augmented_inputs)[1]
            
            for i in [0, 1, 2, 3]:
                # gaussian noise
                mu = 0
                sigma = 1
                noise = 0.1 * np.random.normal(mu, sigma, size=np.shape(augmented_inputs[:, :, i]))    
                augmented_inputs[:, :, i] += noise
            
            # cutoff 
            
            channel_to_cutoff = np.random.randint(0, 4)
            
            begin_of_region = np.random.randint(0, 513 - 10)
            end_of_region = np.random.randint(begin_of_region + 50, 513)
            
            # if channel_to_cutoff != -1:
            augmented_inputs[:, begin_of_region:end_of_region + 1, channel_to_cutoff] = np.full(shape=np.shape(augmented_inputs[:, :, channel_to_cutoff]), fill_value = np.mean(augmented_inputs[:, :, channel_to_cutoff]))
                   

            # second cuttoff
            
            channel_to_cutoff = np.random.randint(0, 4)
            begin_of_region = np.random.randint(0, len_batch - 10)
            end_of_region = np.random.randint(begin_of_region + 50, len_batch)
            augmented_inputs[:, begin_of_region:end_of_region + 1, channel_to_cutoff] = 0 #np.full(shape=np.shape(augmented_inputs[:, :, channel_to_cutoff]), fill_value = np.max(augmented_inputs[:, :, channel_to_cutoff]))
                   
            # add a start of gaussian at the end of beginning of the file:
            add_gausian_tail_to_data = np.random.randint(0, 1)
            if add_gausian_tail_to_data == 1:
                at_the_end_of_data = np.random.randint(0, 1)
                
                
                if at_the_end_of_data == 1:
                    augmented_inputs[:, [len_batch - 20, len_batch]] += gaussian(np.arange(len_batch - 20, len_batch), np.random.randint(len_batch, 65), 0.1)
                else:
                    augmented_inputs[:, [0, 30]] += gaussian(np.arange(0, 30), np.random.randint(0, -32), 0.1)

                        
            return augmented_inputs
        else:
            return inputs

    def get_config(self):
        config = super(CustomDataAugmentation, self).get_config()
        config.update({'num_components': self.num_components, 'amplitude': self.amplitude, 'fs': self.fs})
        return config

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
        
    def decoder_block(self, inputs, skip_connection, filters_num, kernel_size=3, stride=2, output_padding=0, activation="relu"):

        print('Decoder input', np.shape(inputs))
        
        x = self.conv_block(inputs, num_filters=filters_num, kernel_size=2, padding='valid')
        x = Conv1DTranspose(
            filters_num, 
            kernel_size=kernel_size, 
            activation=activation, 
            strides=stride, 
            output_padding=output_padding
        )(x)
        x = self.conv_block(x, num_filters=filters_num, kernel_size=1, padding='valid')
        
        x = Add()([x, skip_connection])
        
        print('x signals', np.shape(x))

        return x


    def mask_decoder_block(self, x, encoder_block1, encoder_block2, encoder_block3, encoder_block4):
        
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
        x = Dropout(0.2)(x)
        x = self.conv_block(x, num_filters=16, kernel_size=1, padding='valid', activation='relu')
        
        print(np.shape(x))
        
        x = self.conv_block(x, num_filters=1, kernel_size=1, stride=1)
                    
        decode_mask = Activation('relu')(x)
            
        return decode_mask

    def signal_decoder_block(self, x, encoder_block1, encoder_block2, encoder_block3, encoder_block4):

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
        
        print(np.shape(x))
        
        x = self.conv_block(x, num_filters=1, kernel_size=1, stride=1)
                    
        decode_signal = Activation('relu')(x)
            
        return decode_signal

    def linknet(self): 
        inputs = Input(batch_shape=self.input_shape)

        inputs = CustomDataAugmentation(num_components=15, amplitude=0.1, fs=1000)(inputs)
        
        
        inputs = Dropout(0.5)(inputs)
        # Encoder
        encoder_block1 = self.encoder_block(inputs, num_filters=64)
        print('Encoder Block 1', np.shape(encoder_block1))
        encoder_block1 = Dropout(0.2)(encoder_block1)

        encoder_block2 = self.encoder_block(encoder_block1, num_filters=128)
        print('Encoder Block 2', np.shape(encoder_block2))
        encoder_block2 = Dropout(0.2)(encoder_block2)

        encoder_block3 = self.encoder_block(encoder_block2, num_filters=256)
        print('Encoder Block 3', np.shape(encoder_block3))
        encoder_block3 = Dropout(0.2)(encoder_block3)

        encoder_block4 = self.encoder_block(encoder_block3, num_filters=512)
        print('Encoder Block 4', np.shape(encoder_block4))
        encoder_block4 = Dropout(0.2)(encoder_block4)

        bottleneck = self.encoder_block(encoder_block4, num_filters=1024)
        print('Bottle neck', np.shape(bottleneck))
        bottleneck = Dropout(0.2)(bottleneck)

    
        mask_decoded = self.mask_decoder_block(bottleneck[:, :, 256:512], encoder_block1, encoder_block2, encoder_block3, encoder_block4)
        signal_decoded = self.signal_decoder_block(bottleneck[:, :, 0:256], encoder_block1, encoder_block2, encoder_block3, encoder_block4)

        # Output
        outputs = tf.concat([signal_decoded, mask_decoded], 2)
        

        print('Output form', np.shape(outputs))

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='linknet')
        
        
        return

    def fit_and_evaluate(self):
        
        self.linknet()
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.init_lr), 
            loss=Loss(self.w_signal, self.w_signal, self.w_combined).loss, # 
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(name='rmse'), 
                'mean_squared_error', 
                Metric.mse_signal, 
                Metric.mse_mask
            ]
        )

        history = self.model.fit(
                self.training_data, 
                self.ground_truth, 
                epochs=self.total_epochs, 
                batch_size=self.batch_size,
                # validation_data=(self.testing_data, self.ground_truth_testing),
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
        
