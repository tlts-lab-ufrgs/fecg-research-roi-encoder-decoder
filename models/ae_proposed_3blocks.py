#%%

import time
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

from utils.masks_function import gaussian



@tf.function
def add_baseline_wandering(x, num_components=7, amplitude=0.1, fs=1000):
    t = tf.range(tf.shape(x)[1], dtype=tf.float32) / fs
    time_stacked = tf.stack([t, t, t], axis=-1)
    baseline_wandering = tf.zeros_like(x, dtype=tf.float32)

    num_components = tf.cast(num_components, tf.int32)
    frequencies = tf.random.uniform(shape=[num_components, 1, 1], minval=0.1, maxval=1, dtype=tf.float32)
    phases = tf.random.uniform(shape=[num_components, 1, 1], minval=0, maxval=2 * np.pi, dtype=tf.float32)

    components = amplitude * tf.sin(2 * np.pi * frequencies * time_stacked + phases)
    components_sum = tf.reduce_sum(components, axis=0)
    baseline_wandering += components_sum

    x_with_baseline = x + baseline_wandering

    min_val = tf.reduce_min(x_with_baseline, axis=1, keepdims=True)
    x_with_baseline -= min_val

    max_val = tf.reduce_max(x_with_baseline, axis=1, keepdims=True)
    max_val = tf.where(tf.equal(max_val, 0), tf.constant(1e-7, dtype=max_val.dtype), max_val)
    x_with_baseline /= max_val

    return x_with_baseline


class CustomDataAugmentation(keras.layers.Layer):
    def __init__(self, num_components=7, amplitude=0.1, fs=500, **kwargs):
        super().__init__(**kwargs)
        self.num_components = num_components
        self.amplitude = amplitude
        self.fs = fs

    def call(self, inputs, training=None):
        if training:
            return self._augment(inputs)
        else:
            return inputs
        

    @tf.function
    def _augment(self, inputs):

        signal_shape = tf.shape(inputs)
        batch_size = signal_shape[0]
        quarter_batch_size = tf.cast(batch_size / 2, tf.int32)

        inputs_float32 = tf.cast(inputs, dtype=np.float32)

        indices_aug = tf.random.uniform(shape=[quarter_batch_size], minval=0, maxval=signal_shape[0], dtype=tf.int32)
        indices_aug = tf.expand_dims(indices_aug, axis=-1) 
        inputs_with_bl = add_baseline_wandering(tf.gather(inputs_float32, indices_aug[:, 0]), self.num_components, self.amplitude, self.fs)
        augmented_inputs = tf.tensor_scatter_nd_update(
            inputs_float32,
            indices_aug,
            inputs_with_bl
        )

        mu = 0.0
        sigma = 1.0
        amplitude = tf.random.uniform(shape=[], minval=0.01, maxval=0.08, dtype=tf.float32)
        noise = amplitude * tf.random.normal(shape=(quarter_batch_size, signal_shape[1], signal_shape[2]), mean=mu, stddev=sigma)

        indices = tf.random.uniform(shape=[quarter_batch_size], minval=0, maxval=signal_shape[0], dtype=tf.int32)
        scattered_noise = augmented_inputs + tf.scatter_nd(
            indices=tf.expand_dims(indices, axis=-1),
            updates=noise,
            shape=tf.shape(augmented_inputs)
        )

        # # Batch indices
        batch_indices = tf.random.uniform(shape=[quarter_batch_size], minval=0, maxval=signal_shape[1]-1, dtype=tf.int32)

        # Precompute all possible indices for the signal length
        all_possible_indices = tf.range(signal_shape[1])

        # Initialize the scatter_updates with the original tensor
        scatter_updates = scattered_noise

        # Function to update the tensor in a loop-compatible way
        def update_tensor(i, scatter_updates):
            # Randomly select channel, begin, and end of the region
            channel_to_cutoff = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
            begin_of_region = tf.random.uniform(shape=[], minval=0, maxval=signal_shape[1] - 50, dtype=tf.int32)
            end_of_region = begin_of_region + tf.random.uniform(shape=[], minval=10, maxval=150, dtype=tf.int32)
            
            # Mask to select the indices within the range [begin_of_region, end_of_region)
            mask = (all_possible_indices >= begin_of_region) & (all_possible_indices < end_of_region)
            region_indices = tf.boolean_mask(all_possible_indices, mask)
            
            # Combine batch, region, and channel indices into a single tensor
            batch_index = tf.fill([tf.size(region_indices)], batch_indices[i])
            channel_index = tf.fill([tf.size(region_indices)], channel_to_cutoff)
            indices_to_update = tf.stack([batch_index, region_indices, channel_index], axis=-1)
            
            # Generate updates (zeros in this case)
            updates = tf.zeros([tf.size(region_indices)], dtype=tf.float32)
            
            # Apply scatter update to the cumulative tensor
            scatter_updates = tf.tensor_scatter_nd_update(scatter_updates, indices_to_update, updates)
            
            return i + 1, scatter_updates

        # Run the loop to update the TensorArray
        _, scatter_updates = tf.while_loop(
            cond=lambda i, _: tf.less(i, quarter_batch_size),
            body=update_tensor,
            loop_vars=[tf.constant(0), scatter_updates]
        )

        return scatter_updates

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(CustomDataAugmentation, self).get_config()
        config.update({
            'num_components': self.num_components, 
            'amplitude': self.amplitude, 
            'fs': self.fs
        })
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
        
        # # corrupt the output to make it work as a DAE

        # mu = 0
        # sigma = 1
        # noise = 0.1 * np.random.normal(mu, sigma, size=np.shape(y_pred))
        # y_pred += noise

        y2_pred_combined = tf.multiply(y_pred[:, :, 0], y_true[:, :, 1])
        y1_pred_combined = tf.multiply(y_true[:, :, 0], y_pred[:, :, 1])

        y_combined =  tf.multiply(y_pred[:, :, 0], y_pred[:, :, 1])
        
        # y_pred_combined = y1_pred_combined + y2_pred_combined

        
        loss_combined = (
            tf.keras.losses.logcosh(y_true_mod, y2_pred_combined) + 
            tf.keras.losses.logcosh(y_true_mod, y1_pred_combined)
            # tf.keras.losses.logcosh(y_true_mod, y_combined)
        )

        # loss_combined = tf.keras.losses.logcosh(y_true_mod, y_combined)
        
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
        epochs_in_patience = 15, 
        limit_epoch_for_lr_scheduler = 10
    ):
        
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

        self.limit_epoch_for_lr_scheduler = limit_epoch_for_lr_scheduler
        
        pass

    def lr_scheduler(self, epoch, lr):
  
        if epoch < self.limit_epoch_for_lr_scheduler:
            return lr
        else:
            return np.float64(lr * np.exp(-0.1))
    
    @staticmethod
    def downsampling(inputs, num_filters, stride):
        
        x = Conv1D(num_filters, kernel_size=1, strides=stride, padding='same')(inputs)
        x = BatchNormalization()(x)
        
        return x

    @staticmethod
    def conv_block(inputs, num_filters, kernel_size=3, stride=1, padding='same', activation='relu'):
        x = Conv1D(
                num_filters, 
                kernel_size, 
                strides=stride, 
                padding=padding, 
                # activity_regularizer=tf.keras.regularizers.L2(l2=1e-5)       
            )(inputs)
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
        
        x = self.conv_block(
            inputs, 
            num_filters=filters_num, 
            kernel_size=2, 
            padding='valid'
        )
        x = Conv1DTranspose(
            filters_num, 
            kernel_size=kernel_size, 
            activation=activation, 
            strides=stride, 
            #output_padding=output_padding
        )(x)
        x = self.conv_block(
            x, 
            num_filters=filters_num, 
            kernel_size=1, 
            padding='valid'
        )
        
        x = Add()([x, skip_connection])
        
        print('x signals', np.shape(x))

        return x


    def signal_decoder_block(self, x, encoder_block1, encoder_block2):

        #decoder = self.decoder_block(x, encoder_block3, 256, kernel_size=4)
        decoder = self.decoder_block(x, encoder_block2, 128, kernel_size=4)
        decoder = self.decoder_block(decoder, encoder_block1, 64, kernel_size=4)

        # decoder = Dropout(0.1)(decoder)

        x = self.conv_block(decoder, num_filters=64, kernel_size=2, padding='valid', activation='relu')

        x = Conv1DTranspose(
            64, 
            kernel_size=4, 
            activation="relu", 
            strides=2,
        )(x)

        print(np.shape(x))

        signal_decoded = self.conv_block(x[:, :, 0:32], num_filters=1, kernel_size=1, padding='valid', activation='relu')
        mask_decoded = self.conv_block(x[:, :, 32::], num_filters=1, kernel_size=1, padding='valid', activation='relu')

        x = tf.concat([signal_decoded, mask_decoded], 2)
                    
        decode_signal = Activation('relu')(x)
            
        return decode_signal

    def linknet(self): 
        inputs = Input(batch_shape=self.input_shape)

        aug_inputs = CustomDataAugmentation(num_components=15, amplitude=0.1, fs=500)(inputs)
        
        
        #inputs = Dropout(0.5)(inputs)
        # Encoder
        encoder_block1 = self.encoder_block(aug_inputs, num_filters=64)
        print('Encoder Block 1', np.shape(encoder_block1))
        encoder_block1 = Dropout(0.2)(encoder_block1)

        encoder_block2 = self.encoder_block(encoder_block1, num_filters=128)
        print('Encoder Block 2', np.shape(encoder_block2))
        encoder_block2 = Dropout(0.2)(encoder_block2)

        encoder_block3 = self.encoder_block(encoder_block2, num_filters=256)
        print('Encoder Block 3', np.shape(encoder_block3))
        encoder_block3 = Dropout(0.2)(encoder_block3)

        # encoder_block4 = self.encoder_block(encoder_block3, num_filters=512)
        # print('Encoder Block 4', np.shape(encoder_block4))
        # encoder_block4 = Dropout(0.2)(encoder_block4)
   
        outputs = self.signal_decoder_block(encoder_block3, encoder_block1, encoder_block2)

        print('Output form', np.shape(outputs))

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='linknet')
        
        #self.model.load_weights('/home/julia/Documents/research/backbone/backbone-b2-dataset-FIXED.weights.h5', skip_mismatch=True, by_name=True)
        
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
                    tf.keras.callbacks.LearningRateScheduler(self.lr_scheduler)
                ],
            )
        
        if self.testing_data is None:
            return history, [], []
        else:
            test = self.model.evaluate(self.testing_data, self.ground_truth_testing)
            
            start = time.time()

            prediction = self.model.predict(self.testing_data)

            end = time.time()

            print(f'Predict duration for {np.shape(self.testing_data)} is {end-start} seconds ')
            
            return history, test, prediction
    
    def save(self, path_dir):
        
        self.model.save(path_dir)
        
