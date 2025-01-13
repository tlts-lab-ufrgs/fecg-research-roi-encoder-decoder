import numpy as np
import tensorflow as tf
import keras

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
        batch_indices = tf.random.uniform(shape=[quarter_batch_size], minval=0, maxval=batch_size, dtype=tf.int32)

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