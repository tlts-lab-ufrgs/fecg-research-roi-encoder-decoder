import tensorflow as tf
import numpy as np

def mse_with_mask(y_true, y_pred):

    y_true_mod = tf.multiply(y_true[:, :, 0], y_true[:, :, 1])
    y_pred_mod = tf.multiply(y_pred[:, :, 0], y_pred[:, :, 1])

    # tf.print(y_true_mod)
    # tf.print(y_pred_mod)

    N = 600 # tf.cast(tf.shape(y_true_mod)[0], tf.float32)
    
    error = y_true_mod - y_pred_mod

    loss = tf.reduce_sum(tf.multiply(error, error)) / N

    return loss