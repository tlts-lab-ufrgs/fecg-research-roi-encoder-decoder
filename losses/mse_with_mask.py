import numpy as np
import tensorflow as tf

def mse_with_mask(y_true, y_pred):

    y_true_mod = y_true[0] * y_true[1]
    y_pred_mod = y_pred[0] * y_true[1]

    loss = tf.sum(tf.power(y_true_mod - y_pred_mod, 2)) / tf.shape(y_true[0])[0]

    return loss