import tensorflow as tf
import numpy as np

def mse_mask(y_true, y_pred):
    
    

    error_mask = y_true[:, :, 1] - y_pred[:, :, 1]
    
    loss_mask = tf.reduce_mean(tf.math.square(error_mask))
    
    return loss_mask
    
    