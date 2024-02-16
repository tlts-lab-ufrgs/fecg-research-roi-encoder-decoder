import tensorflow as tf
import numpy as np

def mse_signal(y_true, y_pred):
    
    

    error = y_true[:, :, 0] - y_pred[:, :, 0]
    
    loss = tf.reduce_mean(tf.math.square(error))
    
    return loss