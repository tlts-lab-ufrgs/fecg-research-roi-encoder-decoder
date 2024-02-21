import tensorflow as tf
import numpy as np

def mse_with_mask(y_true, y_pred):

    y_true_mod = tf.multiply(y_true[:, :, 0], y_true[:, :, 1])
    y_pred_mod = tf.multiply(y_pred[:, :, 0], y_pred[:, :, 1])

    # tf.print(y_true_mod)
    # tf.print(y_pred_mod)
    
    
    # tf.print(tf.where(y_true == 1))
    
    N = 512 # tf.size(tf.where(y_true == 1))
    
    # tf.print(N)
    
    error_signal = y_true[:, :, 0] - y_pred[:, :, 0]
    
    error_mask = y_true[:, :, 1] - y_pred[:, :, 1]

    # N = tf.size(tf.where(y_true[:, :, 1] == 1))# 512 # tf.cast(tf.shape(y_true_mod)[0], tf.float32)
    
    # bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    # loss_mask = bce(y_true[:, :, 1], y_pred[:, :, 1])

    
    error = y_true_mod - y_pred_mod
    
    # error_2 = y_true[:, :, 0] - y_pred_mod
    
    # error = 0.3 * error_signal + 0.4 * error + 0.3 * error_mask

    # loss_combined = tf.reduce_sum(tf.math.square(error)) / N # tf.cast(N, tf.float32)

    loss_combined =  tf.keras.losses.logcosh(y_true_mod, y_pred_mod) 
        
    # loss_signal = tf.reduce_mean(tf.math.square(error_signal)) # tf.cast(N, tf.float32)
    
    loss_signal = tf.keras.losses.logcosh(y_true[:, :, 0], y_pred[:, :, 0]) 
    
    # loss_mask = tf.reduce_sum(tf.math.square(error_mask)) / N
    
    
    
    loss_mask = -tf.math.log(tf.math.reduce_sum(tf.math.sqrt(
        tf.multiply(y_true[:, :, 1], y_pred[:, :, 1]) + 1e-20
    )) )
    
    # if loss_mask < 0:
        
    #     loss_mask
        # 
    loss_mask_mse = tf.keras.losses.logcosh(y_true[:, :, 1], y_pred[:, :, 1]) 
    
    ratio_between_losses = 15 # loss_mask / loss_signal

    
    loss = 0.3 * loss_mask_mse + 0.3 * loss_combined + 0.4 * loss_signal

    return loss

#%%