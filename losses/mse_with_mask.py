import tensorflow as tf

def mse_with_mask(y_true, y_pred):

    # print(y_true)

    y_true_mod = tf.multiply(y_true[:, :, 0], y_true[:, :, 1])
    y_pred_mod = tf.multiply(y_pred[:, :, 0], y_true[:, :, 1])

    N = tf.cast(tf.shape(y_true_mod)[0], tf.float32)

    loss = tf.reduce_sum(tf.pow(y_true_mod - y_pred_mod, 2)) / N

    return loss