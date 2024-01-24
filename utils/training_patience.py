import tensorflow as tf

def callback(to_monitor = 'loss', epochs = 3):

    return tf.keras.callbacks.EarlyStopping(monitor=to_monitor, patience=epochs)
