
import tensorflow as tf

def scheduler(epoch, lr):
  if epoch < 2:
    return lr
  else:
    return lr * tf.math.exp(-0.3)


callback = tf.keras.callbacks.LearningRateScheduler(scheduler)