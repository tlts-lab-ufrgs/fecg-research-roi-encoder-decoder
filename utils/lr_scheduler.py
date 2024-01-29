
import tensorflow as tf
import numpy as np

def scheduler(epoch, lr):
  # if epoch < 15:
  #   return lr
  # else:
  return 0.005 * (np.exp(-epoch/10) + 0.1* np.sin(np.pi * epoch/10)**2)
  # if epoch < 10:
  #   return lr
  # else:
  #   return lr * tf.math.exp(-0.1)

def decayed_learning_rate(step, lr):
  alpha = 0
  initial_decay_lr = 0.001
  step = min(step, 30)
  cosine_decay = 0.5 * (1 + np.cos(np.pi * step / 30))
  decayed = (1 - alpha) * cosine_decay + alpha
  
  return initial_decay_lr * decayed

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)