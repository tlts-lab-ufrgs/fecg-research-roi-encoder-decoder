#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
U Net model to fECG extraction

Created on Sunday Oct 22 08:30:00 2023

@author: juliacremus
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
     
from tensorflow_examples.models.pix2pix import pix2pix


def unet(INPUT_SHAPE, OUTPUT_CHANNELS, DEPTH):
      
  base_model = tf.keras.applications.MobileNetV2(
      input_shape=INPUT_SHAPE, 
      weights=None, 
      include_top=False
      )

  # Use as ativações dessas camadas
  layer_names = [
      'block_1_expand_relu',   # 64x64
      'block_3_expand_relu',   # 32x32
      'block_6_expand_relu',   # 16x16
      'block_13_expand_relu',  # 8x8
      'block_16_project',      # 4x4
  ]
  layers = [base_model.get_layer(name).output for name in layer_names]

  # Crie o modelo de extração de características
  down_stack = tf.keras.Model(
            inputs=base_model.input, 
            outputs=layers
            )

  down_stack.trainable = True
      

  up_stack = [
      pix2pix.upsample(512, DEPTH),  # 4x4 -> 8x8
      pix2pix.upsample(256, DEPTH),  # 8x8 -> 16x16
      pix2pix.upsample(128, DEPTH),  # 16x16 -> 32x32
      pix2pix.upsample(64, DEPTH),   # 32x32 -> 64x64
  ]
      
  def unet_model(output_channels):

    # Esta é a última camada do modelo
    last = tf.keras.layers.Conv2DTranspose(
        1, DEPTH, strides=2,
        padding='same', activation='softmax')  #64x64 -> 128x128

    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)
    x = inputs

    # Downsampling através do modelo
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling e estabelecimento das conexões de salto
    for up, skip in zip(up_stack, skips):
      x = up(x)
      concat = tf.keras.layers.Concatenate()
      x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


  model = unet_model(OUTPUT_CHANNELS)
  model = down_stack
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
      
  return model
      
