""" DOC STRING FOR pix2pix Blocks Module

Block are reusable sequences of layers
"""
from typing import Callable
import functools
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


ConvMed = functools.partial(tf.keras.layers.Conv3D,
                            kernel_size=(3,3,3),
                            padding='same',
                            #use_bias=False
                            )

NormalizeMed = functools.partial(tfa.layers.InstanceNormalization,
                                axis=-1, center=True, scale=True,
                                beta_initializer="random_uniform",
                                gamma_initializer="random_uniform"
                                )


def encoder_block(
    filters: int, 
    name: str="",
    instance_norm_mask: int = 0b11,
    activation: Callable[[], tf.keras.layers.Layer] = tf.keras.layers.ReLU
    ) -> tf.keras.Sequential:
  """ Encoder Block Creator function

  Parameters
  ----------

  filters : int
    Number of filters to use for the block's two convolutional layers.

  name : str, optional
    The name to use for the block to aid in debugging, defaults to "encoder".

  apply_instancenorm : int, default 0b11
    Binary mask to decide where to apply instance normalisation, the bit
    location defines whether or not to apply instance normalisation after
    the corresponding layer, i.e. 0b10 means apply instance normalisation
    after layer 2 but not after layer 1.

  """
  initializer = tf.random_normal_initializer(0.0, 0.02)

  result = tf.keras.Sequential(name=name)

  result.add(ConvMed(filters, strides=1, kernel_initializer=initializer))

  if (instance_norm_mask & 0b01):
    result.add(NormalizeMed())

  result.add(activation())

  result.add(ConvMed(filters, strides=2, kernel_initializer=initializer))

  if (instance_norm_mask & 0b10):
    result.add(NormalizeMed())

  result.add(activation())

  return result


def decoder_block(filters: int, 
                  name: str="",
                  instance_norm_mask: int = 0b11
                  ) -> tf.keras.Sequential:
  """ Decoder Block Creator function

  Parameters
  ----------

  filters : int
    Number of filters to use for the block's convolutional and
    transposed convolutional layers.

  name : str, optional
    The name to use for the block to aid in debugging, defaults to "decoder".

  apply_instancenorm : int, default 0b11
    Binary mask to decide where to apply instance normalisation, the bit
    location defines whether or not to apply instance normalisation after
    the corresponding layer, i.e. 0b10 means apply instance normalisation
    after layer 2 but not after layer 1.

  """
  initializer = tf.random_normal_initializer(0.0, 0.02)

  result = tf.keras.Sequential(name=name)

  result.add(
      tf.keras.layers.Conv3DTranspose(
          filters, (3,3,3), strides=2, padding='same',
          kernel_initializer=initializer, use_bias=False
          )
      )

  if (instance_norm_mask & 0b01):
    result.add(NormalizeMed())

  result.add(tf.keras.layers.ReLU())

  result.add(ConvMed(filters, strides=1, kernel_initializer=initializer))

  if (instance_norm_mask & 0b10):
    result.add(NormalizeMed())

  result.add(tf.keras.layers.ReLU())

  return result
