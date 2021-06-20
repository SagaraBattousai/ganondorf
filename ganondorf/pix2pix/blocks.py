""" DOC STRING FOR pix2pix Blocks Module

Block are reusable sequences of layers
"""
from typing import Callable
import functools
import tensorflow as tf
import tensorflow_addons as tfa


Feature3D = functools.partial(tf.keras.layers.Conv3D,
                              kernel_size=(3,3,3),
                              padding="same",
                              #use_bias=False
                              )

Decode3D = functools.partial(tf.keras.layers.Conv3DTranspose,
                             kernel_size=(3,3,3),
                             padding="same",
                             use_bias=False,
                             )

Feature2D = functools.partial(tf.keras.layers.Conv2D,
                              kernel_size=(3,3),
                              padding="same",
                              #use_bias=False
                              )

Decode2D = functools.partial(tf.keras.layers.Conv2DTranspose,
                             kernel_size=(3,3),
                             padding="same",
                             use_bias=False,
                             )

InstNormalize = functools.partial(tfa.layers.InstanceNormalization,
                                  axis=-1, center=True, scale=True,
                                  beta_initializer="random_uniform",
                                  gamma_initializer="random_uniform"
                                  )

def encoder_block(
    filters: int,
    feature_layer:tf.keras.layers.Layer,
    name: str="",
    instance_norm_mask: int = 0b11,
    activation: Callable[[], tf.keras.layers.Layer] = tf.keras.layers.ReLU
    ) -> tf.keras.Sequential:
  """ Encoder Block Creator function

  Parameters
  ----------

  filters : int
    Number of filters to use for the block"s two convolutional layers.

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

  result.add(feature_layer(filters, strides=1, kernel_initializer=initializer))

  if instance_norm_mask & 0b01:
    result.add(InstNormalize())

  if activation is not None:
    result.add(activation())

  result.add(feature_layer(filters, strides=2, kernel_initializer=initializer))

  if instance_norm_mask & 0b10:
    result.add(InstNormalize())

  if activation is not None:
    result.add(activation())

  return result


def decoder_block(
    filters: int,
    feature_layer:tf.keras.layers.Layer,
    decode_layer:tf.keras.layers.Layer,
    name: str="",
    instance_norm_mask: int = 0b11,
    activation: Callable[[], tf.keras.layers.Layer] = tf.keras.layers.ReLU
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

  result.add(decode_layer(filters, strides=2, kernel_initializer=initializer))

  if instance_norm_mask & 0b01:
    result.add(InstNormalize())

  if activation is not None:
    result.add(activation())

  result.add(feature_layer(filters, strides=1, kernel_initializer=initializer))

  if instance_norm_mask & 0b10:
    result.add(InstNormalize())

  if activation is not None:
    result.add(activation())

  return result

def encoder_block_3D(
    filters: int,
    name: str="",
    instance_norm_mask: int = 0b11,
    activation: Callable[[], tf.keras.layers.Layer] = tf.keras.layers.ReLU
    ) -> tf.keras.Sequential:

  return encoder_block(filters, Feature3D, name, instance_norm_mask, activation)

def decoder_block_3D(
    filters: int,
    name: str="",
    instance_norm_mask: int = 0b11,
    activation: Callable[[], tf.keras.layers.Layer] = tf.keras.layers.ReLU
    ) -> tf.keras.Sequential:

  return decoder_block(filters, 
                       Feature3D, Decode3D,
                       name, instance_norm_mask,
                       activation)

def encoder_block_2D(
    filters: int,
    name: str="",
    instance_norm_mask: int = 0b11,
    activation: Callable[[], tf.keras.layers.Layer] = tf.keras.layers.ReLU
    ) -> tf.keras.Sequential:

  return encoder_block(filters, Feature2D, name, instance_norm_mask, activation)

def decoder_block_2D(
    filters: int,
    name: str="",
    instance_norm_mask: int = 0b11,
    activation: Callable[[], tf.keras.layers.Layer] = tf.keras.layers.ReLU
    ) -> tf.keras.Sequential:

  return decoder_block(filters, 
                       Feature2D, Decode2D,
                       name, instance_norm_mask,
                       activation)






