import functools
from typing import Union
from .typing import arch_block, tensors
import tensorflow as tf
from ganondorf.layers import InstNormalize

__all__ = ['generator_up_model', 'discriminator']

def generator_up_model(input_shape:tuple[int],
                       output_channels:int,
                       dense_shape:tuple[int],
                       up_stack:list[tf.keras.Sequential],
                       #residual_stack:arch_block=None,
                       last_kernel:Union[int, tuple]=3,
                       ):

  dense_depth = functools.reduce(lambda a, b: a * b, dense_shape, 1)

  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(dense_depth, use_bias=False, input_shape=input_shape))
  model.add(InstNormalize())
  model.add(tf.keras.layers.ReLU()) # vs leaky since I normalize 0->1 instead of -1->1

  model.add(tf.keras.layers.Reshape(dense_shape))
  assert model.output_shape == (None, *dense_shape)

  for up in up_stack:
    model.add(up)

  model.add(tf.keras.layers.Conv2D(output_channels,
                                    last_kernel,
                                    strides=1,
                                    padding='same',
                                    activation="tanh"))

  return model



LeakyReLUMed = functools.partial(tf.keras.layers.LeakyReLU, alpha=0.2)

#temp discrim
def discriminator(down_stack:list[tf.keras.Sequential],#arch_block,
                  #residual_stack:arch_block=None,
                  last_kernel:Union[int, tuple]=3,
                  ):

  model = tf.keras.Sequential()

  for down in down_stack:
    model.add(down)
    #Add dropout after

  model.add(tf.keras.layers.Conv2D(32,#?
                                   last_kernel,
                                   strides=1,
                                   padding='valid'))#,
                                   #activation="tanh"))

  model.add(LeakyReLUMed())

  model.add(tf.keras.layers.Dense(units=1))

  return model





