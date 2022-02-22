from typing import Union
from .typing import arch_block, tensors
import tensorflow as tf

__all__ = ['unet_model']

def down_stack_pass(model:arch_block,
                    inputs:tensors
                   ) -> tuple[tensors, tensors]:
  steps = []
  x = inputs

  if isinstance(model, tf.keras.Model):
    steps = model(x)
    x = steps[-1]
  else:
    for stack in model:
      x = stack(x)
      steps.append(x)

  return (x, steps)

def unet_model(output_channels:int,
               down_stack:arch_block,
               up_stack:list[tf.keras.Sequential],
               inputs:tf.keras.layers.Input,
               residual_stack:arch_block=None,
               last_kernel:Union[int, tuple]=3):


  x, skips = down_stack_pass(down_stack, inputs)
  skips = reversed(skips)

  if residual_stack is not None:
    x, _ = down_stack_pass(residual_stack, x)
    # for res in residual_stack:
    #   x = res(x)

  for up, skip in zip(up_stack, skips):
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])
    x = up(x)

  last = tf.keras.layers.Conv2D(output_channels,
                                last_kernel,
                                strides=1,
                                padding='same',
                                activation="tanh")
  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

