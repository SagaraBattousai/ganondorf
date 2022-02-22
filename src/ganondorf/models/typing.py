from typing import Union
import tensorflow as tf

__all__ = ['arch_block', 'tensors']

arch_block = Union[tf.keras.Model, list[tf.keras.Sequential]]
tensors = Union[tf.Tensor, list[tf.Tensor]]
