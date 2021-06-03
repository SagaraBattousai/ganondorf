""" Residual module for definition of Residual layers

"""

from typing import Tuple, Union, ClassVar
import functools
import numpy as np
import tensorflow as tf

Conv3DSpec = Union[int, Tuple[int, int, int]]

class ResidualBottleneckLayer(tf.keras.Model):

  DOWN_FILTER_SCALE: ClassVar[int] = 4
  _Name_Acc = 0

  def __init__(self, 
               filters: Conv3DSpec,
               kernel_size: Conv3DSpec = (3, 3, 3),
               strides: Conv3DSpec = (1, 1, 1),
               name: str = None):

    if name == None:
      ResidualBottleneckLayer._Name_Acc += 1
      name = f"Residual_Bottleneck_{ResidualBottleneckLayer._Name_Acc}"

    super().__init__(name=name)
    self._filter1, self._filter2, self._filter3 = [filters] * 3 \
        if isinstance(filters, int) else filters 
        #TODO set indent if else is below if

    self._kernel_size = [kernel_size] * 3 \
        if isinstance(kernel_size, int) else kernel_size

    self._strides = [strides] * 3 \
        if isinstance(strides, int) else strides

    # Feature Down Convolution 
    self.down_norm = tf.keras.layers.BatchNormalization()
    self.down_act  = tf.keras.layers.ReLU()
    self.down_conv = tf.keras.layers.Conv3D(filters=self._filter1,
                                            kernel_size=1,
                                            strides=strides,
                                            padding="valid")

    # kernel_size Convolution
    self.norm = tf.keras.layers.BatchNormalization()
    self.act  = tf.keras.layers.ReLU()
    self.conv = tf.keras.layers.Conv3D(filters=self._filter2,
                                       kernel_size=kernel_size,
                                       strides=1,
                                       padding="same")

    # Feature Up Convolution
    self.up_norm = tf.keras.layers.BatchNormalization()
    self.up_act  = tf.keras.layers.ReLU()
    self.up_conv = tf.keras.layers.Conv3D(filters=self._filter3,
                                          kernel_size=1,
                                          strides=1,
                                          padding="valid")


  #TODO add insert code for first def after __init__
  def call(self, input_tensor, training=False):
    """ For the first Residual Unit 
    (that follows a stand-alone convolutional layer,
    conv1), we adopt the first activation right after conv1
    and before splitting into two paths; for the last Residual Unit
    (followed by average pooling and a fullyconnected classifier),
    we adopt an extra activation right after its element-wise addition

    """

    x = self.down_norm(input_tensor, training=training)
    x = self.down_act(x)
    x = self.down_conv(x)

    x = self.norm(x, training=training)
    x = self.act(x)
    x = self.conv(x)

    x = self.up_norm(x, training=training)
    x = self.up_act(x)
    x = self.up_conv(x)

    x += input_tensor

    return x


  #TODO: fix indenting for this too
  @classmethod
  def as_residual_bridge(cls, shape: Union[int, Tuple[int, int]]):
    """ Doc String

    input -> [(input_shape -> 1/4) -> (1/4 -> 1/4) -> [1/4 -> output_shape)]

    """
    out_shape = shape if isinstance(shape, int) else shape[1]
    scale = out_shape // cls.DOWN_FILTER_SCALE
    filters = [scale, scale, out_shape]

    return cls(filters, kernel_size=(3, 3, 3), strides=(1,1,1))

  @staticmethod
  def _tuple_reducer(acc, elem) -> str:
    if acc == "":
      return f"({elem})"
    else:
      return f"{acc[:-1]}x{elem})"

  #TODO: Spacing properly i.e. using maths
  def __str__(self):
    kernel_string = functools.reduce(self._tuple_reducer, self._kernel_size, "")

    stride_string = functools.reduce(self._tuple_reducer, self._strides, "")

    return (f"Layer  |  filter->filter |  kernel size |  stride  | Padding\n\n"

            f"Conv 1 |  (input -> {self._filter1})  | "
            f"   (1x1x1)   | {stride_string}  | \"valid\"\n"

            "\t\t\u21D3\n"

            f"Conv 2 |   ({self._filter1} -> {self._filter2})    | "
            f"   {kernel_string}   | (1x1x1)  | \"same\"\n"

            "\t\t\u21D3\n"

            f"Conv 3 |    ({self._filter2} -> {self._filter3})  | "
            f"   (1x1x1)   | (1x1x1)  | \"valid\""
            )

