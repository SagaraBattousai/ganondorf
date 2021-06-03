""" 
  Module containing the interface for loading a dataset
  as a tensorflow dataset and an ArrayLoader as a trivial but useful example
"""

import collections
from abc import ABC, abstractmethod
import attr
import numpy as np
import tensorflow as tf

class Loader(ABC):
  """ Interface for loading a dataset as a tensorflow dataset """

  @abstractmethod
  def load_dataset(self) -> tf.data.Dataset:
    """ 
    Single Interface method defining how to load a dataset as an instance of
    a tensorflow Dataset
    """
    pass

@attr.s(auto_attribs=True, slots=True, frozen=True)
class ArrayLoader(Loader):
  """
  Implementation of Loader that loads numpy arrays which is an array
  of vectors of input and output concatinated at *label_index*
  """
  array: np.array = None
  label_index: int = -1

  def load_dataset(self) -> tf.data.Dataset:

    input_array, label_array = np.hsplit(self.array,
                                         np.array([self.label_index]))

    return tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x = input_array,
            y = label_array
            )
        )
