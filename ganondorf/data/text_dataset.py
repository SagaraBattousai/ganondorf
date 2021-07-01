""" TextDataset class module docstring

"""

import numpy as np
import tensorflow as tf
from . import dataset

class TextDataset(dataset.Dataset):

  @staticmethod
  @tf.function
  def normalise_text(data:tf.Tensor, label_index:int=None) -> tf.Tensor:
    """ ```data``` must have dimension rows be dataset and cols be datapoints 
    """
    min_data_vector = tf.reduce_min(data, axis=0)
    max_data_vector = tf.reduce_max(data, axis=0)

    if label_index is not None:
      min_data_vector[label_index]= 0
      max_data_vector[label_index]= 1

    numerator = tf.subtract(data, min_data_vector)
    denominator = tf.subtract(max_data_array, min_data_array)

    return tf.divide(numerator, denominator)

  @staticmethod
  @tf.function
  def normalise_labeled_text(data:tf.Tensor) -> tf.Tensor:
    return normalise_data(data, label_index=-1)

  @staticmethod
  @tf.function
  def shift_labels(data:tf.Tensor, 
                   label_index:int=-1, shift_amount:int=1
                   ) -> tf.Tensor:
    vector_shape = data.shape[1]
    if index < 0:
      index = vector_shape + index

    shifter = tf.constant([shift_amount])
    padding = tf.constant([[index, vector_shape - 1 - index]])

    subtractor = tf.pad(shifter, padding, "CONSTANT", constant_values=0)

    return tf.subtract(data, sub)

