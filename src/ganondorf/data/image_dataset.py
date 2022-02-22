""" ImageDataset class module docstring

"""

import tensorflow as tf
from . import dataset

class ImageDataset(dataset.Dataset):

  @staticmethod
  @tf.function
  def normalize_image(tensor_image):
    return tf.cast(tensor_image, tf.float32) / 255.0


