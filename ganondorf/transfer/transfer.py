""" Transfer Learning Module DOC STRING

"""

#import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
#import os
#import glob
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

def get_image_and_label(fname:str, label:int) -> Tuple[np.array, int]:
  return (np.asarray(Image.open(fname)), label)

#def resize_img(image, save=False, savename=None):
#  pass


if __name__ == "__main__":

  BATCH_SIZE = 32
  IMG_SIZE = (160, 160)#(256, 256)

  train_dataset = image_dataset_from_directory(
      "..\\data\\datasets\\datasets\\AL_Subset\\train",
      shuffle=True,
      batch_size=BATCH_SIZE,
      image_size=IMG_SIZE
    )

  test_dataset = image_dataset_from_directory(
      "..\\data\\datasets\\datasets\\AL_Subset\\test",
      shuffle=True,
      batch_size=BATCH_SIZE,
      image_size=IMG_SIZE
    )

  train_dataset = train_dataset.prefetch(4)
  test_dataset = test_dataset.prefetch(4)

  data_augmentation = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
      tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
      ])

  preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

  IMG_SHAPE = IMG_SIZE + (3,)

  base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                 include_top=False,
                                                 weights="imagenet")

  image_batch, label_batch = next(iter(train_dataset))
  feature_batch = base_model(image_batch)

  base_model.trainable = False

  global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
  feature_batch_average = global_average_layer(feature_batch)
