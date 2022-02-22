
from typing import Union, Callable, Any, Sequence

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# May not need be array could be any??
Datapoint = Union[tf.data.Dataset, tuple[np.array, np.array]]

identity = lambda x: x

class Visualizer():

  @staticmethod
  def display_image(*display_list, title:Sequence=None):
    #plt.figure(figsize=(15,15))
    if title is None:
      title = range(1, len(display_list) + 1)

    for i in range(len(display_list)):
      plt.subplot(1, len(display_list), i+1)
      plt.title(title[i])
      plt.imshow(display_list[i])
      plt.axis('off')
    plt.show()

#
  @staticmethod
  def display_image_predictions(image, label, prediction):
    Visualizer.display_image(image,
                             label,
                             prediction,
                             title = ['Input', 'Labeled', 'Predicted'])

  @staticmethod
  def show_image_predictions(
      model:tf.keras.Model,
      dataset:Datapoint=None,
      num:int=1,
      format_prediction:Callable[np.array, np.array]=identity
      )->None:

    if isinstance(dataset, tf.data.Dataset):
      for image, label in dataset.take(num):
        print("okay")
        prediction = model.predict(image)
        Visualizer.display_image_predictions(image[0], label[0],
                                             format_prediction(prediction))
    else:
      prediction = model.predict(dataset[0][np.newaxis, ...])
      Visualizer.display_image_predictions(dataset[0], dataset[1],
                                           format_prediction(prediction))  
