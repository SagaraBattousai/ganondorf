import sys

import PIL
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


if __name__ == '__main__':
  if len(sys.argv) < 3:
    print("Requires path to image being passed in."
          " and path to model too."
          " Exiting ....")

  img = PIL.Image.open(sys.argv[1]).convert('RGB').resize((128,128))
  arr = np.asarray(img)[np.newaxis,...]

  model = tf.keras.models.load_model(sys.argv[2])
  predict = model.predict(arr)[0]
  predict = np.argmax(predict, axis=-1)

  plt.imshow(predict)
  plt.show()


  

