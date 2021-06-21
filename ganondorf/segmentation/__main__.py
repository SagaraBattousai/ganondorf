import sys

import PIL
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import ganondorf.data



@tf.function
def normalize(tensor_image):
  return tf.cast(tensor_image, tf.float32) / 255.0


@tf.function
def load_image_test(image, mask):
  input_image = normalize(image)
  return input_image, mask

@tf.function
def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]



if __name__ == '__main__':
  if len(sys.argv) < 3:
    print("Requires path to image being passed in."
          " and path to model too."
          " Exiting ....")

  model = tf.keras.models.load_model(sys.argv[2])

  if sys.argv[1].lower() == "all":
    _, test_dataset = ganondorf.data.Dataset.load("ALSegmentation",
                                                  size=(128,128))
    test_dataset = test_dataset.map(load_image_test)    
    ganondorf.data.Visualizer.show_image_predictions(
      model,
      test_dataset.batch(1),
      num=20,
      format_prediction=create_mask)

  else:

    img = PIL.Image.open(sys.argv[1]).convert('RGB').resize((128,128))
    arr = np.asarray(img)[np.newaxis,...]


    arr = arr.astype(np.float32) / 255.0

    predict = create_mask(model.predict(arr))
    plt.imshow(predict)
    plt.show()



