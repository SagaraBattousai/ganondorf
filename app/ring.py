import sys

import PIL
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import ganondorf.data

model = None

@tf.function
def normalize(tensor_image):
  return tf.cast(tensor_image, tf.float32) / 255.0

@tf.function
def load_image(image, mask):
  input_image = normalize(image)
  return input_image, mask

@tf.function
def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def generate_ring(img_name, network="RingIONetwork"):
  global model
  if model is None:
    model = tf.keras.models.load_model(network)

  img = PIL.Image.open(img_name).convert('RGB')
  img = np.asarray(img)
  if img.shape[0] != img.shape[1]:
    img = ganondorf.data.square_pad(img)
  img = PIL.Image.fromarray(img).resize((128,128))
  img = np.asarray(img)

  arr = (img[np.newaxis,...]).astype(np.float32) / 255.0

  predict = create_mask(model.predict(arr))
  predict = predict.numpy()
  predict = predict.astype(np.uint8)
  predict = predict.reshape(predict.shape[0], predict.shape[1])

  return predict, img

