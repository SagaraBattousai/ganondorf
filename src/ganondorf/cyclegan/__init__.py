
import os
import datetime
import numpy as np
import tensorflow as tf
import ganondorf as gd
#from ganondorf.data import datasets as gds

#from .generator import Generator, generator_loss
#from .discriminator import Discriminator, discriminator_loss
#from .run import fit

def dir_path(dirname: str) -> str:
  return os.path.join(os.path.dirname(__file__), dirname)

@tf.function
def normalize(tensor_image):
  return tf.cast(tensor_image, tf.float32) / 255.0


@tf.function
def load_image_test(image, model):
  input_image = normalize(image)
  return input_image, model

def main():

  cmds = gd.data.Dataset.load("cyclemodel", "ThroughSubsetCVCClinicDB",
                      size=(128,128))
  
  ri = cmds[0].map(normalize)
  mi = cmds[1]

  return ri, mi


#main()
