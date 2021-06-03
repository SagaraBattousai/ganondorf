""" DOC STRING FOR pix2pix GAN Model
"""

from .generator import Generator, generator_loss
from .discriminator import Discriminator, discriminator_loss
from .run import fit

###########################################################################


def dir_path(dirname: str) -> str:
  return os.path.join(os.path.dirname(__file__), dirname)

def patch_dataset(pre, post):
  pre_patches  = gds.split_into_patches(pre)
  post_patches = gds.split_into_patches(post)

  return (pre_patches, post_patches)

def image_norm(image):
  max_image = np.amax(image, axis=0)
  max_image[max_image == 0] = 1
  return (image / max_image)

def main():
  import os
  import datetime
  from typing import Tuple
  import numpy as np
  import SimpleITK as sitk
  import matplotlib.pyplot as plt
  import tensorflow as tf

  from ganondorf.data import datasets as gds


  btl = gds.BrainTumorProgressionLoader()

  BUFFER_SIZE = 400 #1000
  BATCH_SIZE = 1
  IMG_WIDTH = 256
  IMG_HEIGHT = 256
  EPOCHS = 150  #100,000 in paper and another 100,000 after pretraining :O

  train_pre  = tf.data.Dataset.from_tensor_slices(list(map(image_norm, btl.pre_scans[:10])))
  train_post = tf.data.Dataset.from_tensor_slices(list(map(image_norm, btl.post_scans[:10])))

  test_pre  = tf.data.Dataset.from_tensor_slices(list(map(image_norm, btl.pre_scans[10:])))
  test_post = tf.data.Dataset.from_tensor_slices(list(map(image_norm, btl.post_scans[10:])))

  train_dataset = tf.data.Dataset.zip((train_pre, train_post))
  test_dataset  = tf.data.Dataset.zip((test_pre, test_post))

  train_dataset = train_dataset.shuffle(BUFFER_SIZE)
  test_dataset  = test_dataset.shuffle(BUFFER_SIZE)

  #Ignore Batching for now due to patch splitting
  #train_dataset = train_dataset.batch(BATCH_SIZE) 
  #test_dataset  = test_dataset.batch(BATCH_SIZE)


  train_dataset = train_dataset.map(patch_dataset)
  test_dataset  = test_dataset.map(patch_dataset)
  
  generator = p2p.Generator()
  discriminator = p2p.Discriminator()

  generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                                 beta_1=0.5)
  
  discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                                     beta_1=0.5)
  
  checkpoint_dir = dir_path("training_checkpoints")
  
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(
      generator_optimizer=generator_optimizer,
      discriminator_optimizer=discriminator_optimizer,
      generator=generator,
      discriminator=discriminator)


  log_dir = dir_path("logs")

  summary_writer = tf.summary.create_file_writer(
      os.path.join(log_dir,
                   "fit",
                   datetime.datetime.now().strftime("%Y%m%d-%H;%M;%S"))
      )

  p2p.fit(generator=generator,
          generator_loss=p2p.generator_loss,
          generator_optimizer=generator_optimizer,
          discriminator=discriminator,
          discriminator_loss=p2p.discriminator_loss,
          discriminator_optimizer=discriminator_optimizer,
          train_dataset=train_dataset,
          test_dataset=test_dataset,
          epochs=EPOCHS,
          checkpoint=checkpoint,
          summary_writer=summary_writer,
          checkpoint_prefix=checkpoint_prefix)





