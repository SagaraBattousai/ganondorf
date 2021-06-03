import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import datetime
from typing import Tuple
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import tensorflow as tf

from ganondorf.data import datasets as gds
from ganondorf import pix2pix as p2p

def dir_path(dirname: str, *args) -> str:
  return os.path.join(os.path.dirname(__file__), dirname, *args)

def patch_dataset(pre, post):
  pre_patches  = gds.split_into_patches(pre)
  post_patches = gds.split_into_patches(post)

  return (pre_patches, post_patches)

def image_norm(image):
  max_image = np.amax(image, axis=0)
  max_image[max_image == 0] = 1
  return (image / max_image)

def invert(image):
  image = image * (-1)
  image = image + 1
  return image


if __name__ == "__main__":
  btl = gds.BrainTumorProgressionLoader()

  BUFFER_SIZE = 400 #1000
  BATCH_SIZE = 1
  IMG_WIDTH = 256
  IMG_HEIGHT = 256
  EPOCHS = 150  #100,000 in paper and another 100,000 after pretraining :O

  train_pre  = tf.data.Dataset.from_tensor_slices(
      list(map(image_norm, btl.pre_scans[:10]))
      )
  # train_post = tf.data.Dataset.from_tensor_slices(list(map(image_norm, btl.post_scans[:10])))

  test_pre  = tf.data.Dataset.from_tensor_slices(
      list(map(image_norm, btl.pre_scans[10:]))
      )
  # test_post = tf.data.Dataset.from_tensor_slices(list(map(image_norm, btl.post_scans[10:])))

  ##################################################
  # New Invert of target do make it look diff
  train_post_images = list(
      map(invert,
          map(image_norm,
              btl.post_scans[:10]
              )
         )
      )

  test_post_image = list(
      map(invert,
          map(image_norm,
              btl.post_scans[10:]
              )
         )
      )

  train_post = tf.data.Dataset.from_tensor_slices(train_post_images)

  test_post = tf.data.Dataset.from_tensor_slices(test_post_image)

  ##################################################



  train_dataset = tf.data.Dataset.zip((train_pre, train_post))
  test_dataset  = tf.data.Dataset.zip((test_pre, test_post))

  train_dataset = train_dataset.shuffle(BUFFER_SIZE)
  test_dataset  = test_dataset.shuffle(BUFFER_SIZE)

  #Ignore Batching for now due to patch splitting
  #train_dataset = train_dataset.batch(BATCH_SIZE) 
  #test_dataset  = test_dataset.batch(BATCH_SIZE)


  train_dataset = train_dataset.map(patch_dataset)
  test_dataset  = test_dataset.map(patch_dataset)
  
  ##TODO: split and sew code!
  #>>> ds2 = ds1.map(gds.split_into_patches)
  #>>> ids2 = iter(ds2)
  #>>> x = next(ids2)
  #>>> ids2 = iter(ds2)
  #>>> x[0].shape
  #TensorShape([24, 32, 32, 1])
  #>>> xten = tf.stack(x)
  #>>> xten.shape
  #TensorShape([64, 24, 32, 32, 1])

  generator = p2p.Generator()
  discriminator = p2p.Discriminator()

  generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                                 beta_1=0.5)
  
  discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                                     beta_1=0.5)
  
<<<<<<< HEAD
  checkpoint_dir = dir_path("tensorflow_outputs", "training_checkpoints")
=======
  checkpoint_dir = dir_path("training_checkpoints")
>>>>>>> origin/master
  
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(
      generator_optimizer=generator_optimizer,
      discriminator_optimizer=discriminator_optimizer,
      generator=generator,
      discriminator=discriminator)


<<<<<<< HEAD
  log_dir = dir_path("tensorflow_outputs", "logs")
=======
  log_dir = dir_path("logs")
>>>>>>> origin/master

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





