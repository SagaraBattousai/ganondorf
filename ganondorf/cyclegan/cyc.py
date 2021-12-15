"""
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import ganondorf as gd
#from ganondorf.data import datasets as gds

#from .generator import Generator, generator_loss
#from .discriminator import Discriminator, discriminator_loss
#from .run import fit
"""
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import ganondorf as gd

from generator import Generator, generator_loss

from discriminator import Discriminator, discriminator_loss

AUTOTUNE = tf.data.AUTOTUNE
LAMBDA = 10

#example_image = None
#example_mask = None

#TODO: -> image = gd.data.normalize(image) vs -127 etc

# @tf.function
# def create_mask(pred_mask):
#   pred_mask = tf.argmax(pred_mask, axis=-1)
#   pred_mask = pred_mask[..., tf.newaxis]
#   return pred_mask[0]

# class DisplayCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs=None):
#     if (epoch + 1) % 10 == 0:
#       gd.data.Visualizer.show_image_predictions(
#           self.model,
#           dataset=(example_image, example_mask),
#           format_prediction=create_mask
#           )
#     print('\nSample Predictions after epoch {}\n'.format(epoch+1))


# class SaverCallback(tf.keras.callbacks.Callback):
#   def __init__(self):
#     super().__init__()
#     self.best_accuracy = -1
#     self.lowest_loss = 100

#   def on_epoch_end(self, epoch, logs): #logs can't be empty
#     if (epoch + 1) > 59:
#       if logs['val_loss'] < self.lowest_loss:
#         self.model.save("./ring_segmentation_checkpoint/lowest_loss")
#         self.lowest_loss = logs['val_loss']

#       if logs['val_accuracy'] > self.best_accuracy:
#         self.model.save("./ring_segmentation_checkpoint/best_accuracy")
#         self.best_accuracy = logs['val_accuracy']

def random_crop(image, size=(128,128)):
  cropped_image = tf.image.random_crop(
      image, size=[size[0], size[0], 3])

  return cropped_image

# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1 # vs /255.0
  return image

def random_jitter(image):
  # resizing to 143 x 143 x 3 for 128x128 img
  image = tf.image.resize(image, [143, 143],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 128 x 128 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image

def preprocess_image_train(image):
  image = random_jitter(image)
  image = normalize(image)
  return image

def preprocess_image_test(image):
  image = normalize(image)
  return image

def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  return LAMBDA * loss1

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

def generate_images(model, test_input):
  prediction = model(test_input)

  display_list = [test_input[0], prediction[0]]
  title = ["Input Image", "Predicted Image"]

  for i in range(2):
    plt.subplot(1,2,i+1)
    plt.title(title[i])
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

def generate_cycle(model_da, test_da, model_db, test_db):
  prediction_db = model_da(test_da)
  prediction_da = model_db(test_db)

  display_list = [test_da[0], prediction_db[0],
                  test_db[0], prediction_da[0]]
  title = ["Domain A Image", "Predicted",
           "Domain B Image", "Predicted"]

  for i in range(4):
    plt.subplot(2,2,i+1)
    plt.title(title[i])
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()


def transfer_generator(image_shape:tuple[int,int]=(128,128),
                       output_channels:int=3):
  base_model = tf.keras.applications.MobileNetV2(
      input_shape=[*image_shape, 3], 
      include_top=False
      )

  layer_names = [
      'block_1_expand_relu',  #  64x64x96
      'block_3_expand_relu',  #  32x32x144
      'block_6_expand_relu',  #  16x16x192
      'block_13_expand_relu', #  8x8x576
      'block_16_project',     #  4x4x320
      # 'Conv_1', # 4x4x1280
      ]

  base_model_outputs = \
      [base_model.get_layer(name).output for name in layer_names]

  down_stack = tf.keras.Model(inputs=base_model.input,
                              outputs=base_model_outputs)

  down_stack.trainable = False

  residual_stack = [
      # gd.layers.ResidualBottleneckLayer.as_residual_bridge(2, 1280), # 320),
      # gd.layers.ResidualBottleneckLayer.as_residual_bridge(2, 1280), # 320),
      # gd.layers.ResidualBottleneckLayer.as_residual_bridge(2, 1280), # 320),
      gd.layers.ResidualBottleneckLayer.as_residual_bridge(2, 320),
      gd.layers.ResidualBottleneckLayer.as_residual_bridge(2, 320),
      gd.layers.ResidualBottleneckLayer.as_residual_bridge(2, 320),
      tf.keras.layers.ReLU(),
      ]

  up_stack = [
      # gd.layers.decoder_block_2D(1280, name="decode_block_1"), #512
      gd.layers.decoder_block_2D(320, name="decode_block_1"), #512
      gd.layers.decoder_block_2D(576, name="decode_block_2"), #256
      gd.layers.decoder_block_2D(192, name="decode_block_3"), #128
      gd.layers.decoder_block_2D(144,  name="decode_block_4"), #64
      gd.layers.decoder_block_2D(96,  name="decode_block_5"), #64
      ]


  #Could remove magic 3 later
  model_input = tf.keras.layers.Input(shape=[*image_shape, 3])

  return gd.models.unet_model(output_channels,
                              down_stack, up_stack,
                              inputs=model_input,
                              residual_stack=residual_stack)

if __name__ == '__main__':
  EPOCHS = 100
  OUTPUT_CHANNELS = 3
  IMAGE_SHAPE = (128, 128)

  BATCH_SIZE = 1 #8 # 1 is from Tut
  BUFFER_SIZE = 16 #1000
  STEPS_PER_EPOCH = 1
  VAL_SUBSPLITS = 5
  VALIDATION_STEPS = 1

  dataset = gd.data.Dataset.load("cyclemodel",
                                 "ThroughSubsetCVCClinicDB",
                                 "ColonModelRenders",
                                 load_train=True,
                                 load_test=True,
                                 size=IMAGE_SHAPE)

  train_colon, train_model = dataset["trainA"], dataset["trainB"]
  test_colon, test_model = dataset["testA"], dataset["testB"]

  train_colon = train_colon.cache().map(
      preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
          BUFFER_SIZE).batch(BATCH_SIZE)#.repeat()

  train_model = train_model.cache().map(
      preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
          BUFFER_SIZE).batch(BATCH_SIZE)#.repeat()

  #------------------------------------------------------------------

  test_colon = test_colon.map(
      preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
          BUFFER_SIZE).batch(BATCH_SIZE)#.repeat()

  test_model = test_model.map(
      preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
          BUFFER_SIZE).batch(BATCH_SIZE)#.repeat()

  sc = next(iter(train_colon))
  sm = next(iter(train_model))
  

  generator_da = transfer_generator() #Generator()
  generator_db = transfer_generator() #Generator()

  discriminator_da = Discriminator()
  discriminator_db = Discriminator()

  # tf.keras.utils.plot_model(d, show_shapes=True, dpi=64)

  # contrast = 8

  # to_model = generator_da(sc)

  # plt.subplot(121)
  # plt.title('Colon')
  # plt.imshow(sc[0] * 0.5 + 0.5)

  # plt.subplot(122)
  # plt.title('Gen')
  # plt.imshow(to_model[0] * 0.5 * contrast + 0.5)

  # plt.show()


  generator_da_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  generator_db_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


  discriminator_da_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  discriminator_db_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

  checkpoint_path = "./checkpoints/train"

  ckpt = tf.train.Checkpoint(
      generator_da=generator_da,
      generator_db=generator_db,
      discriminator_da=discriminator_da,
      discriminator_db=discriminator_db,
      generator_da_optimizer=generator_da_optimizer,
      generator_db_optimizer=generator_db_optimizer,
      discriminator_da_optimizer=discriminator_da_optimizer,
      discriminator_db_optimizer=discriminator_db_optimizer)

  ckpt_manager = tf.train.CheckpointManager(ckpt,
                                            checkpoint_path,
                                            max_to_keep=5)

  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Latest Checkpoint Restored")


  @tf.function
  def train_step(real_da, real_db):
    #persistent is set to True as GradientTape is used more than once
    with tf.GradientTape(persistent=True) as tape:
      fake_db = generator_da(real_da, training=True)
      cycled_da = generator_db(fake_db, training=True)

      fake_da = generator_db(real_db, training=True)
      cycled_db = generator_da(fake_da, training=True)

      # same_x and same_y are for ident loss
      same_da = generator_db(real_da, training=True)
      same_db = generator_da(real_db, training=True)

      disc_real_da = discriminator_da(real_da, training=True)
      disc_real_db = discriminator_db(real_db, training=True)

      disc_fake_da = discriminator_da(fake_da, training=True)
      disc_fake_db = discriminator_db(fake_db, training=True)

      # Calculate the loss
      gen_da_loss = generator_loss(disc_fake_db)
      gen_db_loss = generator_loss(disc_fake_da)

      total_cycle_loss = calc_cycle_loss(real_da, cycled_da) + \
          calc_cycle_loss(real_db, cycled_db)

      # Total generator loss
      total_gen_da_loss = gen_da_loss + \
          total_cycle_loss + \
          identity_loss(real_db, same_db)

      total_gen_db_loss = gen_db_loss + \
          total_cycle_loss + \
          identity_loss(real_da, same_da)


      disc_da_loss = discriminator_loss(disc_real_da, disc_fake_da)
      disc_db_loss = discriminator_loss(disc_real_db, disc_fake_db)

    # Calculate gradients
    generator_da_gradients = tape.gradient(total_gen_da_loss,
                                           generator_da.trainable_variables)
    generator_db_gradients = tape.gradient(total_gen_db_loss,
                                           generator_db.trainable_variables)

    discriminator_da_gradients = tape.gradient(
        disc_da_loss,
        discriminator_da.trainable_variables)

    discriminator_db_gradients = tape.gradient(
        disc_db_loss,
        discriminator_db.trainable_variables)

    # Apply gradients to optimizer
    generator_da_optimizer.apply_gradients(
        zip(generator_da_gradients,
            generator_da.trainable_variables))

    generator_db_optimizer.apply_gradients(
        zip(generator_db_gradients,
            generator_db.trainable_variables))

    discriminator_da_optimizer.apply_gradients(
        zip(discriminator_da_gradients,
            discriminator_da.trainable_variables))

    discriminator_db_optimizer.apply_gradients(
        zip(discriminator_db_gradients,
            discriminator_db.trainable_variables))

  for epoch in range(EPOCHS):
    start = time.time()

    n = 0
    for image_da, image_db in tf.data.Dataset.zip((train_colon, train_model)):
      train_step(image_da, image_db)
      if n % 10 == 0:
        print(".", end="")
      n += 1

    if (epoch + 1) % 10 == 0:
      generate_cycle(generator_da, sc, generator_db, sm)

    if (epoch + 1) % 5 == 0:
      ckpt_save_path = ckpt_manager.save()
      print("Saving checkpoint for epoch {} at {}".format(epoch + 1,
                                                          ckpt_save_path))

    print("Time taken for epoch {} is {} sec\n".format(epoch + 1,
                                                       time.time() - start))
