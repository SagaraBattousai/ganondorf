import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import ganondorf.data.datasets.segmentation_loader as segloader

from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras.preprocessing import image_dataset_from_directory

simage = None
smask = None

# def nearest_convolve(arr):
#   out = np.empty_like(arr)
#   height = arr.shape[0]
#   width = arr.shape[1]
#   for i in range(height):
#     for j in range(width):
      


def add_sample_weights(image, mask):
  class_weights = tf.constant([1.33, 1.0]) # 1.33 looks awesome i think 1.25 too
  class_weights = class_weights / tf.reduce_sum(class_weights)

  sample_weights = tf.gather(class_weights, indices=tf.cast(mask, tf.int32))

  return image, mask, sample_weights

def normalize(tensor_image):
  return tf.cast(tensor_image, tf.float32) / 255.0

def display(display_list):
  #plt.figure(figsize=(15,15))
  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

@tf.function
def load_image_train(image, mask, image_shape=(128,128)):
  input_image = tf.image.resize(image, image_shape)
  input_mask  = tf.image.resize(mask, image_shape)

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask  = tf.image.flip_left_right(input_mask)

  input_image = normalize(input_image)

  return input_image, input_mask

# Why no @tf.function?
def load_image_test(image, mask, image_shape=(128,128)):
  input_image = tf.image.resize(image, (128,128))
  input_mask  = tf.image.resize(mask, image_shape)

  input_image = normalize(input_image)

  return input_image, input_mask
  

def unet_model(output_channels, down_stack, up_stack):
  inputs = tf.keras.layers.Input(shape=[128,128,3])

  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  last = tf.keras.layers.Conv2DTranspose(output_channels,
                                         3,
                                         strides=2,
                                         padding='same')
  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1) # Choose arg with max probability
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(model, dataset=None, num=1, simage=None, smask=None):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([simage, 
             smask, 
             create_mask(model.predict(simage[tf.newaxis, ...]))])



class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    #clear_output(wait=True)
    if (epoch + 1) % 5 == 0:
      show_predictions(self.model, simage=simage, smask=smask)
    print('\nSample Predictions after epoch {}\n'.format(epoch+1))


if __name__ == '__main__':

  DATASET_PATH = os.path.join(
      "..","data","datasets","datasets","ALSegment"
      )

  TRAIN_PATH = os.path.join(DATASET_PATH, "train")
  TEST_PATH  = os.path.join(DATASET_PATH, "test")

  TRAIN_LENGTH = len(os.listdir(TRAIN_PATH))
  BATCH_SIZE = 64
  BUFFER_SIZE = 1000
  STEPS_PER_EPOCH = 1#TRAIN_LENGTH // BATCH_SIZE #  will be 0, is this an issue?

  train_data = segloader.load_segmentation(TRAIN_PATH)

  test_data = segloader.load_segmentation(TEST_PATH)

  train = train_data.map(load_image_train,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # train = train_data.map(add_sample_weights,
                            # num_parallel_calls=tf.data.experimental.AUTOTUNE)

  test = test_data.map(load_image_test)

  train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
  train_dataset = train_dataset.prefetch(
      buffer_size=tf.data.experimental.AUTOTUNE
      )


  test_dataset = test.batch(BATCH_SIZE)

  simage, smask  = next(iter(test.take(1)))

  OUTPUT_CHANNELS = 2

  base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3],
                                                 include_top=False)

  layer_names = [
      'block_1_expand_relu',  #  64x64
      'block_3_expand_relu',  #  32x32
      'block_6_expand_relu',  #  16x16
      'block_13_expand_relu', #  8x8
      'block_16_project',     #  4x4
      ]

  base_model_outputs = \
      [base_model.get_layer(name).output for name in layer_names]

  down_stack = tf.keras.Model(inputs=base_model.input,
                              outputs=base_model_outputs)

  down_stack.trainable = False

  up_stack = [
      pix2pix.upsample(512, 3),
      pix2pix.upsample(256, 3),
      pix2pix.upsample(128, 3),
      pix2pix.upsample(64, 3),
      ]
#-------------------------------------


  # residual_stack = [
  #     residual.ResidualBottleneckLayer.as_residual_bridge(128),
  #     residual.ResidualBottleneckLayer.as_residual_bridge(128),
  #     residual.ResidualBottleneckLayer.as_residual_bridge(128),
  #     ]

  # after_residual_activation = tf.keras.layers.ReLU()

  # decoder_stack = [
  #     decoder_block_3D(128, name="decode_block_1"),  # (bs,  6,  8,  8, 128)
  #     decoder_block_3D(64,  name="decode_block_2"),  # (bs, 12, 16, 16, 64)
  #     decoder_block_3D(32,  name="decode_block_3"),  # (bs, 24, 32, 32, 32)
  #     ]



#-------------------------------------



  model = unet_model(OUTPUT_CHANNELS, down_stack, up_stack)
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True
                    ),
                metrics=['accuracy'])

  # tf.keras.utils.plot_model(model, show_shapes=True, dpi=64, to_file="plot.svg")

  show_predictions(model, simage=simage, smask=smask)


  EPOCHS = 12 # 25 #15 #  12 looks best maybe 12.5 10 is consistantly best #20
  VAL_SUBSPLITS = 5
  VALIDATION_STEPS = 1#len(os.listdir(TEST_PATH)) // BATCH_SIZE // VAL_SUBSPLITS

  model_history = model.fit(train_dataset.map(add_sample_weights),
                            epochs=EPOCHS,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_steps=VALIDATION_STEPS,
                            validation_data=test_dataset,
                            callbacks=[DisplayCallback()])

  loss = model_history.history['loss']
  val_loss = model_history.history['val_loss']

  plt.figure()
  plt.plot(model_history.epoch, loss, 'r', label='Training loss')
  plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
  plt.title('Training and Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss Value')
  plt.ylim([0, 1])
  plt.legend()
  plt.show()

  # model.save("chkpt/")
  model.save("E{:02}_chkpt/".format(EPOCHS))


