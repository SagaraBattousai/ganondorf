import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import ganondorf.data
from ganondorf.pix2pix import residual
from ganondorf.pix2pix import blocks

example_image = None
example_mask = None

# def nearest_convolve(arr):
#   out = np.empty_like(arr)
#   height = arr.shape[0]
#   width = arr.shape[1]
#   for i in range(height):
#     for j in range(width):


@tf.function
def normalize(tensor_image):
  return tf.cast(tensor_image, tf.float32) / 255.0

@tf.function
def load_image_train(image, mask):

  if tf.random.uniform(()) > 0.5:
    image = tf.image.flip_left_right(image)
    mask  = tf.image.flip_left_right(mask)

  image = normalize(image)

  return image, mask

@tf.function
def load_image_test(image, mask):
  input_image = normalize(image)
  return input_image, mask
  
def unet_model(output_channels, down_stack, up_stack, residual_stack=None):
  inputs = tf.keras.layers.Input(shape=[128,128,3])

  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  if residual_stack is not None:
    for res in residual_stack:
      x = res(x)

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

@tf.function
def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    #clear_output(wait=True)
    if (epoch + 1) % 10 == 0:
      ganondorf.data.Visualizer.show_image_predictions(
          self.model,
          dataset=(example_image, example_mask),
          format_prediction=create_mask
          )
    print('\nSample Predictions after epoch {}\n'.format(epoch+1))


if __name__ == '__main__':
  EPOCHS = 100
  OUTPUT_CHANNELS = 2
  IMAGE_SHAPE = (128, 128)

  TRAIN_LENGTH = 34
  BATCH_SIZE = 64
  BUFFER_SIZE = 1000
  STEPS_PER_EPOCH = 1
  VAL_SUBSPLITS = 5
  VALIDATION_STEPS = 1

  train_dataset, test_dataset = ganondorf.data.Dataset.load("ALSegmentation",
                                                            size=IMAGE_SHAPE)
  train_dataset = train_dataset.map(load_image_train,
                                    num_parallel_calls=tf.data.AUTOTUNE)

  test_dataset = test_dataset.map(load_image_test)

  train_dataset = train_dataset.cache() \
                               .shuffle(BUFFER_SIZE) \
                               .batch(BATCH_SIZE)#.repeat()
  train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

  example_image, example_mask  = next(
      iter(test_dataset.take(6).skip(5))
      )

  test_dataset = test_dataset.batch(BATCH_SIZE)


  base_model = tf.keras.applications.MobileNetV2(input_shape=[*IMAGE_SHAPE, 3],
                                                 include_top=False)
  layer_names = [
      'block_1_expand_relu',  #  64x64x96
      'block_3_expand_relu',  #  32x32x144
      'block_6_expand_relu',  #  16x16x192
      'block_13_expand_relu', #  8x8x576
      'block_16_project',     #  4x4x320
      ]

  base_model_outputs = \
      [base_model.get_layer(name).output for name in layer_names]

  down_stack = tf.keras.Model(inputs=base_model.input,
                              outputs=base_model_outputs)

  down_stack.trainable = False

  residual_stack = [
      residual.ResidualBottleneckLayer.as_residual_bridge(2, 320),
      residual.ResidualBottleneckLayer.as_residual_bridge(2, 320),
      residual.ResidualBottleneckLayer.as_residual_bridge(2, 320),
      tf.keras.layers.ReLU(),
      ]
  
  up_stack = [
      blocks.decoder_block_2D(512, name="decode_block_1"),
      blocks.decoder_block_2D(256, name="decode_block_2"),
      blocks.decoder_block_2D(128, name="decode_block_3"),
      blocks.decoder_block_2D(64,  name="decode_block_4"),
      ]


  model = unet_model(OUTPUT_CHANNELS,
                     down_stack, up_stack, residual_stack=residual_stack)

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True
                    ),
                metrics=['accuracy'])

  tf.keras.utils.plot_model(model, show_shapes=True, dpi=64, to_file="plot.svg")

  ganondorf.data.Visualizer.show_image_predictions(
      model,
      dataset=(example_image, example_mask),
      format_prediction=create_mask
      )


  # 1.33 looks awesome i think 1.25 too
  add_sample_weights = ganondorf.data.Dataset.get_sample_weights_func([3, 1])

  model_history = model.fit(train_dataset.map(add_sample_weights),
                            epochs=EPOCHS,
                            #steps_per_epoch=STEPS_PER_EPOCH,
                            #validation_steps=VALIDATION_STEPS,
                            validation_data=test_dataset,
                            callbacks=[DisplayCallback()])


  ganondorf.data.Visualizer.show_image_predictions(
      model,
      dataset=(example_image, example_mask),
      format_prediction=create_mask
      )

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

  if len(sys.argv) < 2 or sys.argv[1].lower() != "nosave":
    # model.save("chkpt/")
    model.save("E{:02}_chkpt/".format(EPOCHS))


