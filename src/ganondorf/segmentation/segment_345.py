import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import ganondorf as gd

example_image = None
example_mask = None

@tf.function
def load_image_train(image, mask):

  if tf.random.uniform(()) > 0.5:
    image = tf.image.flip_left_right(image)
    mask  = tf.image.flip_left_right(mask)

  image = gd.data.normalize(image)

  return image, mask

@tf.function
def load_image_test(image, mask):
  input_image = gd.data.normalize(image)
  return input_image, mask

@tf.function
def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    if (epoch + 1) % 10 == 0:
      gd.data.Visualizer.show_image_predictions(
          self.model,
          dataset=(example_image, example_mask),
          format_prediction=create_mask
          )
    print('\nSample Predictions after epoch {}\n'.format(epoch+1))


class SaverCallback(tf.keras.callbacks.Callback):
  def __init__(self):
    super().__init__()
    self.best_accuracy = -1
    # self.lowest_loss = 100
    self.model_best_acc = None
    self.epoch_best_acc = 0

  def on_epoch_end(self, epoch, logs): #logs can't be empty
    if logs['val_accuracy'] > self.best_accuracy:
      self.best_accuracy = logs['val_accuracy']
      self.model_best_acc = self.model
      self.epoch_best_acc = epoch
  
  def on_train_end(self, logs=None):
    print("Training has ended, best accuracy epoch was:",
          self.epoch_best_acc + 1, "with accuracy of:", self.best_accuracy)
    self.model_best_acc.save("./ring_segmentation_checkpoint/best_accuracy")


if __name__ == '__main__':
  EPOCHS = 220 #atm 205 is best
  OUTPUT_CHANNELS = 2
  IMAGE_SHAPE = (128, 128)

  TRAIN_LENGTH = 53
  BATCH_SIZE = 8
  BUFFER_SIZE = 16 #1000
  STEPS_PER_EPOCH = 1
  VAL_SUBSPLITS = 5
  VALIDATION_STEPS = 1

  train_dataset, test_dataset = gd.data.Dataset.load("ALRingWithDays345",
                                                     size=IMAGE_SHAPE)
  train_dataset = train_dataset.map(load_image_train,
                                    num_parallel_calls=tf.data.AUTOTUNE)

  test_dataset = test_dataset.map(load_image_test)

  train_dataset = train_dataset.cache() \
                               .shuffle(BUFFER_SIZE) \
                               .batch(BATCH_SIZE).repeat()
  train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

  example_image, example_mask  = next(iter(test_dataset))

  test_dataset = test_dataset.batch(BATCH_SIZE)


  base_model = tf.keras.applications.MobileNetV2(input_shape=[*IMAGE_SHAPE, 3],
                                                 include_top=False)
  layer_names = [
      'block_1_expand_relu',  #  64x64x96
      'block_3_expand_relu',  #  32x32x144
      'block_6_expand_relu',  #  16x16x192
      'block_13_expand_relu', #  8x8x576
      # 'block_16_project',     #  4x4x320
      'Conv_1', # 4x4x1280
      ]

  base_model_outputs = \
      [base_model.get_layer(name).output for name in layer_names]

  down_stack = tf.keras.Model(inputs=base_model.input,
                              outputs=base_model_outputs)

  down_stack.trainable = False

  residual_stack = [
      gd.layers.ResidualBottleneckLayer.as_residual_bridge(2, 1280), # 320),
      gd.layers.ResidualBottleneckLayer.as_residual_bridge(2, 1280), # 320),
      gd.layers.ResidualBottleneckLayer.as_residual_bridge(2, 1280), # 320),
      tf.keras.layers.ReLU(),
      ]

  up_stack = [
      gd.layers.decoder_block_2D(1280, name="decode_block_1"), #512
      gd.layers.decoder_block_2D(576, name="decode_block_2"), #256
      gd.layers.decoder_block_2D(192, name="decode_block_3"), #128
      gd.layers.decoder_block_2D(144,  name="decode_block_4"), #64
      gd.layers.decoder_block_2D(96,  name="decode_block_5"), #64
      ]


  model_input = tf.keras.layers.Input(shape=[*IMAGE_SHAPE, 3])

  model = gd.models.unet_model(OUTPUT_CHANNELS,
                               down_stack, up_stack,
                               inputs=model_input,
                               residual_stack=residual_stack)

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True
                    ),
                metrics=['accuracy'])

  tf.keras.utils.plot_model(model, show_shapes=True, dpi=64, to_file="plot.svg")

  ring_sample_weights = gd.data.Dataset.get_sample_weights_func([4, 3])#[24, 17])

  model_history = model.fit(train_dataset.map(ring_sample_weights),
                            epochs=EPOCHS,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_steps=VALIDATION_STEPS,
                            validation_data=test_dataset,
                            callbacks=[SaverCallback()])#DisplayCallback()])


  gd.data.Visualizer.show_image_predictions(
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

  val_loss_arr = np.array(val_loss)
  val_acc_arr = np.array(model_history.history['val_accuracy'])

  val_acc_best_index = np.argmax(val_acc_arr)
  val_loss_best_index = np.argmin(val_loss_arr)

  print("Highest Accuracy Epoch:", val_acc_best_index + 1,
        "Lowest Loss Epoch:", val_loss_best_index + 1)

  print("Highest Accuracy:", val_acc_arr[val_acc_best_index],
        "Lowest Loss:", val_loss_arr[val_loss_best_index])

  # if len(sys.argv) < 2 or sys.argv[1].lower() != "nosave":
  #   checkpoint_dir = './training_checkpoints'
  #   checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  #   checkpoint = tf.train.Checkpoint(model)
  #   model.save("E{:02}_chkpt/".format(EPOCHS))


