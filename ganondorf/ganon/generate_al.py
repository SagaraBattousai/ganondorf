import sys
import functools
import os
import time
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
def load_image_test(image):
  input_image = (tf.cast(image, tf.float32) - 127.5) / 127.5
  return input_image
  
class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    #clear_output(wait=True)
    if (epoch + 1) % 10 == 0:
      gd.data.Visualizer.show_image_predictions(
          self.model,
          dataset=(example_image, example_mask),
          format_prediction=create_mask
          )
    print('\nSample Predictions after epoch {}\n'.format(epoch+1))


if __name__ == '__main__':
  EPOCHS = 100
  INPUT_CHANNELS = 3
  OUTPUT_CHANNELS = 3
  IMAGE_SHAPE = (128, 128)

  TRAIN_LENGTH = 34
  BATCH_SIZE = 8
  BUFFER_SIZE = 1000
  STEPS_PER_EPOCH = 1
  VAL_SUBSPLITS = 5
  VALIDATION_STEPS = 1

  train_dataset, test_dataset = gd.data.Dataset.load("ALGeneration",
                                                     size=IMAGE_SHAPE)
  train_dataset = train_dataset.map(load_image_test,
                                    num_parallel_calls=tf.data.AUTOTUNE)

  test_dataset = test_dataset.map(load_image_test,
                                  num_parallel_calls=tf.data.AUTOTUNE)

  train_dataset = train_dataset.cache() \
                               .batch(BATCH_SIZE)#.repeat()
                               #.shuffle(BUFFER_SIZE) \
                               #.batch(BATCH_SIZE)#.repeat()
  train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)



  # example_image, example_mask  = next(iter(test_dataset))

  # test_dataset = test_dataset.batch(BATCH_SIZE)



  # residual_stack = [
  #     gd.layers.ResidualBottleneckLayer.as_residual_bridge(2, 1280), # 320),
  #     gd.layers.ResidualBottleneckLayer.as_residual_bridge(2, 1280), # 320),
  #     gd.layers.ResidualBottleneckLayer.as_residual_bridge(2, 1280), # 320),
  #     tf.keras.layers.ReLU(),
  #     ]


  LeakyReLUMed = functools.partial(tf.keras.layers.LeakyReLU, alpha=0.2)

  down_stack = [
      gd.layers.encoder_block_2D(64,
                                 name="encode_disc_block_1",
                                 activation=LeakyReLUMed), # (bs, 64, 64, 64)
      gd.layers.encoder_block_2D(128,
                                 name="encode_disc_block_2",
                                 activation=LeakyReLUMed), # (bs,  32, 32, 128)
      gd.layers.encoder_block_2D(256,
                                 name="encode_disc_block_3",
                                 activation=LeakyReLUMed), # (bs,  16, 16, 256)
      ]

  up_stack = [
      gd.layers.decoder_block_2D(256, name="decode_block_1"),
      gd.layers.decoder_block_2D(128, name="decode_block_2"),
      gd.layers.decoder_block_2D(64, name="decode_block_3"),
      ]

  input_shape = 512
  dense_shape = (16, 16, 256)

  generator = gd.models.generator_up_model(input_shape=(input_shape,),
                                           output_channels=OUTPUT_CHANNELS,
                                           dense_shape=dense_shape,
                                           up_stack=up_stack)

  discriminator = gd.models.discriminator(down_stack)

  cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

  def discriminator_loss(real, fake):
    rl = cross_entropy(tf.ones_like(real), real)
    fl = cross_entropy(tf.zeros_like(fake), fake)
    total_loss = rl + fl
    return total_loss

  def generator_loss(fake):
    return cross_entropy(tf.ones_like(fake), fake)

  generator_optimizer = tf.keras.optimizers.Adam(1e-4)
  discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

  seed = gd.random.normal_in_range([1, input_shape], -1.0, 1.0, 4.5,
                                    clip=True, name="RandomGen")

  @tf.function
  def train_step(images):
    noise = gd.random.normal_in_range([BATCH_SIZE, input_shape], -1.0, 1.0, 4.5,
                                      clip=True, name="RandomGen")

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    grad_of_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    grad_of_disc = disc_tape.gradient(disc_loss,
                                      discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(grad_of_gen, generator.trainable_variables))

    discriminator_optimizer.apply_gradients(
        zip(grad_of_disc, discriminator.trainable_variables))


  def train(dataset, epochs):
    for epoch in range(epochs):
      start = time.time()

      for image_batch in dataset:
        train_step(image_batch)
      
      if (epoch + 1) % 15 == 0:
        generate_and_save_images(generator, epoch + 1, seed)

      print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    generate_and_save_images(generator, epoch + 1, seed)


  def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    #for i in range(predictions.shape[0]):
    pred = tf.cast(predictions[0] * 127.5 + 127.5, tf.uint8)
    inpt = tf.cast(
        next(iter(train_dataset.take(1)))[0] * 127.5 + 127.5,
        tf.uint8)

    plt.subplot(1, 2, 1)
    plt.imshow(pred)
    plt.subplot(1, 2, 2)
    plt.imshow(inpt)
    
    plt.axis('off')

    plt.show()

  train(train_dataset, EPOCHS)




  # generated_image = generator(noise, training=False)
  
  # plt.subplot(121)
  # plt.imshow(tf.reshape(noise, (32,16)))
  # plt.subplot(122)
  # plt.imshow(generated_image[0])
  # plt.show()


if False:
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True
                    ),
                metrics=['accuracy'])


  tf.keras.utils.plot_model(model, show_shapes=True, dpi=64, to_file="plot.svg")

  ring_sample_weights = gd.data.Dataset.get_sample_weights_func([3.075, 1])

  model_history = model.fit(train_dataset.map(ring_sample_weights),
                            epochs=EPOCHS,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_steps=VALIDATION_STEPS,
                            validation_data=test_dataset,
                            callbacks=[DisplayCallback()])


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

  if val_loss_best_index != val_acc_best_index:
    print("Loss and acc not at same location")
    print("Loss Accuracy:", val_acc_arr[val_loss_best_index],
          "Accuracy Loss:", val_loss_arr[val_acc_best_index])

  if len(sys.argv) < 2 or sys.argv[1].lower() != "nosave":
    # model.save("chkpt/")
    model.save("E{:02}_chkpt/".format(EPOCHS))


