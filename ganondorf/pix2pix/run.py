""" Runner Docstring

"""
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import ganondorf.data.datasets as gds

def show_model_output(model, input_data):

  output = model(input_data, training=False)

  plt.subplot(121)
  plt.imshow(input_data[0, 0, ...], cmap="Blues")

  plt.subplot(122)
  plt.imshow(output[0, 0, ...], cmap="Blues")

  plt.show()

def plot_model(model, savename: str = None):
  if savename is None:
    savename = f"{model.name}.png" if model.name != "" else "model_plot.png"

  tf.keras.utils.plot_model(model, show_shapes=True, dpi=64, to_file=savename)

def generate_images(model, test_input, target, epoch:int):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(12, 12))

  test_input = gds.sew_patches(test_input)
  target = gds.sew_patches(target)
  prediction = gds.sew_patches(prediction)

  display_list = [test_input[15], target[15], prediction[15]]
  title = ["Input Image", "Ground Truth", "Predicted Image"]

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])

    plt.imshow(display_list[i])  # May need to window and level
    plt.axis("off")

  plt.savefig(os.path.join(os.path.dirname(__file__),
                           "..",
                           "tensorflow_outputs",
                           "Generator_Output",
                           "{:02}_Generated_Output.jpg".format(epoch)),
              dpi=96)


@tf.function
def train_step_tf_tut(input_image,
               target,
               generator,
               generator_loss,
               generator_optimizer,
               discriminator,
               discriminator_loss,
               discriminator_optimizer,
               epoch,
               summary_writer=None):

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output],
                                          training=True)

    gen_loss, pair_loss, adversarial_loss, _ = generator_loss(target,
                                                        gen_output,
                                                        disc_generated_output)
    disc_loss = discriminator_loss(disc_generated_output,
                                   disc_real_output)

  generator_gradients = gen_tape.gradient(
      gen_loss,
      generator.trainable_variables)

  discriminator_gradients = disc_tape.gradient(
      disc_loss,
      discriminator.trainable_variables)


  generator_optimizer.apply_gradients(
      zip(generator_gradients,
          generator.trainable_variables)
      )

  discriminator_optimizer.apply_gradients(
      zip(discriminator_gradients,
          discriminator.trainable_variables)
      )


  if summary_writer is not None:
    with summary_writer.as_default():
      tf.summary.scalar("gen_total_loss", gen_loss, step=epoch)
      tf.summary.scalar("pair_loss", pair_loss, step=epoch)
      tf.summary.scalar("adversarial_loss", adversarial_loss, step=epoch)
      tf.summary.scalar("disc_loss", disc_loss, step=epoch)





#We adopted Adam [38] as the optimizer to train the model
#with a learning rate of 10−4. Other hyper-parameters were
#empirically set as K = 3, α = 1, λ1 = 0.5, λ2 = 0.05,
#λ3 = 0.1, λ4 = 0.5, and λ5 = 0.5.




@tf.function
def train_step(input_image,
               target,
               generator,
               generator_loss,
               generator_optimizer,
               discriminator,
               discriminator_loss,
               discriminator_optimizer,
               epoch,
               summary_writer=None):

  # Train Discriminator with fixed Generator
  with tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output],
                                          training=True)

    disc_loss = discriminator_loss(disc_generated_output,
                                   disc_real_output)

  discriminator_gradients = disc_tape.gradient(
      disc_loss,
      discriminator.trainable_variables)


  discriminator_optimizer.apply_gradients(
      zip(discriminator_gradients,
          discriminator.trainable_variables)
      )

  with tf.GradientTape() as gen_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output],
                                          training=True)

    gen_loss, pair_loss, adversarial_loss, _ = generator_loss(target,
                                                        gen_output,
                                                        disc_generated_output)
  generator_gradients = gen_tape.gradient(
      gen_loss,
      generator.trainable_variables)

  generator_optimizer.apply_gradients(
      zip(generator_gradients,
          generator.trainable_variables)
      )

  if summary_writer is not None:
    with summary_writer.as_default():
      tf.summary.scalar("gen_total_loss", gen_loss, step=epoch)
      tf.summary.scalar("pair_loss", pair_loss, step=epoch)
      tf.summary.scalar("adversarial_loss", adversarial_loss, step=epoch)
      tf.summary.scalar("disc_loss", disc_loss, step=epoch)





def fit(generator,
        generator_loss,
        generator_optimizer,
        discriminator,
        discriminator_loss,
        discriminator_optimizer,
        train_dataset,
        test_dataset,
        epochs,
        checkpoint,
        batch_size:int = 16, #32 is too large and 8 is a close second
        summary_writer=None,
        checkpoint_prefix="ckpt"):

  for epoch in range(epochs):
    start = time.time()

    for example_input, example_target in test_dataset.take(1):
      generate_images(generator, example_input, example_target, epoch)
    print("Epoch: ", epoch)

    # Train
    for n, (input_image, target) in train_dataset.enumerate():
      print(".", end="")
      if (n + 1) % 100 == 0:
        print()

      for batch_start in range(0, len(input_image), batch_size):
        input_image_patch = input_image[batch_start : batch_start + batch_size]
        target_patch = target[batch_start : batch_start + batch_size]

        train_step(input_image_patch,
                   target_patch,
                   generator,
                   generator_loss,
                   generator_optimizer,
                   discriminator,
                   discriminator_loss,
                   discriminator_optimizer,
                   epoch,
                   summary_writer=summary_writer)
    print()

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

    print(f"Time taken for epoch {epoch + 1} is {time.time() - start} sec\n")

  checkpoint.save(file_prefix=checkpoint_prefix)








