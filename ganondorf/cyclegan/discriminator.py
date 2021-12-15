""" DOC STRING FOR Cycle Discriminator Module
"""
import functools
import tensorflow as tf
from ganondorf.layers import encoder_block_2D, InstNormalize

LeakyReLUMed = functools.partial(tf.keras.layers.LeakyReLU,
                                 alpha=0.2,
                                 )
LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def Discriminator(): #pylint: disable=C0103
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[128,128,3], name="input_image")

  encoder_stack = [
      encoder_block_2D(32,
                    name="encode_disc_block_1",
                    activation=LeakyReLUMed),  # (bs, 64, 64, 32)
      encoder_block_2D(64,
                    name="encode_disc_block_2",
                    activation=LeakyReLUMed),  # (bs, 32, 32, 64)
      encoder_block_2D(128,
                    name="encode_disc_block_3",
                    activation=LeakyReLUMed),  # (bs, 16, 16, 128)
      ]

  #-----------------------

  final_conv = tf.keras.layers.Conv2D(filters=256, #32,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding="valid")# ????

  final_norm = InstNormalize() #?????Do we normalise the final one?? u2u

  final_activation = LeakyReLUMed()

  last = tf.keras.layers.Dense(units=1) # Could change to conv

  # Functional API run of the model

  x = inp #//tf.keras.layers.concatenate([generated, real])

  # Downsample through the model
  for encode in encoder_stack:
    x = encode(x)

  x = final_conv(x)

  #Potentially remove
  #x = final_norm(x)

  x = final_activation(x)

  x = last(x)

  return tf.keras.Model(inputs=inp, outputs=x)



def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)
  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5



def discriminator_loss_med(disc_gen_output, disc_real_output):
  """ DOC

      L_dis = E(x,y)~(Pre, Post)[(D(post)(y) − 1)² + D(post)(G(pre->post)(x))²
                               + (D(pre)(x)  − 1)² + D(pre)(G(post->pre)(y))²]

  """

  return tf.reduce_mean(
      (disc_real_output - 1) ** 2 + \
      (disc_gen_output) ** 2
      )

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss_med_alt(disc_generated_output, disc_real_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(
      tf.zeros_like(
          disc_generated_output
          ),
      disc_generated_output
      )

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss































