""" DOC STRING FOR pix2pix Discriminator Module
"""
import functools
import tensorflow as tf
from .blocks import encoder_block_3D #, InstNormalize

LeakyReLUMed = functools.partial(tf.keras.layers.LeakyReLU,
                                 alpha=0.2,
                                 )

def Discriminator(): #pylint: disable=C0103
  #initializer = tf.random_normal_initializer(0., 0.02)

  generated = tf.keras.layers.Input(shape=[24,32,32,1], name="input_image")
  real      = tf.keras.layers.Input(shape=[24,32,32,1], name="target_image")

  encoder_stack = [
      encoder_block_3D(32,
                    name="encode_disc_block_1",
                    activation=LeakyReLUMed),  # (bs, 12, 16, 16, 32)
      encoder_block_3D(64,
                    name="encode_disc_block_2",
                    activation=LeakyReLUMed),  # (bs,  6,  8,  8, 64)
      encoder_block_3D(128,
                    name="encode_disc_block_3",
                    activation=LeakyReLUMed),  # (bs,  3,  4,  4, 128)
      ]

  final_conv = tf.keras.layers.Conv3D(filters=32,
                                      kernel_size=(3, 3, 3),
                                      strides=(1, 1, 1),
                                      padding="valid")# ????
  #final_norm = InstNormalize() #?????Do we normalise the final one???

  final_activation = LeakyReLUMed()

  last = tf.keras.layers.Dense(units=1)

  # Functional API run of the model

  x = tf.keras.layers.concatenate([generated, real])

  # Downsample through the model
  for encode in encoder_stack:
    x = encode(x)

  x = final_conv(x)
  x = final_activation(x)

  x = last(x)

  return tf.keras.Model(inputs=[generated, real], outputs=x)



def discriminator_loss(disc_gen_output, disc_real_output):
  """ DOC

      L_dis = E(x,y)~(Pre, Post)[(D(post)(y) − 1)² + D(post)(G(pre->post)(x))²
                               + (D(pre)(x)  − 1)² + D(pre)(G(post->pre)(y))²]

  """

  return tf.reduce_mean(
      (disc_real_output - 1) ** 2 + \
      (disc_gen_output) ** 2
      )

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss_alt(disc_generated_output, disc_real_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(
      tf.zeros_like(
          disc_generated_output
          ),
      disc_generated_output
      )

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss































