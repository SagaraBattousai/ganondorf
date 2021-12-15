""" DOC STRING FOR cycleGAN Generator Module
"""
import tensorflow as tf
from ganondorf.layers import ResidualBottleneckLayer
from ganondorf.layers import encoder_block_2D, decoder_block_2D, Feature2D

LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def Generator(): # pylint: disable=C0103
  OUTPUT_CHANNELS = 3 # pylint: disable=C0103

  inputs = tf.keras.layers.Input(shape=[128, 128, 3])

  encoder_stack = [
      encoder_block_2D(32,
                    name="encode_block_1",
                    activation=tf.keras.layers.ReLU),  #(bs, 64, 64, 32)
      encoder_block_2D(64,
                    name="encode_block_2",
                    activation=tf.keras.layers.ReLU),  #(bs, 32, 32, 64)
      encoder_block_2D(128,
                    name="encode_block_3",
                    activation=tf.keras.layers.ReLU),  #(bs, 16, 16, 128)
      ]

  residual_stack = [
      ResidualBottleneckLayer.as_residual_bridge(2, 128),
      ResidualBottleneckLayer.as_residual_bridge(2, 128),
      ResidualBottleneckLayer.as_residual_bridge(2, 128),
      ]

  after_residual_activation = tf.keras.layers.ReLU()

  decoder_stack = [
      decoder_block_2D(128, name="decode_block_1"),  #(bs, 32, 32, 128)
      decoder_block_2D(64,  name="decode_block_2"),  #(bs, 64, 64, 64)
      decoder_block_2D(32,  name="decode_block_3"),  # (bs, 128, 128, 32)
      ]


  initializer = tf.random_normal_initializer(0., 0.02)
  last = Feature2D(filters=OUTPUT_CHANNELS,
                 strides=1,
                 kernel_initializer=initializer,
                 activation="tanh")  # (bs, 128, 128, 3)

  x = inputs

  # Downsample through the model
  skips = []
  for encode in encoder_stack:
    x = encode(x)
    skips.append(x)

  skips.reverse()

  # Residual blocks through the model
  for res_block in residual_stack:
    x = res_block(x)

  x = after_residual_activation(x)

  # Upsample and establish skips
  for decode, skip in zip(decoder_stack, skips):
    x = tf.keras.layers.Concatenate()([x, skip])  # ????? or above?
    x = decode(x)

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)


def generator_loss_med(target, gen_output, disc_gen_output):
  """ DOC

      L_cycle = Not Implemented
        E(x,y)~(Pre_pair, Post_pair)(||G(pre -> post)[G(post -> pre)(x)] - x||₁)

      pair_loss = E(x,y)~(Pre_pair, Post_pair)(||G(pre -> post)(x) − y||₁

      L_adv = E(x,y)~(Pre, Post)[(D(post)(G(pre -> post)(x)) − 1)²]
  """

  pair_loss = tf.reduce_mean(tf.abs(gen_output - target))

  adversarial_loss = tf.reduce_mean((disc_gen_output - 1) ** 2)

  # approx of lambda2 from paper redistributed to 3%
  total_loss = pair_loss + 0.03 * adversarial_loss

  return total_loss, pair_loss, adversarial_loss, None


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss_med_alt(target, gen_output, disc_generated_output):
  gan_loss = loss_object(
      tf.ones_like(
          disc_generated_output
          ),
      disc_generated_output
      )

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (100 * l1_loss)

  return total_gen_loss, gan_loss, l1_loss, None












