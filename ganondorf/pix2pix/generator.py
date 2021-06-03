""" DOC STRING FOR pix2pix Generator Module
"""
import tensorflow as tf
from . import residual
from .blocks import encoder_block, decoder_block, ConvMed 

def Generator():
  OUTPUT_CHANNELS = 1

  inputs = tf.keras.layers.Input(shape=[24,32,32,1])

  encoder_stack = [
      encoder_block(32,
                    name="encode_block_1",
                    activation=tf.keras.layers.ReLU),  # (bs, 12, 16, 16, 32)
      encoder_block(64,
                    name="encode_block_2",
                    activation=tf.keras.layers.ReLU),  # (bs,  6,  8,  8, 64)
      encoder_block(128,
                    name="encode_block_3",
                    activation=tf.keras.layers.ReLU),  # (bs,  3,  4,  4, 128)
      ]

  residual_stack = [
      residual.ResidualBottleneckLayer.as_residual_bridge(128),
      residual.ResidualBottleneckLayer.as_residual_bridge(128),
      residual.ResidualBottleneckLayer.as_residual_bridge(128),
      ]

  after_residual_activation = tf.keras.layers.ReLU()

  decoder_stack = [
      decoder_block(128, name="decode_block_1"),  # (bs,  6,  8,  8, 128)
      decoder_block(64,  name="decode_block_2"),  # (bs, 12, 16, 16, 64)
      decoder_block(32,  name="decode_block_3"),  # (bs, 24, 32, 32, 32)
      ]


  initializer = tf.random_normal_initializer(0., 0.02)
  last = ConvMed(filters=OUTPUT_CHANNELS,
                 strides=1,
                 kernel_initializer=initializer,
                 activation='tanh')  # (bs, 24, 32, 32, 1)

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

def generator_loss(target, gen_output, disc_gen_output):
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


def generator_loss_alt(target, gen_output, disc_generated_output):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (100 * l1_loss)

  return total_gen_loss, gan_loss, l1_loss, None












