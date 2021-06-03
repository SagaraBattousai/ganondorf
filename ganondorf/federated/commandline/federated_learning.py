""" Federated Learning Module Doc String """
import nest_asyncio
nest_asyncio.apply()

import collections

if __name__ == "__main__":
  import os
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

tff.backends.reference.set_reference_context()

def create_batch_spec(input_spec: int, label_spec: int = 1):
  return collections.OrderedDict(
      x = tf.TensorSpec(shape=[None, input_spec], dtype=tf.float32),
      y = tf.TensorSpec(shape=[None, label_spec], dtype=tf.int32))


def create_model(input_shape: int, classes: int):
  return tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=(input_shape,)),
      tf.keras.layers.Dense(classes, kernel_initializer='zeros'),
      tf.keras.layers.Softmax()
      ])

def model_fn(input_shape: int, classes: int, label_spec: int = 1):
  model = create_model(input_shape, classes)
  return tff.learning.from_keras_model(
      model,
      input_spec=create_batch_spec(input_shape, label_spec),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(), #MeanAbsoluteError()
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

if __name__ == "__main__":
  import fbl.data as fbld
  import fbl.pipeline as fblp

  train, test = fbld.load_har_dataset(normalised=True)
  input_shape = 561
  classes = 6
  all_train = fblp.preprocess_client_data(train, input_shape)
  import functools
  mf = functools.partial(model_fn, input_shape=input_shape, classes=classes)

  iterative_process = tff.learning.build_federated_averaging_process(
      mf, #model_fn,
      client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
      server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))


  logdir = os.path.dirname(os.path.abspath(__file__)) + "/logs/"
  summary_writer = tf.summary.create_file_writer(logdir)

  state = iterative_process.initialize()

  NUM_ROUNDS = 2

  # with summary_writer.as_default():
  #   for round_num in range(NUM_ROUNDS):
  #     state, metrics = iterative_process.next(state, all_train)
  #     for name, value in metrics['train'].items():
  #       tf.summary.scalar(name, value, step=round_num)

  for round_num in range(NUM_ROUNDS):
    state, metrics = iterative_process.next(state, all_train)
    print("Round {:2d}, metrics={}".format(round_num, metrics))

  print("Done Training {:2d} rounds".format(NUM_ROUNDS))













