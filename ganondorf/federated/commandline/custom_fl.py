""" Custom Federated Learning Module Doc String """
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

BATCH_SPEC = collections.OrderedDict(
    x = tf.TensorSpec(shape=[None, 23], dtype=tf.float32),
    y = tf.TensorSpec(shape=[None, 1], dtype=tf.int32)
    )

BATCH_TYPE = tff.to_type(BATCH_SPEC)

LOCAL_DATA_TYPE = tff.SequenceType(BATCH_TYPE)

MODEL_SPEC = collections.OrderedDict(
    weights = tf.TensorSpec(shape=[23, 12], dtype=tf.float32),
    bias = tf.TensorSpec(shape=[12], dtype=tf.float32)
    )
MODEL_TYPE = tff.to_type(MODEL_SPEC)

SERVER_MODEL_TYPE = tff.type_at_server(MODEL_TYPE)
CLIENT_DATA_TYPE = tff.type_at_clients(LOCAL_DATA_TYPE)

SERVER_LEARNING_RATE_TYPE = tff.type_at_server(tf.float32)
SERVER_CLASS_COUNT_TYPE = tff.type_at_server(tf.int32)

def create_empty_model(inputs: int, classes: int):
  return collections.OrderedDict(
    weights=np.zeros([inputs, classes], dtype=np.float32),
    bias=np.zeros([classes], dtype=np.float32))


@tf.function
def forward_pass(model, batch, classes):
  predicted_y = tf.nn.softmax(
    tf.matmul(batch['x'], model['weights']) + model['bias']
    )
  return -tf.reduce_mean(
    tf.reduce_sum(
        tf.one_hot(batch['y'], classes) # pylint: disable=no-value-for-parameter
      * tf.math.log(predicted_y),
      axis=[1]))


@tff.tf_computation(MODEL_TYPE, BATCH_TYPE, tf.int32)
def batch_loss(model, batch, classes):
  return forward_pass(model, batch, classes)


@tff.tf_computation(MODEL_TYPE, BATCH_TYPE, tf.float32, tf.int32)
def batch_train(initial_model, batch, learning_rate, classes):
  model_vars = collections.OrderedDict(
    [(name, tf.Variable(name=name, initial_value=value))
    for name, value in initial_model.items()])

  optimizer = tf.keras.optimizers.SGD(learning_rate)

  @tf.function
  def _train_on_batch(model_vars, batch, classes):
    with tf.GradientTape() as tape:
      loss = forward_pass(model_vars, batch, classes)
    grads = tape.gradient(loss, model_vars)
    optimizer.apply_gradients(
      zip(tf.nest.flatten(grads), tf.nest.flatten(model_vars)))
    return model_vars

  return _train_on_batch(model_vars, batch, classes)

@tff.federated_computation(MODEL_TYPE, tf.float32, LOCAL_DATA_TYPE, tf.int32)
def local_train(initial_model, learning_rate, all_batches, classes):

  @tff.federated_computation(MODEL_TYPE, BATCH_TYPE)
  def batch_fn(model, batch):
    return batch_train(model, batch, learning_rate, classes)

  return tff.sequence_reduce(all_batches, initial_model, batch_fn)


@tff.federated_computation(MODEL_TYPE, LOCAL_DATA_TYPE, tf.int32)
def local_eval(model, all_batches, classes):
  return tff.sequence_sum(
    tff.sequence_map(
      tff.federated_computation(lambda b: batch_loss(model, b, classes),
                                    BATCH_TYPE), all_batches))

@tff.federated_computation(SERVER_MODEL_TYPE, CLIENT_DATA_TYPE,
                           SERVER_CLASS_COUNT_TYPE)
def federated_eval(model, data, classes):
  return tff.federated_mean(
    tff.federated_map(local_eval,
                      [tff.federated_broadcast(model), 
                       data, 
                       tff.federated_broadcast(classes)]))

@tff.federated_computation(SERVER_MODEL_TYPE, SERVER_LEARNING_RATE_TYPE,
                           CLIENT_DATA_TYPE, SERVER_CLASS_COUNT_TYPE)
def federated_train(model, learning_rate, data, classes):
  return tff.federated_mean(
    tff.federated_map(local_train,
                      [tff.federated_broadcast(model),
                      tff.federated_broadcast(learning_rate),
                      data,
                      tff.federated_broadcast(classes)]))

if __name__ == "__main__":
  import fbl.data as fbld
  import fbl.pipeline as fblp

  train, test = fbld.load_mhealth_dataset(normalised=True)
  input_shape = 23
  all_train = fblp.preprocess_client_data(train, input_shape)

  # logdir = os.path.dirname(os.path.abspath(__file__)) + "/logs/"
  # summary_writer = tf.summary.create_file_writer(logdir)
  
  # state = iterative_process.initialize()

  # NUM_ROUNDS = 20

  # with summary_writer.as_default():
  #   for round_num in range(NUM_ROUNDS):
  #     state, metrics = iterative_process.next(state, all_train)
  #     for name, value in metrics['train'].items():
  #       tf.summary.scalar(name, value, step=round_num)

  # for round_num in range(NUM_ROUNDS):
  #   state, metrics = iterative_process.next(state, all_train)
  #   print("Round {:2d}, metrics={}".format(round_num, metrics))














