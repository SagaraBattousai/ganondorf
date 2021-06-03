""" DOC STRING """

import collections
from functools import partial
import tensorflow as tf
import tensorflow_federated as tff

def preprocess(dataset,
               input_shape: int,
               label_shape: int = 1,
               num_epochs: int = 5,
               batch_size: int = 1000,
               shuffle_buffer: int = 1000000,
               prefetch_buffer: int = 100
               ):

  def batch_format_fn(element):
    return collections.OrderedDict(
        x=tf.reshape(element["x"], [-1, input_shape]),
        y=tf.cast(tf.reshape(element["y"], [-1, label_shape]), tf.int32))

  return dataset.repeat(num_epochs) \
        .shuffle(shuffle_buffer, reshuffle_each_iteration=True) \
        .batch(batch_size) \
        .map(batch_format_fn) \
        .prefetch(prefetch_buffer)


def preprocess_client_data(client_data: tff.simulation.ClientData,
                           input_shape: int,
                           label_shape: int = 1,
                           num_epochs: int = 5,
                           batch_size: int = 1000,
                           shuffle_buffer: int = 1000000,
                           prefetch_buffer: int = 100
                           ):
  preprocess_client = partial(preprocess,
                              input_shape=input_shape,
                              label_shape=label_shape,
                              num_epochs=num_epochs,
                              batch_size=batch_size,
                              shuffle_buffer=shuffle_buffer,
                              prefetch_buffer=prefetch_buffer)

  return [preprocess_client(client_data.create_tf_dataset_for_client(client))
          for client in client_data.client_ids]
