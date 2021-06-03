""" DOC STRING """
import os
import random
import collections
from typing import Iterable, Callable, Tuple
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.simulation import ClientData as TffClientData

BASE_DATASET_DIR = os.path.dirname(os.path.realpath(__file__)) + "/dataset"

CLIENT_ID_DEFAULT_BASE = "client_{}"
CLIENT_TRAIN_DEFAULT_SUFFIX = "_train_data"
CLIENT_TEST_DEFAULT_SUFFIX = "_test_data"

ARCM_CLIENT_ID_RANGE = (1, 16) # 15 clients
MHEALTH_CLIENT_ID_RANGE = (1, 11) # 10 clients
HAR_CLIENT_ID_RANGE = (1, 7) # 6 clients (actually 6 classes used as clients)

def array_to_dataset(array: np.array, label_index: int = -1) -> tf.data.Dataset:

  input_array, label_array = np.hsplit(array, np.array([label_index]))

  return tf.data.Dataset.from_tensor_slices(
      collections.OrderedDict(
          x = input_array,
          y = label_array
          )
      )

def load_npz_dataset(filename: str, client_ids: Iterable[str],
                     client_train_suffix: str = "",
                     client_test_suffix: str = "", label_index: int = -1
                     ) -> Tuple[TffClientData, TffClientData]:

  npz_dataset = np.load(filename)
  train_dataset = {}
  test_dataset = {}

  for client_id in client_ids:
    train_data = npz_dataset[client_id + client_train_suffix]
    test_data = npz_dataset[client_id + client_test_suffix]

    train_dataset[client_id] = array_to_dataset(train_data, label_index)

    test_dataset[client_id] = array_to_dataset(test_data, label_index)

  train_client_data = TffClientData.from_clients_and_fn(
      client_ids,
      train_dataset.get
      )

  test_client_data = TffClientData.from_clients_and_fn(
      client_ids,
      test_dataset.get
      )

  return train_client_data, test_client_data

def load_builtin_dataset(data_filename: str,
                         relative_dir: str,
                         client_id_range: Tuple[int, int]
                         ) -> Tuple[TffClientData, TffClientData]:

  client_ids = [CLIENT_ID_DEFAULT_BASE.format(i)
                for i in range(*client_id_range)]

  return load_npz_dataset(
      filename = "{}/{}/{}".format(
          BASE_DATASET_DIR, relative_dir, data_filename
          ),

      client_ids = client_ids,
      client_train_suffix = CLIENT_TRAIN_DEFAULT_SUFFIX,
      client_test_suffix = CLIENT_TEST_DEFAULT_SUFFIX
      )

def load_arcm_dataset(normalised: bool = False
                      ) -> Tuple[TffClientData, TffClientData]:

  data_filename = "arcm_data_norm.npz" if normalised else "arcm_data.npz"

  return load_builtin_dataset(data_filename, "arcm", ARCM_CLIENT_ID_RANGE)

def load_mhealth_dataset(normalised: bool = False
                           ) -> Tuple[TffClientData, TffClientData]:

  data_filename = "mHealth_data_norm.npz" if normalised else "mHealth_data.npz"

  return load_builtin_dataset(data_filename, "mHealth", MHEALTH_CLIENT_ID_RANGE)

def load_har_dataset(normalised: bool = False
                     ) -> Tuple[TffClientData, TffClientData]:

  data_filename = "har_data_norm.npz" if normalised else "har_data.npz"

  return load_builtin_dataset(data_filename, "har", HAR_CLIENT_ID_RANGE)
