""" DOC STRING """
import os
import math
import csv
import random
from multiprocessing import Pool
from functools import partial
from typing import Tuple, Dict, Sequence, Iterable, Callable, List
import numpy as np

BASE_DATASET_DIR = os.path.dirname(os.path.realpath(__file__)) + "/dataset"

strip_extension = lambda a: a[:a.rindex(".")]

def csv_line_to_typed_array(csv_line, ptype: type = float,
                            dtype: type = np.float32):
  array_fn = partial(np.array, dtype=dtype)
  return array_fn(list(map(ptype, csv_line)))

def csv_to_array_list(filenames: Iterable[str],
                       fieldnames: Sequence[str] = None,
                       dataset_directory: str = BASE_DATASET_DIR,
                       ptype: type = float, dtype: type = np.float32,
                       num_processes: int = None) -> List[np.array]:
  #array_list = []
  csv_map_fn = partial(csv_to_array,
                       fieldnames=fieldnames,
                       dataset_directory=dataset_directory, ptype=ptype,
                       dtype=dtype, num_processes=num_processes)

  return list(map(csv_map_fn, filenames))


def csv_to_array(filename: str, fieldnames: Sequence[str] = None,
                 dataset_directory: str = BASE_DATASET_DIR,
                 ptype: type = float, dtype: type = np.float32,
                 num_processes: int = None) -> np.array:
  arrays = None
  csv_map_fn = partial(csv_line_to_typed_array, ptype=ptype, dtype=dtype)
  with open("{}/{}".format(dataset_directory, filename)) as csv_file:
    csv_reader = csv.DictReader(csv_file, fieldnames=fieldnames)
    lines = (list(line.values()) for line in csv_reader)
    with Pool(num_processes) as p:
      arrays = p.map(csv_map_fn, lines)

    # for line in csv_reader:
      # arrays.append(np.array(list(map(float, line.values())), dtype=dtype))

  return np.vstack(arrays)

def split_client_data(input_data: Sequence[np.array], classes: Sequence,
                      clientnames: Sequence[str], label_index: int = -1,
                      test_ratio: float = 0.1, randomize: bool = True,
                      num_processes: int = None) -> Dict[str, np.array]:
  dataset = {}
  split_map_fn = partial(split_test_and_train, classes=classes,
                         label_index=label_index, test_ratio=test_ratio,
                         randomize=randomize)
  client_data = None
  with Pool(num_processes) as p:
    client_data = p.map(split_map_fn, input_data)

  for client_index, clientname in enumerate(clientnames):
    dataset[clientname + "_train_data"] = client_data[client_index][0]
    dataset[clientname + "_test_data"] = client_data[client_index][1]

  return dataset


def split_test_and_train(client_data: np.array, classes: Sequence,
                         label_index: int = -1, test_ratio: float = 0.1,
                         randomize: bool = True) -> Tuple[np.array, np.array]:
  train_data_assigned = False
  test_data_assigned = False

  train_data = None
  test_data = None

  for clazz in classes:
    #get all the data for a client with label == clazz
    data = np.array([data for data in client_data
                     if data[label_index] == clazz])

    instance_count = data.shape[0]
    test_indicies_count = math.ceil(instance_count * test_ratio)
    test_indicies = None

    if randomize:
      test_indicies = random.choices(range(instance_count),
                                     k = test_indicies_count)
    else:
      test_indicies = range(instance_count)[-test_indicies_count:]

    for i, arr in enumerate(data):
      if i in test_indicies:
        if not test_data_assigned:
          test_data = arr
          test_data_assigned = True
        else:
          test_data = np.vstack((test_data, arr))
      else:
        if not train_data_assigned:
          train_data = arr
          train_data_assigned = True
        else:
          train_data = np.vstack((train_data, arr))

  return (train_data, test_data)


def combine_npz_arrays(filenames: Iterable[str]) -> Sequence[np.array]:
  data_for_clients = []
  for fname in filenames:
    npz = np.load(fname)
    client_data = np.hstack([npz[f] for f in npz.files])
    data_for_clients.append(client_data)

  return data_for_clients



