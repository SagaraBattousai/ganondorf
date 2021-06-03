""" Module for preprossessing raw datasets such as normalisation """
import numpy as np
from typing import Dict, TypeVar, Mapping

T = TypeVar("T")

#Bit concerend if ,in and max come from labels
def normalise_data(data: np.array, label_index: int = None) -> np.array:
  min_data_array = np.amin(data, axis=0)
  max_data_array = np.amax(data, axis=0)

  if label_index is not None:
    min_data_array[label_index]= 0
    max_data_array[label_index]= 1

  return (data - min_data_array) / (max_data_array - min_data_array)


def normalise_labeled_data(data: np.array) -> np.array:
  return normalise_data(data, label_index = -1)

def normalise_federated_dataset(dataset: Mapping[T, np.array],
                                label_index: int = None) -> Dict[T, np.array]:

  return {key: normalise_data(dataset[key], label_index)
          for key in dataset}

def shift_labels(data: np.array,
                 label_index: int = -1,
                 shift_amount: int = 1) -> np.array:
  shifted_data = data.copy()
  shifted_data[:,label_index] -= shift_amount
  return shifted_data

def shift_federated_labels(dataset: Mapping[T, np.array],
                           label_index: int = -1,
                           shift_amount: int = 1) -> Dict[T, np.array]:

  return {key: shift_labels(dataset[key], label_index, shift_amount)
          for key in dataset}

