""" Dataset class module docstring

"""
from typing import ClassVar, Callable, Union, TypeVar, Mapping, Sequence

import numpy as np
import tensorflow as tf
import ganondorf.core.datacore as datacore
import ganondorf.data.dataset_loader as dl

T = TypeVar("T")
LoadedDataset = Union[tf.data.Dataset, tuple]
DatasetLoaderFunc = Callable[..., LoadedDataset]

def placeholder(*args, **kwargs):
  return (0,)

class Dataset(datacore.Dataset):

  _DATASET_LOADER_MAP: ClassVar[dict[str, DatasetLoaderFunc]] = \
      {# "arcm"
       "arcm": dl.load_arcm,
       # "har"
       "har": dl.load_har,
       # "mHealth"
       "mhealth": dl.load_mhealth,
       # "BrainTumorProgression"
       "braintumorprogression": dl.load_brain_tumor_progression,
       # "BrainLesionsGlial"
       "brainlesionsglial": placeholder,
       # "ALIntraoperative"
       "alintraoperative": placeholder,
       # "ALSegmentation"
       "alsegmentation": dl.load_AL_segmentation,
       }

  @staticmethod
  def load(dataset_name:str, *args, **kwargs)->LoadedDataset: #normalize=False
    return Dataset._DATASET_LOADER_MAP[dataset_name.lower()](*args, **kwargs)

  @staticmethod
  def normalise_data(data:np.array, label_index:int=None) -> np.array:
    """ ```data``` must have dimension rows be dataset and cols be datapoints 
    """
    min_data_array = np.amin(data, axis=0)
    max_data_array = np.amax(data, axis=0)

    if label_index is not None:
      min_data_array[label_index]= 0
      max_data_array[label_index]= 1

    return (data - min_data_array) / (max_data_array - min_data_array)


  @staticmethod
  def normalise_labeled_data(data:np.array) -> np.array:
    return normalise_data(data, label_index=-1)

  @staticmethod
  def normalise_federated_dataset(dataset:Mapping[T, np.array],
                                  label_index: int = None) -> dict[T, np.array]:

    return {key: normalise_data(dataset[key], label_index)
            for key in dataset}

  @staticmethod
  def shift_labels(data:np.array, 
                   label_index:int=-1, shift_amount:int=1
                   ) -> np.array:
    sub = np.zeros(data.shape[1])
    sub[label_index] = shift_amount
    return data - sub

  @staticmethod
  def shift_federated_labels(dataset:Mapping[T, np.array],
                             label_index:int=-1, shift_amount:int=1
                             )-> dict[T, np.array]:
    return {key: shift_labels(dataset[key], label_index, shift_amount)
            for key in dataset}


  @staticmethod
  def get_sample_weights_func(weights:Sequence[float]):
    def sample_weights_func(inputs, targets):
      class_weights = tf.constant(weights)
      class_weights = class_weights / tf.reduce_sum(class_weights)
  
      sample_weights = tf.gather(class_weights, 
                                 indices=tf.cast(targets, tf.int32))
  
      return inputs, targets, sample_weights
  
    return tf.function(sample_weights_func)    


