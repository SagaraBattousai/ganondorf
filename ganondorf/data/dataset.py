""" Dataset class module docstring

"""
from typing import ClassVar, Callable, Union, TypeVar, Mapping, Sequence

import numpy as np
import tensorflow as tf
#import ganondorf.core.datacore as datacore
import ganondorf.data.dataset_loader as dl

T = TypeVar("T")
LoadedDataset = Union[tf.data.Dataset, tuple]
DatasetLoaderFunc = Callable[..., LoadedDataset]

def placeholder(*args, **kwargs):
  return (0,)

class Dataset(): #datacore.Dataset):

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
       # "ALRing"
       "alring": dl.load_AL_ring,
       # "ALRingWithDays345"
       "alringwithdays345": dl.load_AL_ring_345,
       # "ALGeneration"
       "algeneration": dl.load_AL_generation,                 
       # "CycleModel"
       "cyclemodel": dl.load_cycle_model,
       }

  @staticmethod
  def load(dataset_name:str, *args, **kwargs)->LoadedDataset: #normalize=False
    return Dataset._DATASET_LOADER_MAP[dataset_name.lower()](*args, **kwargs)


  @staticmethod
  def get_sample_weights_func(weights:Sequence[float]):
    def sample_weights_func(inputs, targets):
      class_weights = tf.constant(weights)
      class_weights = class_weights / tf.reduce_sum(class_weights)
  
      sample_weights = tf.gather(class_weights, 
                                 indices=tf.cast(targets, tf.int32))
  
      return inputs, targets, sample_weights
  
    return tf.function(sample_weights_func)    


