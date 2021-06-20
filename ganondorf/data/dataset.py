""" Dataset class module docstring

"""
# Do I need?
#----------------------------------------
#from .loader import Loader, ArrayLoader
#from .federated import FederatedLoader, NpzLoader
#from .builtin_numeric_dataset import load_numeric
#from .medical import NiiLoader, load_medical, \
#    BrainTumorProgressionLoader, split_into_patches, sew_patches
#----------------------------------------

from typing import ClassVar, Callable, Union

import attr
import tensorflow as tf
import ganondorf.core.datacore as datacore
import ganondorf.data.dataset_loader as dl

LoadedDataset = Union[tf.data.Dataset, tuple]
DatasetLoaderFunc = Callable[..., LoadedDataset]

def placeholder() -> tuple[int]:
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
    return Dataset._DATASET_LOADER_MAP[dataset_name](*args, **kwargs)

