""" Module for loading datasets revolving around medical images especially
those saved in a medical format such as .nii (NIfTI)

"""
import os
import math
import attr
from typing import Sequence, Union, List, Tuple, ClassVar, Dict, TypeVar
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from .loader import Loader

T = TypeVar("T")
U = TypeVar("U")

@attr.s(auto_attribs=True, slots=True, frozen=True)
class NiiLoader(Loader):
  """
  Implementation of Loader that loads Medical Images in nii or nii.gz format

  """
  filenames: Union[str, Sequence[str]] = None

  def load_dataset(self) -> tf.data.Dataset:
    arr = self.load_images()

    #tensor = tf.convert_to_tensor(arr, dtype=arr.dtype)
    return tf.data.Dataset.from_tensor_slices(arr)


  def load_images(self) -> List[np.array]:
    if isinstance(self.filenames, str):
      return [NiiLoader._load_array(self.filenames)]
    else:
      return NiiLoader._load_multiple_files(self.filenames)


  def load_stacked_images(self) -> np.array:
    if isinstance(self.filenames, str):
      return NiiLoader._load_array(self.filenames)
    else:
      return NiiLoader._stack_multiple_files(self.filenames)


  @staticmethod
  def _load_multiple_files(fnames: Sequence[str]) -> List[np.array]:
    return [NiiLoader._load_array(fname) \
            for fname in fnames]


  @staticmethod
  def _stack_multiple_files(fnames) -> np.array:
    arr = NiiLoader._load_array(fnames[0])
    for i in range(1, len(fnames)):
      arr = np.vstack((arr, NiiLoader._load_array(fnames[i])))
    return arr

  @staticmethod
  def _load_array(filename: str) -> np.array:
    # return sitk.GetArrayFromImage(sitk.ReadImage(filename))
    return sitk.GetArrayFromImage(
        sitk.ReadImage(filename)
        ).astype(np.float32)[..., np.newaxis]



def load_medical(dataset_name: str,
                 image_names: Union[str, Sequence[str]]
                 ) -> NiiLoader:
  """ Loads a builtin numeric (array like) dataset

  Parameters
  ----------
  dataset_name : str
    The name of the dataset

  image_names : Union[str, Sequence[str]]
    The name of the medical image(s) including subdirectory names.
    This will be the path to the image from the directory holding the dataset.

  Returns
  -------
  NiiLoader holding image data

  """
  dir_name = f"{os.path.dirname(__file__)}/datasets/{dataset_name}"

  if isinstance(image_names, str):
    full_path = os.path.join(dir_name, image_names)
    return NiiLoader(full_path)

  else:
    paths = [os.path.join(dir_name, img_name) for img_name in image_names]
    return NiiLoader(paths)

@attr.s(auto_attribs=True, slots=True)
class BrainTumorProgressionLoader(Loader):
  """ Dataset Loader for builtin BrainProgressionTumor Dataset

  """

  DATASET_NAME: ClassVar[str] = "BrainTumorProgression"
  IMAGE_TYPES: ClassVar[int] = 4 # Pre and Post Scans and Pre and Post Masks
  CLIENT_COUNT: ClassVar[int] = 11

  CLIENT_NAMES: ClassVar[str] = sorted(
      ["{:02}".format(i) \
       for i in range(1, CLIENT_COUNT + 1)] * IMAGE_TYPES
      )

  IMAGE_NAMES: ClassVar[str] = ["Pre_Scan_T1.nii.gz",
                                "Pre_Mask.nii.gz",
                                "Post_Scan_T1.nii.gz",
                                "Post_Mask.nii.gz"] * CLIENT_COUNT

  image_paths = [os.path.join(val[0], val[1]) \
                 for val in zip(CLIENT_NAMES, IMAGE_NAMES)]

  _pre_scans:  np.array = None

  _pre_masks:  np.array = None

  _post_scans: np.array = None

  _post_masks: np.array = None

  # would _ store original value of None? will giving property name work?
  dataset_dictionary: Dict[str, np.array] = {"pre_scans"  : _pre_scans,
                                             "pre_masks"  : _pre_masks,
                                             "post_scans" : _post_scans,
                                             "post_masks" : _post_masks
                                             }

  @property
  def pre_scans(self) -> np.array:
    if self._pre_scans is None:
      self._pre_scans = load_medical(self.DATASET_NAME,
                                     self.image_paths[0::4]).load_images()
    return self._pre_scans

  @property
  def pre_masks(self) -> np.array:
    if self._pre_masks is None:
      self._pre_masks = load_medical(self.DATASET_NAME,
                               self.image_paths[1::4]).load_images()
    return self._pre_masks

  @property
  def post_scans(self) -> np.array:
    if self._post_scans is None:
      self._post_scans = load_medical(self.DATASET_NAME,
                                      self.image_paths[2::4]).load_images()
    return self._post_scans

  @property
  def post_masks(self) -> np.array:
    if self._post_masks is None:
      self._post_masks = load_medical(self.DATASET_NAME,
                                      self.image_paths[3::4]).load_images()
    return self._post_masks


  def load_dataset(self) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices(self.dataset_dictionary)

  # @staticmethod
  # def _pairs(index: int,
  #            fst: Sequence[T],
  #            snd: Sequence[U]) -> Tuple[T, U]:
  #   return (fst[index], snd[index])

  # def progression_pairs(self, scan_number: int) -> Tuple[np.array, np.array]:
  #   return BrainTumorProgressionLoader._pairs(
  #       scan_number, self.pre_scans, self.post_scans)

  # def pre_segmentation_pairs(self,
  #                            segmentation_number: int
  #                            ) -> Tuple[np.array, np.array]:
  #   return BrainTumorProgressionLoader._pairs(
  #       scan_number, self.pre_scans, self.pre_masks)

  # def post_segmentation_pairs(self,
  #                             segmentation_number: int
  #                             ) -> Tuple[np.array, np.array]:
  #   return BrainTumorProgressionLoader._pairs(
  #       scan_number, self.post_scans, self.post_masks)


  # def progression_dataset(self,
  #                         scan_range: Tuple[int, int]
  #                         ) -> tf.data.Dataset:
  #   dataset = tf.data.Dataset.from_tensor_slices(list(range(*scan_range)))
  #   dataset = dataset.map(self.progression_pairs)

  #   return dataset

  # def pre_segmentation_dataset(self,
  #                              segmentation_range: Tuple[int, int]
  #                              ) -> tf.data.Dataset:
  #   dataset = tf.data.Dataset.from_tensor_slices(
  #       list(range(*segmentation_range))
  #       )
  #   dataset = dataset.map(self.pre_segmentation_pairs)

  #   return dataset

  # def post_segmentation_dataset(self,
  #                               segmentation_range: Tuple[int, int]
  #                               ) -> tf.data.Dataset:
  #   dataset = tf.data.Dataset.from_tensor_slices(
  #       list(range(*segmentation_range))
  #       )
  #   dataset = dataset.map(self.post_segmentation_pairs)

  #   return dataset

def split_into_patches(image: np.array,
                       patch_size: Tuple[int, int, int] = (24,32,32)
                       ) -> List[np.array]:

  slices, height, width = image.shape[:3]
  (slice_patch, height_patch, width_patch) = patch_size

  slice_range  = math.ceil(slices / slice_patch)
  height_range = math.ceil(height / height_patch)
  width_range  = math.ceil(width  / width_patch)

  patches = []

  for i in range(slice_range):
    slice_start = i * slice_patch
    slice_end   = (i + 1) * slice_patch
    for j in range(height_range):
      height_start = j * height_patch
      height_end   = (j + 1) * height_patch
      for k in range(width_range):
        width_start = k * width_patch
        width_end   = (k + 1) * width_patch

        patches.append(image[slice_start  : slice_end,\
                             height_start : height_end,\
                             width_start  : width_end])

  return patches


def sew_patches(patches: List[np.array],
                image_size: Tuple[int, int, int] = (24,256,256)
                ) -> np.array:

  height, width = image_size[1:]
  (height_patch, width_patch) = patches[0].shape[1:3]

  height_range = math.ceil(height / height_patch)
  width_range  = math.ceil(width  / width_patch)

  width_counts = len(patches) // width_range

  width_patches = \
      [np.concatenate(patches[i * width_range : width_range * (i + 1)], axis=2)\
       for i in range(width_counts)]

  height_counts = len(width_patches) // height_range

  height_patches = \
      [np.concatenate(
          width_patches[i * height_range : (i+1) * height_range],
          axis=1
          ) for i in range(height_counts)]

  return np.concatenate(height_patches, axis=0)





