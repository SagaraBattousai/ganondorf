""" Module for Formating Medical images to use with the machine learning module

"""
import SimpleITK as sitk
import numpy as np
import os
from typing import Sequence
import tensorflow as tf
import ganon.data
#VV This way it apears as if that function origionated here, needed untill all
# data code is transformed 
from ganon.data import window_level

__all__ = ['window_level', 'medical_as_array', 'split_into_patches',
           'sew_patches', 'resize_nii', 'convert_dir_images_to_nii',
           'resize_medical_image', 'square_medical_pad',
           'square_medical_pad_and_save', 'save_image_array',
           'load_image_array']

def medical_as_array(image: sitk.Image) -> np.array:
  return sitk.GetArrayFromImage(image).astype(np.float32)[..., np.newaxis]

def split_into_patches(image: np.array,
                       patch_size: tuple[int, int, int] = (24,32,32)
                       ) -> list[np.array]:

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

def sew_patches(patches: list[np.array],
                image_size: tuple[int, int, int] = (24,256,256)
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

def resize_nii(image_filename: str,
               size: Sequence[int],
               outname: str = "out.nii",
               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) -> None:
  arr = sitk.GetArrayFromImage(sitk.ReadImage(image_filename))
  dtype = arr.dtype

  if arr.ndim == 3:
    arr = arr[..., np.newaxis]

  tensor = tf.convert_to_tensor(arr, dtype=dtype)
  image = tf.image.resize(tensor, size, method=method)

  sitk.WriteImage(sitk.GetImageFromArray(image.numpy()), outname)

def convert_dir_images_to_nii(outname: str = "out.nii",
                              dirname: str = ".") -> None:
  image_filenames = list(
      map(
          lambda fname: os.path.join(dirname, fname), os.listdir(dirname)
          )
      )

  arr = sitk.GetArrayFromImage(sitk.ReadImage(image_filenames[0]))
  for i in range(1, len(image_filenames)):
    arr = np.vstack(
        (arr, sitk.GetArrayFromImage(sitk.ReadImage(image_filenames[i])))
        )

    print(arr.shape)
  img = sitk.GetImageFromArray(arr)
  sitk.WriteImage(img, outname, imageIO="NiftiImageIO")

def resize_medical_image(image_name: str,
                 new_size: tuple[int, int] = (256, 256),
                 interpolator = sitk.sitkNearestNeighbor,
                 ) -> sitk.SimpleITK.Image:
  """ Resizes Images to a new shape preserving slice count

  """
  try:
    image = sitk.ReadImage(image_name)
  except RuntimeError as re:
    print(re)
    return None

  orig_size = image.GetSize()
  orig_spacing = image.GetSpacing()

  resize = (new_size[0], new_size[1], orig_size[2]) if len(orig_size) == 3 \
      else new_size

  new_spacing = [
      ((orig_size[0] - 1) * orig_spacing[0] / (new_size[0] - 1)),
      ((orig_size[1] - 1) * orig_spacing[1] / (new_size[1] - 1)),
      1.0
      ]

  out_image = sitk.Resample(image1=image,
                            size=resize,
                            interpolator=interpolator,
                            transform=sitk.Transform(),
                            outputOrigin=image.GetOrigin(),
                            outputDirection=image.GetDirection(),
                            outputSpacing=new_spacing)
  return out_image


def square_medical_pad(img: np.array) -> np.array:
  if img.ndim == 2:
    h, w = img.shape
    img = img[np.newaxis, ...]
  elif img.ndim < 5:
    h, w = img.shape[1:3]
  else:
    h, w = img.shape[-2:img.ndim]

  (left, right, top, bottom) = (0,0,0,0)

  if h > w:
    diff = h - w
    left = diff // 2
    right = diff - left
  elif h < w:
    diff = w - h
    top = diff // 2
    bottom = diff - top
  else:
    return img

  if img.ndim == 2:
    pad = ((top, bottom), (left,right))
  else:
    extra_dims = img.ndim - 3 # 2 for width and height and 1 for leading
    pad = ((top, bottom), (left,right), *([(0,0)] * extra_dims))

  out_arrs = [np.pad(arr, pad, constant_values=0) for arr in img]

  return np.stack(out_arrs, 0)

def square_medical_pad_and_save(img_name: str, save_name: str = None) -> None:

  save_name = save_name if save_name is not None else img_name
  image = fix_aspect_ratio(img_name)

  sitk.WriteImage(image, save_name)

def load_image_array(filename: str) -> np.array:
  return sitk.GetArrayFromImage(sitk.ReadImage(filename))

def save_image_array(image: np.array, filename: str = "output.nii.gz") -> None:
  sitk.WriteImage(sitk.GetImageFromArray(image), filename)
