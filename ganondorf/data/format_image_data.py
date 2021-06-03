""" Module for Formating images to use with the machine learning packages

"""
import PIL
import SimpleITK as sitk
import numpy as np
import os
from typing import Sequence, Tuple
import tensorflow as tf
import SimpleITK as sitk
from ganondorf.core import datamodule as data

def window_level(hound_image: np.array, window: int, level:int) -> np.array:

  image = hound_image.copy()  # ?????????
  
  if image.ndim == 3:
    half_window = window // 2
    window_min = level - half_window
    window_max = level + half_window


    image[image < window_min] = window_min
    image[image > window_max] = window_max

    image = image - window_min

    batch_count = image.shape[0]

    max_vector = \
        np.amax(np.amax(image, axis=1), axis=1) \
        .reshape(batch_count, -1, 1)

    max_matrix = np.full_like(image, max_vector)

    max_matrix[max_matrix == 0] = -1

    image = image * (255 / max_matrix)

  else:
    image = data.window_level(image, window, level)

  return image

def signed_hex(num: int, bits: int = 16) -> str:
  pass

def color_level(hound_image: np.array) -> np.array:
  pass


def resize_nii(image_filename: str,
               size: Sequence[int],
               outname: str = "out.nii",
               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) -> None:
  arr = sitk.GetArrayFromImage(sitk.ReadImage(image_filename))
  dtype = arr.dtype

  if arr.ndim == 3:
    shape = arr.shape
    arr = arr.reshape(shape[0], shape[1], shape[2], 1)

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

def resize_image(image_name: str,
                 new_size: Tuple[int, int] = (256, 256),
                 interpolator = sitk.sitkNearestNeighbor,
                 ) -> sitk.SimpleITK.Image:
  """ Resizes Images to a new shape preserving slice count

  """
  image = sitk.ReadImage(image_name)

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

def fix_aspect_ratio(img: np.array) -> np.array:
  arrs = [arr for arr in img]
  h, w = arrs[0].shape
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

  out_arrs = [np.pad(arr, ((top, bottom), (left,right)), constant_values=0) \
              for arr in arrs]

  return np.stack(out_arrs, 0)


def fix_med_aspect_ratio(img_name: str) -> sitk.SimpleITK.Image:
  img = sitk.GetArrayFromImage(sitk.ReadImage(img_name))

  return sitk.GetImageFromArray(fix_aspect_ratio(img))


def resize_image_and_save(image_name: str,
                          new_size: Tuple[int, int] = (256, 256),
                          interpolator = sitk.sitkNearestNeighbor,
                          save_name: str = None) -> None:

  save_name = save_name if save_name != None else image_name

  image = resize_image(image_name, new_size, interpolator)

  sitk.WriteImage(image, save_name)

def fix_aspect_ratio_and_save(img_name: str, save_name: str = None) -> None:

  save_name = save_name if save_name != None else img_name
  image = fix_aspect_ratio(img_name)

  sitk.WriteImage(image, save_name)


def load_image_array(filename: str) -> np.array:
  return sitk.GetArrayFromImage(sitk.ReadImage(filename))

def save_image_array(image: np.array, filename: str = "output.nii.gz") -> None:
  sitk.WriteImage(sitk.GetImageFromArray(image), filename)

def square_pad(image: np.array)-> np.array:
  (l, r, t, b) = (0,0,0,0)
  h, w, _ = image.shape
  if h > w:
    diff = h - w
    l = diff // 2
    r = diff - l
  elif h < w:
    diff = w - h
    t = diff // 2
    b = diff - t
  else:
    return image
  return np.pad(image, ((t, b), (l, r), (0, 0)), constant_values=0)

def square_image(filenames: Sequence[str],
                 out_filenames: Sequence[str] = None):
  for i, fname in enumerate(filenames):
    arr = np.asarray(PIL.Image.open(fname))
    img = PIL.Image.fromarray(square_pad(arr))
    outname = fname if out_filenames == None else out_filenames[i]

    img.save(outname)


