""" Module for Formating images to use with the machine learning packages

"""
import PIL
import SimpleITK as sitk
import numpy as np
import os
from typing import Sequence
import tensorflow as tf
from ganondorf.core import datacore

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
    image = datacore.window_level(image, window, level)

  return image


#def signed_hex(num: int, bits: int = 16) -> str:
#  pass

#def color_level(hound_image: np.array) -> np.array:
#  pass

def image_as_array(image: PIL.Image.Image, mode: str=None) -> np.array:
  if mode is not None:
    image = image.convert(mode=mode)
  
  arr = np.asarray(image)
  if mode == '1':
    arr = arr.astype(np.uint8)
    arr[arr > 0] = 1

  if arr.ndim == 2:
    arr = arr[..., np.newaxis]
  return arr

def medical_as_array(image: sitk.Image) -> np.array:
  return sitk.GetArrayFromImage(image).astype(np.float32)[..., np.newaxis]

def as_mask(img):
  arr = np.asarray(img)
  arr = arr.copy()
  arr[arr > 0] = 255
  img = Image.fromarray(arr)
  img = img.convert(mode='1')
  return img

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


def resize_image(image_name: str, #  Not always going to be str
                 new_size: tuple[int, int] = (256, 256),
                 interpolator = PIL.Image.NEAREST,
                 roi: tuple[float, float, float, float] = None,
                 reducing_gap:float=None
                 ) -> PIL.Image.Image:
  image = PIL.Image.open(image_name)
  return image.resize(new_size, interpolator, roi, reducing_gap)


def fix_aspect_ratio(img: np.array) -> np.array:
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


def fix_med_aspect_ratio(img_name: str) -> sitk.SimpleITK.Image:
  img = sitk.GetArrayFromImage(sitk.ReadImage(img_name))

  return sitk.GetImageFromArray(fix_aspect_ratio(img))


def resize_image_and_save(image_name: str,
                          new_size: tuple[int, int] = (256, 256),
                          interpolator = sitk.sitkNearestNeighbor,
                          save_name: str = None) -> None:

  save_name = save_name if save_name is not None else image_name

  image = resize_image(image_name, new_size, interpolator)
  
  if iamge is not None:
    sitk.WriteImage(image, save_name)

def fix_aspect_ratio_and_save(img_name: str, save_name: str = None) -> None:

  save_name = save_name if save_name is not None else img_name
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

def square_images(filenames: Sequence[str],
                 out_filenames: Sequence[str] = None):
  for i, fname in enumerate(filenames):
    try:
      arr = np.asarray(PIL.Image.open(fname))
    except PIL.UnidentifiedImageError as uie:
      print(uie)
      print("Skipping non image file")
      continue

    img = PIL.Image.fromarray(square_pad(arr))
    outname = fname if out_filenames is None else out_filenames[i]

    img.save(outname)


