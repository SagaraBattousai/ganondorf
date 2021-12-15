""" Module for Formating images to use with the machine learning module

"""
from collections.abc import Sequence, Callable
from typing import Union
import pathlib
import PIL
import numpy as np
import os
import tensorflow as tf
from ganondorf.core import datacore
import pythonal

__all__ = ['image_as_array', 'as_mask', 'resize_image',
           'normalize', 'square_images', 'resize_images',
           'square_pad', 'rename_in_sequence', 'images_as_mask',
           'refill_blackout', 'refill_blackout_images']

def image_as_array(image: PIL.Image.Image, mode: str=None) -> np.ndarray:
  if mode is not None:
    image = image.convert(mode=mode)
  
  arr = np.asarray(image)
  if mode == '1':
    arr = arr.astype(np.uint8)
    arr[arr > 0] = 1

  if arr.ndim == 2:
    arr = arr[..., np.newaxis]
  return arr

def as_mask(img: PIL.Image.Image)->PIL.Image.Image:
  arr = np.asarray(img)
  #without copy get read-only error
  arr = arr.copy() 
  arr[arr > 0] = 255
  img = PIL.Image.fromarray(arr)
  img = img.convert(mode='1')
  return img

def resize_image(image: Union[str, np.ndarray, pathlib.Path, PIL.Image.Image],
                 new_size: tuple[int, int] = (512, 512),
                 interpolator = PIL.Image.BOX,
                 roi: tuple[float, float, float, float] = None,
                 reducing_gap:float=None
                 ) -> PIL.Image.Image:

  if isinstance(image, (str, pathlib.Path)):
    try:
      image = PIL.Image.open(image)
    except PIL.UnidentifiedImageError as uie:
      raise ValueError("Non Image File Encountered")
  elif isinstance(image, np.ndarray):
    image = PIL.Image.fromarray(image)
  
  image = image.resize(new_size, interpolator, roi, reducing_gap)
  return image

def square_pad(image: Union[np.ndarray, PIL.Image.Image]
               )-> Union[np.ndarray, PIL.Image.Image]:
  pil_input = False

  if isinstance(image, PIL.Image.Image):
    pil_input = True
    image = np.asarray(image)

  #As previous would conver image to array only need to check this
  if not isinstance(image, np.ndarray):
    raise ValueError("'image' must be of type PIL.Image.Image or numpy array"
                     " but got type: {}".format(type(image)))

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
    return image if not pil_input else PIL.Image.fromarray(image)

  image = np.pad(image, ((t, b), (l, r), (0, 0)), constant_values=0)
  return image if not pil_input else PIL.Image.fromarray(image)

#
def blackout(image: Union[np.ndarray, PIL.Image.Image],
             to_blackout:Sequence[int]=[255,255,255] 
               )-> Union[np.ndarray, PIL.Image.Image]:
  pil_input = False

  if isinstance(image, PIL.Image.Image):
    pil_input = True
    if image.mode == "RGBA":
      image = image.convert(mode="RGB")
    image_arr = np.asarray(image)
    image = image_arr.copy()

  image[image == to_blackout] = 0

  return image if not pil_input else PIL.Image.fromarray(image)

#
def refill_blackout(image: Union[np.ndarray, PIL.Image.Image],
                    dark_range:int=30)-> Union[np.ndarray, PIL.Image.Image]:
  pil_input = False

  if isinstance(image, PIL.Image.Image):
    pil_input = True
    image = np.asarray(image).copy()

  def is_blackout_edge(arr):
    return arr[0] > dark_range or arr[1] > dark_range or arr[2] > dark_range

  h, w, _ = image.shape

  refill_indices = []
  edge_found = False

  fill = np.mean(np.mean(image,1),0)

  #Do left side, top and bottom but not right
  for y in range(h):
    for x in range(w):
      if is_blackout_edge(image[y,x]):
        for i in refill_indices:
          image[y,i] = fill
        edge_found = True
        break
      else:
        refill_indices.append(x)
  #----------------------------------------------------------------
    if edge_found: # Do the right side
      refill_indices.clear()
      for x in range(w - 1, -1, -1):
        if is_blackout_edge(image[y,x]):
          for i in refill_indices:
            image[y,i] = fill
          edge_found = True
          break
        else:
          refill_indices.append(x)
  #----------------------------------------------------------------
    if not edge_found:
      for i in refill_indices:
        image[y,i] = fill
    edge_found = False
    refill_indices.clear()

  return image if not pil_input else PIL.Image.fromarray(image)


####

# TODO: Undecorate and just make plain function, decoration is a tad ott
# def apply_to_images(fapply: Callable[[PIL.Image.Image, ...], PIL.Image.Image]
#                     ) -> Callable[[Sequence[str], Sequence[str]], None]:

def apply_to_images(fapply: Callable[[PIL.Image.Image, ...], PIL.Image.Image]):
  def apply_decorator(unused):
    def apply(filenames: Sequence[str],
            out_filenames: Sequence[str] = None):
      for i, fname in enumerate(filenames):
        try:
          image = PIL.Image.open(fname)
        except PIL.UnidentifiedImageError as uie:
          print(uie)
          print("Skipping non image file")
          continue
        img = fapply(image)
        outname = fname if out_filenames is None else out_filenames[i]
        img.save(outname)
        print("{:02.2f}%".format(((i+1) / len(filenames)) * 100), end="\r")
      print("\n")
    return apply
  return apply_decorator

@apply_to_images(square_pad)
def square_images(filenames: Sequence[str],
                  out_filenames: Sequence[str] = None):
  pass

@apply_to_images(resize_image)
def resize_images(filenames: Sequence[str],
                  out_filenames: Sequence[str] = None):
  pass

@apply_to_images(as_mask)
def images_as_mask(filenames: Sequence[str],
                   out_filenames: Sequence[str] = None):
  pass

@apply_to_images(blackout)
def blackout_images(filenames: Sequence[str],
                    out_filenames: Sequence[str] = None):
  pass

@apply_to_images(refill_blackout)
def refill_blackout_images(filenames: Sequence[str],
                           out_filenames: Sequence[str] = None):
  pass

def rename_in_sequence(start, image_name_format="image_{:02}.png",
                       image_type=".png"):
  images = [x for x in os.listdir() if x.endswith(image_type)]
  for image in images:
    os.rename(image, image_name_format.format(start))
    start += 1
  return start


@tf.function
def normalize(tensor_image):
  return tf.cast(tensor_image, tf.float32) / 255.0






