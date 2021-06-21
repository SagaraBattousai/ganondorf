
""" Module for loading datasets.
Includes dataset transformations for data of types such as:
  - Images including:
    - Standard iamge formats such as Png's and Jpeg's
    - Medical images such as those saved in medical formats such as .nii (NIfTI)
  
  - Raw data saved in numpy format
"""

#stdlib imports
import os
import importlib.resources as resources
# import collections

#external imports
import PIL
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
import ptils

#intra-package imports
import ganondorf.data.format_image_data as fid
import ganondorf.data.datasets as gdds

IMAGE = 0
MASK = 1

def image_as_array(path: str, mode: str=None) -> np.array:
  image = PIL.Image.open(path)
  return fid.image_as_array(image, mode)

def medical_as_array(path: str) -> np.array:
  image = sitk.ReadImage(path)
  return fid.medical_as_array(image)

def dir_to_mask(path):
  image_names = os.listdir(path)
  for x in image_names:
    img = PIL.Image.open(x)
    img = fid.as_mask(img)
    img.save(x)

def load_image_pair_dataset(path:str, 
                            image_dirname:str="image",
                            label_dirname:str="mask",
                            size:tuple[int, int]=None)->tf.data.Dataset:

  dataset_path = resources.files(gdds).joinpath(path) #.open()

  image_files = dataset_path.joinpath(image_dirname).iterdir()
  label_files = dataset_path.joinpath(label_dirname).iterdir()

  images = []
  labels = []

  for img, lab in zip(image_files, label_files):

    if size is not None:
      image = fid.image_as_array(fid.resize_image(img, size))
      label = fid.image_as_array(fid.resize_image(lab, size), mode='1')
    else:
      #TODO: decide if acceptable or not!
      image = image_as_array(img) # Using path but in zip so not safe? 
      label = image_as_array(lab, mode='1') #  Should already be in this form
       

    # dataset.append( {'image': image, 'label': label} )
    images.append(image)
    labels.append(label)
  
  return tf.data.Dataset.from_tensor_slices((images, labels))

def load_medical_dataset(path:str,
                         data_prefix:str = "",
                         data_suffix:str = ""
                         )->tf.data.Dataset:

  dataset_path = resources.files(gdds).joinpath(path)
  patients = (patient for patient in dataset_path.iterdir() \
              if patient.is_dir() and patient.name != ".git")

  dataset = []

  for patient in patients:
    patient_data = [data for data in patient.iterdir() \
                    if data.startswith(data_prefix) and \
                       data.endswith(data_suffix) ]
    dataset.extend(patient_data)

  return tf.data.Dataset.from_tensor_slices(dataset)

def load_npz_dataset(path:str, label_index:int=-1)->tf.data.Dataset:

  data = resources.files(gdds).joinpath(path)

  dataset = {}

  with np.load(data) as datadict:
    for name in datadict:
      arr, label = np.hsplit(datadict[name], np.array([label_index]))
      dataset[name] = (arr, label)
  
  # return tf.data.Dataset.from_tensor_slices(
  #     collections.OrderedDict(
  #         x = input_array,
  #         y = label_array
  #         )
  #     )

  return tf.data.Dataset.from_tensor_slices(dataset)

def load_AL_segmentation(load_train = True,
                         load_test = True,
                         size:tuple[int, int]=None)->tf.data.Dataset:
  path = "ALSegment"
  dataset = []

  if load_train:
    dataset.append(load_image_pair_dataset(path + "/train",
                                           image_dirname="image",
                                           label_dirname="mask",
                                           size=size))
  if load_test:
    dataset.append(load_image_pair_dataset(path + "/test",
                                           image_dirname="image",
                                           label_dirname="mask",
                                           size=size))
  if dataset == []:
    return None
  elif len(dataset) == 1:
    return dataset[0]
  else:
    return tuple(dataset)

def load_brain_tumor_progression(load_pre_scan=True,
                                 load_post_scan=True,
                                 load_pre_mask=True,
                                 load_post_mask=True):

  path = "BrainTumorProgression"

  dataset = []

  if load_pre_scan:
    dataset.append(load_medical_dataset(path, 
                                        data_prefix="Pre_Scan_T1.nii.gz"))

  if load_post_scan:
    dataset.append(load_medical_dataset(path, 
                                        data_prefix="Post_Scan_T1.nii.gz"))
  
  if load_pre_mask:
    dataset.append(load_medical_dataset(path, 
                                        data_prefix="Pre_Mask.nii.gz"))

  if load_post_mask:
    dataset.append(load_medical_dataset(path, 
                                        data_prefix="Post_Mask.nii.gz"))

  if dataset == []:
    return None
  else:
    return tf.data.Dataset.zip(tuple(dataset))

def load_arcm(normalised:bool=False)->tf.data.Dataset:

  path = ("arcm/"
          "arcm_data_norm.npz" if normalised else "arcm_data.npz")

  return load_npz_dataset(path)

def load_mhealth(normalised:bool=False)->tf.data.Dataset:

  path = ("mHealth/"
          "mHealth_data_norm.npz" if normalised else "mHealth_data.npz")

  return load_npz_dataset(path)

def load_har(normalised:bool=False)->tf.data.Dataset:

  path = ("har/"
          "har_data_norm.npz" if normalised else "har_data.npz")

  return load_npz_dataset(path)



