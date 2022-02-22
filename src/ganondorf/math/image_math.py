import numpy as np

__all__ = ['moment', 'center_of_mass']

def moment(img: np.array, p, q):
  acc = 0
  xres, yres = img.shape[0:2]
  for i in range(1, xres + 1):
    for j in range(1, yres + 1):
      acc += (i ** p) * (j ** q) * img[i - 1, j - 1]
  return acc

def center_of_mass(img: np.array):
  m00 = moment(img, 0, 0)
  xc = moment(img, 1, 0) / m00
  yc = moment(img, 0, 1) / m00
  return xc, yc

