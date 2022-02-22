import math
from collections.abc import Callable

import numpy as np
import matplotlib.pyplot as plt
import cv2

import hue 
import relaxation


lower_rgb = (226, 230, 243)
mid_rgb   = (191, 191, 227)
top_rgb   = (180, 160, 181)

# lower_hsv = (113, 18, 181)
# upper_hsv = (149, 40, 243)

# lower_hsv = (100, 15, 181)
# upper_hsv = (160, 255, 243)

lower_hsv = (110, 12, 0)
upper_hsv = (161, 40, 255)

#hextract.extract_in_range(img, lower_hsv, upper_hsv)




def extract_in_range(img, lower, upper):
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  # Threshold the HSV image to get only colors in range
  mask = cv2.inRange(hsv, lower, upper)
  # Bitwise-AND mask and original image
  res = cv2.bitwise_and(img,img,mask=mask)

  return res, mask, hsv

def show_extraction(img, lower, upper):

  res, mask, hsv = extract_in_range(img, lower, upper)
  plt.subplot(2,2,1)
  plt.imshow(img)
  plt.subplot(2,2,2)
  plt.imshow(cv2.bitwise_and(hsv,hsv,mask=mask))
  plt.subplot(2,2,3)
  plt.imshow(mask)
  plt.subplot(2,2,4)
  plt.imshow(res)
  plt.show()

def context_simple(ai, li, aj, lk) -> float:
    if li == 0 and lk == 0:
      return 0.5
    elif li == 1 and lk == 0:
      return 0.2
    elif li == 0 and lk == 1:
      return 0.2
    elif li == 1 and lk == 1:
      return 0.95
    
    print(li, lk)
    raise Exception()

def context_hue(img: np.array)->Callable[..., float]:
    
  def context(ai, li, aj, lk):
    pi = img[ai] if li == 0 else 1 - img[ai]
    pj = img[aj] if lk == 0 else 1 - img[aj]
    
    # return pi * pj
    return (pi + pj) / 2
    # return max(pi, pj)

  return context

def coeff(ai, aj) -> float:
  x, y = ai
  i, j = aj

  w = 1 if i == x or x - 1 == i or x + 1 == i else 0
  h = 1 if j == y or y - 1 == j or y + 1 == j else 0

  return (w and h)
  
def old_main():
  res, mask, hsv = extract_in_range(img, lower_hsv, upper_hsv)

  #assert np.all(h == hsv[:,:,0])

  mask = mask / 255
 
  v2 = v.copy() / np.amax(v)
  v2 = 1.0 - v2

  s2 = s.copy() / np.amax(s)
  # s2 = 1.0 - s2

  #-------------------------------------
  pm = v2 * mask

  pm = (pm / np.amax(pm)) * 0.70
  pm = pm + 0.05
  
  pm = cv2.resize(pm, None, fx=(1/8), fy=(1/8), interpolation=cv2.INTER_AREA)
  probs = relaxation.ImageBiProbabilityMap(pm)
  #--------------------------------------
  # v2 = (v2 * 0.70) + 0.05
  # v2 = cv2.resize(v2, None, fx=(1/8), fy=(1/8), interpolation=cv2.INTER_AREA)

  # mask = (mask * 0.70) + 0.05
  # mask = cv2.resize(mask, None, fx=(1/8), fy=(1/8), interpolation=cv2.INTER_AREA)

  # probs = relaxation.ImageBiProbabilityMap(v2)#mask)#v2)
  #--------------------------------------
    
  i = 1
  count = 10
  # for _ in range(count):
  #   context = context_simple
  #   # context = context_hue(mask)#v2)#mask)
  #   relaxation.relax(probs, context, coeff)
  #   print("{:.2f}%".format(i/count * 100), end="\r")
  #   i += 1


  mask = cv2.resize(mask, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
  rax = probs.probabilities()
  rax = cv2.resize(rax, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)

  #-----------------------------------------------------
  s3 = np.array(list(map(lambda a: a ** 2, s2)))
  v3 = np.array(list(map(lambda a: a ** 2, v2)))

  sv3 = s3 + v3
  sv3 = np.array(list(map(lambda a: np.sqrt(a), sv3)))
  #-----------------------------------------------------


  plt.subplot(221)
  plt.imshow(mask)
  plt.subplot(222)
  plt.imshow(v2)
  plt.subplot(223)
  plt.imshow(sv3)
  plt.subplot(224)
  plt.imshow(rax)
  plt.show()

if __name__ == "__main__":
  # img = cv2.imread("image_09.png")
  img = cv2.imread("masked.png")
  h, s, v = hue.cvt_hsv(img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  res, mask, hsv = extract_in_range(img, lower_hsv, upper_hsv)

  res2, mask2, hsv2 = extract_in_range(img, (0, 12,240), (5, 20, 255))

  img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  brusing = np.count_nonzero(mask)
  total = np.count_nonzero(img_gray)
  print("{:.2f}%".format((brusing / total) * 100))
  remainder = img - res

  plt.subplot(221)
  plt.imshow(img)
  plt.subplot(222)
  plt.imshow(res)
  plt.subplot(223)
  plt.imshow(res2 + res)
  plt.subplot(224)
  plt.imshow(cv2.cvtColor(remainder, cv2.COLOR_RGB2HSV))
  plt.show()
  plt.show()
  


