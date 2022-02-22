import numpy as np
import matplotlib.pyplot as plt
import cv2

def cvt_hsv(img):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h = hsv[:,:,0].copy()
  s = hsv[:,:,1].copy()
  v = hsv[:,:,2].copy()

  return h, s, v

if __name__ == "__main__":
  # img = cv2.imread("image_09.png")
  img = cv2.imread("masked.png")
  h, s, v = cvt_hsv(img)

  hroi = h.copy()
  hroi[hroi < 130] = 0
  hroi[hroi > 150] = 0

  sroi = s.copy()
  sroi[sroi > 38] = 255
  #sroi[sroi < 14] = 255
  sroi = 255 - sroi

  vroi = v.copy()
  vroi[vroi > 225] = 255
  vroi[vroi < 170] = 255
  vroi = 255 - vroi

  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  images=[h, s, v, hroi, sroi, vroi]

  rows = 3
  cols = 3

  #plt.fig(7, 7)

  plt.subplot(rows, cols, 2)
  #[300:360, 210:440]
  plt.imshow((img))# / np.amax(img)))
  # plt.imshow((img[int(300/4):int(360/4), int(210/4):int(440/4)] / np.amax(img[i])))


  for i in range(len(images)):
    plt.subplot(rows, cols, i+4)

    #[300:360, 210:440]
    plt.imshow((images[i]) / np.amax(images[i]))
    # plt.imshow((images[i][int(300/4):int(360/4), int(210/4):int(440/4)] / np.amax(images[i])))

  plt.show()
