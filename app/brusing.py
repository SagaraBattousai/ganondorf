import os
import csv

import numpy as np
import matplotlib.pyplot as plt
import cv2

import ring

lower_hsv = (110, 12, 0)
upper_hsv = (161, 40, 255)

def extract_in_range(img, lower, upper):
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  # Threshold the HSV image to get only colors in range
  mask = cv2.inRange(hsv, lower, upper)
  # Bitwise-AND mask and original image
  res = cv2.bitwise_and(img,img,mask=mask)

  return res, mask, hsv

def extraction_percentage(mask, img) -> float:
  intensity_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  mask_count = np.count_nonzero(mask)
  total = np.count_nonzero(intensity_img)
  return (mask_count / total) * 100

def extract_percentage(img, lower, upper) -> float:
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  mask = cv2.inRange(hsv, lower, upper)
  return extraction_percentage(mask, img)

def main(image_name):
  mask, img = ring.generate_ring(image_name)
  # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  masked = cv2.bitwise_and(img,img,mask=mask)
  brusing_percentage = extract_percentage(masked, lower_hsv, upper_hsv)
  res, _, _ = extract_in_range(masked, lower_hsv, upper_hsv)
  print("{:.2f}".format(brusing_percentage))
  return mask, img, res, brusing_percentage

def generate_percentages(image_names, filename):
  pass


if __name__ == "__main__":
  image_dir = "Leak\\image\\"
  image_names = os.listdir(image_dir) 
  filename = "Leak\\brusing.csv"
  csv_file = open(filename, 'w', newline='')
  fieldnames=["Image_Name", "Brusing_Percentage"]
  writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
  writer.writeheader()
  for image_name in image_names:
    _, _, _, percentage = main("{}\\{}".format(image_dir, image_name))
    writer.writerow({"Image_Name": image_name,
                     "Brusing_Percentage": percentage})

  csv_file.close()

    
    

  #------------------------------------------------------------------------
  # mask, img, res, _ = main('test2.png') #'image_09.png')
  # masked = cv2.bitwise_and(img,img,mask=mask)

  # images = [img, mask, masked, res]
  # rows = 2
  # cols = 2

  # for x in range(len(images)):
  #   plt.subplot(rows, cols, x + 1)
  #   plt.imshow(images[x])

  # plt.show()
  #------------------------------------------------------------------------
