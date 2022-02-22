import os
import csv
import argparse

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

def generate_percentages(image_name, network:str="RingIONetwork"):
  mask, img = ring.generate_ring(image_name, network)
  # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  masked = cv2.bitwise_and(img,img,mask=mask)
  brusing_percentage = extract_percentage(masked, lower_hsv, upper_hsv)
  res, _, _ = extract_in_range(masked, lower_hsv, upper_hsv)
  print("{:.2f}".format(brusing_percentage))
  return mask, img, res, brusing_percentage

def show_data(image_name, network:str="RingIONetwork"):
  mask, img, res, brusing_percentage = generate_percentages(image_name, network)
  masked = cv2.bitwise_and(img,img,mask=mask)

  images = [img, mask, masked, res]
  titles = ["Original Image", "Mask", "Masked Image", "Bruising"]
  fig, axs = plt.subplots(2, 2)
  axs = tuple(axs.flatten())

  fig.suptitle("Bruising Percentage: {:.2f}%".format(brusing_percentage))

  for x in range(len(images)):
    axs[x].set_title(titles[x])
    axs[x].imshow(images[x])

  plt.show()

def bruising_meta(leak_name:str, network:str="RingIONetwork",
                  metadata_name:str="bruising.csv"):

  image_dir = f"{leak_name}{os.sep}image"
  image_names = os.listdir(image_dir) 
  filename = f"{leak_name}{os.sep}{metadata_name}"
  csv_file = open(filename, 'w', newline='')
  fieldnames=["Image_Name", "Brusing_Percentage"]
  writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
  writer.writeheader()
  for image_name in image_names:
    _, _, _, percentage = \
        generate_percentages(f"{image_dir}{os.sep}{image_name}", network)
    
    writer.writerow({"Image_Name": image_name,
                     "Brusing_Percentage": percentage})

  csv_file.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest="subcommand_name",
                                     help="SubCommand help message")

  metadata_parser = subparsers.add_parser("metadata",
                                          help="Metadata generator help")

  metadata_parser.add_argument("leak_type", choices=["NoLeak", "Leak"])
  metadata_parser.add_argument("-n", "--network", default="RingIONetwork")
  metadata_parser.add_argument("-o", "--out_name")

  show_parser = subparsers.add_parser("show",
                                      help="Show generator help")

  show_parser.add_argument("image_name", help="Name of image to use")
  show_parser.add_argument("-n", "--network", default="RingIONetwork")

  args = parser.parse_args()

  if args.subcommand_name == "metadata":
    bruising_meta(args.leak_type, args.network, args.out_name)
  elif args.subcommand_name == "show":
    show_data(args.image_name, args.network)


