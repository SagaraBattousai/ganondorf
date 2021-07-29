import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL

if __name__ == "__main__":
  base_image = PIL.Image.open(sys.argv[1])
  base_image = base_image.resize((128,128))
  blackout = PIL.Image.new(base_image.mode, (128,128))
  mask = PIL.Image.open(sys.argv[2])

  image = PIL.Image.composite(base_image, blackout, mask)

  plt.imshow(image)
  plt.axis('off')
  plt.show()

  image.save("masked.png")

  
