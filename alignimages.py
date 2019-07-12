import cv2
import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

if __name__ == 'x__main__':

  # Read reference image
  refFilename = "correct.jpg"
  print("Reading reference image : ", refFilename)
  imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

  # Read image to be aligned
  imFilename = "input/a.jpg"
  print("Reading image to align : ", imFilename);
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

  print("Aligning images ...")
  # Registered image will be resotred in imReg.
  # The estimated homography will be stored in h.
  imReg, h = alignImages(im, imReference)

  # Write aligned image to disk.
  outFilename = "output/a.jpg"
  print("Saving aligned image : ", outFilename);
  cv2.imwrite(outFilename, imReg)

  # Print estimated homography
  print("Estimated homography : \n",  h)

# This is to list the images in input and process them
from os import listdir
from os.path import join, abspath, isfile
from lib import alignfiles

if __name__ == '__main__':
  for file in listdir('images'):
    if (isfile(join('images', file))):
      alignfiles.alignfiles(abspath('correct.jpg'), abspath(join('images', file)))
