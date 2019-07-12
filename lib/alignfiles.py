import cv2
from os.path import basename, dirname, splitext, join, exists
from os import mkdir
from lib.alignimages import alignimages

def alignfiles(base, rotated):
  dir = dirname(rotated)
  if (not exists(join(dir, 'aligned'))): mkdir(join(dir, 'aligned'))
  if (not exists(join(dir, 'debug'))): mkdir(join(dir, 'debug'))
  sAligned = join(dir, 'aligned', basename(rotated))
  sDebug = join(dir, 'debug', basename(rotated))
  print("Aligning:", rotated)

  # Read the images
  imgBase = cv2.imread(base, cv2.IMREAD_COLOR)
  imgRotated = cv2.imread(rotated, cv2.IMREAD_COLOR)

  # Do the rotation
  imgAligned, homography, imgDebug = alignimages(imgBase, imgRotated)

  # Write the images
  cv2.imwrite(sAligned, imgAligned)
  cv2.imwrite(sDebug, imgDebug)

  print("Wrote aligned image to", sAligned)
