import cv2
import numpy as np

MAX_FEATURES = 2500
FEATURE_QUALITY = 0.15

def alignimages(imgTrain, imgQuery):
  # Just an easier way to handle these
  class train: image=imgTrain
  class query: image=imgQuery

  # Convert images to grayscale, because the feature detection works best with
  # grayscale. Not sure if it's even possible to do with color
  train.gray = cv2.cvtColor(train.image, cv2.COLOR_BGR2GRAY)
  query.gray = cv2.cvtColor(query.image, cv2.COLOR_BGR2GRAY)

  # Create a feature detector
  orb = cv2.ORB_create(2500, 1.1)
  # Detect the features
  train.keypoints, train.descriptors = orb.detectAndCompute(train.gray, None)
  query.keypoints, query.descriptors = orb.detectAndCompute(query.gray, None)

  # Brute Force Matching
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(query.descriptors, train.descriptors)
  matches = sorted(matches, key = lambda x:x.distance)

  # Filter matches by quality
  numGoodMatches = int(len(matches) * FEATURE_QUALITY)
  matches = matches[:numGoodMatches]

  # Draw debug information
  imgDebug = cv2.drawMatches(query.image, query.keypoints, train.image, train.keypoints, matches, None)

  # Get points for homography
  train.points = np.zeros((len(matches), 2), dtype=np.float32)
  query.points = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    train.points[i, :] = train.keypoints[match.trainIdx].pt
    query.points[i, :] = query.keypoints[match.queryIdx].pt

  homography, mask = cv2.findHomography(query.points, train.points, cv2.RANSAC)

  # Transform the image
  height, width, channels = train.image.shape
  imgAligned = cv2.warpPerspective(query.image, homography, (width, height))

  return imgAligned, homography, imgDebug
