import cv2
import numpy as np
import sys

img_file = sys.argv[1]
img = cv2.imread(img_file, cv2.IMREAD_COLOR)
img = cv2.blur(img, (5, 5))

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

thresh0 = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
thresh1 = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
thresh2 = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
thresh = cv2.bitwise_or(thresh0, thresh1)

cv2.imshow('Image-thresh0', thresh0)
cv2.waitKey(0)
cv2.imshow('Image-thresh1', thresh1)
cv2.waitKey(0)
cv2.imshow('Image-thresh2', thresh2)
cv2.waitKey(0)
