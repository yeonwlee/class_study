import cv2
import sys

img = cv2.imread('./data/starry_night.jpg')

cv2.imshow('Display window', img)

cv2.waitKey()
cv2.destroyAllWindows()
