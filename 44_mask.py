import cv2

src = cv2.imread('./data/starry_night.jpg', cv2.IMREAD_COLOR)
mask = cv2.imread('./data/starry_night_mask.jpg', cv2.IMREAD_GRAYSCALE)
dst = cv2.imread('./data/starry_night_BGR.jpg', cv2.IMREAD_COLOR)

cv2.copyTo(src, mask, dst)
cv2.imshow('', dst)
cv2.waitKey()
cv2.destroyAllWindows()