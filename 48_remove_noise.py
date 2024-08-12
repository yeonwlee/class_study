import cv2

# src = cv2.imread('./data/noise.jpg')
# dst = cv2.medianBlur(src, 11) # 홀수로 지정해줘야함

src = cv2.imread('./data/woman_640.jpg')
dst = cv2.bilateralFilter(src, -1, 20, 5)

cv2.imshow('origin', src)
cv2.imshow('', dst)
cv2.waitKey()
cv2.destroyAllWindows()