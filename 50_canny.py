import cv2

src = cv2.imread('./data/galaxy-soho.jpg')
dst = cv2.Canny(src, 50, 150) # 엣지 검출 알고리즘

cv2.imshow('', dst)
cv2.waitKey()