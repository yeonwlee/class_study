import cv2
# 얼굴 검출 - > 잘라내서 필터/모자이크 또는 blur -> 그 위치에 똑같이 붙여넣기

src = cv2.imread('./data/gerbera jamesonii.jpg')
src = cv2.GaussianBlur(src, (0, 0), 3)

cv2.imshow('blur', src)
cv2.waitKey()
cv2.destroyAllWindows()