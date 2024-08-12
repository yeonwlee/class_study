import cv2
import numpy as np

# white_canvas = np.zeros((500, 500, 3), dtype='uint8') + 255 # 하얀색 캔버스 생성

# cv2.line(white_canvas, (20, 20), (250, 250), (255, 0, 0), 3, cv2.LINE_AA)
# cv2.line(white_canvas, (250, 250), (0, 500), (0, 0, 255), 3, cv2.LINE_AA)

# cv2.circle(white_canvas, (250, 250), 150, (0, 255, 0), 3, cv2.LINE_AA)
# # cv2.circle(white_canvas, center=(250, 250), radius=150, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

# cv2.rectangle(white_canvas, (50, 50), (200, 200), (0, 0, 0), 2, cv2.LINE_AA)
# # cv2.rectangle(white_canvas, pt1=(50, 50), pt2=(200, 200), color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

# cv2.imshow('image', white_canvas)


cat_image = cv2.imread('./data/test_cat.jpg')
cv2.rectangle(cat_image, pt1=(145, 65), pt2=(425, 425), color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
cv2.rectangle(cat_image, (10, 10, 300, 200), (255, 0, 0), 2, cv2.LINE_AA)

cv2.putText(cat_image, "Cat", (145, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('image', cat_image)

cv2.waitKey()
cv2.destroyAllWindows()





