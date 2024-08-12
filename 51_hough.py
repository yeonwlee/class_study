import cv2
import numpy as np

src = cv2.imread('./data/building.jpg')

# 엣지를 우선 검출
edges = cv2.Canny(src, 50, 150)
lines = cv2.HoughLinesP(edges, 1.0, np.pi / 180., 160, minLineLength=50, maxLineGap=5)

dst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

for temp in range(lines.shape[0]):
    pt1 = (lines[temp][0][0], lines[temp][0][1])
    pt2 = (lines[temp][0][2], lines[temp][0][3])
    cv2.line(dst, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
    
cv2.imshow('', dst)
cv2.waitKey()
    