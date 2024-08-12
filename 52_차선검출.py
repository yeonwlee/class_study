import cv2
import numpy as np

# road_img = cv2.imread('./data/yeji_road.png')

# # 그레이스케일 변환
# gray = cv2.cvtColor(road_img, cv2.COLOR_BGR2GRAY)

# # 이진화 적용
# _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

# # 검은 영역 찾기
# # 이미지의 각 행의 합을 계산하여 검은 영역 탐지
# row_sum = np.sum(binary, axis=1)
# non_black_indices = np.where(row_sum > 0)[0]

# # 상단과 하단의 검은 영역 인덱스 찾기
# top = non_black_indices[0]
# bottom = non_black_indices[-1]

# # 검은 영역 제거
# cropped_image = road_img[top:bottom+1, :]


# # 이미지 크기 조정 (예: 500x300)
# desired_width = 400
# desired_height = 600
# resized_image = cv2.resize(cropped_image, (desired_width, desired_height))

# road_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
# road_img = cv2.cvtColor(road_img, cv2.COLOR_RGB2GRAY)

# road_blur = cv2.GaussianBlur(road_img, (0, 0), 1)
# road_edge = cv2.Canny(road_blur, 100, 250)

# cv2.imshow('Resized Image', road_edge)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
###################

road_img = cv2.imread('./data/road.jpg')

road_img = cv2.cvtColor(road_img, cv2.COLOR_BGR2RGB)
road_img = cv2.cvtColor(road_img, cv2.COLOR_RGB2GRAY)

road_blur = cv2.GaussianBlur(road_img, (0, 0), 1)
road_edge = cv2.Canny(road_blur, 100, 250)

mask = np.zeros([1098, 1400], dtype='uint8')
point = np.array([[(50, 1098), (600, 700), (900, 700), (1400, 1098)]])
cv2.fillPoly(mask, point, 255)

result_img = cv2.bitwise_and(mask, road_edge) # 마스크 흰색과 기존 엣지 흰색에서 둘다 겹치는 부분만 검출

line_img = np.zeros((1098, 1400, 3), dtype=np.uint8)
road_line = cv2.HoughLinesP(result_img, 3, np.pi/180, 90, minLineLength=1, maxLineGap=5)

for line in road_line:
    for x1, y1, x2, y2 in line:
        cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 3)


cv2.imshow('Resized Image', line_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
