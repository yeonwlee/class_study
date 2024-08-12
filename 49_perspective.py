import cv2
import numpy as np

src = cv2.imread('./data/business-card_640.jpg')

print(src.shape)

width = 720
height = 480

# src_quad = np.array([[160, 207], [395, 112], [487, 221], [239, 329]], np.float32)
# dst_quad = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)

# pers = cv2.getPerspectiveTransform(src_quad, dst_quad)
# dst = cv2.warpPerspective(src, pers, (width, height))

# cv2.imshow('', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


######################## gpt가 알려준 귀퉁이 감지

# 그레이스케일 변환
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
 
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized = clahe.apply(gray)

# 가우시안 블러 적용 (노이즈 제거)
blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

# 적응형 이진화 적용
adaptive_thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
)

# 형태학적 변환 적용 (윤곽선 강화)
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(adaptive_thresh, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# 경계 검출 (캐니 엣지)
edges = cv2.Canny(eroded, 50, 150)

# 윤곽선 찾기
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 가장 큰 윤곽선 찾기
largest_contour = max(contours, key=cv2.contourArea)

# 윤곽선 근사화
epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

# 네 귀퉁이가 맞는지 확인 (4개의 점이 있어야 함)
if len(approx) == 4:
    # 각 점에 원 그리기
    for point in approx:
        cv2.circle(src, tuple(point[0]), 5, (0, 255, 0), -1)
else:
    print("네 귀퉁이를 찾지 못했습니다.")
    # 윤곽선의 모든 점 출력 (디버그용)
    for point in largest_contour:
        cv2.circle(src, tuple(point[0]), 1, (255, 0, 0), -1)

# 결과 이미지 보기
cv2.imshow('Detected Corners', src)
cv2.waitKey(0)
cv2.destroyAllWindows()
