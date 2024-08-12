import cv2
import numpy as np

src = cv2.imread('./data/starry_night.jpg')

dst1 = cv2.add(src, (100, 100, 100, 0)) # 밝기 조절
dst2 = np.clip(src+100., 0, 255).astype(np.uint8)
'''
이 코드는 원본 이미지의 모든 픽셀 값을 100만큼 증가시키고, 
증가된 값이 0과 255 사이에 있도록 잘라내어 밝게 조절된 이미지를 생성합니다. 
이 과정에서 발생할 수 있는 오버플로우 문제를 방지하기 위해 np.clip을 사용하여 값의 범위를 제한합니다. 
최종적으로 np.uint8 형식으로 변환하여 OpenCV에서 사용할 수 있는 이미지 형식으로 만듭니다.

예시
원본 이미지 src의 픽셀 값이 150인 경우:

src + 100. => 250.0
np.clip(250.0, 0, 255) => 250.0
astype(np.uint8) => 250
'''

# b, g, r
b_c,g_c,r_c = cv2.split(src)

cv2.imshow('image1', dst1)
cv2.imshow('image2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()