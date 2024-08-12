import sys
import numpy as np
import cv2

cap = cv2.VideoCapture(0) # 카메라 불러오기

backimg = cv2.imread('./data/starry_night.jpg')
backimg = cv2.resize(backimg, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))) # 카메라로 들어온 사이즈에 맞게 이미지 리사이징

if not cap.isOpened(): # 카메라 현재 사용 가능 여부 확인
    print('카메라를 켜주세요')
    sys.exit()
    
fps = cap.get(cv2.CAP_PROP_FPS) # 카메라 fps 가져오기

chroma = False

def pencil_filter(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BayerGR2GRAY) # 우선 그레이스케일로 바꿔야함
    blr = cv2.GaussianBlur(gray_img, (0,0), 3)
    dst = cv2.divide(gray_img, blr, scale=255)
    return dst

def cartoon_filter(img):
    h, w = img.shape[:2] # height, width 가져오기
    img = cv2.resize(img, (w // 2, h // 2)) # 사이즈 줄이기. 비트 연산 들어가서 느려지기 때문에 1. 줄였다가
    blr = cv2.bilateralFilter(img, -1, 20, 7)
    edge = 255 - cv2.Canny(img, 80, 120) # 엣지 검출. 흑백이미지로 나옴
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)    

    dst = cv2.bitwise_and(blr, edge)
    dst = cv2.resize(dst, (w, h), interpolation=cv2.INTER_NEAREST) # 2. 원래 사이즈로 키움
    return dst
    
    
while True:
    ret, frame = cap.read() # 첫 번째 리턴값 불러오는 걸 성공 했냐여부
    if ret == False:
        print('카메라 영상 수신이 실패하였습니다')
    
    if chroma:
        mask = cv2.inRange(frame, (150, 150, 150), (255, 255, 255)) # frame에서 특정 색상 값을 뽑아낼 것, 보통 최소값
        # cv2.copyTo(backimg, mask, frame) # 백그라운드 이미지에서 마스크 영역을 frame에 복사
        cv2.copyTo(cv2.GaussianBlur(frame, (0, 0), 20), mask, frame) # 백그라운드 이미지에서 마스크 영역을 frame에 복사

    #필터 적용하기
    
    cv2.imshow('chromakey', cartoon_filter(frame))
    key = cv2.waitKey(1)
    if key == ord(' '):
        chroma = not chroma
    elif key == 27: # esc
        break

cap.release()
cv2.destroyAllWindows()
        