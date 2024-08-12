import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh()


face_img = cv2.imread('./data/face_mk.png', cv2.IMREAD_UNCHANGED)
cap = cv2.VideoCapture(0)

while cap.isOpened:
  ret, img = cap.read()
  imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 미디어 파이프에서 인식하려면 RGB 형식이어야 함
  result = face_mesh.process(imgRGB)
  
  if result.multi_face_landmarks:
    for face_lms in result.multi_face_landmarks:
      xy_point = []
      for counting, landmark in enumerate(face_lms.landmark):
        ih, iw, _ = img.shape # 세 번째 값은 채널
        img = cv2.circle(img, (int(landmark.x * iw), int(landmark.y * ih)), radius=1, color=(255, 0, 0), thickness=2)
        xy_point.append([landmark.x, landmark.y])
        
      top_left = np.min(xy_point, axis=0)
      bottom_right = np.max(xy_point, axis=0)
      mean_xy = np.mean(xy_point, axis=0)
      
      cv2.circle(img, (int(top_left[0] * iw), int(top_left[1] * ih)), 4, (0, 0, 255), 3)
      cv2.circle(img, (int(bottom_right[0] * iw), int(bottom_right[1] * ih)), 4, (0, 0, 255), 3)
      cv2.circle(img, (int(mean_xy[0] * iw), int(mean_xy[1] * ih)), 4, (0, 0, 255), 3)
      
      face_width = int(bottom_right[0] * iw) - int(top_left[0] * iw)
      face_height = int(bottom_right[1] * iw) - int(top_left[1] * iw)
      
      if face_width > 0 and face_height > 0: # 얼굴 검출 되면
        face_result = face_overlay(img, face_img, int(mean_xy[0] * iw), int(mean_xy[1]*ih), (face_width+100, face_height+100))
  else:  # 얼굴을 못 찾은 경우
    face_result = img
  
  cv2.imshow('', face_result)
  if cv2.waitKey(1) & 0xFF == 27:
    break

cap.release()
cv2.destroyAllWindows()