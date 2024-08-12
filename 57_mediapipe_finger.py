import cv2
import mediapipe as mp
import numpy as np
import socket

#### 소켓 관련 설정
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # socket 설정, IPv4 127.0.0.1 / UDP 소켓 사용할 것
sendport = ('127.0.0.1', 5054) # 데이터 보낼 IP 및 포트. 클라이언트가 지정해줄 것
##### 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=4)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    frame = cv2.flip(frame, 1) # 좌우반전
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks, hand_lr in zip(results.multi_hand_landmarks, results.multi_handedness):
            # print(hand_lr.classification[0].label) # 왼손 오른손 구분
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            h, w, _ = frame.shape
            # cv2.putText(frame, hand_lr.classification[0].label, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if hand_lr.classification[0].label == 'Right': # 오른손일 때 조절하게 할 것
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                tx = thumb_tip.x * w
                ty = thumb_tip.y * h
                ix = index_finger_tip.x * w
                iy = index_finger_tip.y * h
                
                point1 = np.array([tx, ty])
                point2 = np.array([ix, iy])
                
                distance = np.linalg.norm(point2 - point1)
                # print(distance)
                send_dist =  str(int(distance))
                sock.sendto(str.encode(send_dist), sendport)
                cv2.putText(frame, send_dist, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    cv2.imshow('hand', frame)

    if cv2.waitKey(1) == ord('q'):
        break 
    
cap.release()
cv2.destroyAllWindows()
sock.close()