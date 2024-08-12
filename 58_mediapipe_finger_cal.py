import cv2
import mediapipe as mp
import numpy as np
import socket

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=4)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def calcuate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosangle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    arccos = np.arccos(cosangle)
    degree = np.degrees(arccos)
    return degree

    
while cap.isOpened():
    ret, frame = cap.read()
    
    frame = cv2.flip(frame, 1) # 좌우반전
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks, hand_lr in zip(results.multi_hand_landmarks, results.multi_handedness):
            if1 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            if2 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            if3 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
            print(if1.z)
            index_finger_degree = calcuate_angle([if1.x, if1.y], [if2.x, if2.y], [if3.x, if3.y])
            
            # print(index_finger_degree)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f'index angle: {int(index_finger_degree)}', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('hand', frame)

    if cv2.waitKey(1) == ord('q'):
        break 
    
cap.release()
cv2.destroyAllWindows()