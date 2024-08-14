import cv2
import mediapipe as mp
import numpy as np
import socket
import os 
import csv
import pandas as pd

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=4)
mp_drawing = mp.solutions.drawing_utils


df = pd.read_csv('hand_data.csv')
X_data = df.iloc[:,:-1]
y_data = df.iloc[:,-1]

# 학습을 위해서 문자열을 float형으로 변환
X_data = X_data.to_numpy().astype(np.float32)
y_data = y_data.to_numpy().astype(np.float32)

knn_model = cv2.ml.KNearest_create()
knn_model.train(X_data, cv2.ml.ROW_SAMPLE, y_data)



cap = cv2.VideoCapture(0)

# 각도 계산
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


# csv로 저장
def data_to_csv(data, file_name='hand_data.csv'):
    is_exist = os.path.isfile(file_name)
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        if not is_exist:
            header_row = [f'd{num}' for num in range(1, 16)] + ['label']
            writer.writerow(header_row)
        writer.writerow(data)
        

finger_point = [
            # 엄지 손가락
            [0, 1, 2],  # THUMB_CMC -> THUMB_MCP -> THUMB_IP
            [1, 2, 3],  # THUMB_MCP -> THUMB_IP -> THUMB_TIP
            [2, 3, 4],
            
            # 검지 손가락
            [0, 5, 6],  # WRIST -> INDEX_FINGER_MCP -> INDEX_FINGER_PIP
            [5, 6, 7],  # INDEX_FINGER_MCP -> INDEX_FINGER_PIP -> INDEX_FINGER_DIP
            [6, 7, 8],  # INDEX_FINGER_PIP -> INDEX_FINGER_DIP -> INDEX_FINGER_TIP

            # 중지 손가락
            [0, 9, 10],  # WRIST -> MIDDLE_FINGER_MCP -> MIDDLE_FINGER_PIP
            [9, 10, 11], # MIDDLE_FINGER_MCP -> MIDDLE_FINGER_PIP -> MIDDLE_FINGER_DIP
            [10, 11, 12],# MIDDLE_FINGER_PIP -> MIDDLE_FINGER_DIP -> MIDDLE_FINGER_TIP

            # 약지 손가락
            [0, 13, 14],  # WRIST -> RING_FINGER_MCP -> RING_FINGER_PIP
            [13, 14, 15], # RING_FINGER_MCP -> RING_FINGER_PIP -> RING_FINGER_DIP
            [14, 15, 16], # RING_FINGER_PIP -> RING_FINGER_DIP -> RING_FINGER_TIP

            # 새끼 손가락
            [0, 17, 18],  # WRIST -> PINKY_MCP -> PINKY_PIP
            [17, 18, 19], # PINKY_MCP -> PINKY_PIP -> PINKY_DIP
            [18, 19, 20]  # PINKY_PIP -> PINKY_DIP -> PINKY_TIP
        ]


while cap.isOpened():
    ret, frame = cap.read()
    
    frame = cv2.flip(frame, 1) # 좌우반전
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks, hand_lr in zip(results.multi_hand_landmarks, results.multi_handedness):
            finger_degrees = []
            for fp1, fp2, fp3 in finger_point:
                if1 = hand_landmarks.landmark[fp1]
                if2 = hand_landmarks.landmark[fp2]
                if3 = hand_landmarks.landmark[fp3]
                finger_degree = calcuate_angle([if1.x, if1.y], [if2.x, if2.y], [if3.x, if3.y])
                finger_degrees.append(str(int(finger_degree)))
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # cv2.putText(frame, f'index angle: {int(index_finger_degree)}', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            data = np.array([finger_degrees]).astype(np.float32) # 학습시킬 때와 똑같이 넣어주기
            ret, results, neighbours, dist = knn_model.findNearest(data, 3)
            
            if ret == 1.0:
                cv2.putText(frame, 'close', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif ret == 2.0:
                cv2.putText(frame, 'open', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'bow wow', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            with open('./test.txt', 'w') as save_result:
                save_result.write(str(int(ret)))
                
        
    cv2.imshow('hand', frame)

    if cv2.waitKey(1) & 0xFF == ord('1'):
        finger_degrees.append('1')
        data_to_csv(finger_degrees)
    elif cv2.waitKey(1) & 0xFF == ord('2'):
        finger_degrees.append('2')
        data_to_csv(finger_degrees)
    elif cv2.waitKey(1) & 0xFF == ord('3'):
        finger_degrees.append('3')
        data_to_csv(finger_degrees)   
         
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
    
cap.release()
cv2.destroyAllWindows()