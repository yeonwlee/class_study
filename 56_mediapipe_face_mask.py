import cv2
import mediapipe as mp
import time
import numpy as np

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh

faceMesh = mpFaceMesh.FaceMesh()

faceimg = cv2.imread('./data/face_mk.png', cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(0)

def face_overlay(background_img, img_to_overlay, x, y, overlay_size=None):
    try:
        bg_img = background_img.copy()
        if bg_img.shape[2] == 3:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)
        
        if overlay_size is not None:
            img_to_overlay = cv2.resize(img_to_overlay.copy(), overlay_size)
            
            
        b, g, r, a = cv2.split(img_to_overlay)

        mask = cv2.medianBlur(a, 5)
        
        h, w, _ = img_to_overlay.shape
    
        roi = bg_img[int(y - h/2) : int(y + h/2), int(x - w/2) : int(x + w/2)]
        
        img1_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
        img2_fg = cv2.bitwise_and(img_to_overlay.copy(), img_to_overlay.copy(), mask=mask)
        
        bg_img[int(y - h/2):int(y + h/2), int(x - w/2):int(x + w/2)] = cv2.add(img1_bg, img2_fg)    

        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)
        
        return bg_img
    
    except:
        return background_img

while True:
    ret, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faceMesh.process(imgRGB)
    ih, iw, ic = img.shape
    
    if result.multi_face_landmarks:
        for faceLms in result.multi_face_landmarks:
            
            xy_point = []
            
            for c, lm in enumerate(faceLms.landmark):
                xy_point.append([lm.x, lm.y])
                img = cv2.circle(img, (int(lm.x * iw), int(lm.y * ih)), 1, (255, 0, 0), 2)
                
            
            top_left = np.min(xy_point, axis=0)
            bottom_right = np.max(xy_point, axis=0)
            mean_xy = np.mean(xy_point, axis=0)
            
            cv2.circle(img, (int(mean_xy[0] * iw), int(mean_xy[1] * ih)), 4, (0, 0, 255), 3)
            
            face_width = int(bottom_right[0] * iw) - int(top_left[0] * iw)
            face_height = int(bottom_right[1] * iw) - int(top_left[1] * iw)
            
            if face_width > 0 and face_height > 0 :
                #마스크 오버레이 코드
                face_result = face_overlay(img, faceimg, int(mean_xy[0] * iw), int(mean_xy[1] * ih), (face_width, face_height))
                
    else:
        face_result = img
                
    cv2.imshow('face', face_result)

    if cv2.waitKey(1) == ord('q'):
        break 
    
cap.release()
cv2.destroyAllWindows()