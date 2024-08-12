import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # bgr -> rgb로 꼭 바꿔줘야함
    results = face_detection.process(img_rgb)
    
    # 여러 얼굴을 검출하려면 반복문으로 해야함
    # bbox = results.detections[0].location_data.relative_bounding_box
    # ih, iw, _ = img_rgb.shape
    # x = int(bbox.xmin * iw)
    # y = int(bbox.ymin * ih)
    # w = int(bbox.width * iw)
    # h = int(bbox.height * ih)
    
    # face = img_rgb[y:y+h, w:x+w]
    # face = cv2.resize(face, (10, 10), interpolation=cv2.INTER_LINEAR)
    # face = cv2.resize(face (w, h), interpolation=cv2.INTER_LINEAR)
    # img_rgb[y:y+h, x:x+w] = face    
    # img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    if results.detections:
        for object_index in range(len(results.detections)):
            bbox = results.detections[object_index].location_data.relative_bounding_box # x, y, w, h    
            ih, iw, _ = img_rgb.shape # 원본 이미지의 사이즈 가져옴
            x = int(bbox.xmin * iw) # 실제 좌표를 계산
            y = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)
            
            # 이미지 경계를 넘지 않도록 크기 조정
            x_end = min(x + w, iw)
            y_end = min(y + h, ih)
            w = x_end - x
            h = y_end - y
            
            face = img_rgb[y:y+h, x:x+w] # 얼굴의 영역만큼 잘라서 가져온 다음에
            if face.size:
                face = cv2.resize(face, (10, 10), interpolation=cv2.INTER_LINEAR) # 축소 후
                face = cv2.resize(face, (w, h), interpolation=cv2.INTER_LINEAR) # 다시 확대
                img_rgb[y:y+h, x:x+w] = face # 원래 부위에 붙여넣기
            print(bbox)
        
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    # print(results.detections)
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)

    cv2.imshow('face detect', img_rgb)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()


'''
[label_id: 0
score: 0.835087895
location_data {
  format: RELATIVE_BOUNDING_BOX
  relative_bounding_box {
    xmin: 0.498628676       * 실제 이미지의 width
    ymin: 0.364936709       * 실제 이미지의 height 
    width: 0.203072369      * 실제 이미지의 width
    height: 0.270763457     * 실제 이미지의 height 
  }

'''