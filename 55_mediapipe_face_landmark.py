import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(max_num_faces=1,
                          refine_landmarks=True,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as face_mesh:
  '''
        static_image_mode: Whether to treat the input images as a batch of static
        and possibly unrelated images, or a video stream. See details in
        https://solutions.mediapipe.dev/face_mesh#static_image_mode.
      max_num_faces: Maximum number of faces to detect. See details in
        https://solutions.mediapipe.dev/face_mesh#max_num_faces.
      refine_landmarks: Whether to further refine the landmark coordinates
        around the eyes and lips, and output additional landmarks around the
        irises. Default to False. See details in
        https://solutions.mediapipe.dev/face_mesh#refine_landmarks.
      min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face
        detection to be considered successful. See details in
        https://solutions.mediapipe.dev/face_mesh#min_detection_confidence.
      min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
        face landmarks to be considered tracked successfully. See details in
        https://solutions.mediapipe.dev/face_mesh#min_tracking_confidence.
  '''
  while cap.isOpened():
    ret, frame = cap.read()
    
    if ret == False:
      break
    
    frame = cv2.flip(frame, 1) # 좌우 반전
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    

    # 이미지가 페이스 mesh에서 계산 중인 도중에, 락을 시켜줌(데이터가 변경되지 않도록)  
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
      
        mp_drawing.draw_landmarks(
          image=image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        
        mp_drawing.draw_landmarks(
          image=image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
        )
    
    cv2.imshow('', image)
    
    if cv2.waitKey(1) & 0xFF == 27:
      break
    
cap.release()
cv2.destroyAllWindows()