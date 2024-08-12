import cv2

cap = cv2.VideoCapture(0) # 카메라

width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(width, height)

codec = cv2.VideoWriter_fourcc(*'DIVX') # 코덱 설정?
# codec = cv2.VideoWriter.fourcc(*'DIVX') # 코덱 설정?

fps = cap.get(cv2.CAP_PROP_FPS) # fps 값 가져오기
out = cv2.VideoWriter('out2.avi', codec, fps,(width, height))

count = 0
while True:
    ret, frame = cap.read()
    cv2.imshow('video', frame)
    out.write(frame) # 쓰기
    
    if count % 30 == 0:
        cv2.imwrite(f'frame_{count}.jpg', frame)
    count += 1
    
    if cv2.waitKey(1) == 27: # 'esc'
        break
    
out.release()
cap.release()
cv2.destroyAllWindows()