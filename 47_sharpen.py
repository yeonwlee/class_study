import cv2
import numpy as np

src = cv2.imread('./data/gerbera jamesonii.jpg', cv2.COLOR_BGR2YCrCb) # 컬러 이미지 샤픈

src_float = src[:,:,0].astype(np.float32)
blr = cv2.GaussianBlur(src_float, (0, 0), 6.)
src[:,:,0] = np.clip(2. * src_float - blr, 0, 255).astype(np.uint8)

dst = cv2.cvtColor(src, cv2.COLOR_YCrCb2BGR)
cv2.imshow('origin', src)
cv2.imshow('sharpen', dst)
cv2.waitKey()
cv2.destroyAllWindows()