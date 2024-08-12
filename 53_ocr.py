# pip install easyocr

import easyocr
import cv2

reader = easyocr.Reader(['ko', 'en'])
result = reader.readtext('./data/min.png')

for box, text, conf in result:
    if float(conf) > 0.5:
        print(text)