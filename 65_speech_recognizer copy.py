# pip install SpeechRecognition, PyAudio
import speech_recognition as sr

recognizer = sr.Recognizer()

while True:
    with sr.Microphone() as source: # 마이크로 들어오는 소리 인식
        print('음성을 입력하세요')
        input_audio = recognizer.listen(source)
        txt = recognizer.recognize_google(input_audio, language='ko-KR')
        print(txt)
        