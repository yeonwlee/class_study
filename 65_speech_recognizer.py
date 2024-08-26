# pip install SpeechRecognition, PyAudio
import speech_recognition as sr

recognizer = sr.Recognizer()

# 
script = '초기에는 간단한 규칙 기반 시스템에서 시작해, 이후에 머신러닝과 딥러닝으로 발전했습니다. \
            최근에는 자율주행, 자연어 처리, 컴퓨터 비전 등 다양한 분야에서 활용되고 있습니다.'
#
while True:
    with sr.Microphone() as source: # 마이크로 들어오는 소리 인식
        print('음성을 입력하세요')
        input_audio = recognizer.listen(source)
        txt = recognizer.recognize_google(input_audio, language='ko-KR')
        # write audio to a WAV file
        with open("microphone-results.wav", "wb") as f:
            f.write(input_audio.get_wav_data())        
            print(txt)
        