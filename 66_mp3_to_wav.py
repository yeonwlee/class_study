# pip install pydub
from pydub import AudioSegment
import os
import speech_recognition as sr

file_name = 'test.wav'
if not os.path.exists(file_name):
  audio = AudioSegment.from_mp3('test.mp3')
  audio.export(file_name, format='wav')

# C:\mtoc\mtoc\Lib\site-packages\pydub\utils.py:170: RuntimeWarning: Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work
#   warn("Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work", RuntimeWarning)
# C:\mtoc\mtoc\Lib\site-packages\pydub\utils.py:198: RuntimeWarning: Couldn't find ffprobe or avprobe - defaulting to ffprobe, but may not work
#   warn("Couldn't find ffprobe or avprobe - defaulting to ffprobe, but may not work", RuntimeWarning)
# .....
# FileNotFoundError: [WinError 2] 지정된 파일을 찾을 수 없습니다

# ---> ffmpeg 다운로드받기

# https://www.gyan.dev/ffmpeg/builds/
# git master builds 하위의 ffmpeg-git-essentials.7z 다운로드 - 환경변수 설정 해주기(ffmpeg/bin)

recognizer = sr.Recognizer()
input_audio = sr.AudioFile(file_name)

with input_audio as source:
  audio = recognizer.record(source)
  
result_text = recognizer.recognize_google(audio_data=audio, language='ko-KR')
print(result_text)