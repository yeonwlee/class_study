from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# 키 값 직접 관리하고 있다면 OpenAI(api_key=키값)
client = OpenAI()

## 소리 인식 -> text
audio_file = open('test.mp3', 'rb')
result = client.audio.transcriptions.create(
  file=audio_file,
  model='whisper-1',
  response_format='text',
  temperature=0.0,
)

print(result)

## tts
response = client.audio.speech.create(
  model='tts-1',
  input='오늘은 즐거운 금요일입니다.',
  voice= 'nova',
  response_format='mp3',
  speed=1.1
)

response.stream_to_file('whisper_result.mp3')

