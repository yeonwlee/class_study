# pip install gTTS


from gtts import gTTS

tts = gTTS(text='안녕하세요', lang='ko')
tts.save('test.mp3')

# return FileResponse(file_path, media_type='audio/mpeg') 오디오 돌려주고 싶으면 이런 느낌으로 주면 됨


