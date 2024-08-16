from fastapi import FastAPI
from routers import greeting, user, ai_chat, ai_classification
import uvicorn
import ssl
import os
from dependencies import database


app = FastAPI()
app.include_router(greeting.greeting_router)
app.include_router(user.user_router, prefix='/login')
app.include_router(ai_chat.ai_chat_router, prefix='/chatbot')
app.include_router(ai_classification.ai_classification_router, prefix='/similar')

# 인증서 관련 오류 방지
ssl._create_default_https_context = ssl._create_unverified_context

# 과제1: Port번호 7777로 접속할 수 있도록 FastAPI 서버 구현
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=7777)