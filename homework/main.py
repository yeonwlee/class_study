from fastapi import FastAPI
from routers import greeting, user, ai

import uvicorn

app = FastAPI()
app.include_router(greeting.greeting_router)
app.include_router(user.user_router)
app.include_router(ai.ai_router)

# 과제1: Port번호 7777로 접속할 수 있도록 FastAPI 서버 구현
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=7777)