from fastapi import APIRouter, status

greeting_router = APIRouter()

# 과제2: 홈 화면 접속시 안녕하세요~ 출력
@greeting_router.get('/', status_code=status.HTTP_200_OK)
async def home() -> str:
    return '안녕하세요~'