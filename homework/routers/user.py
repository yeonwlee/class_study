from fastapi import APIRouter, status, Form, Response
from services.user import is_valid_userinfo, generate_token
from schemas.user import UserLoginData

user_router = APIRouter()

# # 과제3: xxx.xxx.xxx.xxx:7777/login 으로 다음과 같은 form데이터를 보내어 로그인 기능 구현
# # {id:meta, pw:ai}
# # 정확히 입력하면 "로그인 성공"을 return
# # 틀리면 "로그인 실패"를 return
@user_router.post('/login')
async def login(response: Response, id:str=Form(...), pw:str=Form(...)) -> str:
    if (user:=is_valid_userinfo(id, pw)):
        generate_token(user) # 토큰 생성
        response.status_code = status.HTTP_200_OK
        return f'로그인 성공'
    else:   
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return '로그인 실패'