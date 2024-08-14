from fastapi import APIRouter, status, Form, Response, Depends
from fastapi.responses import JSONResponse
from services.ai import CustomChatBot
from schemas.user import UserLoginData
from dependencies.auth import check_auth
from starlette.responses import RedirectResponse


ai_router = APIRouter()

tour_chatbot = CustomChatBot(kind_of_model='gpt-4o-mini', model_name='tour', prompt_concept='관광명소 안내', temperature=0.1, history=True)
korean_history_chatbot = CustomChatBot(kind_of_model='gpt-4o-mini', model_name='khistory', prompt_concept='한국 역사', temperature=0.1, history=False)
cook_chatbot = CustomChatBot(kind_of_model='llama3.1', model_name='cook', prompt_concept='세계 전통 요리', temperature=0.2, history=True)

# 과제4:
# xxx.xxx.xxx.xxx:7777/chatbot 으로 다음과 같은 form 데이터를 보낼 수 있도록 챗봇 기능 구현
# {chat:안녕~}
# 챗봇은 관광명소를 추천해주도록 역할을 한정해야 함. 관광명소 추천과 관련없는 질문은 차단.
@ai_router.post('/chatbot')
async def chatbot(response: Response, chat:str=Form(...), user_id:UserLoginData=Depends(check_auth)) -> str:
    return tour_chatbot.exec(chat=chat, user_id=user_id)


@ai_router.post('/chatbot/korean-history')
async def chatbot_korean_history(response: Response, chat:str=Form(...), user_id:UserLoginData=Depends(check_auth)) -> str:
    return korean_history_chatbot.exec(chat=chat, user_id=user_id)


@ai_router.post('/chatbot/cook')
async def chatbot_cook(response: Response, chat:str=Form(...), user_id:UserLoginData=Depends(check_auth)) -> str:
    return cook_chatbot.exec(chat=chat, user_id=user_id)
