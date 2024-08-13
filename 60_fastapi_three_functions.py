from fastapi import FastAPI, Form
import uvicorn
from pydantic import BaseModel
# from langchain_community.llms import OpenAI
from langchain.chat_models import ChatOpenAI
# from langchain_openai import ChatOpenAI
# from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from dotenv import load_dotenv
import os

app = FastAPI()

# .env 파일로 API키를 관리하도록 했어요
load_dotenv()
api_key = os.getenv('OPEN_API_KEY')



# 1. 챗모델 만들기
chat_model = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.1, api_key=api_key)
# 2. 프롬프트 만들기
# 3. 챗+프롬프트 연결 체인 만들기
# 4. 체인에 프롬프트 넣기

class Image(BaseModel):
    name:str
    file:str
    
    
@app.post('/chatbot')
def chatbot(talk:str=Form(...)):
    '''
    이런 식으로 주석을 주면 redoc에서 이 내용이 보입니다.\n
    이렇게 대충 달지 맙시다\n
    챗봇과의 대화
    '''
    messages = [SystemMessage(content='너는 5살짜리 여자아이야. 한국에 살고 있고, 활달한 편이야.') ,
            HumanMessage(content=talk)]
    result = chat_model(messages)
    return {'result': result}


@app.get('/readitem/{id}')
def readitem(id):
    return id


@app.post('/login')
def login(id:str=Form(...), pw:str=Form(...)):
    return {'id':id, 'pw':pw}


@app.post('/upload/')
def upload(image: Image):
    '''
    이미지를 서버에 업로드함
    '''
    return {'result': '업로드 성공'}


@app.get('/aiimage')
def aiimage():
    '''
    이미지를 받아서 해당 이미지를 통해 연예인을 분류함
    '''
    return {'result': '카리나'}


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=7000)