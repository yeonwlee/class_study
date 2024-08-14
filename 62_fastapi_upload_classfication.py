from fastapi import FastAPI, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
from pydantic import BaseModel
# from langchain_community.llms import OpenAI
from langchain.chat_models import ChatOpenAI
# from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import ssl


app = FastAPI()

# .env 파일로 API키를 관리하도록 했어요
load_dotenv()

server_model_path = './models/cel_class_best_model.pth'
server_image_path = './image'
os.makedirs(name=server_image_path, exist_ok=True)


# 인증서 관련 오류 방지
ssl._create_default_https_context = ssl._create_unverified_context

# 모델 불러오기 및 가중치 적용
model = models.resnet34(pretrained=True)
model.fc = nn.Linear(512, 3) # 3개만 분류할거라서 마지막 분류기 부분 수정
model_state_dict = torch.load(server_model_path, map_location='cpu')
model.load_state_dict(model_state_dict)


# 이미지 전처리용
transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])


model.eval()

#####################################
# 1. 챗모델 만들기
chat_model = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.1)

# 2. 프롬프트 만들기
chat_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template('너는 5살짜리 여자 아이야. 한국에 살고 있고, 활발한 성격이야. 5살 짜리 동갑내기가 알아들을 수 있게 답변 해줘'),
        HumanMessagePromptTemplate.from_template('{history}'),
        HumanMessagePromptTemplate.from_template('{input}')
    ]
)

# 버퍼
memory = ConversationBufferMemory()

# 3. 챗모델+프롬프트 연결 체인 만들기
chat_chain = ConversationChain(
    llm=chat_model,
    prompt=chat_prompt,
    memory=memory
)

#######################################

class Chat(BaseModel):
    type:str
    talk:str
    
    
@app.post('/chatbot')
async def chatbot(talk:str=Form(...)):
    '''
    이런 식으로 주석을 주면 redoc에서 이 내용이 보입니다.\n
    이렇게 대충 달지 맙시다\n
    챗봇과의 대화
    '''
    result = chat_chain.run(input=talk)
    return {'result': result}


@app.get('/readitem/{id}')# 이런 식으로 하면 뒤에 오는 값을 받을 수 있음
async def readitem(id:str):
    return id


@app.post('/login')
async def login(id:str=Form(...), pw:str=Form(...)):
    return {'id':id, 'pw':pw}


@app.post('/upload')
async def upload(file:UploadFile): # 서버에 부하 주지 않게 체크도 해주자. ex. 파일형식
    '''
    이미지를 서버에 업로드함
    '''
    if not file.filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
        raise HTTPException(status_code=400, detail='지원되지 않는 파일 형식입니다,')
    
    file_path = os.path.join(server_image_path, file.filename)
    try:
        with open(file_path, 'wb') as buffer: # 왜 저장하느냐. 이것도 학습용으로 야무지게 챙길 예정.
            buffer.write(await file.read())
    except:
        raise HTTPException(status_code=500, detail='파일 처리 중에 오류가 발생했습니다.')
    return FileResponse(file_path, media_type='image/jpeg')

@app.post('/mediapipe')
async def mediapipe():
    with open('./test.txt', 'r') as file:
        text = file.read()
    return JSONResponse(content={'result': text}, status_code=200)


@app.post('/aicnn')
async def aicnn(file:UploadFile):
    if not file.filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
        raise HTTPException(status_code=400, detail='지원되지 않는 파일 형식입니다,')
    
    file_path = os.path.join(server_image_path, file.filename)
    try:
        with open(file_path, 'wb') as buffer:
            buffer.write(await file.read())
        
        image = Image.open(file_path)
    except:
        raise HTTPException(status_code=500, detail='파일 처리 중 오류가 발생했습니다.')
    
    # model.eval() # 위에서 해줌
    try:
        with torch.no_grad():
            image = transform_test(image).unsqueeze(0).to('cpu') # 배치 단위로 실행하기 위해 차원을 하나 늘려줌
            output = model(image)
            pred = torch.max(output, 1)[1]
            class_name = ['마동석', '이도현', '팜하니']
            result = class_name[pred[0]]
            return JSONResponse(content={'result': result}, status_code=200, detail='분류에 성공했습니다. 여긴 이연우 PC입니다')
    except:
        raise HTTPException(status_code=500, detail='이미지 분류 중 오류가 발생했습니다.')
    
@app.get('/aiimage')
async def aiimage():
    '''
    이미지를 받아서 해당 이미지를 통해 연예인을 분류함
    '''
    return {'result': '카리나'}


@app.post('/jsontest')
async def jsontest(chatbot:Chat): # 위의 클래스 참고. 이렇게 해주면 데이터 타입이 맞는지 아닌지도 자동 체크해줍니다. ex. type 에 숫자 넣어보내면 오류남
    return {'type': chatbot.type, 'talk': chatbot.talk}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=7000)