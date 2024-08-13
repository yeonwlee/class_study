from fastapi import FastAPI, Form
import uvicorn
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

app = FastAPI()

# API키 가져오기
load_dotenv()

#####################################
# 1. 챗모델 만들기
chat_model = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)

# 2. 프롬프트 만들기
chat_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template('당신은 초등학교 선생님이고, 5학년을 담당하고 있습니다. 초등학교 5학년이 알아들을 수 있도록 답변해주세요.'),
        HumanMessagePromptTemplate.from_template('선생님, {question}'),
        MessagesPlaceholder(variable_name="history"),
    ]
)

memory = ConversationBufferMemory(memory_key='history', return_messages=True)

# 3. 챗모델+프롬프트 연결 체인 만들기
chat_chain = chat_prompt | chat_model
#######################################

@app.post('/teacher')
def chatbot(question:str=Form(...)):
    '''
    챗봇과의 대화
    '''
    history = memory.load_memory_variables({})['history']
    result = chat_chain.invoke({'question': question, 'history': history})
    
    # 메모리에 대화 기록 추가
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(result)
    
    return {'result': result}

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=7001)