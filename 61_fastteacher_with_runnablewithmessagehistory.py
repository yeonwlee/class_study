from fastapi import FastAPI, Form
import uvicorn
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

app = FastAPI()

# API키 가져오기
load_dotenv()


#### 참고 링크: https://wikidocs.net/235581
# 세션 기록을 저장할 딕셔너리
store = {}  

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    print(session_ids)
    if session_ids not in store:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


#####################################
# 1. 챗모델 만들기
chat_model = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)

# 2. 프롬프트 만들기
chat_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template('당신은 초등학교 선생님이고, 5학년을 담당하고 있습니다. 초등학교 5학년이 알아들을 수 있도록 답변해주세요.'),
        HumanMessagePromptTemplate.from_template('선생님, {question}'),
        MessagesPlaceholder(variable_name='history')
    ]
)

memory = ConversationBufferMemory(memory_key='history', return_messages=True)

# 3. 챗모델+프롬프트 연결 체인 만들기
chat_chain = chat_prompt | chat_model

with_message_history = (
    RunnableWithMessageHistory(  # RunnableWithMessageHistory 객체 생성
        chat_chain,  # 실행할 Runnable 객체
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 입력 메시지의 키
        history_messages_key="history",  # 기록 메시지의 키
    )
)
#######################################

@app.post('/teacher')
def chatbot(question:str=Form(...)):
    '''
    챗봇과의 대화
    '''
    result = with_message_history.invoke(
                                        {'question': question},
                                         config={"configurable": {"session_id": "123"}}
                                         )
    return {'result': result}

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=7001)