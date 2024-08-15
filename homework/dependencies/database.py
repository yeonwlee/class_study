from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session
from fastapi import Depends
from schemas.ai_chat import Base
import os

print(os.getcwd())
DATABASE_URL = 'sqlite:///C:/mtoc/homework/chatbot.db'

# SQLite 데이터베이스 엔진 생성
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# 데이터베이스 테이블 생성
Base.metadata.create_all(bind=engine)

# 세션 로컬 클래스 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_session(db: Session = Depends(get_db)):
    return db