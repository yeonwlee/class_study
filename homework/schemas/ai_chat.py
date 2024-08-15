from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = 'chat_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    user_id = Column(String, index=True)
    model_name = Column(String)
    role = Column(String)  # 'user' / 'ai'
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

    def as_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "model_name": self.model_name,
            "role": self.role,
            "message": self.message,
            "timestamp": self.timestamp.isoformat()  # datetime을 ISO 8601 형식으로 변환
        }

    def __repr__(self):
        return f"<ChatHistory(session_id='{self.session_id}', user_id='{self.user_id}', model_name='{self.model_name}', role='{self.role}', message='{self.message}', timestamp='{self.timestamp}')>"
