from sqlalchemy.orm import Session
from schemas.ai_chat import ChatHistory

def save_message_to_db(db: Session, session_id: str, user_id: str, model_name:str, role: str, message: str):
    db_message = ChatHistory(session_id=session_id, user_id=user_id, model_name=model_name, role=role, message=message)
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message

def get_chat_history(db: Session, session_id: str):
    return db.query(ChatHistory).filter(ChatHistory.session_id == session_id).all()
