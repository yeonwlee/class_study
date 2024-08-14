from schemas.user import UserLoginData, Token
import bcrypt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt
from constants.config import TokenConfig

# CryptContext에서 deprecated될 시, 자동으로 새로운 알고리즘으로 마이그레이션하도록.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


INMEMORY_USER_DB = {
    'meta': UserLoginData(
        id='meta',
        pw=pwd_context.hash('ai'),
        token=None
    ),
    'yeonw': UserLoginData(
        id='yeonw',
        pw=pwd_context.hash('ai'),
        token=None
    ),
}

def is_valid_userinfo(id:str, pw:str) -> UserLoginData:
    user_data = INMEMORY_USER_DB.get(id)
    if user_data and pwd_context.verify(pw, user_data.pw):
        return user_data
    return None


def generate_token(login_data: UserLoginData) -> None:
    expire_time: datetime = datetime.utcnow() + timedelta(minutes=TokenConfig.TOKEN_EXPIRE_MINUTES)
    payload = {
        "id": login_data.id,
        "exp": expire_time
    }
    token:str=jwt.encode(payload, TokenConfig.SECRET_KEY, TokenConfig.ALGORITHM)
    login_data.token = Token(access_token=token, token_type="bearer")
