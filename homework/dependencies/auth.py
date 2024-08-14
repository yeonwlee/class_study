from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt

from constants.config import TokenConfig
from schemas.user import UserLoginData, Token

#의존성 주입
oauth2_scheme: OAuth2PasswordBearer = OAuth2PasswordBearer(tokenUrl='login')

def check_auth(token:str=Depends(oauth2_scheme)) -> UserLoginData:
    try:
        payload = jwt.decode(token, TokenConfig.SECRET_KEY, algorithms=[TokenConfig.ALGORITHM])
        print(payload)
        if 'exp' not in payload or payload['exp'] - datetime.now().timestamp() < 0:
            raise Exception
        return payload['id']
    except:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='유효하지 않은 토큰')