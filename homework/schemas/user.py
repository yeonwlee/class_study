from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Token(BaseModel):
    access_token: str
    token_type: str
    
    
class UserLoginData(BaseModel):
    id:str
    pw:str
    token:Optional[Token]
    
    
