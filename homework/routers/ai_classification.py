from fastapi import APIRouter, status, Response, Depends, UploadFile
from fastapi.responses import JSONResponse
from schemas.user import UserLoginData
from services.ai_classification import CelebrityImageClassificationModel
from dependencies.auth import check_auth
from starlette.responses import RedirectResponse


ai_classification_router = APIRouter()

celebrity_classification_model = CelebrityImageClassificationModel('resnet34')

# 과제5:
# xxx.xxx.xxx.xxx:7777/similar 으로 다음과 같은 이미지 데이터를 보낼 수 있도록 연예인 얼굴 분류 기능 구현
# {imgFile: image.jpg}
# 파일 형식은 jpg, jpeg, png를 받을 수 있도록 설정
# 얼굴은 차은우, 카리나, 마동석, 김혜자
@ai_classification_router.post('')
async def similar(response: Response, imgFile:UploadFile, user_id:UserLoginData=Depends(check_auth)) -> JSONResponse:
    await celebrity_classification_model.upload(imgFile)
    return await celebrity_classification_model.exec(imgFile)
