import os
import re
import time
import logging
import inspect
import aiofiles  # 비동기 파일 작업을 위한 라이브러리

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse


valid_models = {'resnet34'}
valid_model_names = {'celebrity_classification'}
server_model_path = './models'
server_image_path = './image'

os.makedirs(name=server_image_path, mode=0o777, exist_ok=True)


class ImageClassificationModel():
    def __init__(self, kind_of_model: str, model_name: str, classes: list[str], device: str='cuda' if torch.cuda.is_available() else 'cpu', ) -> None:
        assert kind_of_model in valid_models, f'모델의 종류는 {valid_models} 중 선택하세요'
        
        self.device = device
        self.classes = classes
        
        if kind_of_model == 'resnet34' and model_name == 'celebrity_classification':
            self.model = models.resnet34(pretrained=True)
            self.model.fc = nn.Linear(512, 4)
            
            model_file_path = os.path.join(server_model_path, model_name, f'{kind_of_model}.pth')
            model_state_dict = torch.load(model_file_path, map_location=self.device)
            self.model.load_state_dict(model_state_dict)
            
            # 입력들어올 이미지 전처리용
            self.transform_test = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
            
            self.image_file_path = os.path.join(server_image_path, model_name)
            
        # 평가모드로 전환    
        self.model.eval()
    
    
    def _validate_image_type(self, image: UploadFile, valid_file_format: set[str]={'PNG', 'JPEG', 'JPG', 'GIF'}) -> None:
        try:
            img = Image.open(image.file)
            img.verify()  # 이미지가 손상되었거나 잘못된 형식이라면 여기서 예외가 발생.
            if img.format not in valid_file_format: # 원하는 형식만 허용
                logging.warning(f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}: 이미지 형식이 유효하지 않습니다. 허용된 형식: {valid_file_format}, 입력된 형식: {img.format}")
                raise HTTPException(status_code=400, detail=f'파일 형식은 {valid_file_format}만 지원합니다')
        except (IOError, SyntaxError) as e:
            logging.warning(f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}: 이미지가 손상되었거나 잘못된 형식입니다. 오류 메시지: {str(e)}")
            raise HTTPException(status_code=400, detail=f'이미지가 손상되었거나 형식이 잘못 되었습니다.')
        finally:
            image.file.seek(0)  # 파일 포인터를 원래 위치로 초기화
    
    
    #기본 max_size는 10mb
    def _validate_file_size(self, image: UploadFile, max_size: int=10*1024*1024) -> None:
        file_size = os.fstat(image.file.fileno()).st_size
        if file_size > max_size:
            logging.warning(f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}: 파일 크기 초과. 허용된 최대 크기: {max_size / (1024 * 1024):.2f}MB, 실제 크기: {file_size / (1024 * 1024):.2f}MB")
            raise HTTPException(status_code=400, detail=f'파일의 크기는 {max_size / (1024 * 1024):.2f}MB 보다 작아야 합니다')
    
    
    def _sanitize_filename(self, image_name: str) -> str:
        # 파일 이름에서 허용되지 않는 문자를 필터링
        return re.sub(r'[^a-zA-Z0-9_.-]', '_', image_name)
    
    
    def _generate_unique_filename(self, image_name: str) -> str:
        # 현재 타임스탬프를 기반으로 고유한 파일 이름 생성
        ext = os.path.splitext(image_name)[1]
        timestamp = int(time.time())
        unique_filename = f"{timestamp}{ext}"
        return unique_filename
    
    
    async def upload(self, image: UploadFile) -> None:
        self._validate_image_type(image)
        self._validate_file_size(image)
        os.makedirs(self.image_file_path, mode=0o777, exist_ok=True)
        unique_filename = self._generate_unique_filename(self._sanitize_filename(image.filename))
        file_path = os.path.join(self.image_file_path, unique_filename)
        try:
            image.file.seek(0)  # 파일 포인터를 원래 위치로 초기화
            async with aiofiles.open(file_path, 'wb') as buffer:
                content = await image.read()
                await buffer.write(content)
        except OSError as e:
            logging.error(f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}: 파일 저장 중 오류 발생: {str(e)}")
            raise HTTPException(status_code=500, detail=f'파일 처리 중 오류가 발생했습니다: {str(e)}')
        
        
    def _preprocess_image(self, image: UploadFile) -> Image:
        image = Image.open(image.file)
        if image.mode == 'L' or image.mode == 'RGBA':
            image = image.convert('RGB')
        image = self.transform_test(image).unsqueeze(0).to(self.device) # 배치 단위로 실행하기 위해 차원을 하나 늘려줌
        return image
    
    
    async def exec(self, image: UploadFile) -> JSONResponse:
        try:
            with torch.no_grad():
                image = self._preprocess_image(image)
                output = self.model(image)
                pred = torch.max(output, 1)[1]
                result = self.classes[pred[0]]
                return JSONResponse(content={'result': result}, status_code=200)
        except (IOError, ValueError) as e:
                raise HTTPException(status_code=400, detail=f'유효한 이미지를 입력해 주세요: {str(e)}')
        except Exception as e:
            logging.critical(f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}: 이미지 분류 중 오류가 발생했습니다. 오류 메시지: {str(e)}")
            raise HTTPException(status_code=500, detail=f'이미지 분류 중 오류가 발생했습니다: {str(e)}')
        
        
        
class CelebrityImageClassificationModel(ImageClassificationModel):
    def __init__(self, kind_of_model:str) -> None:
        super().__init__(kind_of_model, 'celebrity_classification', ['김혜자', '마동석', '차은우', '카리나'])
    
        
    def _validate_image_type(self, image: UploadFile) -> None:
        super()._validate_image_type(image, valid_file_format={'PNG', 'JPEG', 'JPG'})


    def _validate_file_size(self, image: UploadFile) -> None:
        super()._validate_file_size(image, max_size=3*1024*1024) # 기본 용량 제한 3MB
        
    
   
            
