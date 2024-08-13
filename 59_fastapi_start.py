from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get('/hello')
async def root():
    '''
    안녕 월드를 리턴함
    '''
    print('허허')
    return {'message': '안녕 월드'}


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)