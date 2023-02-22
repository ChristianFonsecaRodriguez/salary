import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def hola():
    return {'message':'Hola Mundo'}

@app.get('/{name}')
def get_name(name:str):
    return {'message':f'Hola {name}'}

@app.post('/guardar')
def savenames(name:str, lastname:str):
    return f'Se ha guardado a {name} {lastname}'

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)