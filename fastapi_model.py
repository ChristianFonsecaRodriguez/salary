import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
import uvicorn
from fastapi import FastAPI

# 1er Paso
app = FastAPI()

# 2do Paso
model = load_model('salary')

# 3er Paso
@app.post('/predict')
def predict(age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week):
    data = pd.DataFrame([[age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week]])
    data.columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    prediction = predict_model(model, data=data)
    return {'prediction': np.round(prediction['Score'][0],4)}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)