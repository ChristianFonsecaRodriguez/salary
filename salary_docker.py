
import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model('salary_docker')

# Define predict function
@app.post('/predict')
def predict(age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week):
    data = pd.DataFrame([[age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week]])
    data.columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    predictions = predict_model(model, data=data) 
    return {'prediction': list(predictions['Label'])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)