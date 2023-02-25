from flask import Flask, render_template, request, url_for, jsonify
from pycaret.classification import *
import pandas as pd
import numpy as np


model = load_model('salary')
cols  = ['age', 'fnlwgt', 'education-num', 
         'capital-gain', 'capital-loss', 
         'hours-per-week']

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    features_array = np.array(features)
    data2predict = pd.DataFrame([features_array], columns=cols)
    prediction = predict_model(model, data=data2predict)
    prediction = prediction['Score'][0]
    return render_template('home.html', pred=prediction)

@app.rounte('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data = pd.DataFrame([data])
    prediction = predict_model(model, data=data)
    prediction = prediction['Score'][0]
    return jsonify(prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
