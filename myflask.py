from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import xgboost

app = Flask(__name__)
xgb_model = joblib.load('bestxgbLunchModel.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    inputs = request.form
    inputs = inputs.to_dict(flat=True)
    inputs = pd.DataFrame(inputs,index=[0])
    y = xgb_model.predict(inputs)
    return render_template('result.html', y= y[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)