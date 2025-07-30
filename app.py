from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('insurence_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        sex = int(request.form['sex'])  # 0: female, 1: male
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])  # 0: no, 1: yes
        region = int(request.form['region'])  # 0: southeast, 1: southwest, 2: northeast, 3: northwest

        input_data = np.array([[age, sex, bmi, children, smoker, region]])
        prediction = model.predict(input_data)[0]
        prediction = round(prediction, 2)

        return render_template('index.html', prediction_text=f'Estimated Insurance Charge: ${prediction}')
    except:
        return render_template('index.html', prediction_text='Invalid Input. Please check your values.')

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
