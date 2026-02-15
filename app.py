from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])

        features = np.array([[age, sex, cp, trestbps, chol, fbs,
                              restecg, thalach, exang, oldpeak,
                              slope, ca, thal]])

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)

        if prediction[0] == 1:
            result = "⚠️ High Risk: Patient likely has Heart Disease"
        else:
            result = "✅ Low Risk: Patient likely has No Heart Disease"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text="Error: " + str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
