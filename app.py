from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("knn_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        amount = float(request.form['amount'])
        time = float(request.form['time'])
        location = int(request.form['location'])
        prev_fraud = int(request.form['prev_fraud'])

        features = np.array([[amount, time, location, prev_fraud]])
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]
        result = "Fraud" if pred == 1 else "Not Fraud"

        return render_template("index.html", prediction=result)
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
