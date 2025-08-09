from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

app = Flask(__name__)

# Load dataset
df = pd.read_csv("knn.csv")

# Encode categorical variables
label_encoders = {}
for column in ['Transaction_Type', 'Location', 'Time']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Features & Target
X = df[['Amount', 'Transaction_Type', 'Location', 'Time']]
y = df['Is_Fraud'].map({'No': 0, 'Yes': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Save model & encoders
pickle.dump(model, open('knn_model.pkl', 'wb'))
pickle.dump(label_encoders, open('label_encoders.pkl', 'wb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model = pickle.load(open('knn_model.pkl', 'rb'))
    label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

    amount = float(request.form['amount'])
    transaction_type = label_encoders['Transaction_Type'].transform([request.form['transaction_type']])[0]
    location = label_encoders['Location'].transform([request.form['location']])[0]
    time = label_encoders['Time'].transform([request.form['time']])[0]

    prediction = model.predict([[amount, transaction_type, location, time]])[0]
    result = 'Fraud' if prediction == 1 else 'Not Fraud'
    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
