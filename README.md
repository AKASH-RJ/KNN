# Credit Card Fraud Detection using KNN & Flask

## Overview

This project detects whether a **credit card transaction** is **Fraud** or **Not Fraud** using the **K-Nearest Neighbors (KNN)** algorithm. The model is trained on a dataset of 200 transactions and deployed as a **Flask web application** with HTML & CSS.

-----

## Features

  - **KNN Classifier** for detecting fraudulent transactions.
  - **Flask backend** for serving predictions.
  - **Interactive HTML form** to input transaction details.
  - **CSS styling** for a clean and user-friendly interface.
  - **CSV dataset** with 200 labeled transactions.

-----

## Project Structure

```
credit_card_fraud_knn/
│
├── model.py             # Trains and saves the KNN model
├── app.py               # Flask application for predictions
├── templates/
│   ├── index.html       # Main input form
│   └── result.html      # Displays prediction result
├── static/
│   └── style.css        # CSS for styling
├── dataset.csv          # Dataset (200 credit card transactions)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

-----

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```
Flask==3.0.0
pandas==2.1.4
scikit-learn==1.3.2
numpy==1.26.2
```

-----

## Dataset

The dataset (`dataset.csv`) contains credit card transactions labeled as Fraud or Not Fraud.

Example:

```
amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,label
1200,1500,300,0,1200,Not Fraud
5600,6000,400,2000,7600,Fraud
```

Features:

  - `amount`: Transaction amount.
  - `oldbalanceOrg`: Sender's balance before the transaction.
  - `newbalanceOrig`: Sender's balance after the transaction.
  - `oldbalanceDest`: Receiver's balance before the transaction.
  - `newbalanceDest`: Receiver's balance after the transaction.
  - `label`: Fraud or Not Fraud.

-----

## How It Works

### Model Training (`model.py`)

  - Loads dataset from `dataset.csv`.
  - Prepares features and labels.
  - Trains a KNN classifier.
  - Saves the trained model as `model.pkl`.

### Web Application (`app.py`)

  - Loads `model.pkl`.
  - Accepts transaction details from the HTML form.
  - Predicts Fraud or Not Fraud.
  - Displays the result to the user.

-----

## Running the Project

1.  **Train the Model**
    ```bash
    python model.py
    ```
2.  **Run Flask App**
    ```bash
    python app.py
    ```
3.  **Open in Browser**
    Go to: `http://127.0.0.1:5000/`

-----

## Screenshots
---
Home Page

<img width="661" height="476" alt="Screenshot 2025-08-12 120419" src="https://github.com/user-attachments/assets/ab714bd6-ad00-4b2d-b99e-52e07351e0da" />

---
Prediction Result

<img width="624" height="525" alt="Screenshot 2025-08-12 120432" src="https://github.com/user-attachments/assets/f7a92d9e-e636-4490-be31-1780f14dbb8a" />
