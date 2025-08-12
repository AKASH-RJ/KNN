import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("knn.csv")

# Features & target
X = df[['transaction_amount', 'transaction_time', 'location_mismatch', 'num_prev_frauds']]
y = df['label'].map({'Not Fraud': 0, 'Fraud': 1})

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_scaled, y)

# Save
joblib.dump(model, "knn_fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved.")
