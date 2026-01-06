import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

# 1. Load Dataset
# Pastikan file diabetes_cleaning.csv ada di folder yang sama
df = pd.read_csv("diabetes_cleaning.csv")

# Sesuaikan target column (contoh di sini 'Outcome')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. MLflow Tracking
mlflow.set_experiment("Diabetes_CI_Experiment")

with mlflow.start_run() as run:
    # Train Model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict & Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model Accuracy: {accuracy}")

    # Log Metrics & Model
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    # [PENTING] Simpan Run ID ke file txt agar bisa dibaca oleh GitHub Actions
    # untuk keperluan build docker image
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

    print(f"Run ID: {run.info.run_id} saved to run_id.txt")