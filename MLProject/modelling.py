import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv("diabetes_clean.csv")

# 2. Preprocessing
# Target kolom Anda adalah 'Diabetes_binary'
target_col = 'Diabetes_binary'

print(f"Menggunakan kolom target: {target_col}")

# Pisahkan Fitur (X) dan Target (y)
X = df.drop(target_col, axis=1)
y = df[target_col]

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. MLflow Tracking
# Set nama eksperimen (opsional, tapi rapi)
mlflow.set_experiment("Diabetes_CI_Experiment")

with mlflow.start_run() as run:
    print("Training model...")
    # Train Model
    model = LogisticRegression(max_iter=1000) # Tambah max_iter agar konvergensi aman
    model.fit(X_train, y_train)

    # Predict & Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model Accuracy: {accuracy}")

    # Log Metrics & Model ke MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    # [PENTING] Simpan Run ID ke file txt
    # File ini akan dibaca oleh GitHub Actions untuk build Docker Image
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

    print(f"Run ID: {run.info.run_id} berhasil disimpan ke run_id.txt")