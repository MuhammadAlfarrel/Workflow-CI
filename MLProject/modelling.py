import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import numpy as np
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Menentukan path file dataset
    # Jika argumen ke-3 tidak ada, default ke diabetes_clean.csv di folder yang sama
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(script_dir, "diabetes_clean.csv")
    
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)

    # PENTING: Ganti "Outcome" dengan nama kolom target di diabetes_clean.csv kamu
    target_column = "Outcome" 

    if target_column not in data.columns:
        print(f"Error: Kolom target '{target_column}' tidak ditemukan di dataset.")
        sys.exit(1)

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )

    # Input example untuk signature model
    input_example = X_train.iloc[0:5]

    # Mengambil parameter dari argumen (jika ada) atau default
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    print(f"Training dengan n_estimators={n_estimators}, max_depth={max_depth}")

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        # Logging Model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        # Logging Metrics
        accuracy = model.score(X_test, y_test)
        print(f"Accuracy: {accuracy}")
        mlflow.log_metric("accuracy", accuracy)