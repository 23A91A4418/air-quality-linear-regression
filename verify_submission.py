import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def verify_all_contracts():
    print("=== Starting Contract Verification ===")
    
    # Contract 1: Docker files
    assert os.path.exists("docker-compose.yml"), "docker-compose.yml missing"
    assert os.path.exists("Dockerfile"), "Dockerfile missing"
    print("[OK] Contract 1 Passed: docker-compose.yml and Dockerfile exist")

    # Contract 2: .env.example
    assert os.path.exists(".env.example"), ".env.example missing"
    with open(".env.example") as f:
        env_content = f.read()
    assert "DATA_PATH" in env_content and "OUTPUT_PATH" in env_content, ".env.example missing required variables"
    print("[OK] Contract 2 Passed: .env.example contains DATA_PATH and OUTPUT_PATH")

    # Contract 3: Executable run.py
    assert os.path.exists("run.py"), "run.py missing"
    print("[OK] Contract 3 Passed: run.py script exists")

    # Contract 4 & 5: processed_data.csv
    proc_path = os.path.join("output", "processed_data.csv")
    assert os.path.exists(proc_path), "processed_data.csv missing"
    df_proc = pd.read_csv(proc_path)
    
    # Check nulls
    assert df_proc.isnull().sum().sum() == 0, "processed_data.csv contains missing values!"
    
    # Check required original columns
    orig_cols = ["pm25", "temperature", "humidity", "wind_speed", "traffic_volume_per_hour", "timestamp"]
    for col in orig_cols:
        assert col in df_proc.columns, f"Column '{col}' missing from processed_data.csv"
        
    # Check required engineered columns
    eng_cols = ["hour_of_day", "day_of_week", "temp_x_wind"]
    for col in eng_cols:
        assert col in df_proc.columns, f"Engineered column '{col}' missing from processed_data.csv"
    
    assert pd.api.types.is_integer_dtype(df_proc["hour_of_day"]), "hour_of_day is not integer type"
    assert pd.api.types.is_integer_dtype(df_proc["day_of_week"]), "day_of_week is not integer type"
    print(f"[OK] Contracts 4 & 5 Passed: processed_data.csv schema ({df_proc.shape}) valid, clean & feature engineered")

    # Contract 6: Model serialization
    model_path = os.path.join("output", "model.joblib")
    assert os.path.exists(model_path), "model.joblib missing"
    model = joblib.load(model_path)
    assert isinstance(model, LinearRegression), f"Loaded model is type {type(model)}, expected LinearRegression"
    print("[OK] Contract 6 Passed: model.joblib is LinearRegression instance")

    # Contract 7: Predictions
    pred_path = os.path.join("output", "predictions.csv")
    assert os.path.exists(pred_path), "predictions.csv missing"
    df_pred = pd.read_csv(pred_path)
    assert list(df_pred.columns) == ["predicted_pm25"], f"Predictions header mismatch: {df_pred.columns}"
    assert pd.api.types.is_numeric_dtype(df_pred["predicted_pm25"]), "predicted_pm25 is non-numeric"
    print(f"[OK] Contract 7 Passed: predictions.csv contains {len(df_pred)} numeric predictions")

    # Contract 8: Metrics JSON
    metrics_path = os.path.join("output", "metrics.json")
    assert os.path.exists(metrics_path), "metrics.json missing"
    with open(metrics_path) as f:
        metrics = json.load(f)
    required_keys = ["r2_score", "mean_squared_error", "mean_absolute_error"]
    for k in required_keys:
        assert k in metrics, f"Metric key '{k}' missing from metrics.json"
        assert isinstance(metrics[k], (float, int)), f"Metric value for '{k}' is not numeric"
    print(f"[OK] Contract 8 Passed: metrics.json valid -> {metrics}")

    # Contract 9: Coefficients
    coef_path = os.path.join("output", "coefficients.csv")
    assert os.path.exists(coef_path), "coefficients.csv missing"
    df_coef = pd.read_csv(coef_path)
    assert list(df_coef.columns) == ["feature", "coefficient"], f"Coefficients header mismatch: {df_coef.columns}"
    expected_num_features = len([c for c in df_proc.columns if c not in ["pm25", "timestamp"]])
    assert len(df_coef) == expected_num_features, f"Coefficients row count mismatch ({len(df_coef)} vs {expected_num_features})"
    print(f"[OK] Contract 9 Passed: coefficients.csv valid with {len(df_coef)} feature coefficients")

    # Contract 10: Train/Test splits
    x_train_p = os.path.join("output", "X_train.csv")
    y_train_p = os.path.join("output", "y_train.csv")
    x_test_p = os.path.join("output", "X_test.csv")
    y_test_p = os.path.join("output", "y_test.csv")

    assert os.path.exists(x_train_p) and os.path.exists(y_train_p), "Train split files missing"
    assert os.path.exists(x_test_p) and os.path.exists(y_test_p), "Test split files missing"

    X_train = pd.read_csv(x_train_p)
    y_train = pd.read_csv(y_train_p)
    X_test = pd.read_csv(x_test_p)
    y_test = pd.read_csv(y_test_p)

    assert len(X_train) == len(y_train), f"Train X ({len(X_train)}) and y ({len(y_train)}) length mismatch"
    assert len(X_test) == len(y_test), f"Test X ({len(X_test)}) and y ({len(y_test)}) length mismatch"
    assert len(df_pred) == len(X_test), f"Predictions ({len(df_pred)}) and test set ({len(X_test)}) length mismatch"

    test_ratio = len(X_test) / len(df_proc)
    assert 0.18 <= test_ratio <= 0.22, f"Test ratio {test_ratio:.2f} is not approximately 20%"
    print(f"[OK] Contract 10 Passed: Train/test split files valid (Train: {len(X_train)}, Test: {len(X_test)}, Split ratio: {test_ratio:.2%})")

    print("\n=============================================")
    print("ALL 10 CORE CONTRACT SPECIFICATIONS VERIFIED!")
    print("=============================================")

if __name__ == "__main__":
    verify_all_contracts()
