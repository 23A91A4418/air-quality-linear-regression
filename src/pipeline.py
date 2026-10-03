import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data(data_dir: str):
    """
    Loads raw CSV datasets from data directory.
    """
    pollution_path = os.path.join(data_dir, "pollution.csv")
    weather_path = os.path.join(data_dir, "weather.csv")
    traffic_path = os.path.join(data_dir, "traffic.csv")

    if not (os.path.exists(pollution_path) and os.path.exists(weather_path) and os.path.exists(traffic_path)):
        raise FileNotFoundError(f"One or more required raw CSV files missing in '{data_dir}'")

    df_pollution = pd.read_csv(pollution_path)
    df_weather = pd.read_csv(weather_path)
    df_traffic = pd.read_csv(traffic_path)

    # Ensure timestamp is datetime
    for df in [df_pollution, df_weather, df_traffic]:
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df_pollution, df_weather, df_traffic

def clean_and_merge_data(df_pollution: pd.DataFrame, df_weather: pd.DataFrame, df_traffic: pd.DataFrame) -> pd.DataFrame:
    """
    Merges pollution, weather, and traffic datasets on timestamp via inner join.
    Handles missing values and filters/caps nonsensical values.
    """
    # Inner join on timestamp
    df_merged = df_pollution.merge(df_weather, on="timestamp", how="inner")
    df_merged = df_merged.merge(df_traffic, on="timestamp", how="inner")

    # Sort by timestamp
    df_merged = df_merged.sort_values("timestamp").reset_index(drop=True)

    # Impute missing values (ffill then bfill, fallback to median if any remain)
    df_merged = df_merged.ffill().bfill()
    numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_merged[col].isnull().sum() > 0:
            df_merged[col] = df_merged[col].fillna(df_merged[col].median())

    # Filter/Cap nonsensical data points
    if "humidity" in df_merged.columns:
        df_merged["humidity"] = df_merged["humidity"].clip(lower=0.0, upper=100.0)
    
    if "wind_speed" in df_merged.columns:
        df_merged["wind_speed"] = df_merged["wind_speed"].clip(lower=0.0)

    if "pm25" in df_merged.columns:
        df_merged["pm25"] = df_merged["pm25"].clip(lower=0.0)

    if "traffic_volume_per_hour" in df_merged.columns:
        df_merged["traffic_volume_per_hour"] = df_merged["traffic_volume_per_hour"].clip(lower=0.0)

    return df_merged

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates time-based and interaction features.
    Required features:
    - hour_of_day (int)
    - day_of_week (int)
    - interaction feature (temp_x_wind)
    """
    df = df.copy()
    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"])
        df["hour_of_day"] = timestamps.dt.hour.astype(int)
        df["day_of_week"] = timestamps.dt.dayofweek.astype(int)
    else:
        raise KeyError("'timestamp' column is required for feature engineering.")

    # Interaction feature: Temperature x Wind Speed
    if "temperature" in df.columns and "wind_speed" in df.columns:
        df["temp_x_wind"] = df["temperature"] * df["wind_speed"]
    else:
        raise KeyError("'temperature' and 'wind_speed' columns required for temp_x_wind interaction feature.")

    return df

def run_pipeline(data_dir: str = "data", output_dir: str = "output", test_size: float = 0.2, random_state: int = 42):
    """
    Executes the entire end-to-end Machine Learning pipeline:
    1. Loading raw data
    2. Data cleaning & merging
    3. Feature engineering
    4. Saving processed data
    5. Train/Test splitting & saving splits
    6. Training Linear Regression model & serializing model
    7. Making test set predictions
    8. Evaluating performance metrics (MSE, MAE, R²) & saving metrics
    9. Extracting and saving feature coefficients
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    df_pollution, df_weather, df_traffic = load_data(data_dir)

    # 2. Clean and Merge
    df_merged = clean_and_merge_data(df_pollution, df_weather, df_traffic)

    # 3. Feature Engineering
    df_processed = engineer_features(df_merged)

    # Ensure no null values exist
    if df_processed.isnull().sum().sum() > 0:
        raise ValueError("Processed dataset contains null values after cleaning!")

    # Save output/processed_data.csv
    processed_path = os.path.join(output_dir, "processed_data.csv")
    df_processed.to_csv(processed_path, index=False)
    print(f"[OK] Saved clean merged processed data to '{processed_path}' ({len(df_processed)} rows)")

    # 4. Prepare Features (X) and Target (y)
    target_col = "pm25"
    feature_cols = [c for c in df_processed.columns if c not in [target_col, "timestamp"]]
    
    X = df_processed[feature_cols]
    y = df_processed[target_col]

    # 5. Split Data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Save train/test splits
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    print(f"[OK] Saved data splits: Train size = {len(X_train)}, Test size = {len(X_test)}")

    # 6. Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Serialize model to joblib
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"[OK] Serialized trained model to '{model_path}'")

    # 7. Make Predictions on Test Set
    y_pred = model.predict(X_test)
    df_predictions = pd.DataFrame({"predicted_pm25": y_pred})
    predictions_path = os.path.join(output_dir, "predictions.csv")
    df_predictions.to_csv(predictions_path, index=False)
    print(f"[OK] Saved test predictions to '{predictions_path}' ({len(df_predictions)} rows)")

    # 8. Evaluate Metrics
    mse = float(mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "r2_score": r2,
        "mean_squared_error": mse,
        "mean_absolute_error": mae
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[OK] Saved metrics to '{metrics_path}': R²={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")

    # 9. Extract and Save Coefficients
    df_coefs = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": model.coef_
    })
    coef_path = os.path.join(output_dir, "coefficients.csv")
    df_coefs.to_csv(coef_path, index=False)
    print(f"[OK] Saved feature coefficients to '{coef_path}'")

    print("\n[SUCCESS] Pipeline execution complete! All output artifacts generated successfully.")

if __name__ == "__main__":
    run_pipeline()
