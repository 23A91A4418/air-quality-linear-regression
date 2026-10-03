import os
import pandas as pd
import numpy as np

def generate_synthetic_data(output_dir="data", num_hours=1500, random_seed=42):
    """
    Generates synthetic environmental and traffic datasets with realistic hourly patterns,
    seasonal noise, missing values (NaNs), and out-of-bound edge cases for testing pipeline robustness.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(random_seed)

    start_date = pd.Timestamp("2025-01-01 00:00:00")
    timestamps = [start_date + pd.Timedelta(hours=i) for i in range(num_hours)]
    
    # 1. Weather Data Generation
    # Temperature: 5°C to 30°C with diurnal cycle
    hours = np.array([ts.hour for ts in timestamps])
    days = np.array([(ts - start_date).days for ts in timestamps])
    
    temp_base = 15 + 10 * np.sin((hours - 8) * np.pi / 12) + np.sin(days * np.pi / 30) * 3
    temp_noise = np.random.normal(0, 2.0, num_hours)
    temperature = np.round(temp_base + temp_noise, 2)
    
    # Humidity: 30% to 95% inversely correlated with temperature
    humidity_base = 80 - 1.5 * (temperature - 15)
    humidity_noise = np.random.normal(0, 5.0, num_hours)
    humidity = np.round(humidity_base + humidity_noise, 2)
    
    # Inject out-of-bound humidity (e.g. > 100%) to test filtering
    humidity[15] = 105.4
    humidity[120] = 102.1

    # Wind speed: 0 to 15 m/s
    wind_base = 5 + 3 * np.cos(hours * np.pi / 12)
    wind_noise = np.random.normal(0, 1.5, num_hours)
    wind_speed = np.round(np.maximum(0, wind_base + wind_noise), 2)
    
    # Inject negative wind speed to test filtering
    wind_speed[45] = -2.5

    # 2. Traffic Data Generation
    # Traffic volume per hour: morning (8am) and evening (6pm) rush hour peaks
    traffic_peak_morning = np.exp(-((hours - 8) ** 2) / 4) * 800
    traffic_peak_evening = np.exp(-((hours - 18) ** 2) / 6) * 950
    traffic_base = 200 + traffic_peak_morning + traffic_peak_evening
    traffic_noise = np.random.normal(0, 50, num_hours)
    traffic_volume = np.round(np.maximum(50, traffic_base + traffic_noise)).astype(float)
    
    # 3. PM2.5 Pollution Data Generation
    # PM2.5 influenced by traffic (+), temperature (+), wind_speed (-), and random noise
    pm25_base = (
        15.0 
        + 0.035 * traffic_volume 
        + 0.8 * temperature 
        - 1.2 * wind_speed 
        + 0.05 * humidity
    )
    pm25_noise = np.random.normal(0, 4.0, num_hours)
    pm25 = np.round(np.maximum(5.0, pm25_base + pm25_noise), 2)

    # 4. Inject Missing Values (NaNs) across datasets
    # Pollution NaNs
    pm25[np.random.choice(num_hours, size=20, replace=False)] = np.nan
    # Weather NaNs
    temperature[np.random.choice(num_hours, size=15, replace=False)] = np.nan
    humidity[np.random.choice(num_hours, size=15, replace=False)] = np.nan
    wind_speed[np.random.choice(num_hours, size=15, replace=False)] = np.nan
    # Traffic NaNs
    traffic_volume[np.random.choice(num_hours, size=15, replace=False)] = np.nan

    # Create DataFrames
    df_pollution = pd.DataFrame({
        "timestamp": timestamps,
        "pm25": pm25
    })
    
    df_weather = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed
    })

    df_traffic = pd.DataFrame({
        "timestamp": timestamps,
        "traffic_volume_per_hour": traffic_volume
    })

    # Save to CSV files
    pollution_path = os.path.join(output_dir, "pollution.csv")
    weather_path = os.path.join(output_dir, "weather.csv")
    traffic_path = os.path.join(output_dir, "traffic.csv")

    df_pollution.to_csv(pollution_path, index=False)
    df_weather.to_csv(weather_path, index=False)
    df_traffic.to_csv(traffic_path, index=False)

    print(f"Generated synthetic datasets in '{output_dir}/':")
    print(f" - pollution.csv: {len(df_pollution)} rows")
    print(f" - weather.csv: {len(df_weather)} rows")
    print(f" - traffic.csv: {len(df_traffic)} rows")

if __name__ == "__main__":
    generate_synthetic_data()
