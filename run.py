import os
import sys
from dotenv import load_dotenv

# Load environment variables if .env exists
load_dotenv()

from generate_data import generate_synthetic_data
from src.pipeline import run_pipeline

def main():
    """
    Main entry point for executing the air quality prediction pipeline.
    """
    data_path = os.getenv("DATA_PATH", "data")
    output_path = os.getenv("OUTPUT_PATH", "output")
    test_size = float(os.getenv("TEST_SIZE", "0.2"))
    random_state = int(os.getenv("RANDOM_STATE", "42"))

    # Ensure data directory exists and has input files; generate if missing
    pollution_file = os.path.join(data_path, "pollution.csv")
    weather_file = os.path.join(data_path, "weather.csv")
    traffic_file = os.path.join(data_path, "traffic.csv")

    if not (os.path.exists(pollution_file) and os.path.exists(weather_file) and os.path.exists(traffic_file)):
        print(f"[!] Raw dataset files not found in '{data_path}'. Generating synthetic data...")
        generate_synthetic_data(output_dir=data_path, random_seed=random_state)

    print(f"[*] Starting Machine Learning Pipeline...")
    print(f" - Data Directory:   {data_path}")
    print(f" - Output Directory: {output_path}")
    print(f" - Test Split Ratio: {test_size}")
    print(f" - Random State:     {random_state}\n")

    run_pipeline(
        data_dir=data_path,
        output_dir=output_path,
        test_size=test_size,
        random_state=random_state
    )

if __name__ == "__main__":
    main()
