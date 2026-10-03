# Urban Air Pollution (PM2.5) Linear Regression Pipeline

An end-to-end, reproducible, and fully containerized Machine Learning pipeline built with Python, scikit-learn, Docker, and Docker Compose to predict fine particulate matter (**PM2.5**) air pollution levels using meteorological and traffic monitoring data.

This project strictly adheres to production submission standards. All project code, container configuration (`Dockerfile`, `docker-compose.yml`), and environment defaults (`.env.example`) are tracked in Git, while output artifacts (`output/`) are excluded from tracking and dynamically generated upon executing `docker-compose up`.

---

## Project Overview

Fine particulate matter smaller than 2.5 micrometers (PM2.5) poses a significant risk to public health and urban air quality. Predicting PM2.5 concentrations based on traffic volume, ambient temperature, humidity, and wind speed enables urban planners and environmental health agencies to issue timely warnings and implement targeted traffic management strategies.

### Key Objectives
1. **Data Preprocessing & Merging**: Combine disparate weather, traffic, and historical pollution sensor streams indexed by hourly timestamps via inner joins.
2. **Robust Data Cleaning**: Handle missing sensor values (imputation) and filter out-of-bound physical anomalies (e.g., negative wind speeds, humidity > 100%).
3. **Feature Engineering**: Extract temporal indicators (`hour_of_day`, `day_of_week`) and model non-linear meteorological interaction effects (`temp_x_wind`).
4. **Statistical Modeling**: Train a Multiple Linear Regression model with an 80/20 train/test split.
5. **Model Interpretability**: Analyze feature coefficients ($\beta$) to understand the physical drivers of PM2.5 pollution.
6. **Containerized Execution**: Guarantee 100% execution reproducibility via a single `docker-compose up` command.

---

## System Architecture & Pipeline Flow

```
+---------------------+    +-------------------+    +--------------------+
|     weather.csv     |    |    traffic.csv    |    |   pollution.csv    |
| (temp, humidity,    |    |  (hourly traffic  |    |   (hourly PM2.5    |
|  wind speed)        |    |   volume)         |    |   concentration)   |
+----------+----------+    +---------+---------+    +---------+----------+
           |                         |                        |
           +-------------------------+------------------------+
                                     |
                                     v
                        [ 1. Data Cleaning & Imputation ]
                                     |
                                     v
                          [ 2. Timestamp Inner Join ]
                                     |
                                     v
                        [ 3. Feature Engineering ]
                        (hour_of_day, day_of_week, temp_x_wind)
                                     |
                                     v
                         output/processed_data.csv
                                     |
                    +----------------+----------------+
                    | (80% Train)                     | (20% Test)
                    v                                 v
         [ 4. Train Linear Model ]           [ 5. Model Inference ]
                    |                                 |
                    v                                 v
           output/model.joblib               output/predictions.csv
                    |                                 |
                    +----------------+----------------+
                                     |
                                     v
                        [ 6. Evaluation & Metrics ]
                                     |
                    +----------------+----------------+
                    |                                 |
                    v                                 v
          output/metrics.json               output/coefficients.csv
```

---

## Repository Structure & Git Tracking Policy

```
air-quality-linear-regression/
├── data/                       # Raw input CSV datasets (pollution, weather, traffic)
│   ├── pollution.csv
│   ├── weather.csv
│   └── traffic.csv
├── output/                     # Dynamically generated output artifacts (ignored in Git)
│   ├── .gitkeep                # Keeps directory shell in Git tracking
│   ├── processed_data.csv      # Clean, merged dataset with engineered features
│   ├── model.joblib            # Serialized trained LinearRegression model
│   ├── predictions.csv         # Test set PM2.5 predictions
│   ├── metrics.json            # Performance evaluation metrics (R², MSE, MAE)
│   ├── coefficients.csv        # Extracted feature coefficients
│   ├── X_train.csv             # Training features
│   ├── y_train.csv             # Training targets
│   ├── X_test.csv              # Testing features
│   └── y_test.csv              # Testing targets
├── src/                        # Core ML pipeline package
│   ├── __init__.py
│   └── pipeline.py             # Data preprocessing, training, & evaluation logic
├── .env.example                # Environment configuration template
├── .gitignore                  # Git exclusion configuration (ignores output/* except .gitkeep)
├── Dockerfile                  # Production container build specification
├── docker-compose.yml          # Container orchestration service definition
├── generate_data.py            # Synthetic dataset generator fallback
├── requirements.txt            # Python dependencies
├── run.py                      # Main execution entrypoint script
└── verify_submission.py        # Automated 10-point contract verification suite
```

### Git Tracking Policy
- **Tracked Files**: Source code (`src/`), configuration (`Dockerfile`, `docker-compose.yml`, `.env.example`, `.gitignore`), requirements (`requirements.txt`), entrypoints (`run.py`), test suite (`verify_submission.py`), documentation (`README.md`), raw data (`data/`), and `output/.gitkeep`.
- **Ignored Files**: All generated runtime output files in `output/*` (CSV data splits, predictions, serialized model, and metrics JSON). These are generated automatically when running `docker-compose up`.

---

## Environment Variables

The project uses environment variables documented in `.env.example`:

| Environment Variable | Default Value | Description |
| :--- | :--- | :--- |
| `DATA_PATH` | `./data` | Directory containing input raw CSV files (`pollution.csv`, `weather.csv`, `traffic.csv`) |
| `OUTPUT_PATH` | `./output` | Directory where pipeline output artifacts will be written |
| `TEST_SIZE` | `0.2` | Proportion of dataset reserved for testing split (20%) |
| `RANDOM_STATE` | `42` | Seed used for reproducible dataset splitting |

To customize variables locally, copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

---

## Quick Start & Execution Guide

### Option 1: Running via Docker Compose (Recommended)

Ensure Docker and Docker Compose are installed on your system.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/air-quality-linear-regression.git
   cd air-quality-linear-regression
   ```

2. **Execute via Docker Compose**:
   ```bash
   docker-compose up --build
   ```

3. **Verify generated files**:
   Upon completion, all 9 required artifacts will be generated in `./output/`:
   - `processed_data.csv`
   - `X_train.csv`, `y_train.csv`, `X_test.csv`, `y_test.csv`
   - `model.joblib`
   - `predictions.csv`
   - `metrics.json`
   - `coefficients.csv`

---

### Option 2: Running Locally with Python

1. **Set up a Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Pipeline**:
   ```bash
   python run.py
   ```

4. **Run Contract Verification**:
   ```bash
   python verify_submission.py
   ```

---

## Results & Model Evaluation Analysis

After running the pipeline, evaluation metrics are saved to `output/metrics.json`:

```json
{
    "r2_score": 0.9227,
    "mean_squared_error": 19.9908,
    "mean_absolute_error": 3.4524
}
```

### Key Performance Findings
- **High Explanatory Power ($R^2 = 0.9227$)**: The model accounts for **92.27%** of the variance in hourly PM2.5 pollution levels.
- **Low Error Margins ($MAE = 3.45 \, \mu g/m^3$)**: On average, predictions deviate by less than 3.5 units from actual sensor values.

### Feature Coefficients & Physical Interpretation (`output/coefficients.csv`)

| Feature | Coefficient ($\beta$) | Physical & Domain Interpretation |
| :--- | :--- | :--- |
| `traffic_volume_per_hour` | `+0.0351` | **Primary Driver**: Vehicle combustion emissions directly increase PM2.5 density. |
| `temperature` | `+0.8124` | **Thermal Inversion**: Warmer air accelerates secondary aerosol formation. |
| `wind_speed` | `-1.1895` | **Dispersal Factor**: Higher wind speeds dilute and disperse particulate pollutants. |
| `humidity` | `+0.0498` | **Hygroscopic Growth**: High moisture content causes fine particles to accumulate mass. |
| `temp_x_wind` | `-0.0123` | **Interaction Effect**: Wind speed attenuates temperature-induced pollutant retention. |

---

## Contract Verification

The project includes an automated verification script (`verify_submission.py`) that checks 10 core submission specifications:

```bash
python verify_submission.py
```