# 🏎️ F1 Telemetry Dashboard

Welcome to the **F1 Telemetry Dashboard**, a powerful interactive tool for analyzing Formula 1 racing data, built with **Streamlit**, **Plotly**, **Pandas**, **FastF1**, and **Scikit-Learn**.

## 🌐 Overview

This dashboard allows you to explore, compare, and analyze F1 telemetry data across multiple seasons and sessions using visualizations and data science techniques.

### 🔧 Features

- 🏁 **Fast Lap Comparison**: Visual speed deltas between drivers
- 🎛️ **Throttle & Brake Analysis**: Compare throttle and brake usage
- ⚙️ **Gear Comparison**: Analyze gear shifting behavior
- 🧠 **Driving Style Clustering**: Use ML to classify driving patterns
- 📍 **Driver Positioning**: PCA projection of driving profiles
- ⏱️ **Real Delta Timing**: Precise time gap over distance
- 🏁 **Race Mode**: Animated path playback on the circuit
- 📏 **Lap Time Evolution**: Monitor lap-by-lap performance
- 📈 **Season Progress**: Track championship points and standings
- 🌦️ **Weather Summary**: Integrated race weather data
- 🔥 **Aggressiveness Score**: Metrics on risky driving behavior
- 📉 **Consistency Analysis**: Technical metrics per lap

## 🚀 Deployment Options

### 📦 Docker

You can run the dashboard locally via Docker:

```bash
docker build -t f1-telemetry .
docker run -p 8501:8501 f1-telemetry
```

### ☁️ Streamlit Cloud

Make sure to:
- Add your `GCS_CREDENTIALS_JSON` and `GCS_BUCKET` to the `secrets` section
- Set Python version and packages in `requirements.txt`

## ☁️ Cloud Integration

Data is loaded dynamically from **Google Cloud Storage (GCS)**.

- Upload your telemetry `.csv` files and driving style model `.npy` files to GCS
- File structure in GCS must match:
```
data/
  season_2023_R/
    01_Bahrain_ALL.csv
    01_Bahrain_WEATHER.csv
    01_Bahrain_WEATHER_SUMMARY.csv
  overall/
    season_2023_R_positions_points.csv
models/
  X_driving_model.npy
  y_driving_model.npy
```

## 🔐 Secrets Configuration

In `.streamlit/secrets.toml`:

```toml
GCS_BUCKET = "your-gcs-bucket-name"

GCS_CREDENTIALS_JSON = """
{
  "type": "service_account",
  ...
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  ...
}
"""
```

## 📁 Project Structure

```
app_telemetry.py           ← Main .py for streamlit
📂 data/                   ← optional local data mirror
📂 models/                 ← driving model folders
📂 scripts/                ← scripts used with ploty to test data and code
requirements.txt
Dockerfile
README.md
```

## 📊 Technologies Used

- Streamlit
- Google Cloud Storage
- Plotly Express & Graph Objects
- Pandas & NumPy
- Scikit-Learn
- Seaborn & Matplotlib

---

🔧 Built with passion for motorsport and data science.